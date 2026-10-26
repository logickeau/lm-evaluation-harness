//! Resident inference process. Every score request starts with an empty KV cache.
//! JSON-lines stdout is reserved for the protocol; errors are returned explicitly.
use serde_json::{json, Value};
use std::{
    fs,
    io::{self, BufRead, Write},
    time::Instant,
};
use vitni_tensor::{
    model::{
        config::Config, forward::RunState, forward_quantized, gguf::GgufFile,
        quant_weights::QuantizedWeights,
    },
    storage::Storage,
};

type R<T> = Result<T, String>;

fn number(v: &Value, key: &str) -> R<usize> {
    v[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(format!("invalid {key}"))
}
fn tokens(v: &Value) -> R<Vec<u32>> {
    v["tokens"]
        .as_array()
        .ok_or("tokens must be an array")?
        .iter()
        .map(|n| {
            n.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("invalid token".into())
        })
        .collect()
}

// One defined normalization for BOTH engines. f64 ordered accumulation, pinned libm.
// Ties choose the lowest token id; all nonfinite logits fail closed.
fn logprob(row: &[f32], target: u32) -> R<(f64, usize)> {
    if row.is_empty() || target as usize >= row.len() || row.iter().any(|x| !x.is_finite()) {
        return Err("invalid/nonfinite logits or target".into());
    }
    let mut top = 0;
    for i in 1..row.len() {
        if row[i] > row[top] {
            top = i;
        }
    }
    let max = row[top] as f64;
    let mut sum = 0.0_f64;
    for &x in row {
        sum += libm::exp(x as f64 - max);
    }
    Ok((row[target as usize] as f64 - max - libm::log(sum), top))
}

#[derive(Default)]
struct Trace {
    hash: blake3::Hasher,
    rows: Vec<String>,
    target_logprobs: Vec<f64>,
    argmax: Vec<usize>,
    sum: f64,
    greedy: bool,
}
impl Trace {
    fn new() -> Self {
        Self {
            greedy: true,
            ..Self::default()
        }
    }
    fn row(&mut self, row: &[f32], target: Option<u32>) -> R<()> {
        if row.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite logits".into());
        }
        let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.hash.update(&bytes);
        self.rows.push(blake3::hash(&bytes).to_hex().to_string());
        if let Some(target) = target {
            let (lp, top) = logprob(row, target)?;
            self.sum += lp;
            self.target_logprobs.push(lp);
            self.argmax.push(top);
            self.greedy &= top == target as usize;
        }
        Ok(())
    }
    fn finish(&self) -> Value {
        json!({"logprob": self.sum, "logprob_bits": format!("{:016x}", self.sum.to_bits()),
            "greedy": self.greedy, "trace_blake3": self.hash.finalize().to_hex().to_string(),
            "row_blake3": self.rows, "target_logprobs": self.target_logprobs,
            "target_logprob_bits": self.target_logprobs.iter().map(|x| format!("{:016x}",x.to_bits())).collect::<Vec<_>>(),
            "argmax": self.argmax, "forward_steps": self.rows.len()})
    }
}

fn normalize(v: &Value) -> R<Value> {
    let ids = tokens(v)?;
    let ctx = number(v, "context_len")?;
    let vocab = number(v, "vocab_size")?;
    if ctx == 0 || ctx > ids.len() || ids.iter().any(|&id| id as usize >= vocab) {
        return Err("invalid context or token".into());
    }
    let data = fs::read(v["file"].as_str().ok_or("file required")?).map_err(|e| e.to_string())?;
    let steps = if ctx == ids.len() { 0 } else { ids.len() - 1 };
    if vocab == 0 || data.len() != steps * vocab * 4 {
        return Err("wrong logit file length".into());
    }
    let mut trace = Trace::new();
    for (pos, bytes) in data.chunks_exact(vocab * 4).enumerate() {
        let row: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        trace.row(
            &row,
            if pos + 1 >= ctx {
                Some(ids[pos + 1])
            } else {
                None
            },
        )?;
    }
    Ok(trace.finish())
}

fn serve(mut f: impl FnMut(&Value) -> R<Value>) -> R<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let result = line
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string()))
            .and_then(|v| f(&v));
        let out = match result {
            Ok(v) => v,
            Err(e) => json!({"error": e}),
        };
        writeln!(stdout, "{out}").map_err(|e| e.to_string())?;
        stdout.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn run() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--normalizer") {
        return serve(normalize);
    }
    if args.len() != 4 {
        return Err("usage: vitni-lm-eval MODEL cpu|metal MAX_LENGTH, or --normalizer".into());
    }
    let start = Instant::now();
    let blob = fs::read(&args[1]).map_err(|e| e.to_string())?;
    let gguf = GgufFile::parse(&blob).map_err(|e| format!("{e:?}"))?;
    if gguf
        .metadata_str("general.architecture")
        .map_err(|e| format!("{e:?}"))?
        != "llama"
    {
        return Err("only Llama GGUF supported by this prototype".into());
    }
    let mut cfg = Config::from_gguf(&gguf).map_err(|e| format!("{e:?}"))?;
    let max: usize = args[3].parse().map_err(|_| "invalid max length")?;
    if max == 0 || max > cfg.seq_len {
        return Err("max length exceeds model context".into());
    }
    cfg.seq_len = max;
    let w = QuantizedWeights::from_gguf(&gguf, &cfg).map_err(|e| format!("{e:?}"))?;
    let backend = args[2].as_str();
    if !matches!(backend, "cpu" | "metal") {
        return Err("unknown backend".into());
    }
    #[cfg(not(feature = "metal"))]
    if backend == "metal" {
        return Err("Metal not compiled".into());
    }
    #[cfg(feature = "metal")]
    let mut metal = if backend == "metal" {
        // The pinned upstream has an unknown-dtype fallback; reject it here.
        let supported = |t: &vitni_tensor::model::quant_weights::QuantTensor| {
            let dtype = format!("{:?}", t.dtype);
            let block = if matches!(dtype.as_str(), "Q4_K" | "Q6_K") {
                256
            } else {
                32
            };
            matches!(dtype.as_str(), "Q4_K" | "Q6_K" | "Q8_0" | "Q5_0")
                && t.shape[0] as usize % block == 0
        };
        if !matches!(
            format!("{:?}", w.token_embedding_table.dtype).as_str(),
            "F32" | "Q4_0" | "Q8_0" | "Q5_0" | "Q4_K" | "Q6_K"
        ) || !supported(w.wcls.as_ref().unwrap_or(&w.token_embedding_table))
            || w.layers.iter().any(|l| {
                [&l.wq, &l.wk, &l.wv, &l.wo, &l.w1, &l.w2, &l.w3]
                    .iter()
                    .any(|t| !supported(t))
            })
        {
            return Err("unsupported Metal model dtype/width".into());
        }
        Some(vitni_tensor_metal::MetalForward::new(&cfg, &w).map_err(|e| e.to_string())?)
    } else {
        None
    };
    println!(
        "{}",
        json!({"ready":true,"backend":backend,"vocab_size":cfg.vocab_size,"max_length":max,
        "model_blake3":blake3::hash(&blob).to_hex().to_string(),"load_seconds":start.elapsed().as_secs_f64()})
    );
    io::stdout().flush().map_err(|e| e.to_string())?;
    serve(|v| {
        let ids = tokens(v)?;
        let ctx = number(v, "context_len")?;
        if ctx == 0 || ctx > ids.len() || ids.iter().any(|&n| n as usize >= cfg.vocab_size) {
            return Err("invalid context or token".into());
        }
        let steps = if ctx == ids.len() { 0 } else { ids.len() - 1 };
        if steps > max {
            return Err("request exceeds max length; no implicit truncation".into());
        }
        let start = Instant::now();
        let mut cpu = if backend == "cpu" {
            Some(RunState::new(&cfg))
        } else {
            None
        };
        #[cfg(feature = "metal")]
        if let Some(m) = &mut metal {
            m.reset();
        }
        let mut trace = Trace::new();
        for pos in 0..steps {
            let row: Vec<f32> = if let Some(state) = &mut cpu {
                let tensor = forward_quantized::step(&cfg, &w, state, ids[pos], pos)
                    .map_err(|e| format!("{e:?}"))?;
                let Storage::Cpu(s) = tensor.storage() else {
                    return Err("non-CPU output".into());
                };
                s.as_f32_slice().to_vec()
            } else {
                #[cfg(feature = "metal")]
                {
                    metal.as_mut().unwrap().step(ids[pos], pos)
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err("unbuilt Metal backend".into());
                }
            };
            trace.row(
                &row,
                if pos + 1 >= ctx {
                    Some(ids[pos + 1])
                } else {
                    None
                },
            )?;
        }
        let mut out = trace.finish();
        out["seconds"] = json!(start.elapsed().as_secs_f64());
        Ok(out)
    })
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_and_tie_contract() {
        let (lp, top) = logprob(&[0.0, 0.0], 1).unwrap();
        assert_eq!(top, 0);
        assert!((lp + std::f64::consts::LN_2).abs() < 1e-15);
        assert!(logprob(&[f32::NAN], 0).is_err());
        assert!(logprob(&[0.0], 1).is_err());
    }
}
