"""Run the real harness, with pinned weights, dataset revision and normalization.

Each invocation runs ONE backend so memory/GPU scheduling do not confound timing.
The llama.cpp comparator uses the exact same tokenizer and scoring protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

from llama_cpp import Llama, llama_print_system_info

from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.vitni import JsonProcess, VitniLM
from lm_eval.tasks._yaml_loader import load_yaml


MODEL_SHA256 = "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0"


class LlamaBaseline(VitniLM):
    """Matched baseline: real CPU or Metal llama.cpp, no saved answer replay."""

    def __init__(self, pretrained, binary, backend, max_length, audit_path, n_batch=1):
        LM.__init__(self)
        self.pretrained = str(Path(pretrained).resolve())
        with open(self.pretrained, "rb") as model:
            self.model_sha256 = hashlib.file_digest(model, "sha256").hexdigest()
        if self.model_sha256 != MODEL_SHA256:
            raise ValueError("model SHA256 mismatch")
        self.max_length = max_length
        start = time.perf_counter()
        self._vocab = Llama(
            model_path=self.pretrained,
            n_gpu_layers=0 if backend == "cpu" else -1,
            n_ctx=max_length,
            n_batch=n_batch,
            n_ubatch=n_batch,
            n_threads=8,
            n_threads_batch=8,
            logits_all=True,
            seed=0,
            flash_attn=False,
            verbose=True,
        )
        self._native = JsonProcess([binary, "--normalizer"])
        self.info = {
            "backend": backend,
            "load_seconds": time.perf_counter() - start,
            "system_info": llama_print_system_info().decode(),
            "n_batch": n_batch,
            "n_threads": 8,
            "flash_attn": False,
            "model_sha256": self.model_sha256,
        }
        self.records = []
        self.audit_path = Path(audit_path)
        self._device = backend

    def score_tokens(self, context, continuation):
        ids = list(context) + list(continuation)
        if not context:
            raise ValueError("context required")
        if continuation and len(ids) - 1 > self.max_length:
            raise ValueError("context overflow")
        start = time.perf_counter()
        self._vocab.reset()
        if continuation:
            self._vocab.eval(ids[:-1])
        steps = len(ids) - 1 if continuation else 0
        # Raw f32 outputs normalized by the SAME Rust code as Vitni, in id order.
        with tempfile.NamedTemporaryFile(suffix=".f32") as raw:
            self._vocab.scores[:steps].astype("<f4").tofile(raw)
            raw.flush()
            result = self._native.call(
                {
                    "file": raw.name,
                    "tokens": ids,
                    "context_len": len(context),
                    "vocab_size": self._vocab.n_vocab(),
                }
            )
        result["seconds"] = time.perf_counter() - start
        record = {"tokens": ids, "context_len": len(context), **result}
        self.records.append(record)
        with self.audit_path.open("a") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        return result["logprob"], result["greedy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["vitni", "llama"], required=True)
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--n-batch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "requests.jsonl").exists():
        raise FileExistsError("use a new output directory, never mix runs")
    proof_root = Path(__file__).resolve().parent
    dataset_info = json.loads((proof_root / "results/dataset-info.json").read_text())
    config = load_yaml(
        proof_root.parent.parent / "lm_eval/tasks/winogrande/default.yaml"
    )
    config["dataset_kwargs"] = {"revision": dataset_info["sha"]}
    task = ConfigurableTask(config=config)
    docs = [task.validation_docs()[i] for i in range(args.samples)]
    (args.output / "documents.json").write_text(
        json.dumps(docs, sort_keys=True, indent=2)
    )
    start = time.perf_counter()
    common = {
        "pretrained": args.model,
        "binary": args.binary.resolve(),
        "backend": args.backend,
        "max_length": args.max_length,
        "audit_path": args.output / "requests.jsonl",
    }
    lm = (
        VitniLM(**common, model_sha256=MODEL_SHA256)
        if args.engine == "vitni"
        else LlamaBaseline(**common, n_batch=args.n_batch)
    )
    load_seconds = time.perf_counter() - start
    print(
        json.dumps(
            {
                "event": "loaded",
                "engine": args.engine,
                "backend": args.backend,
                "info": lm.info,
            }
        ),
        flush=True,
    )
    try:
        start = time.perf_counter()
        result = simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=0,
            samples={"winogrande": list(range(args.samples))},
            bootstrap_iters=0,
            torch_random_seed=None,
            use_cache=None,
            cache_requests=False,
            log_samples=True,
            apply_chat_template=False,
        )
        seconds = time.perf_counter() - start
        forward_steps = sum(r["forward_steps"] for r in lm.records)
        summary = {
            "engine": args.engine,
            "backend": args.backend,
            "info": lm.info,
            "dataset": dataset_info,
            "sample_ids": list(range(args.samples)),
            "requests": len(lm.records),
            "forward_steps": forward_steps,
            "engine_seconds": sum(r["seconds"] for r in lm.records),
            "evaluation_seconds": seconds,
            "load_wall_seconds": load_seconds,
            "results": result["results"],
            "max_length": args.max_length,
            "documents_sha256": hashlib.sha256(
                (args.output / "documents.json").read_bytes()
            ).hexdigest(),
        }
        (args.output / "harness-results.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        lm.close()


if __name__ == "__main__":
    main()
