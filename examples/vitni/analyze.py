"""Audit completed runs and quantify reproducibility, decisions and runtime."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "results"


def load(name):
    path = ROOT / name
    summary = json.loads((path / "summary.json").read_text())
    records = [
        json.loads(s) for s in (path / "requests.jsonl").read_text().splitlines()
    ]
    samples = json.loads((path / "harness-results.json").read_text())["samples"][
        "winogrande"
    ]
    # Independently check that the actual native results determined the task metric.
    assert len(records) == 2 * len(samples) == summary["requests"] == 64
    correct = 0
    for i, sample in enumerate(samples):
        scores = [r["logprob"] for r in records[2 * i : 2 * i + 2]]
        assert scores == [r[0][0] for r in sample["resps"]]
        prediction = 0 if scores[0] >= scores[1] else 1
        correct += prediction == int(sample["doc"]["answer"]) - 1
    assert correct / len(samples) == summary["results"]["winogrande"]["acc,none"]
    assert sum(r["forward_steps"] for r in records) == summary["forward_steps"]
    return summary, records, samples


def compare(a, b):
    sa, ra, samples_a = a
    sb, rb, samples_b = b
    assert len(ra) == len(rb)
    assert sa["documents_sha256"] == sb["documents_sha256"]
    assert sa["info"]["model_sha256"] == sb["info"]["model_sha256"]
    changed_docs = []
    for x, y in zip(samples_a, samples_b, strict=True):
        if x["acc"] != y["acc"]:
            changed_docs.append(
                {
                    "id": x["doc_id"],
                    "document": x["doc"],
                    "a_scores": x["resps"],
                    "b_scores": y["resps"],
                }
            )
    for x, y in zip(ra, rb, strict=True):
        assert x["tokens"] == y["tokens"]
        assert x["context_len"] == y["context_len"]
        assert x["forward_steps"] == y["forward_steps"] == len(x["tokens"]) - 1
    return {
        "requests": len(ra),
        "logit_trace_mismatches": sum(
            x["trace_blake3"] != y["trace_blake3"] for x, y in zip(ra, rb, strict=True)
        ),
        "logit_row_mismatches": sum(
            u != v
            for x, y in zip(ra, rb, strict=True)
            for u, v in zip(x["row_blake3"], y["row_blake3"], strict=True)
        ),
        "likelihood_bit_mismatches": sum(
            x["logprob_bits"] != y["logprob_bits"] for x, y in zip(ra, rb, strict=True)
        ),
        "target_logprob_bit_mismatches": sum(
            u != v
            for x, y in zip(ra, rb, strict=True)
            for u, v in zip(
                x["target_logprob_bits"], y["target_logprob_bits"], strict=True
            )
        ),
        "target_argmax_mismatches": sum(
            u != v
            for x, y in zip(ra, rb, strict=True)
            for u, v in zip(x["argmax"], y["argmax"], strict=True)
        ),
        "max_abs_request_likelihood_delta": max(
            abs(x["logprob"] - y["logprob"]) for x, y in zip(ra, rb, strict=True)
        ),
        "changed_decisions": changed_docs,
    }


def main():
    runs = {
        name: load(name)
        for name in ["vitni-cpu", "vitni-metal", "llama-cpu", "llama-metal"]
    }
    comparisons = {
        "vitni_cpu_vs_metal": compare(runs["vitni-cpu"], runs["vitni-metal"]),
        "llama_cpu_vs_metal": compare(runs["llama-cpu"], runs["llama-metal"]),
        "vitni_vs_llama_cpu": compare(runs["vitni-cpu"], runs["llama-cpu"]),
        "vitni_vs_llama_metal": compare(runs["vitni-metal"], runs["llama-metal"]),
    }
    canonical = comparisons["vitni_cpu_vs_metal"]
    assert (
        canonical["logit_trace_mismatches"]
        == canonical["likelihood_bit_mismatches"]
        == 0
    )
    performance = {
        name: {
            key: value
            for key, value in run[0].items()
            if key
            in [
                "engine_seconds",
                "evaluation_seconds",
                "load_wall_seconds",
                "forward_steps",
                "results",
            ]
        }
        for name, run in runs.items()
    }
    costs = {
        backend: runs[f"vitni-{backend}"][0]["engine_seconds"]
        / runs[f"llama-{backend}"][0]["engine_seconds"]
        for backend in ["cpu", "metal"]
    }
    output = {
        "comparisons": comparisons,
        "performance": performance,
        "prototype_engine_time_ratios": costs,
    }
    (ROOT / "comparison.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
