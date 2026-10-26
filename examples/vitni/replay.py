"""Independently recompute benchmark requests in reverse order in a new process.

Saved data supplies input token IDs and the expected fingerprint, never outputs
to the engine. Every compared answer is freshly computed from the real weights.
"""

import argparse
import json
from pathlib import Path

from run_comparison import MODEL_SHA256, LlamaBaseline

from lm_eval.models.vitni import VitniLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["vitni", "llama"], required=True)
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output path")
    root = Path(__file__).resolve().parent / "results"
    # The exact two documents whose baseline decisions changed across devices.
    indices = [63, 62, 37, 36]
    source = root / f"{args.engine}-{args.backend}" / "requests.jsonl"
    expected = [json.loads(line) for line in source.read_text().splitlines()]
    common = {
        "pretrained": args.model,
        "binary": args.binary.resolve(),
        "backend": args.backend,
        "max_length": 512,
        "audit_path": str(args.output) + ".requests.jsonl",
    }
    lm = (
        VitniLM(**common, model_sha256=MODEL_SHA256)
        if args.engine == "vitni"
        else LlamaBaseline(**common)
    )
    checks = []
    try:
        for index in indices:
            previous = expected[index]
            cut = previous["context_len"]
            lm.score_tokens(previous["tokens"][:cut], previous["tokens"][cut:])
            actual = lm.records[-1]
            checks.append(
                {
                    "index": index,
                    "trace_equal": actual["trace_blake3"] == previous["trace_blake3"],
                    "score_bits_equal": actual["logprob_bits"]
                    == previous["logprob_bits"],
                    "forward_steps": actual["forward_steps"],
                    "seconds": actual["seconds"],
                }
            )
        result = {
            "engine": args.engine,
            "backend": args.backend,
            "checks": checks,
            "all_equal": all(
                c["trace_equal"] and c["score_bits_equal"] for c in checks
            ),
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        if not result["all_equal"]:
            raise AssertionError("fresh-process replay mismatch")
    finally:
        lm.close()


if __name__ == "__main__":
    main()
