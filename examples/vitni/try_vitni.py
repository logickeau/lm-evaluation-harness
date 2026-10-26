"""One-command, real-weight replay of the published Vitni evaluation trial.

Requires Python 3.12, Rust/Cargo and native C/C++ build tools. Creates an isolated
environment and downloads approximately 638 MiB of public model weights. Nothing
is uploaded. Recorded benchmark outputs are comparison-only, never engine inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import venv
from pathlib import Path


MODEL_SHA256 = "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0"
MODEL_URL = (
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/"
    "52e7645ba7c309695bec7ac98f4f005b139cf465/"
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)


def run(command, root):
    subprocess.run([str(value) for value in command], cwd=root, check=True)  # noqa: S603 - explicit argv, no shell


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, help="reuse the exact GGUF instead of downloading it"
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "metal", "both"],
        default="both"
        if sys.platform == "darwin" and platform.machine() == "arm64"
        else "cpu",
    )
    parser.add_argument(
        "--use-current-env",
        action="store_true",
        help="skip environment setup; dependencies must already be installed",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("use python3.12 for this pinned trial")
    if not shutil.which("cargo"):
        parser.error(
            "Rust/Cargo is required; install it from https://rustup.rs then rerun"
        )
    if args.backend != "cpu" and (
        sys.platform != "darwin" or platform.machine() != "arm64"
    ):
        parser.error(
            "the Metal trial requires Apple Silicon macOS; use --backend cpu for experimental external replication"
        )
    root = Path(__file__).resolve().parents[2]
    example = root / "examples/vitni"
    work = example / "trial-work"
    work.mkdir(exist_ok=True)
    model = (
        args.model.expanduser().resolve()
        if args.model
        else work / "tinyllama.Q4_K_M.gguf"
    )
    if args.model and not model.is_file():
        parser.error(f"model does not exist: {model}")
    if not model.exists():
        print("Downloading the pinned TinyLlama GGUF (~638 MiB)...", flush=True)
        partial = model.with_suffix(".download")
        with (
            urllib.request.urlopen(MODEL_URL, timeout=60) as source,
            partial.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        if digest(partial) != MODEL_SHA256:
            partial.unlink()
            raise ValueError("downloaded model SHA256 mismatch")
        partial.replace(model)
    if digest(model) != MODEL_SHA256:
        raise ValueError(
            "model SHA256 mismatch; the trial requires the exact published GGUF"
        )
    python = Path(sys.executable)
    if not args.use_current_env:
        environment = work / "venv"
        if not environment.exists():
            print("Creating the isolated trial environment...", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        run(
            [python, "-m", "pip", "install", "-r", example / "requirements-lock.txt"],
            root,
        )
    build = [
        "cargo",
        "build",
        "--locked",
        "--release",
        "--manifest-path",
        example / "native/Cargo.toml",
    ]
    if args.backend != "cpu":
        build.extend(["--features", "metal"])
    run(build, root)
    binary = (
        example
        / "native/target/release"
        / ("vitni-lm-eval.exe" if sys.platform == "win32" else "vitni-lm-eval")
    )
    backends = ["cpu", "metal"] if args.backend == "both" else [args.backend]
    output = Path(tempfile.mkdtemp(prefix="replay-", dir=work))
    for backend in backends:
        print(
            f"Recomputing four real benchmark continuations on Vitni {backend}...",
            flush=True,
        )
        run(
            [
                python,
                example / "replay.py",
                "--engine",
                "vitni",
                "--backend",
                backend,
                "--model",
                model,
                "--binary",
                binary,
                "--output",
                output / f"{backend}.json",
            ],
            root,
        )
    if len(backends) == 2:
        records = [
            [
                json.loads(line)
                for line in (output / f"{b}.json.requests.jsonl")
                .read_text()
                .splitlines()
            ]
            for b in backends
        ]
        for cpu, metal in zip(*records, strict=True):
            if (
                cpu["trace_blake3"] != metal["trace_blake3"]
                or cpu["logprob_bits"] != metal["logprob_bits"]
            ):
                raise AssertionError("CPU/Metal numerical mismatch")
    print(
        f"PASS: fresh inference matched every published fingerprint. Results: {output}",
        flush=True,
    )
    print(
        "Scope: four continuation requests from two Winogrande documents, not the full 32-document trial.",
        flush=True,
    )
    print(
        "The published measurement covers CPU/Metal on one M3 Max; other machines are external replication.",
        flush=True,
    )


if __name__ == "__main__":
    main()
