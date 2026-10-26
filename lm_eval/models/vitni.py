"""Experimental deterministic Llama GGUF likelihood backend.

Inference runs in the pinned Vitni Rust engine. llama.cpp is used ONLY as a
vocabulary reader/tokenizer, so comparisons can hold token ids fixed. No result
cache, prefix reuse, sampling, implicit truncation, or chat template is applied.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


class JsonProcess:
    def __init__(self, command, *, handshake=False):
        self.process = subprocess.Popen(  # noqa: S603 - caller selects the binary; argv has no shell
            [str(x) for x in command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.ready = self._read() if handshake else None

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"native process exited: {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def call(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self._read()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()


@register_model("vitni")
class VitniLM(LM):
    def __init__(
        self,
        pretrained,
        binary,
        backend="cpu",
        max_length=512,
        model_sha256=None,
        audit_path=None,
        batch_size=1,
        max_batch_size=None,
        device=None,
    ):
        super().__init__()
        if backend not in ("cpu", "metal"):
            raise ValueError("backend must be cpu or metal")
        if str(batch_size) != "1" or max_batch_size not in (None, 1):
            raise ValueError("prototype requires batch_size=1")
        if device is not None and str(device) != backend:
            raise ValueError("device must match the selected backend")
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        self.pretrained = str(Path(pretrained).resolve())
        with open(self.pretrained, "rb") as model:
            self.model_sha256 = hashlib.file_digest(model, "sha256").hexdigest()
        if model_sha256 is not None and model_sha256 != self.model_sha256:
            raise ValueError("model SHA256 mismatch")
        from llama_cpp import Llama

        self._vocab = Llama(model_path=self.pretrained, vocab_only=True, verbose=False)
        self._native = JsonProcess(
            [binary, self.pretrained, backend, self.max_length], handshake=True
        )
        self.info = self._native.ready
        self.info["model_sha256"] = self.model_sha256
        self.audit_path = Path(audit_path) if audit_path else None
        self.records = []
        self._device = backend

    @property
    def tokenizer_name(self):
        return "gguf-vocab-" + self.model_sha256

    def tok_encode(self, text, add_special_tokens=True):
        return self._vocab.tokenize(
            text.encode("utf-8"), add_bos=bool(add_special_tokens), special=False
        )

    def encode_pair(self, context, continuation):
        if not continuation:
            return self.tok_encode(context) or [self._vocab.token_bos()], []
        if not context:
            return [self._vocab.token_bos()], self.tok_encode(continuation, False)
        spaces = len(context) - len(context.rstrip(" "))
        if spaces:
            continuation = context[-spaces:] + continuation
            context = context[:-spaces]
        ctx = self.tok_encode(context)
        whole = self.tok_encode(context + continuation)
        # Token-boundary ambiguity must be explicit, never score unrelated tokens.
        if whole[: len(ctx)] != ctx:
            raise ValueError("context is not a token prefix of the full input")
        return ctx, whole[len(ctx) :]

    def score_tokens(self, context, continuation):
        ids = list(context) + list(continuation)
        if not context:
            raise ValueError("at least one context token required")
        if continuation and len(ids) - 1 > self.max_length:
            raise ValueError("request exceeds max_length; no implicit truncation")
        result = self._native.call({"tokens": ids, "context_len": len(context)})
        record = {"tokens": ids, "context_len": len(context), **result}
        self.records.append(record)
        if self.audit_path:
            with self.audit_path.open("a") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        return result["logprob"], result["greedy"]

    def loglikelihood(self, requests, **kwargs):
        return [self.score_tokens(*self.encode_pair(*req.args)) for req in requests]

    def loglikelihood_rolling(self, requests, **kwargs):
        results = []
        for req in requests:
            windows = get_rolling_token_windows(
                self.tok_encode(req.args[0], False),
                self._vocab.token_bos(),
                self.max_length,
                context_len=1,
            )
            total = 0.0
            for window in windows:
                total += self.score_tokens(*make_disjoint_window(window))[0]
            results.append(total)
        return results

    def generate_until(self, requests, **kwargs):
        raise NotImplementedError("prototype supports likelihood tasks only")

    def close(self):
        self._native.close()
        self._vocab.close()
