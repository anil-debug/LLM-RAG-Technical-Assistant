"""Local generation with PyTorch and Transformers.

``model.eval()`` turns off dropout. ``torch.inference_mode()`` turns off
autograd, which is the right mode for serving: the forward pass does not
store activations for a backward pass. Decoding itself is ``generate``, which
pre-fills the prompt once and then reuses the key/value cache for each new
token.
"""

import time

from core.device import resolve_device
from core.errors import ModelUnavailableError
from generation.provider import GenerationResult

_DTYPES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}


class TransformersLocalProvider:
    """Load a causal LM and generate from chat messages."""

    def __init__(self, model_name: str, device: str = "cpu", dtype: str = "float32") -> None:
        if not model_name:
            raise ModelUnavailableError(
                "TRANSFORMERS_MODEL is empty. Set it to a Hugging Face causal LM id."
            )
        if dtype not in _DTYPES:
            raise ModelUnavailableError(f"Unsupported dtype '{dtype}'.")
        self.model_name = model_name
        self.device_request = device
        self.dtype_name = dtype
        self._model = None
        self._tokenizer = None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        import torch

        model, tokenizer = self._load()
        prompt = self._render(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
        do_sample = temperature > 0
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_length = int(inputs["input_ids"].shape[1])
        new_tokens = output[0][prompt_length:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_length,
            completion_tokens=int(new_tokens.shape[0]),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _render(self, tokenizer, messages: list[dict[str, str]]) -> str:
        template = getattr(tokenizer, "chat_template", None)
        if template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lines = [f"{message['role']}: {message['content']}" for message in messages]
        return "\n".join(lines) + "\nassistant:"

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = resolve_device(self.device_request)
        dtype = getattr(torch, self.dtype_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load Transformers model {self.model_name}: {exc}"
            ) from exc
        model.to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        self._model = model
        return model, tokenizer
