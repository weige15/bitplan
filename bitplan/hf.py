"""Optional Transformers adapter for the two pinned models in configs/m1.json.

This module is intentionally lazy-imported: the CPU smoke path has no third
party dependency. Quantization here is fake-dequantized native PyTorch weight
rounding, which is suitable for paired numerical instrumentation but not for a
packed inference or systems benchmark.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

from .models import ForwardOutput


def _quantize_torch_model(model: Any, bits: int) -> Any:
    import torch

    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.is_floating_point():
                continue
            values = parameter.detach().float()
            if bits == 16:
                rounded = values.to(dtype=torch.bfloat16).to(dtype=parameter.dtype)
            else:
                qmax = (1 << (bits - 1)) - 1
                scale = values.abs().max() / qmax
                if float(scale) == 0.0:
                    rounded = values
                else:
                    rounded = (values / scale).round().clamp(-qmax, qmax) * scale
                rounded = rounded.to(dtype=parameter.dtype)
            parameter.copy_(rounded)
    return model


class TransformersCausalModel:
    """BoundaryModel adapter for GPT-2 and Qwen2/Llama-style decoder models."""

    def __init__(self, model: Any, *, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.vocab_size = int(model.config.vocab_size)
        self._base = getattr(model, "model", None) or getattr(model, "transformer", None)
        if self._base is None:
            raise ValueError("unsupported Transformers model: no decoder base")
        self._layers = getattr(self._base, "layers", None) or getattr(self._base, "h", None)
        if self._layers is None or len(self._layers) != num_layers:
            raise ValueError("unsupported decoder layer layout")

    def forward(self, token_ids: list[int], boundaries: list[int]) -> ForwardOutput:
        import torch

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) < self.num_layers + 1:
            raise ValueError("model did not return all boundary hidden states")
        captured = {
            boundary: hidden_states[boundary][0].float().cpu().tolist()
            for boundary in boundaries
        }
        return ForwardOutput(logits=output.logits[0, -1].float().cpu().tolist(), hidden_by_boundary=captured)

    def _causal_mask(self, hidden: Any, position_ids: Any) -> Any:
        import torch

        attention_mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        updater = getattr(self._base, "_update_causal_mask", None)
        if updater is not None:
            try:
                return updater(
                    attention_mask,
                    hidden,
                    position_ids[0],
                    None,
                    False,
                )
            except (TypeError, RuntimeError):
                pass
        # GPT-2 blocks accept an additive [batch, heads, query, key] mask.
        length = hidden.shape[1]
        mask = torch.full((1, 1, length, length), torch.finfo(hidden.dtype).min, device=hidden.device)
        return torch.triu(mask, diagonal=1)

    def replay_suffix(self, boundary: int, hidden_states: list[list[float]]) -> list[float]:
        import torch

        if boundary < 0 or boundary > self.num_layers:
            raise ValueError("boundary is outside model depth")
        hidden = torch.tensor([hidden_states], dtype=next(self.model.parameters()).dtype, device=self.model.device)
        position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        attention_mask = self._causal_mask(hidden, position_ids)
        rotary = getattr(self._base, "rotary_emb", None)
        position_embeddings = rotary(hidden, position_ids) if rotary is not None else None
        for layer in self._layers[boundary:]:
            parameters = inspect.signature(layer.forward).parameters
            kwargs: dict[str, Any] = {}
            if "attention_mask" in parameters:
                kwargs["attention_mask"] = attention_mask
            if "position_ids" in parameters:
                kwargs["position_ids"] = position_ids
            if "cache_position" in parameters:
                kwargs["cache_position"] = torch.arange(hidden.shape[1], device=hidden.device)
            if "position_embeddings" in parameters:
                kwargs["position_embeddings"] = position_embeddings
            if "past_key_value" in parameters:
                kwargs["past_key_value"] = None
            if "use_cache" in parameters:
                kwargs["use_cache"] = False
            if "output_attentions" in parameters:
                kwargs["output_attentions"] = False
            result = layer(hidden, **kwargs)
            hidden = result[0] if isinstance(result, tuple) else result
        norm = getattr(self._base, "norm", None) or getattr(self._base, "ln_f", None)
        # Qwen2's final hidden-state boundary is already post-norm in
        # output.hidden_states; earlier boundaries need the final norm here.
        if norm is not None and boundary < self.num_layers:
            hidden = norm(hidden)
        logits = self.model.lm_head(hidden[:, -1, :])
        return logits[0].float().detach().cpu().tolist()


def load_transformers_conditions(
    config: dict[str, Any], model_key: str, devices: list[str] | None = None
) -> tuple[Any, dict[str, TransformersCausalModel]]:
    """Load one pinned model per condition on an explicit device assignment."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised only with optional backend
        raise RuntimeError(
            "Transformers execution requires the exact optional environment in environment/m1-lock.json"
        ) from error

    spec = config["models"][model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["name"], revision=spec["tokenizer_revision"], use_fast=True
    )
    condition_count = len(config["quantization"]["conditions"])
    if devices is None:
        devices = ["cuda:0" if torch.cuda.is_available() else "cpu"] * condition_count
    if len(devices) != condition_count:
        raise ValueError(f"exactly {condition_count} devices are required, got {len(devices)}")
    if any(device.startswith("cuda") for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA device assignment requested but torch.cuda.is_available() is false")

    loaded: dict[str, TransformersCausalModel] = {}
    for condition, device in zip(config["quantization"]["conditions"], devices):
        name = condition["name"]
        model = AutoModelForCausalLM.from_pretrained(
            spec["name"], revision=spec["revision"], torch_dtype=torch.bfloat16
        )
        if name != "bf16":
            _quantize_torch_model(model, int(condition["bits"]))
        model.to(device)
        model.eval()
        loaded[name] = TransformersCausalModel(model, num_layers=spec["transformer_layers"])
    return tokenizer, loaded
