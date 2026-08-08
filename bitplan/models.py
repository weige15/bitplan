"""A dependency-free deterministic fixture and the boundary model protocol.

The fixture is intentionally not a benchmark model. It makes the smoke path
network-free while exercising the same boundary/suffix contracts used by the
optional Transformers adapter.
"""

from __future__ import annotations

import copy
import math
import struct
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ForwardOutput:
    logits: list[float]
    hidden_by_boundary: dict[int, list[list[float]]]


class BoundaryModel(Protocol):
    num_layers: int
    vocab_size: int

    def forward(self, token_ids: list[int], boundaries: list[int]) -> ForwardOutput: ...

    def replay_suffix(self, boundary: int, hidden_states: list[list[float]]) -> list[float]: ...


def _bf16_round(value: float) -> float:
    """Round a Python float through IEEE float32/bfloat16 without torch."""
    packed = struct.pack(">f", float(value))
    bits = int.from_bytes(packed, "big")
    lower = bits & 0xFFFF
    upper = bits >> 16
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper += 1
    return struct.unpack(">f", (upper << 16).to_bytes(4, "big"))[0]


def _quantize(values: list[float], bits: int) -> list[float]:
    if bits == 16:
        return [_bf16_round(value) for value in values]
    maximum = max((abs(value) for value in values), default=0.0)
    if maximum == 0.0:
        return list(values)
    qmax = (1 << (bits - 1)) - 1
    scale = maximum / qmax
    return [max(-qmax, min(qmax, round(value / scale))) * scale for value in values]


def _quantize_matrix(matrix: list[list[float]], bits: int) -> list[list[float]]:
    flattened = [value for row in matrix for value in row]
    quantized = _quantize(flattened, bits)
    width = len(matrix[0]) if matrix else 0
    return [quantized[row * width : (row + 1) * width] for row in range(len(matrix))]


def _matrix(rows: int, columns: int, scale: float, offset: int) -> list[list[float]]:
    return [
        [scale * (((row * 17 + column * 7 + offset) % 37) - 18) for column in range(columns)]
        for row in range(rows)
    ]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]


class ToyTransformer:
    """Small causal, layered fixture with explicit hidden-state boundaries."""

    def __init__(self, *, vocab_size: int = 32, hidden_size: int = 8, num_layers: int = 2, bits: int | None = None):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bits = bits
        self.embeddings = _matrix(vocab_size, hidden_size, 0.035, 3)
        self.layers = [
            {
                "weight": _matrix(hidden_size, hidden_size, 0.08 / (layer + 1), layer + 5),
                "bias": [0.01 * (((index + layer * 3) % 7) - 3) for index in range(hidden_size)],
            }
            for layer in range(num_layers)
        ]
        self.head = _matrix(vocab_size, hidden_size, 0.055, 11)

    def copy_for_bits(self, bits: int) -> "ToyTransformer":
        result = copy.deepcopy(self)
        result.bits = bits
        result.embeddings = _quantize_matrix(result.embeddings, bits)
        result.head = _quantize_matrix(result.head, bits)
        for layer in result.layers:
            layer["weight"] = _quantize_matrix(layer["weight"], bits)
            layer["bias"] = _quantize(layer["bias"], bits)
        return result

    @staticmethod
    def _layer(states: list[list[float]], layer: dict[str, Any]) -> list[list[float]]:
        outputs: list[list[float]] = []
        for position, state in enumerate(states):
            context = [
                sum(previous[index] for previous in states[: position + 1]) / (position + 1)
                for index in range(len(state))
            ]
            transformed = _matvec(layer["weight"], state)
            outputs.append([
                math.tanh(value + 0.12 * context[index] + layer["bias"][index])
                for index, value in enumerate(transformed)
            ])
        return outputs

    def _head_logits(self, states: list[list[float]]) -> list[float]:
        return _matvec(self.head, states[-1])

    def forward(self, token_ids: list[int], boundaries: list[int]) -> ForwardOutput:
        if not token_ids:
            raise ValueError("a prefix must contain at least one token")
        if any(token < 0 or token >= self.vocab_size for token in token_ids):
            raise ValueError("token is outside fixture vocabulary")
        requested = set(boundaries)
        if any(boundary < 0 or boundary > self.num_layers for boundary in requested):
            raise ValueError("boundary is outside model depth")
        states = [list(self.embeddings[token]) for token in token_ids]
        hidden: dict[int, list[list[float]]] = {}
        if 0 in requested:
            hidden[0] = copy.deepcopy(states)
        for index, layer in enumerate(self.layers):
            states = self._layer(states, layer)
            boundary = index + 1
            if boundary in requested:
                hidden[boundary] = copy.deepcopy(states)
        return ForwardOutput(logits=self._head_logits(states), hidden_by_boundary=hidden)

    def replay_suffix(self, boundary: int, hidden_states: list[list[float]]) -> list[float]:
        if boundary < 0 or boundary > self.num_layers:
            raise ValueError("boundary is outside model depth")
        states = copy.deepcopy(hidden_states)
        for layer in self.layers[boundary:]:
            states = self._layer(states, layer)
        return self._head_logits(states)


def stable_tokenize(text: str, *, vocab_size: int = 32) -> list[int]:
    """A stable fixture tokenizer; it deliberately does not model a HF tokenizer."""
    words = text.split()
    if not words:
        return [1]
    return [1 + sum((index + 1) * ord(char) for index, char in enumerate(word)) % (vocab_size - 1) for word in words]


def argmax_token(logits: list[float]) -> int:
    return max(range(len(logits)), key=lambda index: (logits[index], -index))
