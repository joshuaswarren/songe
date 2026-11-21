"""Regression: sleep/probe.py must consume the train_step 3-tuple.

The torch and MLX backends return (loss, gradients, loss_tokens) from
train_step (see backends/training.py, crossplatform_smoke.py,
test_torch_backend.py). run_probe unpacked two values and crashed every
candidate probe.
"""
import json
from types import SimpleNamespace

from songe.sleep import probe


class _Generation:
    generation_tps = 10.0
    prompt_tps = 20.0


class _Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1, 2, 3]


class _FakeBackend:
    """Same surface run_probe uses; train_step honors the real contract."""

    def __init__(self, config, check):
        self.config = SimpleNamespace(response_tokens=64)
        self.tokenizer = _Tokenizer()
        self.applied = None

    def prepare(self):
        pass

    def sample(self, prompt, seed=None):
        return None, None, _Generation()

    def optimizer(self, lr):
        return {"lr": lr}

    def train_step(self, step, optimizer, seed=None):
        return 0.5, {"w": 0.1}, 3  # (loss, gradients, loss_tokens)

    def apply(self, gradients, optimizer=None):
        self.applied = gradients

    def peak_memory(self):
        return 1.0e9


def test_probe_handles_train_step_three_tuple(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "support_reason", lambda model: None)
    monkeypatch.setattr("songe.backends.choose", lambda config, platform: _FakeBackend)
    output = tmp_path / "probe.json"
    probe.run_probe(tmp_path / "model", output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["loss_finite"] is True
    assert result["generation_tps"] == 10.0
