"""Regression: the record stage must expect the proxy's reply contract.

ModelRename gives the agent back the model name it asked for, so a proxied
reply carries config.public_model, not the upstream's served id. The check
compared against upstream.request_model(base) and failed every vLLM run with
"Recording used the wrong base" while the correct invariant (the stored
binding) is verified two lines below.
"""
import asyncio
import contextlib
from types import SimpleNamespace

import scripts.e2e_linux as e2e


class _Response:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _Client:
    """Stands in for httpx.AsyncClient; replies carry the requested name."""

    def __init__(self, model):
        self._model = model

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json=None):
        return _Response(
            {
                "model": self._model,
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        )


class _Upstream:
    async def load(self, version):
        pass

    def request_model(self, version):
        return "Qwen/Qwen3-0.6B"  # served id, not the requested name


class _Store:
    def __init__(self):
        self.calls = 0

    def version(self):
        return {"id": "base", "identifier": "Qwen/Qwen3-0.6B"}

    def exchanges(self):
        self.calls += 1
        if self.calls == 1:  # the pre-record snapshot
            return []
        return [
            {"id": "r1", "binding": "base", "complete": True},
            {"id": "r2", "binding": "base", "complete": True},
        ]


def test_record_experience_accepts_renamed_reply(tmp_path, monkeypatch):
    config = SimpleNamespace(
        public_model="songe/e2e-abc", upstream_timeout=5, guard_min_interval=0.0
    )
    monkeypatch.setattr(e2e, "foreground_scheduler", lambda *a: object())
    monkeypatch.setattr("songe.proxy.create_app", lambda *a, **k: object())

    @contextlib.asynccontextmanager
    async def socket(app):
        yield "http://test"

    monkeypatch.setattr(e2e, "proxy_socket", socket)
    monkeypatch.setattr(
        "httpx.AsyncClient", lambda **k: _Client(config.public_model)
    )
    monkeypatch.setattr(
        "songe.steps.extract",
        lambda rows, public_model: [SimpleNamespace(hindsight=[{"role": "user"}])],
    )
    result = asyncio.run(
        e2e.record_experience(
            config, _Store(), _Upstream(), SimpleNamespace(), conversations=1
        )
    )
    assert result["conversations"] == 1
    assert result["learnable_experiences"] == 1
    assert result["binding"] == "base"
