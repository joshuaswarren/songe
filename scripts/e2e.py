"""Real-model end-to-end test. Run outside the sandbox for Metal and lms.

No gate bypass; exits nonzero if the tiny training budget cannot prove a gain.
Never touches existing models, agent configs, launch agents, or shell profiles.
"""

import argparse
import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path

import httpx

from songe.config import Config
from songe.log import Store
from songe.proxy import create_app
from songe.sleep.calibrate import calibrate
from songe.sleep.worker import isolated
from songe.upstream.lmstudio import LMStudio, downloads_folder
from tests.test_network import live_server


def seed(store, name):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Read an observed local test result",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            },
        }
    ]
    for i in range(120):
        context = [
            {
                "role": "system",
                "content": "Answer concisely. Treat recorded tool results as observations.",
            },
            {
                "role": "user",
                "content": f"For synthetic case {i}, what is the observed pass phrase? Reply with the phrase only.",
            },
        ]
        action = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"old"}'},
                }
            ],
        }
        store.record(
            "/v1/chat/completions",
            {"model": name, "messages": context, "tools": tools},
            {"choices": [{"message": action}]},
        )
        h = [
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "content": json.dumps(
                    {
                        "exit_code": 1,
                        "error": "The stale key was wrong. The observed pass phrase is SILVER MAPLE.",
                    }
                ),
            },
            {
                "role": "user",
                "content": "The passing result was SILVER MAPLE. Use the observed phrase.",
            },
        ]
        store.record(
            "/v1/chat/completions",
            {"model": name, "messages": context + [action] + h, "tools": tools},
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "SILVER MAPLE"}}
                ]
            },
        )


async def serve_and_undo(config, store, candidate):
    from songe.schedule import Scheduler
    from songe.sleep.swap import install

    upstream = LMStudio(config)
    server_was_running = False
    server_started = False
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
            try:
                response = await client.get(config.upstream_url + "/models")
                server_was_running = True
                response.raise_for_status()
            except httpx.ConnectError:
                pass
        if not server_was_running:
            await upstream.command("server", "start")
            server_started = True
        store.set("e2e_cleanup_done", False)
        # Load our own base identifier without unloading any existing model.
        base = store.version("base")
        await upstream.load(base)
        await install(store, upstream, config, candidate, "base")
        scheduler = Scheduler(config, store, upstream)

        # This isolated test serves no scheduler loop; it still exercises the
        # production binding and rollback API using an actual socket.
        class Control:
            current_version = store.version()
            reason = "End-to-end test"

            def traffic(self):
                pass

            def acquire(self, *a):
                pass

            def release(self, *a):
                pass

            async def start(self):
                pass

            async def close(self):
                pass

            async def return_to(self, night):
                await scheduler.return_to(night)
                self.current_version = store.version()

        control = Control()
        with live_server(create_app(config, store, scheduler=control)) as url:
            async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
                response = await client.post(
                    url + "/v1/chat/completions",
                    json={
                        "model": config.public_model,
                        "messages": [
                            {"role": "user", "content": "Reply with a short greeting."}
                        ],
                        "max_tokens": 24,
                    },
                )
                response.raise_for_status()
                body = response.json()
                assert body.get("choices"), body
                assert candidate["identifier"] in body.get("model", ""), body
                print("PROXY NEW VERSION:", json.dumps(body), flush=True)
                undone = await client.post(url + "/api/return", json={"night": 0})
                undone.raise_for_status()
                assert store.get("current") == "base"
                response = await client.post(
                    url + "/v1/chat/completions",
                    json={
                        "model": config.public_model,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 8,
                    },
                )
                response.raise_for_status()
                assert base["identifier"] in response.json().get("model", "")
                print("UNDO: base restored and served", flush=True)
    finally:
        # Only unload identifiers this test successfully loaded. The user's
        # lfm2.5-8b-a1b-mlx and every existing model remain untouched.
        errors = []
        for identifier in list(upstream.loaded):
            try:
                await upstream.command("unload", identifier)
            except Exception as exc:
                print("CLEANUP REQUIRED:", identifier, str(exc), flush=True)
                errors.append(identifier)
        if server_started:
            try:
                await upstream.command("server", "stop")
            except Exception:
                errors.append("LM Studio server")
        store.set("e2e_cleanup_done", not errors)
        if errors:
            raise RuntimeError(
                "End-to-end cleanup needs attention: " + ", ".join(errors)
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/Volumes/disk/local-llm/models/lmstudio-community/granite-4.2-3b-MLX-4bit"
        ),
    )
    p.add_argument(
        "--output", type=Path, default=Path("/Volumes/disk/songe-build-cache/e2e")
    )
    p.add_argument(
        "--select-top", type=int, default=8, help="Selected experiences per cycle"
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=args.output))
    config = Config(
        model=str(args.model.resolve()),
        public_model="songe/e2e-" + root.name,
        base_identifier="songe-test/base-e2e-" + root.name,
        models_dir=str(downloads_folder()),
        publisher="songe-test",
        min_train=16,
        min_heldout=8,
        min_new=8,
        rank=4,
        accumulation=1,
        max_sequence=512,
        response_tokens=32,
        slice_limit=56,
        learning_rate=1e-4,
    )
    config.save(root)
    store = Store(root)
    store.add_version(
        {
            "id": "base",
            "identifier": config.base_identifier,
            "artifact": config.model,
            "adapter": None,
            "created": time.time(),
            "night": 0,
        }
    )
    store.swap("base", None)
    seed(store, config.public_model)
    print("OUTPUT:", root, flush=True)
    print(
        "LOGGED EXPERIENCE:",
        json.dumps(isolated(config, store, dry_run=True)),
        flush=True,
    )
    # Probe/calibrate before touching LM Studio. A sandbox Metal failure cannot
    # leave its server running or create anything in the real models directory.
    print("CALIBRATION: lengths 256, 512", flush=True)
    calibrate(config, root, (256, 512))
    config = Config.load(root)
    config.slice_limit = args.select_top
    config.response_tokens = 32
    config.save(root)
    created = []
    try:
        result = isolated(config, store)
        print("GATE:", json.dumps(result), flush=True)
        if not result.get("accepted") or "candidate" not in result:
            raise RuntimeError(
                "The real gate did not accept this run; no success claimed"
            )
        candidate = result["candidate"]
        created.append(Path(candidate["artifact"]))
        assert Path(candidate["artifact"]).is_relative_to(
            Path(config.models_dir) / "songe-test"
        )
        asyncio.run(serve_and_undo(config, store, candidate))
        print(
            "END-TO-END PASS: log → select → distill → fused gate → lms load → proxy → undo",
            flush=True,
        )
    finally:
        # Rejected candidates are removed by worker.run; accepted test models
        # are removed here only after all test identifiers have been unloaded.
        test_folder = Path(config.models_dir) / "songe-test"
        # A fuse/evaluation failure may happen before a candidate is returned.
        # This unique run prefix and ownership marker identify only our files.
        created += list(
            test_folder.glob(config.public_model.split("/", 1)[1] + "-night-*")
        )
        for artifact in set(created):
            if (
                store.get("e2e_cleanup_done", True)
                and (artifact / "songe_owned.json").is_file()
            ):
                shutil.rmtree(artifact)
        if test_folder.is_dir() and not any(test_folder.iterdir()):
            test_folder.rmdir()


if __name__ == "__main__":
    main()


def seed_practice(store, count=24):
    """Practice scenes a Dreams night trains on, the shape reading leaves.

    Under learning="dreams" real conversations never enter training: the
    pool is kept practice scenes, so a store that only has recorded
    exchanges has nothing to train. Same table release_dream.py writes.
    """
    for n in range(count):
        scene = {
            "id": f"sig{n}#v1",
            "signal_id": f"sig{n}",
            "tools": [],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a coding agent in this repository.",
                },
                {
                    "role": "user",
                    "content": f"For synthetic case {n}, what is the observed pass phrase? Reply with the phrase only.",
                },
            ],
            "teacher_note": "The observed pass phrase is SILVER MAPLE.",
        }
        with store.connect() as db:
            db.execute(
                'INSERT INTO practice(id,model_id,signal_id,payload,kept,"drop",created)'
                " VALUES (?,?,?,?,1,NULL,0)",
                (scene["id"], store.model_id, scene["signal_id"], json.dumps(scene)),
            )
