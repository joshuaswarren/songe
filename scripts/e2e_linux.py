"""Real Linux GPU night, including publication, a socket proxy request and undo.

Run against a disposable local upstream. All Songe files and HOME are isolated;
the original model is read-only. A rejected production gate is a failed test.
"""

import argparse
import asyncio
import contextlib
import functools
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

# Support both `python scripts/e2e_linux.py` and `python -m scripts.e2e_linux`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
STOP = threading.Event()


def check_cancelled():
    require(not STOP.is_set(), "Harness interrupted; stopping guarded child")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def model_digest(path):
    """Check immutable contents; servers may refresh cache-file timestamps."""
    path = Path(path)
    if path.is_file():
        return file_digest(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file_digest(file).encode())
    return digest.hexdigest()


def preserve_diagnostics(root, output):
    """Retain private child failures before the disposable HOME is removed."""
    files = [*Path(root).glob("songe.log*"), Path(root) / "calibration.json"]
    for file in files:
        if file.is_file():
            shutil.copy2(file, Path(output) / file.name)


def remove_orphan_upload(folder, digest):
    """Remove only a known new upload after its test tags/runners are gone.

    /api/blobs has no delete endpoint. A failed /api/create can leave an upload
    without a tag; never remove a layer referenced by any remaining manifest.
    """
    for manifest in (folder / "manifests").rglob("*"):
        if manifest.is_file():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            layers = [*metadata.get("layers", []), metadata.get("config", {})]
            if any(layer.get("digest") == "sha256:" + digest for layer in layers):
                return
    (folder / "blobs" / ("sha256-" + digest)).unlink(missing_ok=True)


class Reporter:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def emit(self, stage, seconds, **values):
        row = (
            dict(
                stage=stage,
                status="passed",
                seconds=seconds,
                peak_memory_bytes=None,
                heldout_change=None,
                drift=None,
            )
            | values
        )
        line = json.dumps(row, allow_nan=False)
        with (self.output / "stages.jsonl").open("a") as out:
            out.write(line + "\n")
        print(line, flush=True)

    @contextlib.contextmanager
    def stage(self, name):
        """Sample actual process-tree RSS, including converter grandchildren."""
        import psutil

        started, peak, done = time.monotonic(), [0], threading.Event()
        sampling_errors = []

        def sample():
            process = psutil.Process()
            while not done.is_set():
                total = 0
                try:
                    children = process.children(recursive=True)
                except (psutil.Error, OSError) as exc:
                    children = []
                    if not sampling_errors:
                        sampling_errors.append(str(exc))
                for child in [process, *children]:
                    with contextlib.suppress(psutil.Error, OSError):
                        total += child.memory_info().rss
                peak[0] = max(peak[0], total)
                done.wait(0.1)  # Sampling resolution; not a memory capacity limit.

        watcher = threading.Thread(target=sample, daemon=True)
        watcher.start()
        values = {}
        try:
            yield values
        except BaseException as exc:
            values.update(status="failed", error=str(exc))
            raise
        finally:
            done.set()
            watcher.join()
            if sampling_errors:
                values.update(
                    memory_sampling_error=sampling_errors[0],
                    memory_kind="sampled_parent_rss",
                )
            self.emit(
                name,
                time.monotonic() - started,
                **(
                    {"peak_memory_bytes": peak[0], "memory_kind": "sampled_tree_rss"}
                    | values
                ),
            )


@contextlib.contextmanager
def isolated_environment(root):
    """Keep source libraries readable while isolating every writable cache."""
    old = os.environ.copy()
    home = root / "home"
    home.mkdir()
    os.environ.update(
        HOME=str(home),
        USERPROFILE=str(home),
        SONGE_DATA_DIR=str(root / "data"),
        XDG_DATA_HOME=str(home / ".local/share"),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"),
        HF_HOME=str(home / ".cache/huggingface"),
        HF_HUB_CACHE=old.get(
            "HF_HUB_CACHE",
            str(Path(old.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"),
        ),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TORCH_HOME=str(home / ".cache/torch"),
        TRITON_CACHE_DIR=str(home / ".cache/triton"),
        CUDA_CACHE_PATH=str(home / ".cache/cuda"),
    )
    try:
        yield old
    finally:
        os.environ.clear()
        os.environ.update(old)


def gpu_probe(root):
    import torch

    require(
        torch.cuda.is_available(), "A CUDA/ROCm GPU is required; CPU fallback refused"
    )
    torch.cuda.reset_peak_memory_stats()
    tensor = torch.ones(16, device="cuda")
    require(float((tensor @ tensor).item()) == 16, "GPU tensor probe failed")
    row = dict(
        vendor="rocm" if torch.version.hip else "cuda",
        torch=torch.__version__,
        runtime=torch.version.hip or torch.version.cuda,
        device=torch.cuda.get_device_name(0),
        free_bytes=torch.cuda.mem_get_info()[0],
        total_bytes=torch.cuda.mem_get_info()[1],
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
    )
    (root / "gpu.json").write_text(json.dumps(row) + "\n", encoding="utf-8")


def night_child(root):
    import resource

    """Measure real calls; wrappers do not alter inputs, budgets or verdicts."""
    from songe.backends.torch import TorchBackend
    from songe.config import Config
    from songe.log import Store
    from songe.sleep import fuse, gate, select, worker

    metrics = {}

    def observed(name, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            started = time.monotonic()
            row = metrics.setdefault(name, dict(seconds=0.0, calls=0, status="passed"))
            try:
                result = function(*args, **kwargs)
                if name == "select":
                    row.update(selected=len(result[0]), skipped=result[1])
                elif name == "distill":
                    require(args[0].device.type == "cuda", "Training ran off the GPU")
                    row.update(result)
                elif name == "gate":
                    row.update(scale=result[0], evaluations=result[2])
                    decision = result[1]
                    if decision is not None:
                        row.update(
                            heldout_change=decision.heldout_change,
                            drift=decision.general_drift,
                        )
                    if result[0] is None:
                        row["status"] = "failed"
                return result
            except BaseException as exc:
                row.update(status="failed", error=str(exc))
                raise
            finally:
                row["seconds"] = (
                    row.get("measured_seconds", 0) + time.monotonic() - started
                )
                row["measured_seconds"] = row["seconds"]
                row["calls"] += 1
                import torch

                row["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
                row["memory_kind"] = "torch_process_peak_allocated"

        return call

    try:
        with (
            patch.object(select, "select", observed("select", select.select)),
            patch.object(
                TorchBackend, "train", observed("distill", TorchBackend.train)
            ),
            patch.object(gate, "choose_scale", observed("gate", gate.choose_scale)),
            patch.object(
                fuse, "export_candidate", observed("export", fuse.export_candidate)
            ),
        ):
            result = worker.run(Config.load(root), Store(root))
        (root / "gate.json").write_text(
            json.dumps(result, allow_nan=False) + "\n", encoding="utf-8"
        )
    finally:
        (root / "night-metrics.json").write_text(
            json.dumps(metrics, allow_nan=False) + "\n", encoding="utf-8"
        )
        (root / "night-rss.json").write_text(
            json.dumps(
                {
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                }
            )
            + "\n",
            encoding="utf-8",
        )


def child_run(mode, root, output, timeout):
    from songe.memory import guarded_run

    with (output / (mode + ".log")).open("w") as log:
        result = guarded_run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                mode,
                "--data-dir",
                str(root),
            ],
            timeout=timeout,
            check=check_cancelled,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    require(
        result.returncode == 0,
        f"{mode} child failed ({result.returncode}); see {output / (mode + '.log')}",
    )


def calibration_run(config, root, output, lengths, timeout):
    from songe.memory import guarded_run

    config.save(root)
    with (output / "calibrate.log").open("w") as log:
        result = guarded_run(
            [
                sys.executable,
                "-m",
                "songe.sleep.calibrate",
                "--data-dir",
                str(root),
                "--lengths",
                ",".join(map(str, lengths)),
                "--child",
            ],
            timeout=timeout,
            check=check_cancelled,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    require(
        result.returncode == 0, f"Calibration failed; see {output / 'calibrate.log'}"
    )
    return json.loads((root / "calibration.json").read_text(encoding="utf-8"))["rows"]


@contextlib.asynccontextmanager
async def proxy_socket(app):
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        deadline = time.monotonic() + 10  # Local socket startup deadline.
        while not server.started:
            require(
                not task.done() and time.monotonic() < deadline, "Proxy did not start"
            )
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 10)
        finally:
            sock.close()


async def verify_binding(config, store, version, client, proxy_url, upstream=None):
    response = await client.post(
        proxy_url + "/v1/chat/completions",
        json={
            "model": config.public_model,
            "messages": [{"role": "user", "content": "Reply with a short greeting."}],
            "max_tokens": 24,
        },
    )
    response.raise_for_status()
    body = response.json()
    require(bool(body.get("choices")), f"No completion: {body}")
    expected_identifier = (
        upstream.request_model(version)
        if hasattr(upstream, "request_model")
        else version["identifier"]
    )
    require(
        body.get("model") == expected_identifier, f"Wrong served identifier: {body}"
    )
    # The proxy logs asynchronously. Check the actual persisted binding too.
    deadline = time.monotonic() + config.upstream_timeout
    while True:
        rows = [r for r in store.exchanges() if r.get("binding") == version["id"]]
        if rows:
            break
        require(time.monotonic() < deadline, "Proxy did not log the version binding")
        await asyncio.sleep(config.guard_min_interval)
    if config.upstream == "llamacpp":
        url = (
            upstream.request_url(version)
            if hasattr(upstream, "request_url")
            else config.upstream_url
        ).removesuffix("/v1")
        props = await client.get(url + "/props")
        props.raise_for_status()
        require(
            Path(props.json()["model_path"]).resolve()
            == Path(config.model_ref["path"]).resolve(),
            "llama.cpp changed the original base",
        )
        adapters = await client.get(url + "/lora-adapters")
        adapters.raise_for_status()
        active = [a for a in adapters.json() if a["scale"] != 0]
        if version["id"] == "base":
            require(not active, "Undo left a llama.cpp adapter active")
        else:
            desired = Path(version["adapter"]).resolve()
            require(
                len(active) == 1
                and Path(active[0]["path"]).resolve() == desired
                and active[0]["scale"] == 1,
                "llama.cpp served the wrong adapter",
            )
    elif config.upstream == "vllm" and version["id"] != "base":
        models = await client.get(config.upstream_url + "/models")
        models.raise_for_status()
        entry = next(
            m for m in models.json()["data"] if m["id"] == version["identifier"]
        )
        require(
            Path(entry["root"]).resolve() == Path(version["artifact"]).resolve(),
            "vLLM identifier points at different weights",
        )
    elif config.upstream == "ollama":
        tags = await client.get(config.upstream_url.removesuffix("/v1") + "/api/tags")
        tags.raise_for_status()
        require(
            expected_identifier in {m["name"] for m in tags.json()["models"]},
            "Ollama test version was not imported",
        )
        from songe.models.discovery import ollama

        refs = await asyncio.to_thread(
            ollama, folder=config.ollama_models_dir, url=config.upstream_url
        )
        imported = next(
            (ref for ref in refs if ref.serving_id == expected_identifier), None
        )
        require(imported is not None, "Ollama imported weights could not be located")

        expected, actual = await asyncio.gather(
            asyncio.to_thread(file_digest, version["artifact"]),
            asyncio.to_thread(file_digest, imported.path),
        )
        require(expected == actual, "Ollama imported different GGUF weights")
    return {
        "identifier": body["model"],
        "binding": version["id"],
        "completion_tokens": body.get("usage", {}).get("completion_tokens"),
    }


def foreground_scheduler(config, store, upstream):
    from songe.schedule import Scheduler

    # Keep production request accounting and undo. The harness drives one night
    # explicitly, so an idle loop must not begin another in the background.
    class ForegroundScheduler(Scheduler):
        async def start(self):
            pass

        async def close(self):
            pass  # Outer cleanup owns the upstream and parked snapshot.

    return ForegroundScheduler(config, store, upstream)


async def record_experience(config, store, upstream, reporter, *, conversations=1):
    """Capture model replies and explicit correction feedback through the proxy."""
    import httpx

    from songe.proxy import create_app
    from songe.steps import extract

    base = store.version()
    await upstream.load(base | {"serving": True})
    scheduler = foreground_scheduler(config, store, upstream)
    before = {r["id"] for r in store.exchanges()}
    async with (
        proxy_socket(create_app(config, store, scheduler=scheduler)) as url,
        httpx.AsyncClient(trust_env=False, timeout=config.upstream_timeout) as client,
    ):
        for case in range(conversations):
            context = [
                {"role": "system", "content": "Answer concisely."},
                {
                    "role": "user",
                    "content": f"For recorded case {case}, what is the observed pass phrase? /no_think",
                },
            ]
            for followup in (False, True):
                response = await client.post(
                    url + "/v1/chat/completions",
                    json={
                        "model": config.public_model,
                        "messages": context,
                        "max_tokens": 64,
                    },
                )
                response.raise_for_status()
                body = response.json()
                # The proxy returns the requested name (ModelRename contract);
                # the base binding is verified on the stored rows below.
                require(
                    body.get("model") == config.public_model,
                    "Recording used the wrong base",
                )
                context.append(body["choices"][0]["message"])
                if not followup:
                    context.append(
                        {
                            "role": "user",
                            "content": "Actually, the observed passing result was SILVER MAPLE. Use that phrase. /no_think",
                        }
                    )
        deadline = time.monotonic() + config.upstream_timeout
        while True:
            rows = [r for r in store.exchanges() if r["id"] not in before]
            if len(rows) >= 2 * conversations:
                break
            require(
                time.monotonic() < deadline,
                "Proxy did not persist the recorded conversations",
            )
            await asyncio.sleep(config.guard_min_interval)
    require(
        all(r["binding"] == base["id"] and r["complete"] for r in rows),
        "Recording lost the base binding",
    )
    learnable = [step for step in extract(rows, config.public_model) if step.hindsight]
    require(
        len(learnable) == conversations,
        "Recorded feedback did not yield learnable experience",
    )
    return {
        "exchanges": len(rows),
        "conversations": conversations,
        "learnable_experiences": len(learnable),
        "binding": base["id"],
        "upstream_identifier": base["identifier"],
    }


async def serve_and_undo(config, store, upstream, candidate, reporter):
    import httpx

    from songe.proxy import create_app
    from songe.sleep.swap import install

    scheduler = foreground_scheduler(config, store, upstream)
    with reporter.stage("swap") as row:
        require(
            await install(store, upstream, config, candidate, "base"),
            "Swap not committed",
        )
        scheduler.current_version = store.version()
        row.update(identifier=candidate["identifier"], kind=candidate["kind"])
    async with (
        proxy_socket(create_app(config, store, scheduler=scheduler)) as url,
        httpx.AsyncClient(trust_env=False, timeout=config.upstream_timeout) as client,
    ):
        with reporter.stage("proxy") as row:
            row.update(
                await verify_binding(config, store, candidate, client, url, upstream)
            )
        with reporter.stage("undo") as row:
            response = await client.post(url + "/api/return", json={"night": 0})
            response.raise_for_status()
            require(
                store.get("current") == "base", "Undo did not restore the base pointer"
            )
            row.update(
                await verify_binding(
                    config, store, store.version(), client, url, upstream
                )
            )


async def cleanup_ollama(upstream, config, store):
    """Delete exact journaled test imports, including interrupted publication."""
    import httpx

    from songe.upstream.ollama import _tag

    imports = store.get("runtime_imports", {})
    for identifier, version in list(imports.items()):
        require(
            version.get("installation_id") == store.get("installation_id"),
            "Import belongs to another installation",
        )
        require(
            Path(version["artifact"]).resolve().is_relative_to(store.root.resolve())
            or (
                version.get("id") == "base"
                and Path(version["artifact"]).resolve()
                == Path(config.model_ref["path"]).resolve()
            ),
            "Import escaped the isolated test root",
        )
        tags = (await upstream._request("GET", "/api/tags")).get("models", [])
        if _tag(identifier) in {_tag(m["name"]) for m in tags}:
            require(
                await upstream.same_weights(version),
                "Test identifier now refers to other weights; preserved",
            )
            # The durable journal is the ownership proof after a harness restart.
            await upstream._warm(identifier, 0)
            try:
                await upstream._request(
                    "DELETE", "/api/delete", json={"model": identifier}
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
        tags = (await upstream._request("GET", "/api/tags")).get("models", [])
        running = await upstream._running()
        require(
            _tag(identifier) not in {_tag(m["name"]) for m in tags},
            "Test tag remains after cleanup",
        )
        require(
            _tag(identifier)
            not in {_tag(m.get("name", m.get("model"))) for m in running},
            "Test runner remains after cleanup",
        )
        imports.pop(identifier)
        store.set("runtime_imports", imports)
    await upstream.close()


async def exercise(args, scratch, reporter, ollama_folder):
    import httpx

    from scripts.e2e import seed
    from songe.config import Config
    from songe.log import Store
    from songe.models import trainable_base
    from songe.platform import Rooted, current
    from songe.sleep.worker import preview
    from songe.upstream import create
    from songe.upstream.supervisor import listening_server, provision_endpoint

    root = scratch / "data"
    root.mkdir()
    platform = Rooted(current(), root)
    token = uuid.uuid4().hex
    config = Config(
        upstream=args.upstream,
        upstream_timeout=min(args.timeout, 600),
        upstream_url=args.url
        or {
            "ollama": "http://127.0.0.1:11434/v1",
            "llamacpp": "http://127.0.0.1:8080/v1",
            "vllm": "http://127.0.0.1:8000/v1",
        }[args.upstream],
        backend="torch",
        torch_options={"device": "cuda"},
        ollama_models_dir=ollama_folder,
        public_model="songe/e2e-" + token,
        publisher="songe-test",
        models_dir=str(root / "exports"),
        min_train=16,
        min_heldout=8,
        min_new=8,
        rank=4,
        accumulation=1,
        max_sequence=max(args.lengths),
        response_tokens=32,
        slice_limit=56,
        learning_rate=args.learning_rate,
    )
    with reporter.stage("discover") as row:
        discovery = create(config, root)
        refs = await asyncio.to_thread(discovery.discover)
        matches = [ref for ref in refs if ref.serving_id == args.model]
        require(
            len(matches) == 1,
            f"Expected one discovered model {args.model!r}; got {[r.serving_id for r in refs]}",
        )
        ref = matches[0]
        original_fingerprint = model_digest(ref.path)
        require(
            ref.path.exists(), f"Upstream model is not locally accessible: {ref.path}"
        )
        # Ollama discovers local manifests; also prove this tag exists in the running daemon.
        async with httpx.AsyncClient(
            trust_env=False, timeout=config.upstream_timeout
        ) as client:
            path = "/api/tags" if args.upstream == "ollama" else "/v1/models"
            response = await client.get(config.upstream_url.removesuffix("/v1") + path)
            response.raise_for_status()
            body = response.json()
            ids = (
                {m["name"] for m in body["models"]}
                if args.upstream == "ollama"
                else {m["id"] for m in body["data"]}
            )
            require(
                ref.serving_id in ids,
                "Discovered model is absent from the running upstream",
            )
        config.model_ref = asdict(ref) | {"path": str(ref.path.resolve())}
        config.base_identifier = (
            f"songe/e2e-{token}:base" if args.upstream == "ollama" else ref.serving_id
        )
        command = json.loads(args.server_command) if args.server_command else []
        if args.upstream == "llamacpp":
            process = await asyncio.to_thread(listening_server, config)
            require(
                process is not None, "Cannot identify the local llama-server process"
            )
            actual = process.cmdline()
            require(
                not command or actual == command,
                "llama-server command differs from --server-command",
            )
            command = actual
        config.server_command = command
        if args.upstream == "llamacpp":
            provision_endpoint(config)
        row.update(config.model_ref)
    upstream, parked, candidate, touched = None, [], None, False
    orphan_digest = None
    store = Store(root)
    try:
        # model activates production llama.cpp supervision before conversion.
        config.model = str(ref.path.resolve())
        upstream = create(config, root)
        if args.upstream == "llamacpp":
            await upstream.ensure(
                {
                    "identifier": config.base_identifier,
                    "artifact": str(ref.path.resolve()),
                    "kind": "gguf-full",
                }
            )
        with reporter.stage("park") as row:
            touched = True
            parked = await upstream.park(
                on_snapshot=lambda items: store.set("parked_models", items)
            )
            row.update(models=len(parked))
        with reporter.stage("gpu") as row:
            await asyncio.to_thread(
                child_run, "gpu", root, reporter.output, args.timeout
            )
            row.update(json.loads((root / "gpu.json").read_text(encoding="utf-8")))
            row["memory_kind"] = "torch_process_peak_allocated"
            shutil.copy2(root / "gpu.json", reporter.output)
        with reporter.stage("prepare") as row:
            from songe.artifacts.runtime import guarded_run as artifact_run

            def convert_run(argv, **kwargs):
                return artifact_run(argv, check=check_cancelled, **kwargs)

            config.model = str(
                await asyncio.to_thread(
                    trainable_base, ref, "torch", platform, runner=convert_run
                )
            )
            require(Path(config.model).is_dir(), "No trainable safetensors base")
            row.update(format=ref.format, dequantized=ref.format == "gguf")
        config.save(root)
        store.add_version(
            dict(
                id="base",
                identifier=config.base_identifier,
                artifact=str(ref.path.resolve()),
                adapter=None,
                kind="gguf-full" if ref.format == "gguf" else "base",
                created=time.time(),
                night=0,
            )
        )
        store.swap("base", None)
        with reporter.stage("record") as row:
            row.update(
                await record_experience(
                    config,
                    store,
                    upstream,
                    reporter,
                    conversations=120 if args.live_experience else 1,
                )
            )
            # Reclaim only the test-owned serving instances before calibration.
            # These instances were created for recording. Reclaim them without
            # adding them to the original restore snapshot: doing so would
            # recreate deleted test imports during final cleanup.
            await upstream.park()
            store.set("parked_models", parked)
        with reporter.stage("log") as row:
            if not args.live_experience:
                seed(store, config.public_model)
            info, _, _ = preview(config, store)
            require(
                info["train"] >= config.min_train
                and info["heldout_conversations"] >= config.min_heldout,
                "Synthetic conversations did not yield enough independent hindsight",
            )
            row.update(info)
        with reporter.stage("calibrate") as row:
            rows = await asyncio.to_thread(
                calibration_run,
                config,
                root,
                reporter.output,
                args.lengths,
                args.timeout,
            )
            row.update(
                rows=rows,
                peak_memory_bytes=int(max(r["peak_memory_gb"] for r in rows) * 1e9),
                memory_kind="torch_process_peak_allocated",
            )
            shutil.copy2(root / "calibration.json", reporter.output)
        config = Config.load(root)
        config.slice_limit = args.select_top
        config.save(root)
        try:
            await asyncio.to_thread(
                child_run, "night", root, reporter.output, args.timeout
            )
        finally:
            for name in ("night-metrics.json", "night-rss.json", "gate.json"):
                if (root / name).is_file():
                    shutil.copy2(root / name, reporter.output)
            metrics = (
                json.loads((root / "night-metrics.json").read_text(encoding="utf-8"))
                if (root / "night-metrics.json").is_file()
                else {}
            )
            for stage, values in metrics.items():
                reporter.emit(
                    stage,
                    **{k: v for k, v in values.items() if k != "measured_seconds"},
                )
        verdict = json.loads((root / "gate.json").read_text(encoding="utf-8"))
        check_cancelled()
        with reporter.stage("accept") as row:
            row.update(
                heldout_change=verdict.get("heldout_change"),
                drift=verdict.get("general_drift"),
                evaluations=verdict.get("scale_evaluations"),
                accepted=verdict.get("accepted", False),
            )
            require(
                verdict.get("accepted") and verdict.get("candidate"),
                "The production gate rejected the run; no acceptance claimed",
            )
            candidate = verdict["candidate"]
            require(
                Path(candidate["artifact"]).resolve().is_relative_to(root),
                "Export escaped the isolated root",
            )
            require(
                candidate["kind"] == upstream.serves[0],
                "Unexpected exported artifact kind",
            )
        with reporter.stage("restore"):
            await upstream.restore(parked, set())
            store.set("parked_models", [])
        if args.upstream == "ollama":
            digest = await asyncio.to_thread(file_digest, candidate["artifact"])
            if not (Path(ollama_folder) / "blobs" / ("sha256-" + digest)).exists():
                orphan_digest = digest
        await serve_and_undo(config, store, upstream, candidate, reporter)
    finally:
        with reporter.stage("cleanup") as row:
            errors = []
            parked = parked or store.get("parked_models", [])

            async def attempt(action):
                try:
                    await action()
                except Exception as exc:
                    errors.append(str(exc))

            if upstream and args.upstream == "ollama":
                await attempt(lambda: cleanup_ollama(upstream, config, store))
                if orphan_digest and not errors:
                    try:
                        remove_orphan_upload(Path(ollama_folder), orphan_digest)
                    except Exception as exc:
                        errors.append(str(exc))
            elif upstream and candidate and args.upstream == "vllm":
                await attempt(lambda: upstream.unload(candidate))
            if upstream and touched:
                if args.upstream == "llamacpp":
                    await attempt(upstream.close)
                else:
                    await attempt(lambda: upstream.restore(parked, set()))
            if upstream:
                await attempt(upstream.close)
            row.update(errors=errors)
            require(
                model_digest(ref.path) == original_fingerprint,
                "The original model files changed during the test",
            )
            require(not errors, "Cleanup failed: " + "; ".join(errors))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", choices=("ollama", "llamacpp", "vllm"))
    p.add_argument("--model", help="Exact id discovered in the running upstream")
    p.add_argument("--url", help="Loopback OpenAI-compatible base, including /v1")
    p.add_argument(
        "--live-experience",
        action="store_true",
        help="Capture all learning conversations through the live proxy (default for vLLM)",
    )
    p.add_argument("--ollama-models-dir", type=Path)
    p.add_argument("--server-command", help="Exact server argv as a JSON array")
    p.add_argument("--output", type=Path, default=Path("test-output/linux-e2e"))
    p.add_argument(
        "--select-top", type=int, default=8, help="Selected experiences per cycle"
    )
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 512])
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument(
        "--timeout", type=int, default=7200, help="Deadline per heavy child, seconds"
    )
    p.add_argument("--child", choices=("gpu", "night"), help=argparse.SUPPRESS)
    p.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.child:
        require(
            args.data_dir is not None and sys.platform == "linux",
            "Child requires Linux and --data-dir",
        )
        (gpu_probe if args.child == "gpu" else night_child)(args.data_dir)
        return
    args.live_experience = args.live_experience or args.upstream == "vllm"
    if not args.upstream or not args.model:
        p.error("--upstream and --model are required")
    if min(args.select_top, args.timeout) <= 0 or min(args.lengths) <= 64:
        p.error("budgets must be positive; sequence lengths must exceed 64")
    if sys.platform != "linux":
        p.error("This harness requires Linux with a CUDA/ROCm GPU")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: STOP.set())
    # Discover the original Ollama library before HOME changes (including the
    # system service's distinct home on Linux).
    folder = args.ollama_models_dir or os.environ.get("OLLAMA_MODELS")
    if args.upstream == "ollama" and not folder:
        folder = Path.home() / ".ollama/models"
        if not folder.exists():
            import pwd

            with contextlib.suppress(KeyError):
                folder = Path(pwd.getpwnam("ollama").pw_dir) / ".ollama/models"
    output = args.output.resolve() / (args.upstream + "-" + uuid.uuid4().hex)
    reporter = Reporter(output)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="songe-linux-e2e-") as temporary:
            scratch = Path(temporary)
            with isolated_environment(scratch):
                try:
                    asyncio.run(exercise(args, scratch, reporter, str(folder or "")))
                finally:
                    preserve_diagnostics(scratch / "data", output)
        reporter.emit(
            "complete",
            time.monotonic() - started,
            output=str(output),
            scratch_removed=True,
        )
    except BaseException as exc:
        reporter.emit(
            "complete",
            time.monotonic() - started,
            status="failed",
            error=str(exc),
            output=str(output),
        )
        raise


if __name__ == "__main__":
    main()
