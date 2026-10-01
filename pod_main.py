# pod_main.py
# RunPod entry point for the SLIP server. Wraps app.py (unchanged) in a plain
# ASGI guard that adds what an on-demand, pay-per-second pod needs and a dev
# box does not:
#   - a shared-secret check (the pod's proxy URL is reachable from the internet),
#   - chunked uploads (RunPod's HTTP proxy caps a request body near 100 MB; a
#     CT volume is larger), and
#   - a watchdog that deletes this pod when nobody has used it for a while.
#     The gateway normally stops the pod first; this is the backstop for when
#     the gateway is down, so a forgotten pod can never bill for days.
#
# License note: GPL v3, like app.py which it imports.
import asyncio
import hmac
import json
import os
import re
import tempfile
import threading
import time
import urllib.request

from app import app as slip_app  # loads the model at import time

TOKEN = os.environ.get("SLIP_TOKEN", "")
IDLE_S = int(os.environ.get("SLIP_IDLE_SHUTDOWN_S", "420"))
MAX_LIFETIME_S = int(os.environ.get("SLIP_MAX_LIFETIME_S", "28800"))
RELAY_MAX_BYTES = int(os.environ.get("SLIP_RELAY_MAX_BYTES", str(2 * 1024 ** 3)))
RELAY_DIR = tempfile.mkdtemp(prefix="relay-")
RELAY_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

_started = time.monotonic()
_last_activity = time.monotonic()
_relays = {}  # upload id -> {"next": int, "size": int}


def _touch() -> None:
    global _last_activity
    _last_activity = time.monotonic()


def _authorized(scope) -> bool:
    if not TOKEN:
        return False  # fail closed: a pod without a token serves nobody
    want = b"Bearer " + TOKEN.encode()
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            return hmac.compare_digest(value, want)
    return False


async def _reply(send, status: int, body: bytes, content_type: bytes = b"application/json", extra=()) -> None:
    headers = [(b"content-type", content_type), (b"content-length", str(len(body)).encode()), *extra]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _reply_json(send, status: int, obj) -> None:
    await _reply(send, status, json.dumps(obj).encode())


async def _dispatch(path: str, content_type: str, body: bytes):
    """Run one request through the wrapped app in-process and collect its response."""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "",
        "headers": [(b"content-type", content_type.encode()), (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 0), "server": ("127.0.0.1", 1529),
    }
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()  # no client behind this call, so never a disconnect

    out = {"status": 500, "headers": [], "body": bytearray()}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = list(message.get("headers", []))
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    await slip_app(scope, receive, send)
    return out


def _relay_path(upload_id: str) -> str:
    return os.path.join(RELAY_DIR, upload_id)


def _drop_relay(upload_id: str) -> None:
    _relays.pop(upload_id, None)
    try:
        os.remove(_relay_path(upload_id))
    except OSError:
        pass


async def _relay_chunk(upload_id: str, seq: int, receive, send) -> None:
    """PUT /_relay/<id>/<seq>: append one chunk. Chunks must arrive in order from 0."""
    state = _relays.get(upload_id)
    if seq == 0:
        _drop_relay(upload_id)
        state = _relays[upload_id] = {"next": 0, "size": 0}
    if state is None or seq != state["next"]:
        await _reply_json(send, 409, {"error": "chunk out of order", "expected": state["next"] if state else 0})
        return
    with open(_relay_path(upload_id), "ab") as f:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                _drop_relay(upload_id)
                return
            chunk = message.get("body", b"")
            state["size"] += len(chunk)
            if state["size"] > RELAY_MAX_BYTES:
                f.close()
                _drop_relay(upload_id)
                await _reply_json(send, 413, {"error": "upload too large"})
                return
            f.write(chunk)
            if not message.get("more_body"):
                break
    state["next"] += 1
    await _reply_json(send, 200, {"received": state["size"]})


async def _relay_commit(upload_id: str, scope, send) -> None:
    """POST /_relay/<id>/commit: replay the assembled body against the real route."""
    if upload_id not in _relays:
        await _reply_json(send, 404, {"error": "unknown upload"})
        return
    headers = {k.decode("latin1"): v.decode("latin1") for k, v in scope.get("headers", [])}
    path = headers.get("x-relay-path", "")
    content_type = headers.get("x-relay-content-type", "application/octet-stream")
    if not path.startswith("/") or path.startswith("/_relay"):
        _drop_relay(upload_id)
        await _reply_json(send, 400, {"error": "bad x-relay-path"})
        return
    with open(_relay_path(upload_id), "rb") as f:
        body = f.read()
    _drop_relay(upload_id)
    out = await _dispatch(path, content_type, body)
    del body
    content_type_out = next((v for k, v in out["headers"] if k == b"content-type"), b"application/octet-stream")
    await _reply(send, out["status"], bytes(out["body"]), content_type_out)


async def asgi(scope, receive, send):
    if scope["type"] != "http":
        await slip_app(scope, receive, send)  # lifespan
        return
    path = scope["path"]
    # Unauthenticated liveness probe: says nothing about the session and does
    # not count as activity, so polling it cannot keep the pod alive.
    if path == "/ping":
        await _reply(send, 200, b"ok", b"text/plain")
        return
    if not _authorized(scope):
        await _reply_json(send, 401, {"error": "unauthorized"})
        return
    _touch()
    parts = path.strip("/").split("/")
    if parts[0] == "_relay" and len(parts) == 3 and RELAY_ID.match(parts[1]):
        if parts[2] == "commit" and scope["method"] == "POST":
            await _relay_commit(parts[1], scope, send)
        elif parts[2].isdigit() and scope["method"] == "PUT":
            await _relay_chunk(parts[1], int(parts[2]), receive, send)
        else:
            await _reply_json(send, 404, {"error": "not found"})
        _touch()
        return
    await slip_app(scope, receive, send)
    _touch()  # a long embedding pass counts as activity until it returns


def _delete_self() -> bool:
    pod = os.environ.get("RUNPOD_POD_ID")
    key = os.environ.get("RUNPOD_API_KEY")
    if not pod or not key:
        print("watchdog: not on RunPod (no RUNPOD_POD_ID / RUNPOD_API_KEY), nothing to delete", flush=True)
        return True
    attempts = (
        ("DELETE", f"https://rest.runpod.io/v1/pods/{pod}", None),
        ("POST", "https://api.runpod.io/graphql",
         json.dumps({"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pod}).encode()),
    )
    for method, url, data in attempts:
        try:
            # An explicit User-Agent is required: RunPod's API is behind
            # Cloudflare, which answers Python's default one with 403 / 1010.
            req = urllib.request.Request(url, data=data, method=method, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "User-Agent": "seg-slip-pod/1",
            })
            with urllib.request.urlopen(req, timeout=30) as r:
                print(f"watchdog: {method} {url} -> HTTP {r.status}", flush=True)
                return True
        except Exception as e:  # noqa: BLE001 - any failure moves on to the next way
            print(f"watchdog: {method} {url} failed: {e}", flush=True)
    return False


def _watchdog() -> None:
    while True:
        time.sleep(15)
        now = time.monotonic()
        idle, alive = now - _last_activity, now - _started
        if idle < IDLE_S and alive < MAX_LIFETIME_S:
            continue
        reason = f"idle {int(idle)}s" if idle >= IDLE_S else f"max lifetime {int(alive)}s"
        print(f"watchdog: {reason} -> deleting this pod", flush=True)
        while not _delete_self():
            time.sleep(60)  # keep trying: exiting instead would just restart the container
        if not os.environ.get("RUNPOD_POD_ID"):
            os._exit(0)  # local run: stop the container
        time.sleep(3600)


if IDLE_S > 0:
    threading.Thread(target=_watchdog, name="idle-watchdog", daemon=True).start()
