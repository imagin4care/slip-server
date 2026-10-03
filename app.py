# app.py
# FastAPI wrapper exposing IRCAD SLIP (https://github.com/IRCAD/SLIP) over the
# nnInteractive wire protocol used by @imagin/seg-core's SLIPProvider.
# Single global session (one volume, one object), like the SAM/nnInteractive servers.
#
# License note: SLIP is GPL v3; this wrapper imports it and is therefore also
# GPL v3. It ships only as a deploy-side service — the browser toolkit never
# links against it.
import asyncio
import gc
import gzip
import io
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from scipy import ndimage

from SLIP_model.inference_module import Inference_module

CKPT = os.environ.get("SLIP_CKPT", "/weights/SLIP_ckpt.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_COMPILE = os.environ.get("SLIP_TORCH_COMPILE", "1") != "0"
# SLIP downsizes any volume whose largest axis exceeds max_size (isotropically,
# mapping clicks in / masks out itself). 512 is upstream's default; clients may
# request a smaller per-volume value ("coarse" upload field) so one 32x192x192
# patch covers more anatomy and click propagation doesn't truncate large organs.
MAX_SIZE = int(os.environ.get("SLIP_MAX_SIZE", "512"))

app = FastAPI()

# keep_largest_component=False: interactive editing must not silently drop
# disconnected regions the user clicked into existence.
_predictor = Inference_module(
    device=DEVICE,
    checkpoint=CKPT,
    max_size=MAX_SIZE,
    activate_propagation=True,
    keep_largest_component=False,
    torch_compile=TORCH_COMPILE,
    vis=False,
)

# Session state. The coarse fields implement "coarse first click, then refine at
# full resolution": the first click of an object runs on downsized embeddings so
# propagation can span the whole structure, then the session is promoted — full
# resolution embeddings, with that coarse mask injected as every patch's prior.
STATE = {
    "volume": None,        # float32 [nz,ny,nx] as uploaded
    "shape": None,         # uploaded shape (masks are returned in it)
    "ready": False,        # embeddings computed
    "coarse_req": 0,       # cap the client asked for on this volume (0 = none)
    "coarse_active": False,  # current embeddings come from a downsized pass
    "rearm_coarse": False,   # next first click should re-embed coarse
    "clicks": [],          # [([z,y,x], label)] of the current object, for replay
    "seed": None,          # the label's voxels the object started from, uint8, or None
    "last_mask": None,     # accumulated object mask, uploaded shape, uint8
    "history": [],         # packed accumulated masks, one per click, for undo
}
_lock = threading.Lock()
# Logit magnitude for injected patch priors. SLIP stores mask logits (not
# binaries) and feeds them back as SAM-style mask prompts; +-8 reads as a
# confident prior without saturating the decoder.
SEED_LOGIT = 8.0

# SLIP's sliding window asserts the volume — after it moves the smallest axis to
# the front — is at least this big. Typical clinical volumes are thinner than
# 192 in plane (the demo head is 128x128x88), so short axes get padded.
PATCH = (32, 192, 192)


def _padded_shape(shape):
    """Smallest shape >= `shape` that satisfies SLIP's patch assertion.

    SLIP permutes the smallest axis to position 0, so that axis only needs
    PATCH[0]; the other two need PATCH[1]/PATCH[2]. Padding never shrinks an
    axis, and the axis left small stays the smallest (the others end up >= 192),
    so the permutation SLIP picks afterwards is still the one assumed here.
    """
    out = list(shape)
    order = np.argsort(shape)  # smallest axis first, matching SLIP's argmin
    for rank, axis in enumerate(order):
        out[axis] = max(shape[axis], PATCH[rank])
    return tuple(out)


def _free_session() -> None:
    """Release everything the previous volume pinned on the GPU.

    process_image builds the new embedding set while the old one is still
    referenced by the predictor, so without this the peak VRAM of a second
    upload is old + new (per-patch embeddings, plus the full-res volume and GT
    copies vol_original/gt3D_original) — the classic second-image OOM.
    """
    for attr in ("bbox_data", "combiner", "vol", "vol_original", "gt3D", "gt3D_original"):
        if hasattr(_predictor, attr):
            setattr(_predictor, attr, None)
    _predictor.bbox_reverse_patches = []
    _predictor.action_history = []
    _predictor.click_points = None
    _predictor.click_labels = None
    _predictor.click_points_original = None
    _predictor.click_labels_original = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def _effective_max_size(shape, requested: int) -> int:
    """Largest-axis cap SLIP can actually honor for `shape` (0 = default).

    DownsizeIfLarge rescales every axis by cap/largest_axis, and SLIP then
    asserts the result still holds one 32x192x192 patch (smallest axis first).
    So the cap has a shape-dependent floor; tio.Resize truncates with int(),
    so we bump the cap until the truncated dims genuinely satisfy the patch.
    Returns the largest axis itself (i.e. no downsizing) when no valid cap
    below it exists — e.g. extreme aspect ratios.
    """
    cap = max(192, min(requested, MAX_SIZE) if requested > 0 else MAX_SIZE)
    s = sorted(shape)
    need = (32, 192, 192)  # per sorted rank, matching SLIP's patch
    while cap < s[2]:
        if all(int(d * cap / s[2]) >= n for d, n in zip(s, need)):
            return cap
        cap += 1
    return s[2]


def _process(vol: np.ndarray, coarse: int = 0) -> None:
    """(Re)compute patch embeddings and reset all interaction state.

    `coarse` (0 = off) requests a cap on the volume's largest axis before
    encoding: SLIP's own DownsizeIfLarge resamples isotropically and its
    coordinate transforms keep clicks/masks in original space, so a coarse
    session is transparent to the client — masks come back full-size, just
    from a lower-res model pass. The requested cap is raised to the smallest
    value whose downsized volume still fits SLIP's 32x192x192 patch.
    """
    _free_session()
    target = _padded_shape(vol.shape)
    _predictor.max_size = _effective_max_size(target, coarse)
    if coarse > 0:
        print(f"coarse requested {coarse} -> effective max_size {_predictor.max_size}", flush=True)
    if target != vol.shape:
        # Pad at the END of each axis only, so voxel coordinates — and therefore
        # the click coords the client sends — are unchanged. 'edge' replicates
        # border voxels instead of inventing a bright/dark rim.
        vol = np.pad(vol, [(0, t - s) for s, t in zip(vol.shape, target)], mode="edge")
        print(f"padded {STATE['shape']} -> {target} for SLIP's {PATCH} patch", flush=True)
    _predictor.process_image(
        image_path=None, gt_path=None, target_label=None, numpy_image=vol
    )
    # Downsizing only happened if the cap is below the volume's largest axis;
    # that is exactly when a later full-resolution promotion is worth doing.
    STATE["coarse_active"] = _predictor.max_size < max(target)
    STATE["ready"] = True


def _to_processed_space(mask: np.ndarray) -> torch.Tensor:
    """Uploaded-space uint8 mask -> (Z,H,W) float tensor in SLIP's processed space.

    Mirrors process_image / transform_mask_to_original_space in reverse: pad to
    the patch-assertion shape, apply the axis permutation SLIP chose, then
    resize to the transformed shape it encoded.
    """
    target = _padded_shape(mask.shape)
    if target != mask.shape:
        mask = np.pad(mask, [(0, t - s) for s, t in zip(mask.shape, target)], mode="edge")
    t = torch.from_numpy(mask.astype(np.float32)).to(_predictor.device)[None, None]
    if _predictor.axis_permutation is not None:
        t = t.permute([0, 1] + [2 + int(a) for a in _predictor.axis_permutation]).contiguous()
    want = tuple(int(s) for s in _predictor.transformed_shape)
    if tuple(t.shape[2:]) != want:
        t = F.interpolate(t, size=want, mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0)


def _seed_patch_priors(mask: np.ndarray) -> None:
    """Inject `mask` (uploaded space) as the prior of every patch.

    SLIP has no seed-mask API, but each patch keeps its own mask *logits*
    (`previous_masks` / `high_res_masks`, plus the half-resolution
    `low_res_masks` that is fed back to the decoder as a mask prompt). Writing
    those three fields makes the combined output equal the seed, and makes the
    next click refine the seed instead of starting from nothing.
    """
    p = _predictor
    if not getattr(p, "bbox_data", None):
        return
    vol_logits = (_to_processed_space(mask) > 0.5).to(p.dtype) * (2 * SEED_LOGIT) - SEED_LOGIT
    nz, ny, nx = vol_logits.shape
    lo = tuple(s // 2 for s in p.patch_size)
    for data in p.bbox_data.values():
        (z0, z1), (x0, x1), (y0, y1) = data["bbox_coords"]
        patch = torch.full(p.patch_size, -SEED_LOGIT, device=p.device, dtype=p.dtype)
        # Patches may hang off the volume (SLIP's sliding window allows negative
        # starts); fill only the in-bounds intersection, leave the rest background.
        sz0, sx0, sy0 = max(0, int(z0)), max(0, int(x0)), max(0, int(y0))
        sz1, sx1, sy1 = min(nz, int(z1)), min(ny, int(x1)), min(nx, int(y1))
        if sz1 <= sz0 or sx1 <= sx0 or sy1 <= sy0:
            continue
        dz, dx, dy = sz0 - int(z0), sx0 - int(x0), sy0 - int(y0)
        patch[dz:dz + (sz1 - sz0), dx:dx + (sx1 - sx0), dy:dy + (sy1 - sy0)] = \
            vol_logits[sz0:sz1, sx0:sx1, sy0:sy1]
        data["previous_masks"] = patch
        data["high_res_masks"] = patch
        data["low_res_masks"] = F.interpolate(
            patch[None, None].float(), size=lo, mode="trilinear", align_corners=False
        ).squeeze(0).squeeze(0).to(p.dtype)


def _start_object(seed=None) -> None:
    """Forget the current object's clicks; the object starts as `seed`.

    `seed` is the label's existing voxels (or None): clicks then refine it
    (_accumulate) and undoing every click comes back to it. Coarse is re-armed
    only when a promotion has replaced the coarse embeddings — right after an
    upload they still are coarse, and embedding them again just made the first
    click wait. The re-embed is deferred to that click (see add_point) so
    starting a new object stays instant.
    """
    STATE["clicks"] = []
    STATE["seed"] = seed
    STATE["last_mask"] = seed
    STATE["history"] = []
    STATE["rearm_coarse"] = STATE["coarse_req"] > 0 and not STATE["coarse_active"]


def _begin_object(seed=None) -> None:
    """A new object on the current embeddings, starting from `seed`."""
    _predictor.reset()
    _start_object(seed)
    # A pending coarse re-embed would drop the priors; add_point seeds them then.
    if seed is not None and not STATE["rearm_coarse"]:
        _seed_patch_priors(seed)


def _replay_clicks(clicks):
    """Re-issue recorded clicks so history, undo stack and CC filtering are real."""
    mask = None
    for (z, y, x), label in clicks:
        mask = _predictor.click_inference([[z, y, x]], [label])
    return mask


def _promote_to_full_res() -> None:
    """Re-encode at full resolution with the coarse result as the prior.

    Runs in a worker thread right after the first click answers, so the
    embedding pass overlaps with the user looking at that first mask; a second
    click simply waits on the lock. On any failure the coarse session is
    rebuilt so the object stays usable.
    """
    with _lock:
        if not (STATE["coarse_active"] and STATE["clicks"] and STATE["last_mask"] is not None):
            return
        seed, clicks, vol = STATE["last_mask"], list(STATE["clicks"]), STATE["volume"]
        try:
            _process(vol, coarse=0)
            _seed_patch_priors(seed)
            _replay_clicks(clicks)
            # The accumulated object mask is deliberately NOT replaced by the
            # replay's output: the replay exists to rebuild SLIP's internal
            # state, and the client already holds the coarse result.
            # Volumes bigger than MAX_SIZE are still downsized by SLIP itself,
            # but this session is now as fine as it gets: no further promotion.
            STATE["coarse_active"] = False
            print(f"promoted to full resolution ({len(clicks)} click(s) replayed)", flush=True)
        except Exception as e:  # noqa: BLE001 - keep the session alive whatever fails
            print(f"promotion failed ({e}); staying coarse", flush=True)
            try:
                _process(vol, coarse=STATE["coarse_req"])
                _seed_patch_priors(seed)
                _replay_clicks(clicks)
            except Exception as e2:  # noqa: BLE001
                print(f"coarse rebuild failed too: {e2}", flush=True)


def _to_client_mask(mask_t) -> np.ndarray:
    """(1,1,Z,H,W) torch mask -> uint8 [nz,ny,nx] in the uploaded shape."""
    m = (mask_t.squeeze().detach().to("cpu").numpy() > 0).astype(np.uint8)
    nz, ny, nx = STATE["shape"]
    # In a coarse session SLIP's downsize->upsize round-trip can come back a
    # voxel short per axis (int truncation on both legs): replicate the edge
    # up to the uploaded shape, then crop any patch-assertion padding.
    pads = [(0, max(0, t - s)) for s, t in zip(m.shape, (nz, ny, nx))]
    if any(p[1] for p in pads):
        m = np.pad(m, pads, mode="edge")
    return np.ascontiguousarray(m[:nz, :ny, :nx])


def _mask_body(m: np.ndarray) -> Response:
    """uint8 [nz,ny,nx] mask -> gzipped .npy response body."""
    buf = io.BytesIO()
    np.save(buf, m)
    return Response(
        content=gzip.compress(buf.getvalue()),
        media_type="application/octet-stream",
    )


def _region_at(mask: np.ndarray, zyx) -> np.ndarray:
    """The connected part of `mask` that holds voxel `zyx` (empty when none does)."""
    z, y, x = (min(max(int(c), 0), s - 1) for c, s in zip(zyx, mask.shape))
    out = np.zeros(mask.shape, dtype=np.uint8)
    if not mask[z, y, x]:
        return out
    # Label the mask's bounding box only: labelling a whole CT takes seconds.
    box = []
    for others in ((1, 2), (0, 2), (0, 1)):
        hit = np.flatnonzero(mask.any(axis=others))
        box.append(slice(int(hit[0]), int(hit[-1]) + 1))
    box = tuple(box)
    labels, _ = ndimage.label(mask[box])
    at = labels[z - box[0].start, y - box[1].start, x - box[2].start]
    out[box] = labels == at
    return out


def _accumulate(raw: np.ndarray, positive: bool, click) -> np.ndarray:
    """Fold one click's raw SLIP output into the object mask, monotonically.

    SLIP re-decodes and re-blends every patch on each click, and a patch the
    decoder calls empty carries a -1024 logit. The combiner averages
    overlapping patches before thresholding, so one such verdict erases what
    earlier clicks (or an injected prior, which is only +-8) had established
    there — that is why click 2 could destroy click 1's result. Clicks are
    therefore applied as deltas against the accumulated object instead of
    replacing it: a positive click may only ADD, a negative may only REMOVE,
    and only the region it was placed on — the same blending drops parts far
    from the click, which would otherwise erase an existing segmentation.
    """
    prev = STATE["last_mask"]
    if prev is None or prev.shape != raw.shape:
        return raw
    out = (prev | raw) if positive else (prev & (1 - _region_at(prev & (1 - raw), click)))
    added = int((out & ~prev).sum())
    removed = int((prev & ~out).sum())
    ignored = int((raw != out).sum())
    print(f"click {'+' if positive else '-'}: +{added} -{removed} voxels "
          f"({ignored} changes SLIP proposed were not applied)", flush=True)
    return out


def _mask_response(mask_t, positive: bool | None = None, click=None) -> Response:
    m = _to_client_mask(mask_t)
    if positive is not None:
        STATE["history"].append(np.packbits(STATE["last_mask"])
                                if STATE["last_mask"] is not None else None)
        m = _accumulate(m, positive, click)
    STATE["last_mask"] = m
    return _mask_body(m)


@app.get("/")
def root():
    return {
        "status": "ok",
        "device": DEVICE,
        "ready": STATE["ready"],
        # Session detail — handy for debugging a live box, and what the test
        # harness waits on to know a coarse->full-res promotion has landed.
        "max_size": _predictor.max_size,
        "coarse_active": STATE["coarse_active"],
        "clicks": len(STATE["clicks"]),
    }


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...), coarse: int = Form(0)):
    raw = await file.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    vol = np.load(io.BytesIO(raw)).astype(np.float32)  # [nz,ny,nx]

    def work():
        with _lock:
            STATE["volume"] = vol
            STATE["shape"] = vol.shape
            STATE["ready"] = False
            STATE["coarse_req"] = coarse
            STATE["rearm_coarse"] = False
            STATE["clicks"] = []
            STATE["seed"] = None
            STATE["last_mask"] = None
            STATE["history"] = []
            _process(vol, coarse=coarse)

    # Off the event loop: GPU work holds `_lock` for seconds (and the background
    # promotion can hold it for longer), and blocking the loop would freeze even
    # the status endpoint for the duration.
    await asyncio.to_thread(work)
    return JSONResponse({
        "status": "ok",
        "shape": list(vol.shape),
        "coarse": _predictor.max_size,
        "coarse_active": STATE["coarse_active"],
    })


@app.post("/upload_segment")
async def upload_segment(file: UploadFile = File(...)):
    # Start a new object from the label's existing voxels (all zeros for an
    # empty label). SLIP has no seed-mask API: the seed becomes the object's
    # accumulated mask, which every click refines (_accumulate), and every
    # patch's prior (_seed_patch_priors), which SLIP's decoder refines.
    raw = await file.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    seed = np.load(io.BytesIO(raw))

    def work():
        with _lock:
            if STATE["volume"] is None:
                return JSONResponse({"error": "no image uploaded"}, status_code=409)
            if seed.shape != STATE["shape"]:
                return JSONResponse({"error": f"segment shape {list(seed.shape)} is not the image's "
                                              f"{list(STATE['shape'])}"}, status_code=400)
            # reset() clears the memory bank, undo stack, and click history but keeps
            # the patch embeddings — a new object is instant instead of a re-embed.
            # Any coarse re-arm is deferred to the first click (see add_point).
            if not STATE["ready"]:
                _process(STATE["volume"], coarse=STATE["coarse_req"])
            start = (seed > 0).astype(np.uint8) if seed.any() else None
            _begin_object(start)
            if start is not None:
                print(f"upload_segment: object starts from {int(start.sum())} existing voxels", flush=True)
            return JSONResponse({"status": "ok"})

    return await asyncio.to_thread(work)


@app.post("/add_point_interaction")
async def add_point(req: Request):
    p = await req.json()
    if not STATE["ready"]:
        return JSONResponse({"error": "no image processed"}, status_code=503)
    z, y, x = (int(round(c)) for c in p["voxel_coord"])  # client sends zyx order
    label = 1 if p.get("positive_click", True) else 0

    def work():
        with _lock:
            # An object that went back to empty (new object / undo of everything)
            # re-embeds coarse lazily, here, so its first click gets the wide view.
            if STATE["rearm_coarse"] and not STATE["clicks"]:
                _process(STATE["volume"], coarse=STATE["coarse_req"])
                STATE["rearm_coarse"] = False
                if STATE["seed"] is not None:  # the re-embed dropped the seed's priors
                    _seed_patch_priors(STATE["seed"])
            first = not STATE["clicks"]
            mask = _predictor.click_inference([[z, y, x]], [label])
            STATE["clicks"].append(([z, y, x], label))
            return _mask_response(mask, positive=bool(label), click=(z, y, x)), first and STATE["coarse_active"]

    body, promote = await asyncio.to_thread(work)
    if promote:
        # Full-res re-encode overlaps with the user reading this first mask.
        threading.Thread(target=_promote_to_full_res, daemon=True).start()
    return body


@app.post("/undo_interaction")
async def undo_interaction():
    """Revert the last click server-side (SLIP keeps a per-click undo stack).

    Returns the recombined full-volume mask after removal — same body format as
    add_point_interaction — so the client can apply it directly instead of
    replaying the whole click chain.
    """
    if not STATE["ready"]:
        return JSONResponse({"error": "no image processed"}, status_code=503)

    def work():
        with _lock:
            if not _predictor.action_history:
                return JSONResponse({"error": "nothing to undo"}, status_code=409)
            if len(_predictor.action_history) == 1:
                # Undoing the only click: skip upstream undo (its recombine assumes
                # surviving per-patch masks) and start the object over from its seed.
                seed = STATE["seed"]
                _begin_object(seed)
                return _mask_body(seed if seed is not None else np.zeros(STATE["shape"], dtype=np.uint8))
            # Roll SLIP's own state back so later clicks behave, but return the
            # ACCUMULATED mask this object had before the undone click — the
            # recombine cannot reproduce it (see _accumulate).
            _predictor.undo(_predictor.action_history)
            if STATE["clicks"]:
                STATE["clicks"].pop()
            packed = STATE["history"].pop() if STATE["history"] else None
            n = int(np.prod(STATE["shape"]))
            restored = (np.unpackbits(packed, count=n).reshape(STATE["shape"])
                        if packed is not None else np.zeros(STATE["shape"], dtype=np.uint8))
            STATE["last_mask"] = restored
            return _mask_body(restored)

    return await asyncio.to_thread(work)


@app.post("/add_bbox_interaction")
async def add_bbox():
    return JSONResponse({"error": "SLIP supports point prompts only"}, status_code=501)


@app.post("/add_scribble_interaction")
async def add_scribble():
    return JSONResponse({"error": "SLIP supports point prompts only"}, status_code=501)


@app.post("/add_lasso_interaction")
async def add_lasso():
    return JSONResponse({"error": "SLIP supports point prompts only"}, status_code=501)
