# tests/test_app.py
# The wrapper's session logic, on CPU, against a scripted stand-in for SLIP's
# Inference_module: the model itself is not under test, what the wrapper does
# with its output is.
#   pip install pytest httpx fastapi python-multipart numpy torch scipy
#   pytest tests
import gzip
import io
import sys
import threading
import time
import types

import numpy as np
import pytest
import torch


class FakeSLIP:
    """Inference_module as app.py uses it. `answer(zyx, label)` scripts the raw
    mask a click returns (padded, processed-space shape == uploaded+padding)."""

    def __init__(self, **kw):
        self.max_size = kw.get("max_size", 512)
        self.device = "cpu"
        self.dtype = torch.float32
        self.patch_size = (32, 192, 192)
        self.axis_permutation = None
        self.transformed_shape = None
        self.bbox_data = None
        self.action_history = []
        self.bbox_reverse_patches = []
        self.embeds = 0                 # process_image calls so far
        self.embeds_at_click = []       # value of `embeds` when each click ran
        self.answer = None

    def process_image(self, image_path, gt_path, target_label, numpy_image=None):
        self.embeds += 1
        self.shape = numpy_image.shape
        self.transformed_shape = numpy_image.shape
        pz, py, px = self.patch_size  # one patch, at the origin
        self.bbox_data = {"p0": {"bbox_coords": ((0, pz), (0, py), (0, px))}}

    def click_inference(self, clicks, labels):
        self.embeds_at_click.append(self.embeds)
        self.action_history.append({"type": "include" if labels[-1] else "exclude", "coords": clicks})
        raw = self.answer(clicks[-1], labels[-1]) if self.answer else np.zeros(self.shape, bool)
        return torch.from_numpy(raw.astype(np.float32))[None, None]

    def reset(self):
        self.action_history = []
        for d in (self.bbox_data or {}).values():
            for k in ("previous_masks", "high_res_masks", "low_res_masks"):
                d.pop(k, None)

    def undo(self, action_history):
        action_history.pop()
        return torch.zeros((1, 1) + self.shape)


fake_module = types.ModuleType("SLIP_model.inference_module")
fake_module.Inference_module = FakeSLIP
sys.modules.setdefault("SLIP_model", types.ModuleType("SLIP_model"))
sys.modules["SLIP_model.inference_module"] = fake_module

import app  # noqa: E402  (after the stand-in is registered)
from fastapi.testclient import TestClient  # noqa: E402

SHAPE = (4, 8, 8)  # padded to SLIP's 32x192x192 patch inside the wrapper


def npy(a, gz=False):
    buf = io.BytesIO()
    np.save(buf, a)
    return gzip.compress(buf.getvalue()) if gz else buf.getvalue()


def mask_of(r):
    assert r.status_code == 200, r.text
    raw = r.content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return np.load(io.BytesIO(raw)).astype(bool)


def box(*slices, shape=SHAPE):
    m = np.zeros(shape, bool)
    m[slices] = True
    return m


def padded(m):
    """A mask in uploaded space, as the fake must return it (padded space)."""
    out = np.zeros(app._padded_shape(m.shape), bool)
    out[tuple(slice(0, s) for s in m.shape)] = m
    return out


@pytest.fixture()
def client(monkeypatch):
    app._predictor = FakeSLIP()
    for k in list(app.STATE):
        app.STATE[k] = [] if isinstance(app.STATE[k], list) else None
    app.STATE.update(ready=False, coarse_req=0, coarse_active=False, rearm_coarse=False)
    # Promotions run on a background thread: each test waits for its own, so
    # none lands on the next test's session.
    promotions = []
    promote = app._promote_to_full_res

    def tracked():
        promotions.append(threading.Event())
        try:
            promote()
        finally:
            promotions[-1].set()

    monkeypatch.setattr(app, "_promote_to_full_res", tracked)
    yield TestClient(app.app)
    for done in promotions:
        done.wait(10)


def upload(client, shape=SHAPE, coarse=0):
    data = {"coarse": str(coarse)} if coarse else {}
    r = client.post("/upload_image", files={"file": ("image.npy", npy(np.zeros(shape, np.int16)))}, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def start_object(client, seed=None, shape=SHAPE):
    seed = np.zeros(shape, np.uint8) if seed is None else seed.astype(np.uint8)
    r = client.post("/upload_segment", files={"file": ("segment.npy.gz", npy(seed, gz=True))})
    assert r.status_code == 200, r.text


def click(client, zyx, positive=True):
    return client.post("/add_point_interaction", json={"voxel_coord": list(zyx), "positive_click": positive})


# --- an existing segmentation is where the object starts ----------------------

def test_a_positive_click_keeps_the_existing_segmentation(client):
    upload(client)
    seed = box(slice(0, 2), slice(0, 3), slice(0, 3))
    found = box(slice(2, 4), slice(5, 8), slice(5, 8))
    app._predictor.answer = lambda zyx, label: padded(found)  # SLIP sees only the new structure
    start_object(client, seed)
    assert (mask_of(click(client, (3, 6, 6))) == (seed | found)).all()


def test_a_negative_click_removes_only_what_it_touches(client):
    upload(client)
    seed = box(slice(0, 4), slice(0, 8), slice(0, 4))
    near = box(slice(0, 4), slice(0, 2), slice(0, 4))   # region around the click, dropped
    far = box(slice(0, 4), slice(6, 8), slice(0, 4))    # dropped by a patch re-judged empty
    app._predictor.answer = lambda zyx, label: padded(seed & ~near & ~far)
    start_object(client, seed)
    assert (mask_of(click(client, (1, 1, 1), positive=False)) == (seed & ~near)).all()


def test_a_negative_click_outside_the_object_removes_nothing(client):
    upload(client)
    seed = box(slice(0, 4), slice(0, 4), slice(0, 4))
    app._predictor.answer = lambda zyx, label: np.zeros(app._padded_shape(SHAPE), bool)
    start_object(client, seed)
    assert (mask_of(click(client, (1, 6, 6), positive=False)) == seed).all()


def test_undoing_the_only_click_goes_back_to_the_existing_segmentation(client):
    upload(client)
    seed = box(slice(0, 2), slice(0, 3), slice(0, 3))
    app._predictor.answer = lambda zyx, label: padded(box(slice(2, 4), slice(5, 8), slice(5, 8)))
    start_object(client, seed)
    click(client, (3, 6, 6))
    assert (mask_of(client.post("/undo_interaction")) == seed).all()


def test_the_existing_segmentation_becomes_the_patch_prior(client):
    upload(client)
    seed = box(slice(0, 2), slice(0, 3), slice(0, 3))
    start_object(client, seed)
    prior = app._predictor.bbox_data["p0"]["previous_masks"]
    assert (prior[:4, :8, :8].numpy() > 0).tolist() == seed.tolist()


def test_an_empty_segmentation_starts_an_empty_object(client):
    upload(client)
    found = box(slice(0, 1), slice(0, 1), slice(0, 1))
    app._predictor.answer = lambda zyx, label: padded(found)
    start_object(client)
    assert (mask_of(click(client, (0, 0, 0))) == found).all()


# --- the first click does not wait for an embedding it already has -----------

COARSE_SHAPE = (64, 384, 384)  # coarse cap 192 genuinely downsizes this one


def test_the_first_click_after_upload_reuses_the_coarse_embedding(client):
    info = upload(client, COARSE_SHAPE, coarse=192)
    assert info["coarse_active"] is True
    start_object(client, shape=COARSE_SHAPE)
    assert click(client, (1, 1, 1)).status_code == 200
    assert app._predictor.embeds_at_click == [1]  # the upload's embedding, not a second one


def test_a_new_object_after_promotion_gets_the_coarse_view_again(client):
    upload(client, COARSE_SHAPE, coarse=192)
    start_object(client, shape=COARSE_SHAPE)
    click(client, (1, 1, 1))
    deadline = time.time() + 10
    while app.STATE["coarse_active"] and time.time() < deadline:  # background promotion
        time.sleep(0.05)
    assert app.STATE["coarse_active"] is False
    start_object(client, shape=COARSE_SHAPE)
    click(client, (2, 2, 2))
    upload_embed, promotion = 1, 1
    assert app._predictor.embeds_at_click[-1] == upload_embed + promotion + 1
