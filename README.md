# slip-server

[SLIP](https://github.com/IRCAD/SLIP) (IRCAD, [arXiv 2607.22332](https://arxiv.org/abs/2607.22332))
— interactive 3D medical-image segmentation from point clicks — packaged as a
container meant to run as an **on-demand GPU pod**: started when someone needs
it, deleted when nobody does.

```
docker pull ghcr.io/imagin4care/slip-server:latest
```

## What is in the image

| File | Role |
| --- | --- |
| `app.py` | FastAPI wrapper exposing SLIP over the nnInteractive-style wire protocol: upload a volume once, then each click returns the refined mask |
| `pod_main.py` | Guard around `app.py` for a pod that is reachable from the internet and billed per second |
| `Dockerfile` | `python:3.11-slim` + SLIP at a pinned commit + the checkpoint baked in (about 3.6 GB compressed) |

`pod_main.py` adds three things:

- **Token check.** Every request needs `Authorization: Bearer $SLIP_TOKEN`, or
  the token alone in `X-Slip-Token` (for a serverless endpoint, where RunPod
  takes the `Authorization` header for the account's API key). Without
  `SLIP_TOKEN` set the server answers nobody. `GET /ping` is the only open
  route (liveness, returns `ok`).
- **Chunked uploads.** RunPod's HTTP proxy rejects request bodies near 100 MB
  and a CT volume is larger. A client sends `PUT /_relay/<id>/<n>` chunks in
  order, then `POST /_relay/<id>/commit` with `X-Relay-Path` and
  `X-Relay-Content-Type`; the body is reassembled and run through the real route.
- **Idle self-delete.** After `SLIP_IDLE_SHUTDOWN_S` seconds without an
  authenticated request the pod deletes itself through the RunPod API, using
  the pod-scoped key RunPod injects. A forgotten pod cannot bill for days.

## Endpoints

| Route | Method | Body | Response |
| --- | --- | --- | --- |
| `/ping` | GET | – | `ok` (no token needed) |
| `/` | GET | – | `{status, device, ready, …}` |
| `/upload_image` | POST | multipart `.npy` volume `[nz,ny,nx]` (optionally gzipped) | `{status, shape}` — computes patch embeddings |
| `/upload_segment` | POST | multipart `.npy.gz` seed mask | starts a new object |
| `/add_point_interaction` | POST | `{voxel_coord: [z,y,x], positive_click: bool}` | gzipped uint8 `.npy` mask `[nz,ny,nx]` |
| `/undo_interaction` | POST | – | mask after undoing the last click |

SLIP is point-only: the bbox, scribble and lasso routes answer `501`.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `SLIP_TOKEN` | *(required)* | shared secret for the token check |
| `SLIP_IDLE_SHUTDOWN_S` | `420` | idle seconds before the pod deletes itself; `0` disables the watchdog |
| `SLIP_MAX_LIFETIME_S` | `28800` | hard cap on a pod's life, whatever the activity |
| `SLIP_TORCH_COMPILE` | `0` | `1` trades a slow first click for faster later ones |
| `SLIP_MAX_SIZE` | `512` | largest volume side before SLIP downsizes |

## Running it

Needs an NVIDIA GPU with about 10 GB of VRAM and a **CUDA 13** driver: SLIP
pins `torch==2.11`, whose wheels are built against CUDA 13.

```bash
docker run --gpus all -p 1529:1529 -e SLIP_TOKEN=change-me -e SLIP_IDLE_SHUTDOWN_S=0 \
  ghcr.io/imagin4care/slip-server:latest
curl http://localhost:1529/ping
```

On RunPod, as a pod: expose `1529/http`, set `SLIP_TOKEN`, pick a 16 GB+ card
on a CUDA 13.0 host. No volume is needed — nothing is stored.

On RunPod, as a serverless **load-balancing endpoint**: same image and port,
with `PORT=1529`, `PORT_HEALTH=1529`, `SLIP_TOKEN` and `SLIP_IDLE_SHUTDOWN_S=0`
(the endpoint scales to zero by itself). RunPod keeps the image pulled on the
endpoint's idle workers, so a start is the model load alone — 40 to 75 s
instead of the three to four minutes a fresh pod needs to download the image.
Call it at `https://<endpoint id>.api.runpod.ai` with the RunPod API key as the
bearer and the token in `X-Slip-Token`. Requests are capped at 30 MB there, so
larger uploads go through `/_relay` as above. Set **max workers to 1**: the
session lives in the worker's memory, and RunPod's load balancer spreads
requests over every running worker, so with two of them clicks reach one that
never saw the volume ("no image processed").

## Tests

The wrapper's session logic runs on CPU against a scripted stand-in for SLIP:

```bash
pip install pytest httpx fastapi python-multipart numpy torch scipy
python -m pytest tests
```

Volumes smaller than SLIP's 32×192×192 patch are edge-padded before
embedding and the padding is cropped off every returned mask, so voxel
coordinates are unchanged.

## License

SLIP is **GPL v3** (its modified SAM 2 components are Apache 2.0). This
wrapper imports it and is GPL v3 as well — see [LICENSE](LICENSE). The model
weights are downloaded from IRCAD at build time and remain under IRCAD's terms.
