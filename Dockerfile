# syntax=docker/dockerfile:1
# SLIP server image for an on-demand RunPod pod.
#   docker build -t slip-server .
#
# - python:slim base, not pytorch/pytorch: SLIP pins torch==2.11.0, so pip
#   would replace a base image's torch anyway and its 8 GB would be dead
#   weight. A pod is created fresh on every start, so image size is cold-start
#   time.
# - The checkpoint is baked in for the same reason: no volume survives a
#   deleted pod, and a volume that did would be billed while idle.
# - pod_main.py guards the app (token, chunked uploads, idle self-delete).
# SLIP is GPL v3 (modified SAM 2 parts Apache 2.0).
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git wget ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ARG SLIP_REF=d22df6e5acf5469808392333f68197e0d43d7ad0
RUN git clone https://github.com/IRCAD/SLIP.git /opt/SLIP \
    && git -C /opt/SLIP checkout --quiet ${SLIP_REF} \
    && rm -rf /opt/SLIP/.git
# The cache mount keeps the multi-GB wheels out of the image and lets a build
# that died on a flaky download resume instead of starting over.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 -e /opt/SLIP \
    && pip install fastapi uvicorn python-multipart
ARG SLIP_CKPT_URL=https://cloud.ircad.fr/s/8KW4wa8JGrwm7GP/download
RUN mkdir -p /weights && wget -q -O /weights/SLIP_ckpt.pth "${SLIP_CKPT_URL}"
COPY app.py pod_main.py ./
# torch's pip wheels carry their own CUDA libraries; the host only has to
# expose the driver, which these two variables ask the NVIDIA runtime for.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    SLIP_CKPT=/weights/SLIP_ckpt.pth \
    SLIP_TORCH_COMPILE=0 \
    PYTHONUNBUFFERED=1
EXPOSE 1529
CMD ["uvicorn", "pod_main:asgi", "--host", "0.0.0.0", "--port", "1529"]
