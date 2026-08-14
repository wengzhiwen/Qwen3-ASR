# GB10 (DGX Spark, sm_121) deployment image.
#
# Why this base: NGC vLLM 26.02 ships CUDA 13.1 with sm_121 kernels and pins
# vllm 0.15.1 / transformers 4.57.5 / torch 2.11 — exactly the stack this
# fork's compatibility patches target. Upstream HF wheels (cu129/cu130 generic)
# crash on GB10 with missing sm_121 kernels, and pip dependency resolution
# would replace the image's pinned stack — hence every install below uses
# --no-deps with an explicitly flattened closure.
FROM nvcr.io/nvidia/vllm:26.02-py3

# Runtime deps for qwen-asr's import chain + the flask streaming server.
# All --no-deps: never touch the image's vllm/transformers/torch pins.
RUN pip install --no-cache-dir --no-deps \
    nagisa soynlp librosa soundfile flask pytz qwen-omni-utils \
    lazy_loader msgpack scikit-learn scipy joblib narwhals threadpoolctl soxr \
    werkzeug itsdangerous click jinja2 markupsafe blinker

# VLLM_WORKER_MULTIPROC_METHOD=spawn: EngineCore must not fork after CUDA init
#   ("Cannot re-initialize CUDA in forked subprocess").
# PYTORCH_CUDA_ALLOC_CONF: on GB10 unified memory the torch CUDA allocator
#   ratchets up and never returns pages to the OS; expandable segments keeps
#   long sessions from leaking address space.
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# qwen-asr source is live-mounted at /ws by docker-compose (editable install,
# code changes only need a container restart). Model weights mount at /models.
WORKDIR /ws
EXPOSE 8000
