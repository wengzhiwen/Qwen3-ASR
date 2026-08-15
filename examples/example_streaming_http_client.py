# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example: HTTP client for the deployed streaming service (GB10 deployment).

Sibling of example_qwen3_asr_vllm_streaming.py, which streams in-process
(this machine loads the model). This one talks to the hardened streaming
server over HTTP — the same protocol a browser client uses:

    POST /api/start                          -> {"session_id": ...}
    POST /api/chunk?session_id=...           -> {"language": ..., "text": ...}
         body: 16 kHz mono float32 LE PCM, application/octet-stream
         text is the ACCUMULATED transcript so far
    POST /api/finish?session_id=...          -> final transcript
    GET  /api/stats                          -> diagnostics (sessions/RMS/...)

Prerequisites:
  - The service must be running, e.g. via deploy/docker-compose.yml
  - pip install requests soundfile numpy

Real-time clients (see the web demo served at "/" of the service) capture
mic audio and send a chunk every 500 ms; while a chunk request is in
flight they DROP new audio rather than queueing it. This file replays a
wav sequentially for simplicity, which the server serializes anyway.
"""

import argparse
import io
import urllib.request
from typing import Tuple

import numpy as np
import requests
import soundfile as sf

URL_EN = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
SAMPLE_RATE = 16000


def _download_audio_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Simple resample to 16k if needed (uses linear interpolation; good enough for a test)."""
    if sr == SAMPLE_RATE:
        return wav.astype(np.float32, copy=False)
    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * SAMPLE_RATE))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def stream_over_http(base_url: str, wav16k: np.ndarray, step_ms: int) -> None:
    step = int(round(step_ms / 1000.0 * SAMPLE_RATE))

    resp = requests.post(f"{base_url}/api/start", timeout=30)
    resp.raise_for_status()
    session_id = resp.json()["session_id"]

    print(f"\n===== HTTP streaming step = {step_ms} ms =====")
    pos = 0
    call_id = 0
    while pos < wav16k.shape[0]:
        seg = wav16k[pos : pos + step]
        pos += seg.shape[0]
        call_id += 1
        # float32 little-endarian PCM bytes = what the browser sends as
        # Float32Array.buffer over fetch().
        body = seg.astype("<f4", copy=False).tobytes()
        resp = requests.post(
            f"{base_url}/api/chunk",
            params={"session_id": session_id},
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"[call {call_id:03d}] language={result['language']!r} text={result['text']!r}")

    resp = requests.post(f"{base_url}/api/finish", params={"session_id": session_id}, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    print(f"[final] language={result['language']!r} text={result['text']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8001", help="streaming service base URL")
    parser.add_argument("--audio", default=None, help="local wav path (default: download the sample)")
    parser.add_argument("--step-ms", type=int, default=500, help="chunk size in ms (recommended: 500)")
    args = parser.parse_args()

    if args.audio:
        wav, sr = _read_wav_from_bytes(open(args.audio, "rb").read())
    else:
        wav, sr = _read_wav_from_bytes(_download_audio_bytes(URL_EN))
    wav16k = _resample_to_16k(wav, sr)

    # Health probe first: a clear error beats a stack trace mid-stream.
    stats = requests.get(f"{args.url}/api/stats", timeout=5).json()
    print(f"service ready: {stats['sessions']} active session(s)")

    stream_over_http(args.url.rstrip("/"), wav16k, args.step_ms)


if __name__ == "__main__":
    main()
