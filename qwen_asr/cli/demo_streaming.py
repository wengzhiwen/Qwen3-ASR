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
Minimal web demo for Qwen3ASRModel Streaming Inference (vLLM backend).

Install:
  pip install qwen-asr[vllm]

Run:
  python streaming/demo_qwen3_asr_vllm_streaming.py
Open:
  http://127.0.0.1:7860
"""
import argparse
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from flask import Flask, Response, jsonify, request
from vllm import SamplingParams

from qwen_asr import Qwen3ASRModel, parse_asr_output


@dataclass
class Session:
    """窗口式增量转写的会话状态。

    核心思路：不依赖上游 streaming_transcribe 的"前缀续写"机制（该机制在
    连续解说场景下模型常判定无新内容、回滚逐句吃掉前文，实测不可靠）。
    改为每 HOP_SEC 秒把「最近 WINDOW_SEC 音频 + 已定稿上文」从头转写一次
    （离线 generate 路径，稳定），句级匹配把新完整句并入 committed_text。

    - buffer：自上次"全部定稿"以来累积的未定稿音频（有 MAX_BUFFER_SEC 上限）
    - committed_text：已定稿全文，只增不改（客户端字幕的稳定来源）
    - window_text / window_matched：最近一次窗口转写结果及其已并入前缀长度
    """
    created_at: float
    last_seen: float
    committed_text: str = ""
    committed_language: str = ""
    buffer: np.ndarray = None
    new_audio_samples: int = 0      # 距上次窗口转写新增的音频量
    window_text: str = ""
    window_matched: int = 0
    # 音频能量监控：区分"无语音"与"上游送静音"（麦克风权限丢失等）。
    last_rms: float = 0.0
    silent_chunks: int = 0
    silent_samples: int = 0

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = np.zeros((0,), dtype=np.float32)


app = Flask(__name__)


# CORS：允许浏览器从其他源（如 meetingEZ 页面）直连本服务的流式 API。
# before_request 拦截 OPTIONS 预检直接返回 204；after_request 给所有响应加跨域头。
@app.before_request
def _handle_cors_preflight():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        origin = request.headers.get("Origin")
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Access-Control-Max-Age"] = "3600"
        return resp


@app.after_request
def _enable_cors(resp):
    origin = request.headers.get("Origin")
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Max-Age"] = "3600"
    return resp


global asr
global GEN_PARAMS

SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60
# 并发 session 上限（防页面刷新/多标签泄漏 session）。
MAX_SESSIONS = 8

SAMPLE_RATE = 16000
# ---- 窗口式增量转写参数 ----
WINDOW_SEC = 10.0          # 每步新鲜转写的音频窗口长度
HOP_SEC = 2.0              # 攒够多少新音频触发一次窗口转写
SILENCE_COMMIT_SEC = 2.0   # 连续静音秒数 → 整窗收束定稿
MAX_BUFFER_SEC = 45.0      # 未定稿音频缓冲上限，超限整窗强制定稿
CONTEXT_CHARS = 400        # 转写提示携带的已定稿上文长度
VOICE_RMS = 0.015          # 低于此 RMS 视为无人声
MIN_SENTENCE_CHARS = 4     # 定稿句的最短长度（过滤杂散标点）

_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;.\n])")

# 串行化所有 vLLM generate 调用。单 GPU 场景下并发请求只会叠加激活内存
# 峰值、互相拖慢；RLock 防御嵌套获取。
_ASR_LOCK = threading.RLock()


def _split_sentences(text: str) -> list:
    """按句末标点切句，保留标点在句尾。"""
    return [p for p in _SENT_SPLIT_RE.split(text or "") if p.strip()]


def _gc_sessions():
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[Session]:
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s


def _transcribe(audio: np.ndarray, context: str):
    """对一段音频做一次「新鲜」转写（离线 generate 路径）。

    与上游 streaming_transcribe 的前缀续写不同：每次调用都从零转写整段
    窗口音频，已定稿文本只作为 prompt 的 context（背景提示），不参与
    续写。模型无法"判定无新内容"，聋态从机制上不存在。
    """
    prompt = asr._build_text_prompt(context=context, force_language=None)
    outputs = asr.model.generate(
        [{"prompt": prompt, "multi_modal_data": {"audio": [audio]}}],
        sampling_params=GEN_PARAMS,
        use_tqdm=False,
    )
    raw = outputs[0].outputs[0].text
    return parse_asr_output(raw, user_language=None)


def _merge_window_sentences(s: Session, window_text: str) -> None:
    """把窗口转写文本里的新完整句并入 committed_text。

    窗口与已定稿内容有重叠（同一音频被再次转写），靠逐句 endswith 匹配
    跳过已定稿句子；遇到首个未完成尾句即停（留给下一窗口补全）。
    全部句子都已并入时，buffer 可整体释放。
    """
    matched = 0
    text = window_text or ""
    for sentence in _split_sentences(text):
        if s.committed_text.endswith(sentence):
            matched += len(sentence)
            continue
        if len(sentence.strip()) >= MIN_SENTENCE_CHARS:
            s.committed_text += sentence
            matched += len(sentence)
        else:
            break  # 未完成/过短的尾句，留在窗口里
    s.window_text = text
    s.window_matched = matched
    if matched >= len(text.rstrip()):
        # 窗口文本全部定稿：对应音频可释放，从下一句重新累积。
        s.buffer = np.zeros((0,), dtype=np.float32)
        s.new_audio_samples = 0


def _run_window(s: Session) -> None:
    """攒够 HOP_SEC 新音频后，对最近 WINDOW_SEC 音频做一次窗口转写。"""
    window = s.buffer[-int(WINDOW_SEC * SAMPLE_RATE):]
    if len(window) < int(0.5 * SAMPLE_RATE):
        return
    context = s.committed_text[-CONTEXT_CHARS:]
    language, text = _transcribe(window, context)
    if language:
        s.committed_language = language
    _merge_window_sentences(s, text or "")
    s.new_audio_samples = 0


def _finalize_buffer(s: Session, reason: str) -> None:
    """整窗收束：转写全部未定稿音频并整体定稿（静音停顿/缓冲上限/结束时调用）。"""
    buf_sec = len(s.buffer) / SAMPLE_RATE
    if buf_sec < 0.4:
        return
    context = s.committed_text[-CONTEXT_CHARS:]
    language, text = _transcribe(s.buffer, context)
    text = (text or "").strip()
    if language:
        s.committed_language = language
    if len(text) >= MIN_SENTENCE_CHARS:
        if not _SENT_SPLIT_RE.split(text)[-1].strip() or text[-1] not in "。！？!?；;.\n":
            text += "。"
        s.committed_text += text
    print(
        f"[finalize:{reason}] buffer={buf_sec:.1f}s, 定稿={len(text)}字, "
        f"累计={len(s.committed_text)}字",
        flush=True,
    )
    s.buffer = np.zeros((0,), dtype=np.float32)
    s.new_audio_samples = 0
    s.window_text = ""
    s.window_matched = 0


def _merged_result(s: Session) -> dict:
    """对外结果：合并全文 + 分段视图。

    committed_text 只增不改（客户端字幕的稳定来源）；segment_text 是最近
    一次窗口转写里尚未定稿的尾句（live 行，随时可能被下一窗口改写）。
    """
    live_tail = s.window_text[s.window_matched:] if s.window_matched < len(s.window_text) else ""
    return {
        "language": s.committed_language,
        "text": s.committed_text + live_tail,
        "committed_text": s.committed_text,
        "segment_text": live_tail,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Qwen3-ASR Streaming</title>
  <style>
    :root{
      --bg:#ffffff;
      --card:#ffffff;
      --muted:#5b6472;
      --text:#0f172a;
      --border:#e5e7eb;
      --ok:#059669;
      --warn:#d97706;
      --danger:#e11d48;
    }

    html, body { height: 100%; }

    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans";
      background: var(--bg);
      color:var(--text);
    }

    .wrap{
      height: 100vh;
      max-width: none;
      margin: 0;
      padding: 16px;
      box-sizing: border-box;
      display: flex;
    }

    .card{
      width: 100%;
      height: 100%;
      background: var(--card);
      border:1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-sizing: border-box;
      box-shadow: 0 10px 30px rgba(0,0,0,.06);

      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0;
    }

    h1{ font-size: 16px; margin: 0; letter-spacing:.2px;}

    .row{ display:flex; gap:12px; align-items:center; flex-wrap: wrap; }

    button{
      border:1px solid var(--border); border-radius: 12px;
      padding: 10px 14px; cursor:pointer; color:var(--text);
      background: #f8fafc;
      transition: transform .05s ease, background .15s ease, border-color .15s ease;
      font-weight: 700;
    }
    button:hover{ background: #f1f5f9; border-color:#cbd5e1; }
    button:active{ transform: translateY(1px); }
    button.primary{ border-color: rgba(5,150,105,.35); background: rgba(5,150,105,.10); }
    button.danger{ border-color: rgba(225,29,72,.35); background: rgba(225,29,72,.10); }
    button:disabled{ opacity:.5; cursor:not-allowed; }

    .pill{
      font-size: 12px; padding: 6px 10px; border-radius: 999px;
      border:1px solid var(--border); color: var(--muted);
      background: #f8fafc;
      user-select:none;
    }
    .pill.ok{ color: #065f46; border-color: rgba(5,150,105,.35); background: rgba(5,150,105,.10); }
    .pill.warn{ color: #92400e; border-color: rgba(217,119,6,.35); background: rgba(217,119,6,.10); }
    .pill.err{ color: #9f1239; border-color: rgba(225,29,72,.35); background: rgba(225,29,72,.10); }

    .panel{
      border:1px solid var(--border);
      border-radius: 12px;
      background: #ffffff;
      padding: 12px;
    }

    .panel.textpanel{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }

    .label{ color:var(--muted); font-size: 12px; margin-bottom: 6px; }
    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }

    #text{
      flex: 1;
      min-height: 0;
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 15px;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #f8fafc;
      overflow: auto;
    }

    a{ color: #2563eb; text-decoration:none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Qwen3-ASR Streaming</h1>

      <div class="row">
        <button id="btnStart" class="primary">Start / 开始</button>
        <button id="btnStop" class="danger" disabled>Stop / 停止</button>
        <span id="status" class="pill warn">Idle / 未开始</span>
        <a href="javascript:void(0)" id="btnClear" class="mono" style="margin-left:auto;">Clear / 清空</a>
      </div>

      <div class="panel">
        <div class="label">Language / 语言</div>
        <div id="lang" class="mono">—</div>
      </div>

      <div class="panel textpanel">
        <div class="label">Text / 文本</div>
        <div id="text"></div>
      </div>
    </div>
  </div>

<script>
(() => {
  const $ = (id) => document.getElementById(id);

  const btnStart = $("btnStart");
  const btnStop  = $("btnStop");
  const btnClear = $("btnClear");
  const statusEl = $("status");
  const langEl   = $("lang");
  const textEl   = $("text");

  const CHUNK_MS = 500;
  const TARGET_SR = 16000;

  let audioCtx = null;
  let processor = null;
  let source = null;
  let mediaStream = null;

  let sessionId = null;
  let running = false;

  let buf = new Float32Array(0);
  let pushing = false;

  function setStatus(text, cls){
    statusEl.textContent = text;
    statusEl.className = "pill " + (cls || "");
  }

  function lockUI(on){
    btnStart.disabled = on;
    btnStop.disabled = !on;
  }

  function concatFloat32(a, b){
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function resampleLinear(input, srcSr, dstSr){
    if (srcSr === dstSr) return input;
    const ratio = dstSr / srcSr;
    const outLen = Math.max(0, Math.round(input.length * ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++){
      const x = i / ratio;
      const x0 = Math.floor(x);
      const x1 = Math.min(x0 + 1, input.length - 1);
      const t = x - x0;
      out[i] = input[x0] * (1 - t) + input[x1] * t;
    }
    return out;
  }

  async function apiStart(){
    const r = await fetch("/api/start", {method:"POST"});
    if(!r.ok) throw new Error(await r.text());
    const j = await r.json();
    sessionId = j.session_id;
  }

  async function apiPushChunk(float32_16k){
    const r = await fetch("/api/chunk?session_id=" + encodeURIComponent(sessionId), {
      method: "POST",
      headers: {"Content-Type":"application/octet-stream"},
      body: float32_16k.buffer
    });
    if(!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  async function apiFinish(){
    const r = await fetch("/api/finish?session_id=" + encodeURIComponent(sessionId), {method:"POST"});
    if(!r.ok) throw new Error(await r.text());
    return await r.json();
  }

  btnClear.onclick = () => { textEl.textContent = ""; };

  async function stopAudioPipeline(){
    try{
      if (processor){ processor.disconnect(); processor.onaudioprocess = null; }
      if (source) source.disconnect();
      if (audioCtx) await audioCtx.close();
      if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    }catch(e){}
    processor = null; source = null; audioCtx = null; mediaStream = null;
  }

  btnStart.onclick = async () => {
    if (running) return;

    textEl.textContent = "";
    langEl.textContent = "—";
    buf = new Float32Array(0);

    try{
      setStatus("Starting… / 启动中…", "warn");
      lockUI(true);

      await apiStart();

      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        },
        video: false
      });

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      source = audioCtx.createMediaStreamSource(mediaStream);

      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));

      processor.onaudioprocess = (e) => {
        if (!running) return;
        const input = e.inputBuffer.getChannelData(0);
        const resampled = resampleLinear(input, audioCtx.sampleRate, TARGET_SR);
        buf = concatFloat32(buf, resampled);
        if (!pushing) pump();
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      running = true;
      setStatus("Listening… / 识别中…", "ok");

    }catch(err){
      console.error(err);
      setStatus("Start failed / 启动失败: " + err.message, "err");
      lockUI(false);
      running = false;
      sessionId = null;
      await stopAudioPipeline();
    }
  };

  async function pump(){
    if (pushing) return;
    pushing = true;

    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));

    try{
      while (running && buf.length >= chunkSamples){
        const chunk = buf.slice(0, chunkSamples);
        buf = buf.slice(chunkSamples);

        const j = await apiPushChunk(chunk);
        langEl.textContent = j.language || "—";
        textEl.textContent = j.text || "";
        if (running) setStatus("Listening… / 识别中…", "ok");
      }
    }catch(err){
      console.error(err);
      if (running) setStatus("Backend error / 后端错误: " + err.message, "err");
    }finally{
      pushing = false;
    }
  }

  btnStop.onclick = async () => {
    if (!running) return;

    running = false;
    setStatus("Finishing… / 收尾中…", "warn");
    lockUI(false);

    await stopAudioPipeline();

    try{
      if (sessionId){
        const j = await apiFinish();
        langEl.textContent = j.language || "—";
        textEl.textContent = j.text || "";
      }
      setStatus("Stopped / 已停止", "");
    }catch(err){
      console.error(err);
      setStatus("Finish failed / 收尾失败: " + err.message, "err");
    }finally{
      sessionId = null;
      buf = new Float32Array(0);
      pushing = false;
    }
  };
})();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")


@app.route("/api/start", methods=["POST", "OPTIONS"])
def api_start():
    _gc_sessions()
    if len(SESSIONS) >= MAX_SESSIONS:
        return jsonify({"error": "too many sessions, retry later"}), 429
    session_id = uuid.uuid4().hex
    now = time.time()
    SESSIONS[session_id] = Session(created_at=now, last_seen=now)
    return jsonify({"session_id": session_id})


@app.route("/api/chunk", methods=["POST", "OPTIONS"])
def api_chunk():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    if request.mimetype != "application/octet-stream":
        return jsonify({"error": "expect application/octet-stream"}), 400

    raw = request.get_data(cache=False)
    if len(raw) % 4 != 0:
        return jsonify({"error": "float32 bytes length not multiple of 4"}), 400

    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

    # 音频能量监控：区分"无语音"与"上游送静音"。
    s.last_rms = round(float(np.sqrt(np.mean(np.square(wav)))), 4) if wav.size else 0.0
    if s.last_rms < VOICE_RMS:
        s.silent_chunks += 1
        s.silent_samples += int(wav.size)
    else:
        s.silent_chunks = 0
        s.silent_samples = 0

    with _ASR_LOCK:
        s.buffer = np.concatenate([s.buffer, wav.astype(np.float32)])
        s.new_audio_samples += int(wav.size)
        buf_sec = len(s.buffer) / SAMPLE_RATE

        if s.silent_samples >= int(SILENCE_COMMIT_SEC * SAMPLE_RATE) and buf_sec >= 0.4:
            # 停顿：整窗收束定稿（把半句也定格，下次说话重新累积）。
            _finalize_buffer(s, "silence")
        elif buf_sec >= MAX_BUFFER_SEC:
            _finalize_buffer(s, "max_buffer")
        elif s.new_audio_samples >= int(HOP_SEC * SAMPLE_RATE):
            _run_window(s)

    return jsonify(_merged_result(s))


@app.route("/api/finish", methods=["POST", "OPTIONS"])
def api_finish():
    session_id = request.args.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return jsonify({"error": "invalid session_id"}), 400

    with _ASR_LOCK:
        _finalize_buffer(s, "finish")
    out = _merged_result(s)
    SESSIONS.pop(session_id, None)
    return jsonify(out)


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """诊断端点：session 数与各会话缓冲/定稿状态。"""
    return jsonify(
        {
            "sessions": len(SESSIONS),
            "detail": [
                {
                    "session_id": sid,
                    "age_sec": round(time.time() - s.created_at, 1),
                    "idle_sec": round(time.time() - s.last_seen, 1),
                    "buffer_sec": round(len(s.buffer) / SAMPLE_RATE, 1),
                    "committed_chars": len(s.committed_text),
                    "live_tail_chars": max(0, len(s.window_text) - s.window_matched),
                    "last_rms": s.last_rms,
                    "silent_chunks": s.silent_chunks,
                }
                for sid, s in SESSIONS.items()
            ],
        }
    )


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR Streaming Web Demo (vLLM backend)")
    p.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-1.7B", help="Model name or local path")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")
    return p.parse_args()


def main():
    args = parse_args()

    global asr
    global GEN_PARAMS

    asr = Qwen3ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # 窗口式转写下单次请求 ≤45s 音频（约 600 audio tokens）+ 上文，16k 上限
        # 绰绰有余，同时把内存 profiler 的音频预算压下来，降低常驻内存。
        max_model_len=16384,
        # 单用户流式场景：限制并发调度序列数，防止偶发并发请求叠加激活内存峰值。
        max_num_seqs=4,
    )
    # 窗口转写是"新鲜"转写：每步要输出整个窗口的文本（而非增量），上限给足。
    GEN_PARAMS = SamplingParams(temperature=0.0, max_tokens=512)
    print("Model loaded.")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()