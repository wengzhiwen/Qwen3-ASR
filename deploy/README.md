# GB10 (DGX Spark) 部署

在 NVIDIA GB10（sm_121，统一内存）上运行强化版流式转写服务。

## 快速开始

```bash
cd deploy
docker compose up -d --build        # 首次：构建镜像（烘焙依赖层，之后重启秒起）
docker compose logs -f qwen3-asr    # 看模型加载进度（约 1 分钟）
curl localhost:8001/api/stats       # 就绪探测
```

前提：模型权重位于 `~/models/Qwen3-ASR-1.7B`（可在 compose 里改挂载路径），
且宿主机已配好 nvidia container toolkit（`nvidia-ctk runtime configure`）。

## API

- `POST /api/start` → `{session_id}`
- `POST /api/chunk?session_id=...`（body = 16kHz 单声道 float32 LE PCM，
  `Content-Type: application/octet-stream`）→ `{language, text}`（累积全文）
- `POST /api/finish?session_id=...` → 最终全文
- `GET /api/stats` → 诊断：各 session 的音频秒数 / 已收段字数 / 最近 RMS /
  连续静音块数（排查"模型聋"还是"上游送静音"）
- 已开 CORS，浏览器可跨源直连

## 为什么是这些组件（改动语义详见 git log 与主 README）

| 组件 | 原因 |
|---|---|
| 基础镜像 `nvcr.io/nvidia/vllm:26.02-py3` | 自带 sm_121 kernel（cu13.1）+ 本 fork 补丁对准的 vllm 0.15.1；HF 通用 wheel 在 GB10 启动即崩 |
| 全部 `--no-deps` 安装 | 防止 pip 把镜像内 vllm/transformers/torch 替换掉 |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | EngineCore fork 子进程无法重新初始化 CUDA |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | GB10 统一内存下 torch 分配器只涨不还，长会话内存棘轮 |
| `--ipc=host --ulimit memlock=-1` | 默认 64MB SHMEM 限制会让 vLLM 崩（NVIDIA 官方建议） |
| `--gpu-memory-utilization 0.2` | 统一内存语义与独显不同，按共存负载调整 |

## 调参

- 段策略（VAD 静音切断 2s / 最长 60s / 短段 3 字过滤）默认值已在
  `qwen_asr/cli/demo_streaming.py` 内；`--session-rotate-sec` 可调最长段。
- 分段日志：`docker logs qwen3-asr | grep rotate`，reason =
  `silence`（正常，切在句界）/ `max_duration`（连续说话兜底）/ `stall`（停滞兜底）。

## 硬件 / 环境实测记录

- GB10 / aarch64 / Ubuntu 24.04 / 驱动 580.173.02 / Docker 29.2.1
- 启动基线：~27GiB used（含 vLLM 0.2 配额 ≈19GiB CUDA）+ 容器 RSS ~3.6GiB
- 长会话验证：分段策略 + 串行锁 + expandable_segments 后内存平稳
