# Qwen3-ASR · GB10 部署版

把 **Qwen3-ASR-1.7B 流式转写服务**跑在 **NVIDIA GB10（DGX Spark，sm_121 统一内存）**上的自维护 fork。

> 源项目：**[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)**（模型介绍、训练细节、能力列表请看上游）

## 本 fork 改了什么

上游的流式 demo 在 GB10 上跑不起来、也撑不住长会话。本 fork 三个 commit 解决全部问题：

| 改动 | 内容 |
|---|---|
| **vLLM 0.15.x 兼容** | 修复 `_CONFIG_REGISTRY` 缺失模块导致的启动崩溃；适配 `MMEncoderAttention` / `get_vit_attn_backend` 签名变化 |
| **流式服务强化** | VAD 静音分段（切在句界，根治"聋段"）+ 60s 兜底 + 短段过滤 + 生成串行锁 + 停滞检测 + CORS + `/api/stats` 诊断端点 |
| **部署配方** | `deploy/` 目录：Dockerfile（依赖烘焙）+ docker-compose 一键启动 |

本 fork 独立维护，**不向上游回流**；想同步上游时 `git fetch origin && git merge origin/main`。

## 环境要求

- NVIDIA GB10 / DGX Spark（aarch64，sm_121）——其他架构未验证
- Docker ≥ 20.10 + docker compose v2，且已配置 nvidia container toolkit：
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  ```
- 模型权重放在宿主机 `~/models/Qwen3-ASR-1.7B/`（可改 compose 挂载路径）

## 快速开始

```bash
cd deploy
docker compose up -d --build        # 首次：构建镜像并烘焙依赖（几分钟）
docker compose logs -f qwen3-asr    # 看加载进度，出现 "Running on" 即就绪（约 1 分钟）
curl localhost:8000/api/stats       # 健康探测：{"detail":[],"sessions":0,...}
```

改了源码后只需 `docker compose restart`（仓库根目录 live-mount 在容器 `/ws`，无需 rebuild）。

## API

浏览器可跨源直连（已开 CORS）。

| 端点 | 说明 |
|---|---|
| `POST /api/start` | 建会话 → `{session_id}` |
| `POST /api/chunk?session_id=...` | body = **16kHz 单声道 float32 LE PCM**（`application/octet-stream`）→ `{language, text}`，text 为**从头累积的全文** |
| `POST /api/finish?session_id=...` | flush 尾部 → 最终全文 |
| `GET /api/stats` | 诊断：各会话音频秒数 / 已收段字数 / 最近 chunk 的 RMS / 连续静音块数 |

建议客户端节奏：每 500ms 发 8000 样本（4KB × 4 字节）；上一个请求未返回时应丢弃新块（服务端也做了串行化，重复保护是双保险）。

## 配置

改 `deploy/docker-compose.yml` 的 command 参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gpu-memory-utilization` | 0.2 | **统一内存语义与独显不同**：0.2 ≈ 24GiB。GB10 与其他模型共存时从 0.2 起，实测内存平稳后可上调 |
| `--session-rotate-sec` | 60 | 单段音频上限；静音切断（2s 无声）通常先触发，此值兜底连续说话 |
| `--host/--port` | 0.0.0.0:8000 | |

分段策略核心参数（VAD 阈值 0.015 / 静音 2s / 短段 3 字过滤）在 `qwen_asr/cli/demo_streaming.py` 顶部常量区。

## 运维速查

**分段日志**：`docker logs qwen3-asr | grep rotate`

- `[rotate:silence]` —— 正常：切在句子停顿处
- `[rotate:max_duration]` —— 正常兜底：连续说话满 60s
- `[rotate:stall]` —— 兜底生效：段内文本停滞被强制收段（偶发正常，频发需查）
- `(短段过滤)` —— 该段 <3 字，按噪声丢弃

**转写卡住时先看 `/api/stats` 定责**：

| 现象 | 结论 |
|---|---|
| `last_rms` ≈ 0、`silent_chunks` 大 | **上游送的是静音**——查浏览器麦克风权限 / 采集链路，不是本服务的问题 |
| `last_rms` 正常（说话时 >0.02）但文本不涨 | 模型聋段，停滞兜底应在 ~8s 内自愈；不自愈把 stats 输出提 issue |

**内存**：启动基线约 27GiB used（含 vLLM 预分配 ≈19GiB CUDA）。长会话应保持平稳；只涨不跌说明 `expandable_segments` 没生效，检查镜像是否旧版本构建。

## 已知特性（非故障）

- GPU 占用随段内音频增长呈**锯齿状规律变化**——每次推理全量重编码段内音频，是上游伪流式架构的固有开销，已被 60s 分段封顶
- 段边界（静音 2s 切断处）头几个字可能轻微重复——新段无前缀上下文，属正常代价
- `nvidia-smi` 在 GB10 上看不到显存占用（显示 Not Supported），用 `free -h` + `/api/stats` 监控

## 目录

```
deploy/               GB10 部署（本 README 的展开版：组件选型理由与实测数据）
examples/
  example_streaming_http_client.py  流式服务的 HTTP 客户端示例（对拍线上协议，推荐先看）
  example_qwen3_asr_*.py            进程内推理示例（vLLM / Transformers / 流式 / 对齐，上游原样）
qwen_asr/
  cli/demo_streaming.py    强化版流式服务（本 fork 主要改动）
  inference/qwen3_asr.py   vLLM 0.15.x 兼容补丁
  core/vllm_backend/       vLLM API 漂移适配
```
