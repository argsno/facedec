# 海报演员识别系统设计方案

> 目标：输入一张海报 + 一组演员（姓名、角色、头像/剧照），输出海报中每个可见人脸对应的演员、置信度与位置；无法确认的人脸明确标为 unknown；支持无头服务端运行。

## 0. 摘要

一句话：**用「多尺度人脸检测 + ArcFace 特征 + 质量感知的开放集匹配 + 全局分配 + 人工复核」做识别，用「FastAPI + ONNX Runtime/Triton + Redis/S3 + 异步任务」做服务端。**

关键判断：

1. 这不是「人脸检测」问题，而是**小 gallery、开放集、跨域人脸验证**问题。
2. 检测召回是效果上限；海报必须做多尺度/切片检测。
3. 生产环境里 gallery 质量往往比模型更影响结果；头像要检测、筛选、做一致性校验。
4. cosine 相似度不是概率；必须做归一化、校准和分桶阈值。
5. 允许 unknown；不要给每张脸强行配一个演员。
6. 必须有人工复核和标注集；没有标注集就无法可靠定阈值。

---

## 1. 目标、输入输出与边界

### 1.1 输入

| 输入 | 内容 | 要求 |
| --- | --- | --- |
| 海报 | URL 或上传文件 | JPG/PNG/WebP；建议长边 ≥ 1600px；单人脸建议 ≥ 64px |
| 演员列表 | `cast_id`、姓名、角色（可选）、1..N 张头像/剧照 URL | 每个演员至少一张可用正脸；`cast_id` 必须稳定唯一 |
| 选项 | 精度目标、最大人脸数、是否允许多次出现、是否返回调试图 | 见接口定义 |

注意：`name` 可能重名（例如「张伟」），匹配结果必须以 `cast_id` 为准，`name` 只用于展示。

### 1.2 输出

- 每张人脸的 bbox、关键点、质量分。
- 匹配到的演员：`cast_id`、姓名、角色、校准后的概率、状态（accepted / review / unknown）。
- Top-3 候选，供复核。
- 未匹配人脸列表（unknown）。
- 演员列表中未在海报中出现的人（cast_not_found）。
- 警告（头像不可用、海报疑似手绘、部分下载失败等）。
- 模型 bundle 版本、各阶段耗时。

### 1.3 非目标

- 不做通用名人识别；只对调用方给定的 cast 负责。
- 不做人脸生成、换脸、身份推断。
- 不承诺纯动画/手绘海报的识别效果；这类海报需要单独的动漫人脸模型，或直接返回「不支持」。
- 不追求 100% 全自动；低质量人脸必须进入复核。

### 1.4 建议的初始指标

以下数字是产品目标的起点，必须用自己的标注集校准：

| 指标 | 建议目标 |
| --- | --- |
| 人脸检测召回（IoU≥0.5，人脸≥48px） | ≥ 95% |
| 自动接受准确率 | ≥ 97% |
| 自动接受覆盖率（accepted / 可识别人脸） | ≥ 70% |
| 主演召回（海报中出现的 cast 被找到） | ≥ 90% |
| 误接受率（unknown 被强行匹配） | < 1% |
| 人工复核率 | < 20% |
| p95 延迟（GPU，不含图片下载） | < 2s |

---

## 2. 问题本质与难点

1. **域差异大**：演员头像是棚拍/活动照，海报是电影级打光、浓妆、造型、调色，甚至黑白。同一个人的两张图，embedding 距离可能比两个不同的人还大。
2. **开放集**：海报里可能有路人、群演、雕塑、海报内的照片；cast 里也可能有人根本没出现在海报上。不能默认「每张脸一定属于 cast」。
3. **小脸、侧脸、遮挡**：群像海报里人脸可能只有 30–60px；侧脸、墨镜、手部遮挡会让识别率骤降。
4. **Gallery 噪声**：百科头像可能低清、裁错、是合照、是旧照，甚至挂错人。Gallery 质量是生产环境最常见的失败原因。
5. **相似演员/双胞胎**：Top-1 和 Top-2 分数接近时，单靠阈值无法可靠区分。
6. **校准困难**：0.6 的 cosine 在不同人脸质量、不同演员上含义不同；直接把 cosine 当概率会严重误导产品。
7. **服务端工程**：图片下载失败、模型冷启动、GPU 批处理、缓存、版本回滚、生物特征数据合规，都是必须处理的问题。

---

## 3. 总体架构

```text
                         ┌──────────────────────────────┐
                         │  API / Job Service           │
                         │  FastAPI + Auth + RateLimit  │
                         └──────────────┬───────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
    ┌─────────────────┐        ┌──────────────────┐        ┌─────────────────┐
    │  Image Fetcher  │        │  Gallery Builder │        │  Result Cache   │
    │  fetch/validate │        │  cast portraits  │        │  Redis          │
    └────────┬────────┘        └────────┬─────────┘        └─────────────────┘
             │                          │
             ▼                          ▼
    ┌─────────────────────────────────────────────────────┐
    │  Face Pipeline                                      │
    │  detect → align → quality → embed                   │
    └──────────────────────┬──────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │  Matching                                           │
    │  cosine → cohort normalize → calibrate →            │
    │  Hungarian assignment with unknown                  │
    └──────────────────────┬──────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │  Output / Review / Feedback                         │
    └─────────────────────────────────────────────────────┘
```

两条路径：

- **离线/预构建**：演员头像 → 检测、筛选、对齐、embedding → `gallery_id` 缓存。同一个剧的多张海报/剧集可以复用。
- **在线**：海报 → 人脸检测 → 对齐/质量 → embedding → 与 gallery 匹配 → 决策 → 输出。

这样设计的好处：gallery 构建成本只付一次；海报识别可以水平扩展；缓存键清晰；模型升级时统一失效。

---

## 4. 识别流水线

### 4.1 输入获取与规范化

- 使用 `httpx` 异步下载，设置 UA、Referer、超时、重试、最大 20MB；单请求并发下载数限制在 8 左右，避免打爆源站。
- **SSRF 防护**：只允许 http/https；解析 DNS 后拒绝内网/回环/链路本地地址；限制重定向次数；校验 `Content-Type`。
- 解码：PIL 或 libvips；处理 EXIF 方向；统一转 sRGB；透明 PNG 合成到中性灰底；动图只取第一帧或拒绝；限制最大像素数，防解压炸弹。
- 检测用图：长边压到 4096 以内（保留原图用于 crop 和 bbox 映射）。
- 如果第一次检测到 0 张脸，尝试 90°/180°/270° 旋转后再检测，选检测分最高的一组。
- 下载失败可降级为调用方上传；不要无限重试，也不要在服务端做反爬绕过。
- 对海报字节做 `sha256`，用于缓存。

### 4.2 海报人脸检测

推荐 **SCRFD-10G**（InsightFace 自带），对小脸较稳；更高精度可评估 SCRFD-34G 或检测器融合，但先不要上复杂度。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| 检测输入 | 640×640 | 切片后缩放 |
| 全局检测长边 | 1280–1600 | 先跑一遍全局 |
| 切片 | 1024×1024，重叠 128px | 长边 >1600 或多人海报时启用 |
| 检测阈值 | 0.5 | 召回优先可降到 0.4，靠后续质量/校准过滤 |
| NMS IoU | 0.4 | 跨切片合并 |
| 最小检测人脸 | 24px（原图） | 低于此值不进入识别 |
| 最小识别人脸 | 48px（原图） | 低于此值标记 `low_quality`，不自动接受 |
| 最大人脸数 | 64 | 防异常图拖垮服务 |

关键点：

- **切片是必须的**。举例：3000px 高的海报缩到 640，缩放系数约 0.21；原图 120px 的脸只剩约 25px，接近检测下限。切成 1024 的块后缩放系数约 0.63，同一张脸还有约 75px。群像海报整图缩到 640，小脸基本必漏。
- 切片之间要有重叠，否则边界人脸会被切断；合并时用全局坐标做 NMS，并辅以关键点距离去重。
- 检测召回决定了系统上限：漏掉的脸，后面再强的识别模型也救不回来。先把检测召回打上去，再谈识别。
- 不要用「人脸修复/超分」来救小脸；生成模型会伪造身份细节，识别结果可能更差。低于阈值就标记低质量，交给复核。

### 4.3 对齐与质量评估

用 5 个关键点做相似变换，对齐到 112×112（ArcFace 模板）；另存 224×224 裁剪图用于复核和可选的遮挡分析。

质量特征：

| 特征 | 计算 | 用途 |
| --- | --- | --- |
| 尺寸 | `min(bbox_w, bbox_h)` | 硬门槛 + 校准 |
| 清晰度 | Laplacian 方差 | 校准 |
| 姿态 | 3D landmarks/PnP 的 yaw、pitch | 校准、复核路由 |
| 遮挡 | 人脸解析（可选）或关键点可见度 | 校准 |
| 检测分 | detector score | 校准 |
| 曝光 | 亮度直方图 | 可选 |
| 综合质量 | `q = f(size, blur, pose, occlusion, det)` | 分桶、复核路由 |

原则：**质量分主要用于校准和路由，不要用它粗暴过滤掉所有「看起来不好」的脸**。侧脸、黑白、浓妆的脸仍可能匹配成功；过早丢弃会降低召回。

### 4.4 演员 Gallery 构建

每个演员：

1. 下载最多 5–8 张头像/剧照。
2. 每张图检测人脸；选最大、最靠中心、最清晰的一张。
3. 如果一张图里有多张较大的脸（例如合照），标记 `ambiguous`：只有当最大脸明显占主导（例如面积 > 第二大的 2 倍）时才使用；否则默认不用，避免把同框的另一个人当成该演员。
4. 如果检测不到人脸，标记 `no_face`，不要把整张图硬裁成脸。
5. 对齐、提特征；按质量过滤。
6. 多样性选择：在质量合格的候选里做贪心 max-min，选 2–5 张覆盖不同姿态/造型的图，避免全是同一张脸的重复。
7. 跨图一致性检查：计算同演员 gallery 内两两 cosine；如果平均相似度很低（例如 <0.3），标记 `gallery_inconsistent`，提示可能挂错人/年龄跨度大/画风差异大。
8. 缓存 embedding，键为 `sha256(图片字节) + bundle_id`。

重要的工程经验：

- **优先使用剧中的角色剧照**，其次才是活动照/百科头像。剧照的造型、打光、年代感更接近海报，域差异更小。
- Gallery 不是越多越好。1 张低质量图 + 4 张高质量图，比 10 张杂图更稳。
- 如果同一个演员有多张图，打分时用 `top-2 mean` 或 `max`，不要简单平均所有图；错误图会把平均分拉低。
- 头像下载失败、检测不到脸、质量太低，都要在响应里明确告诉调用方，而不是静默跳过。生产环境里「某个演员头像挂错」比模型不准更常见。

### 4.5 特征提取

| 模型 | Backbone | 维度 | 用途 | 相对成本 |
| --- | --- | --- | --- | --- |
| InsightFace `buffalo_l`（`w600k_r50`） | ResNet50 | 512 | 默认，性价比高 | 1× |
| InsightFace `antelopev2`（`glintr100`） | ResNet100 | 512 | 难例、高精度场景 | ~2× |

- 特征 L2 归一化，用 cosine 相似度。
- **TTA**：原图 + 水平翻转各提一次特征，embedding 平均后重新归一化，或分数取 max。先用验证集确认是否真的涨点。
- **模型融合**：不同模型的 embedding 空间不通用，不要直接平均 embedding；应在分数层融合（加权平均 cosine 或校准后的概率）。
- 服务端用 ONNX Runtime：GPU 用 FP16/CUDA/TensorRT EP，CPU 用 INT8/OpenVINO；量化后必须跑一遍基准，确认精度下降可接受。
- 批处理：同一请求内的人脸、同一批海报的人脸可以拼 batch；gallery 构建也同理。

### 4.6 匹配打分与归一化

对海报人脸 `f` 和演员 `i`：

```text
s_fi = max_j cos(x_f, g_ij)          # 或 top-2 mean，取决于 gallery 大小
z_fi = (s_fi - μ_f) / σ_f            # 用 cast 内所有演员的分数做 cohort 归一化
```

- `μ_f`、`σ_f` 用该人脸对全部 cast 的分数统计。cast 很少（<5，尤其 1–2 人）时归一化不稳定，改用固定校准模型或外部 cohort。
- 可选 **T-norm**：对每个演员，用一组「非 cast」的 cohort 人脸算 impostor 分数的均值和方差，得到该演员自己的阈值 `τ_i = μ_i + kσ_i`。这对「大众脸」演员特别有用——他们的分数天然偏高，统一阈值会误接受。
- 归一化不是可选项。海报域和头像域的分数分布差异很大，不做归一化，阈值无法跨海报复用。

### 4.7 校准：把分数变成可信的概率

用带标签的验证集拟合一个校准模型（逻辑回归或 isotonic），输入特征建议：

- `z_top1`、`z_top1 - z_top2`（margin）
- 人脸质量 `q_f`、gallery 质量 `q_g`
- 姿态差（海报人脸 yaw 与 gallery yaw 的差）
- 检测分、人脸像素尺寸
- gallery 一致性
- 海报风格（照片/黑白/手绘，可选）

输出 `P(correct)`。要求：

- 按**演员身份**或**海报**分组做交叉验证，不能按人脸随机切分，否则会泄漏。
- 标注数据少时用 Platt scaling；数据足够再考虑 isotonic。
- 分桶校准：小脸、侧脸、黑白海报分别看可靠性曲线；如果某桶样本太少，用全局模型 + 质量特征，不要硬拟合。
- 如果产品需要统计保证，可以用 conformal prediction 输出候选集，而不是强行给单点答案。

**不要把 raw cosine 直接展示成「相似度 98%」**。如果没做校准，就只展示高/中/低置信度，不要展示百分比。

### 4.8 开放集决策与全局分配

不要对每张脸独立取 argmax 后结束。正确做法是把它看成一个带「未知」选项的二分图匹配：

- 左：海报人脸；右：cast 演员。
- 边权：校准后的对数几率（log-odds），不是原始概率；这样最大化总和才等价于最大化联合似然。
- 每张脸最多匹配一个演员；每个演员默认最多匹配一张脸（海报里同一演员出现两次是少数情况，可用选项放开）。
- 加虚拟节点：每张脸有「unknown」选项，每个演员有「not found」选项。unknown 虚拟边的权重设为 0 log-odds（等价于 P=0.5），或按目标精度调整。
- 用 Hungarian（`scipy.optimize.linear_sum_assignment`）求全局最优。

这样做能解决两个常见问题：

1. 两张脸都像同一个演员时，独立 argmax 会重复分配；全局分配会强制二选一，另一张进 unknown/复核。
2. 某个演员的分数普遍偏高（hub face）时，全局分配会把它留给最像的那张脸，而不是每张都给它。

决策状态：

| 条件 | 状态 | 处理 |
| --- | --- | --- |
| `P ≥ 0.95` 且 margin 足够 且质量合格 | `accepted` | 自动输出 |
| `0.5 ≤ P < 0.95` 或 margin 低 或质量一般 | `review` | 返回 Top-3，人工确认 |
| `P < 0.5` 或人脸太小/太糊 | `unknown` | 不强行匹配，标注原因 |

阈值是示例，必须用标注集校准。默认目标是「高精度优先」：宁可多进复核，不要自动给错人。

**难例升级**：对 `review` 的人脸，可以再用更强的模型（antelopev2）、TTA、或双模型一致性检查重新打分；仍不确定就进人工复核。这样 p50 延迟低，难例才付额外成本。

### 4.9 输出与复核

- JSON 返回 bbox、landmarks、质量、匹配结果、Top-3 候选、unknown、cast_not_found、warnings。
- 可选返回调试图：在海报上画框、名字、分数；方便快速定位问题。
- 复核界面：海报 + 框 + 每个候选的演员头像；支持「接受 / 改成另一个 / 不在列表中 / 未知」。
- 复核结果落库，作为后续校准和 fine-tune 的数据。
- **随机抽检自动接受的结果**，用来估计真实准确率。只统计复核队列会低估错误率，因为高置信度样本根本没进复核。
- `cast_not_found` 的语义是「没有匹配到」，不等价于「一定不在海报里」：可能是检测漏脸、头像不可用、或cast 列表不完整。响应里要写清楚，避免调用方误判。
- 如果 cast 不完整，unknown 人脸可能仍是未提供的演员；不要强行归因，建议补充 cast 后重跑，gallery 和海报特征可以复用。

---

## 5. 服务端设计

### 5.1 部署形态

**MVP（推荐先做）**

```text
Client → FastAPI（同步接口）
           ├─ 图片下载（httpx）
           ├─ ONNX Runtime（检测 + 识别）
           ├─ Redis（gallery / 结果缓存）
           └─ S3/MinIO（可选：调试图、原始图）
```

- 单容器/单服务，Docker Compose 即可跑。
- 适合每天几千到几万张海报；CPU 也能跑，GPU 更快。
- 先不要上 Triton、Kafka、向量数据库；cast 只有几十人，NumPy 矩阵足够。

**规模化**

```text
API → 队列（Redis Streams / SQS）→ GPU Worker Pool
                                  ├─ Face Pipeline
                                  └─ 可选 Triton（动态 batching）
         ↘ Postgres（任务/反馈）  ↘ S3/MinIO  ↘ Redis
```

- API 无状态，按队列深度扩缩容（KEDA/HPA）。
- GPU worker 按并发和显存定副本数；模型常驻，避免冷启动。
- 只有做「全局名人库」（10 万+ 身份）时才需要 FAISS/Milvus/pgvector；按剧的 cast 不需要。

### 5.2 API 设计

建议把「gallery 注册」和「海报识别」分开：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| POST | `/v1/galleries` | 提交 cast + 头像，构建 gallery，返回 `gallery_id` 和每个成员的状态 |
| GET | `/v1/galleries/{id}` | 查询 gallery 状态 |
| DELETE | `/v1/galleries/{id}` | 删除（合规/过期） |
| POST | `/v1/identify` | 海报 + `gallery_id`（或 inline cast），同步识别 |
| POST | `/v1/jobs` | 批量海报异步任务 |
| GET | `/v1/jobs/{id}` | 查询任务结果 |
| GET | `/healthz` `/readyz` `/metrics` | 健康检查/指标 |

- 同步接口设置 15s 超时；批量走异步。
- 支持 `Idempotency-Key`，避免重试重复扣费/重复计算。
- `gallery_id` 可复用：同一部剧的多张海报只构建一次 gallery。

### 5.3 缓存

| 缓存 | Key | TTL | 说明 |
| --- | --- | --- | --- |
| 识别结果 | `sha256(poster) + gallery_id + options_hash + bundle_id` | 7 天 | 完全确定性 |
| Gallery | `gallery_id`（其中包含 `bundle_id`） | 30 天 | 含 embedding 和成员状态；换模型后重建 |
| 单图 embedding | `sha256(image) + bundle_id` | 30 天 | 跨 gallery 复用 |
| 原始图片 | 可选 | 1 天 | 最小化留存 |

- 用 Redis 锁/singleflight 防缓存击穿。
- 模型 bundle 升级时，`bundle_id` 变化自然失效，不要手工清缓存。

### 5.4 模型服务

- MVP：ONNX Runtime 进程内推理。检测和识别分成两个 session；GPU 用 CUDA EP，CPU 用 OpenVINO/oneDNN。
- 规模上来再拆 Triton：动态 batching、多模型 ensemble、指标更完整；但 Python 的预处理/对齐仍放在 worker 里更简单。
- 模型 bundle 包含 `manifest.json`（文件、sha256、输入尺寸、预处理参数、校准版本），启动时校验并 warmup。
- 响应里带 `bundle_id`，出问题可复现。
- 新模型先 shadow（影子流量）对比，再 canary 5%，最后全量。

### 5.5 容量与延迟预算

典型海报 5 张脸、cast 10 人、GPU：

| 阶段 | 预估 |
| --- | --- |
| 海报下载 | 100–300ms（网络为主） |
| 解码/预处理 | 20ms |
| 检测 | 30–60ms |
| 对齐/裁剪 | 5ms |
| 识别（5 张脸） | 15ms |
| 匹配/分配 | 2ms |
| 合计（不含下载） | ~100–200ms |

- 1×T4/L4 级别 GPU，批处理下大约 5–15 张海报/秒；实际以压测为准。
- Cast gallery 构建：10 人 × 3 张图 ≈ 30 次检测+识别，GPU 批处理约 0.3–1s；缓存后不重复付。
- CPU（8 vCPU）：大约慢 3–10 倍，低流量场景完全够用。
- 真正的瓶颈通常不是 GPU，而是图片下载和人工复核。

### 5.6 安全与合规

- **SSRF**：URL 允许列表、DNS 解析后校验、禁止内网 IP、限制重定向、限制响应大小。
- **图片炸弹**：最大字节数、最大像素数、MIME 校验、解码超时。
- **鉴权与配额**：API Key/OAuth；按租户限流；租户数据隔离。
- **生物特征数据**：人脸 embedding 按敏感数据对待；加密存储、最小化留存、访问审计、TTL 删除；日志里不要打 embedding 和原图。
- **数据源合规**：百度百科有反爬和 ToS 限制，生产上优先让调用方上传头像，或使用有授权的数据源；抓取适配器只作为可选模块，遵守 robots/ToS、限速、缓存、署名。
- **模型许可**：InsightFace 代码通常是 MIT，但官方预训练模型的许可需要逐个核对，常见版本限制为非商业研究用途；商用前必须确认许可，或替换为可商用模型/自训模型。
- **肖像权/版权**：海报和演员头像的使用需要相应授权；不要用识别结果做生成、换脸或商业背书。

### 5.7 可观测性

- 指标：请求量、错误码、各阶段延迟、检测人脸数分布、accept/review/unknown 比例、cast 头像不可用率、缓存命中率、GPU 利用率/显存、队列深度。
- 质量指标：随机抽检准确率、复核纠正率、分数分布漂移、各质量桶的准确率。
- 链路追踪：OpenTelemetry，按阶段打 span；日志带 `request_id`、`gallery_id`、`bundle_id`。
- 告警：p95 延迟、错误率、GPU OOM、队列积压、自动接受准确率下降。

### 5.8 测试与发布

- 单元测试：坐标变换、NMS、对齐、归一化、校准、Hungarian 分配。
- 集成测试：mock 图片服务，跑完整流程。
- Golden test：固定海报/cast，断言 Top-1 和状态；允许分数小幅波动。
- 负载测试：100 并发海报，观察 p95 和 GPU 显存。
- 模型回归：每次换模型/bundle 必须跑标注集，指标下降超过阈值就阻止发布。
- 资源保护：单请求限制人脸数/像素数；GPU OOM 时降级 batch/tile 并重试；超时取消，避免拖垮整个 worker。
- 安全测试：SSRF、超大图、畸形图、超长 URL。
- 发布：离线基准 → 影子 → canary → 全量；保留一键回滚。

### 5.9 自建 vs 云服务

- 云服务（AWS Rekognition Face Search、腾讯云/阿里云人脸搜索等）可以快速做 baseline：把演员头像建库，拿海报人脸去搜索。优点是省模型和运维；缺点是单次成本、数据合规/出境、阈值和质控可控性弱，而且对「海报域 vs 头像域」的差异不一定处理得更好。
- 自建适合：量大、要控成本、要自定义质量门控/校准/复核、数据不能出内网、要做剧照 gallery 和域适配。
- 建议：先用云服务或开源 baseline 在 30–50 张目标海报上做对照评测，再决定是否自建。不要在没评测的情况下直接押注某一种。
- VLM/通用大模型不适合做身份主判据；可以用来读海报文字、判断海报风格，降低 OCR/风格分类的工程量。

### 5.10 当前阶段：CPU-only 可行性验证（研究）

约束：研究用途、无 GPU、头像由调用方上传、精度优先。这个阶段的目的是回答「在目标海报上，现成模型 + 简单 gallery 能不能达到可用精度」，而不是搭生产平台。

技术裁剪：

- 识别只用 `buffalo_l`（ArcFace R50）；不上 antelopev2、双模型融合、Triton。
- 检测用 SCRFD-10G；长边 >1600 才启用切片，避免小图无谓变慢。
- 单进程 ONNX Runtime CPU；不做 Redis/队列/数据库/缓存。
- 输出原始 cosine + `accepted/review/unknown`；不输出伪造概率。
- 默认阈值 `accept=0.45`、`review=0.30`、`margin=0.08`，只是占位值，必须用标注集校准。
- FastAPI 只做同步上传识别；上传文件请求结束即删。

本机实测（Apple Silicon / Python 3.12 / `buffalo_l` / CPU）：

| 场景 | 实测 |
| --- | --- |
| 模型加载（已缓存） | ~0.4s |
| 512×512 单脸检测+识别 | ~0.23s |
| CLI 端到端（单图、单演员） | ~0.37s |
| 2048×2048 切片+去重 | ~2.5s |

结论：CPU 足够跑 10–50 张海报的离线可行性验证；瓶颈通常不是算力，而是 gallery 质量和域差异。

配套工程已提供 `facedec` CLI、`facedec-eval` 评测脚本和同步上传 FastAPI；评测 manifest 格式与指标见 `README.md`。

实际单海报验证（《蒸蒸日上》+ 星图 92 人）见 `docs/validation-zzrs.md`：5/5 人脸检出、5/5 Top-1 正确、4 人自动接受且全对、1 人进复核、0 误接受。样本只有一张海报，不能当作泛化精度。

验证顺序：

1. 10–20 张目标海报，人工标所有人脸身份（含 unknown）。
2. 跑 baseline，记录检测召回、Top-1、自动接受精度/覆盖率/复核率。
3. 先修 gallery 和检测漏脸，再谈换模型。
4. 精度优先：自动接受精度 < 97% 就提高阈值，不要为了覆盖率牺牲精度。
5. 扩到 50 张确认，再决定是否做校准、难例升级或轻量域适配。

Go/no-go 建议：人脸≥48px 的检测召回 ≥90%；自动接受精度 ≥97%；覆盖率 ≥40–50%；复核率可承受。达不到时先定位是检测、gallery 还是域差异，不要盲目上更大模型。

---

## 6. 接口契约

### 6.1 构建 gallery

```http
POST /v1/galleries
Content-Type: application/json
```

```json
{
  "name": "蒸蒸日上",
  "members": [
    {
      "cast_id": "c_01",
      "name": "演员A",
      "role": "角色A",
      "portrait_urls": [
        "https://example.com/a_headshot.jpg",
        "https://example.com/a_still.jpg"
      ]
    },
    {
      "cast_id": "c_02",
      "name": "演员B",
      "portrait_urls": ["https://example.com/b_headshot.jpg"]
    }
  ]
}
```

响应：

```json
{
  "gallery_id": "g_7f3c2a",
  "members": [
    {
      "cast_id": "c_01",
      "status": "ok",
      "usable_images": 2,
      "quality": 0.86,
      "consistency": 0.72
    },
    {
      "cast_id": "c_02",
      "status": "low_quality",
      "usable_images": 1,
      "quality": 0.31,
      "consistency": null,
      "warning": "人脸尺寸过小"
    }
  ],
  "bundle_id": "fr-2026.11.1"
}
```

### 6.2 识别海报

```http
POST /v1/identify
Content-Type: application/json
```

```json
{
  "poster": {"url": "https://example.com/poster.jpg"},
  "gallery_id": "g_7f3c2a",
  "options": {
    "precision_target": 0.97,
    "max_faces": 64,
    "allow_duplicate_cast": false,
    "return_debug_image": true
  }
}
```

也可以不传 `gallery_id`，直接内联 `cast`（适合单次调用）。

响应：

```json
{
  "request_id": "req_01J...",
  "poster": {"width": 2000, "height": 3000, "style": "photo"},
  "faces": [
    {
      "face_id": "f_1",
      "bbox": [820, 410, 1010, 640],
      "landmarks": [[880, 470], [960, 470], [920, 520], [885, 580], [955, 580]],
      "quality": {"score": 0.83, "size_px": 190, "blur": 0.91, "yaw": -8, "occlusion": 0.03},
      "match": {
        "cast_id": "c_01",
        "name": "演员A",
        "role": "角色A",
        "score": 0.61,
        "z_score": 3.4,
        "probability": 0.98,
        "status": "accepted",
        "candidates": [
          {"cast_id": "c_01", "name": "演员A", "probability": 0.98},
          {"cast_id": "c_07", "name": "演员G", "probability": 0.01}
        ]
      }
    }
  ],
  "unknown_faces": [
    {"face_id": "f_2", "bbox": [120, 900, 240, 1040], "reason": "low_quality"}
  ],
  "cast_not_found": [
    {"cast_id": "c_03", "name": "演员C"}
  ],
  "warnings": [
    {"code": "CAST_PORTRAIT_UNUSABLE", "cast_id": "c_03", "message": "头像未检测到可用人脸"}
  ],
  "model_bundle": "fr-2026.11.1",
  "timings_ms": {"fetch": 120, "detect": 80, "embed": 40, "match": 5, "total": 260}
}
```

### 6.3 错误码

| HTTP | code | 说明 |
| --- | --- | --- |
| 400 | `INVALID_REQUEST` | 参数缺失/格式错误 |
| 401/403 | `UNAUTHORIZED` | 鉴权失败 |
| 413 | `POSTER_TOO_LARGE` | 文件/像素超限 |
| 422 | `POSTER_DECODE_FAILED` | 图片损坏/格式不支持 |
| 422 | `CAST_EMPTY` | 没有可用 cast |
| 404 | `GALLERY_NOT_FOUND` | `gallery_id` 不存在或已过期 |
| 429 | `RATE_LIMITED` | 限流 |
| 502 | `UPSTREAM_FETCH_FAILED` | 图片下载失败 |
| 200 | `NO_FACE_DETECTED`（warning） | 请求成功，但没检测到人脸 |
| 200 | `ILLUSTRATION_SUSPECTED`（warning） | 疑似手绘/动画海报，结果不可靠 |

原则：**部分失败不要整体 500**。例如 30 个演员里 2 个头像下载失败，仍然返回其余结果 + warnings。

---

## 7. 评估与阈值校准

### 7.1 标注集

- 规模：100–200 张海报，800–2000 张人脸；覆盖主演、配角、群像、单人、黑白、浓妆、侧脸、小脸、手绘。
- 每张海报标：所有人脸 bbox + 身份（cast_id 或 unknown）；无法判断的标 ignore。
- Cast 头像从与线上一致的来源取，保留低质量样本，不要只挑好图。
- 切分：按**演员身份**或**海报**分组，不能随机按人脸切；校准集和测试集必须分开。
- 标注质量：双人标注 + 仲裁；记录 IoU 和身份一致性。

### 7.2 指标

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| Detection Recall | IoU≥0.5 的召回 | 系统上限 |
| Auto-accept Precision | 自动接受中正确的比例 | 产品安全 |
| Coverage | 自动接受人脸 / 可识别人脸 | 自动化率 |
| Cast Recall | 找到的在场演员 / 在场演员 | 业务核心 |
| FPIR | 被错误接受的 impostor 比例 | 开放集安全 |
| ECE | 校准误差 | 概率可信度 |
| Review Rate | 进复核的人脸比例 | 运营成本 |
| p95 Latency | 端到端延迟 | 体验 |

自动接受准确率必须靠**随机抽检**估计；只看复核队列会有偏差。

### 7.3 阈值校准流程

1. 在验证集上计算所有人脸-演员分数和特征。
2. 扫阈值，画 precision–coverage 曲线。
3. 选满足精度目标（例如 97%，用 Wilson 下界）的工作点。
4. 按质量桶（人脸尺寸、模糊、姿态）分别定阈值；样本不足的桶回退到全局。
5. 拟合校准模型，按身份分组交叉验证。
6. 在独立测试集上验证；上线后监控分数分布和抽检准确率。
7. 模型/bundle 变化后重新校准，不能沿用旧阈值。

### 7.4 消融实验

按顺序做，每步都记录精度、覆盖率和延迟：

1. Baseline：SCRFD-10G + buffalo_l + 全局 cosine 阈值。
2. + 切片检测。
3. + 翻转 TTA。
4. + cast cohort 归一化。
5. + 校准模型。
6. + Hungarian 全局分配。
7. + 难例升级（antelopev2/双模型）。
8. + 域适配（adapter/fine-tune）。
9. + 剧照 gallery / gallery 扩展。

不要一次上全部；先确认每一步是否真的带来收益。

### 7.5 错误分析分桶

- 人脸尺寸：<48、48–96、96–192、>192 px。
- 姿态：正脸、半侧、侧脸、仰俯。
- 画质：清晰、模糊、低照度、过曝、黑白、JPEG 伪影。
- 遮挡：无、头发、手、墨镜、口罩、文字/水印。
- 海报类型：写实、风格化、手绘/动画。
- Gallery：单图、多图、低清、合照、疑似挂错。
- 身份：主演/配角、儿童、相似演员/双胞胎。

每类都看 Top-1 错误案例；很多问题不是换模型能解决的，而是 gallery 或检测问题。

---

## 8. 迭代路线

### Phase 0：离线验证（1–2 周）

- 选 30–50 张目标海报，人工标注。
- 跑通 SCRFD + buffalo_l + cosine 的 baseline。
- 测检测召回、Top-1、误接受；确认域差异有多大。
- 产出 go/no-go 和失败案例集。

### Phase 1：可上线 MVP（2–3 周）

- FastAPI + ONNX Runtime + Redis + gallery API。
- 切片检测、质量评估、校准、Hungarian 分配。
- 100–200 张海报的评测集和阈值校准。
- 简单复核界面 + 反馈落库。
- Docker Compose 部署；基础监控。

### Phase 2：提升覆盖率（2–4 周）

- 难例升级（antelopev2/TTA/双模型）。
- 剧照 gallery、gallery 一致性、低质量头像提示。
- 批量任务、gallery 复用、缓存优化。
- 域适配：用标注数据训练轻量 adapter，或对识别模型做少量 fine-tune。
- 影子发布 + canary。

### Phase 3：规模化与增强

- 海报文字 OCR：识别演员名/角色名，作为候选集先验或弱特征（不要单独作为身份依据）。
- 跨海报 gallery 扩展：把人工确认过的高置信度海报人脸加入该演员的 in-domain gallery，改善后续海报。
- 全局名人索引（如果产品需要）：FAISS/pgvector + 分片。
- 主动学习：按不确定性/多样性挑样本送审，持续校准。

---

## 9. 风险、边界与「不要做」

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| 手绘/动画海报 | 人脸检测和识别都不可靠 | 风格分类器预警；单独模型或拒绝 |
| 小脸/远景群像 | 无法识别 | 切片 + 提高分辨率；低于 48px 标记低质量，不自动接受 |
| 侧脸/遮挡/浓妆 | 召回下降 | 质量特征 + 复核；gallery 加入侧脸/剧照 |
| 相似演员/双胞胎 | 误匹配 | margin 检查 + 全局分配 + 复核 |
| Gallery 挂错/旧照 | 系统性错误 | 跨图一致性 + 人工确认 + 多来源 |
| 黑白海报 | 识别率下降 | 训练/校准覆盖；不要强行二值化 |
| 水印/文字压脸 | 遮挡 | 质量标记 + 复核；不要 inpainting 后再识别 |
| 模型许可 | 商用风险 | 核对许可；替换/自训 |
| 数据源 ToS | 法律风险 | 授权数据/用户上传；限速缓存 |
| 生物特征合规 | 合规风险 | 加密、TTL、访问审计、最小化 |
| 评审成本 | 运营压力 | 只复核不确定样本；随机抽检；持续校准 |

明确不要做：

1. **不要用 CLIP/VLM 直接判断「这张脸是不是某个演员」**。CLIP 衡量语义/风格，不是身份；VLM 对细粒度身份不可靠且贵。VLM 可以用来读海报文字、判断风格，不要做身份主判据。
2. **不要用人脸超分/修复（GFPGAN、CodeFormer 等）提升识别率**。生成模型会补出并不存在的身份细节，可能让结果更自信但更错。显示可以修复，识别不要。
3. **不要把 cosine 当概率**。必须校准，否则「98%」会误导产品和用户。
4. **不要给每张脸强行匹配**。unknown 是正确输出；cast_not_found 也是正确输出。
5. **不要为几十个演员上向量数据库**。NumPy 矩阵足够；全局名人库才需要。
6. **不要只做单张人脸 argmax**。全局分配能解决重复匹配和 hub face 问题。
7. **不要在没有标注集的情况下拍阈值**。阈值是产品参数，不是模型默认值。
8. **不要用性别/年龄做硬过滤**。估计不可靠、有偏差，而且对中性化造型的演员不友好。

---

## 10. 技术选型表

| 组件 | 默认选择 | 备选 | 说明 |
| --- | --- | --- | --- |
| 人脸检测 | SCRFD-10G | SCRFD-34G、RetinaFace、YOLO-face | 先切片，再考虑融合 |
| 人脸识别 | ArcFace `buffalo_l`（R50） | `antelopev2`（R100）、AdaFace、自训 | 先 baseline，难例再升级 |
| 对齐 | 5 点相似变换 | 3D landmarks | 112×112 ArcFace 模板 |
| 质量 | 尺寸/模糊/姿态/检测分 | 人脸解析、3D 姿态 | 用于校准和路由 |
| 匹配 | cosine + cohort 归一化 + 校准 | T-norm、双模型融合 | 不要 raw cosine 阈值 |
| 分配 | Hungarian + unknown | 逐脸 argmax + 冲突后处理 | 小 cast 很快 |
| 服务 | FastAPI + ONNX Runtime | Triton、TorchServe | MVP 先进程内 |
| 队列 | Redis Streams / Arq | SQS、Celery | 批量任务 |
| 缓存 | Redis | Memcached | 结果/gallery/embedding |
| 存储 | S3/MinIO | 本地盘 | 调试图、原始图 |
| 数据库 | Postgres | MySQL | 任务、反馈、审计 |
| 部署 | Docker Compose → K8s | ECS/云函数 | GPU 节点池 |
| 监控 | Prometheus + Grafana + OTel | 云监控 | 分阶段指标 |

---

## 11. 需要产品/业务确认的问题

1. **商用还是研究**：决定模型许可，InsightFace 官方预训练模型可能仅限非商业研究。
2. **日处理量和延迟目标**：决定 CPU 还是 GPU、是否需要队列和 Triton。
3. **海报类型**：是否包含手绘/动画/漫画海报；如果是，需要单独模型或明确拒绝。
4. **数据来源**：头像由调用方上传，还是服务端抓百度百科；后者有 ToS/反爬风险。
5. **精度与覆盖率的取舍**：自动接受 97% 精度、70% 覆盖，还是更高覆盖但更多复核？
6. **部署环境**：云/本地、GPU 型号、是否允许外网访问图片源。

---

## 12. 最小落地清单

如果现在开始做，按这个顺序：

- [ ] 收集 30–50 张目标海报，人工标人脸和身份。
- [ ] 跑通 SCRFD-10G + buffalo_l + cosine baseline，记录检测召回和 Top-1。
- [ ] 加切片检测，确认小脸召回提升。
- [ ] 写 gallery 构建：检测、选脸、embedding、质量状态。
- [ ] 加 cohort 归一化和逻辑回归校准，画 precision–coverage 曲线。
- [ ] 加 Hungarian + unknown，处理重复匹配。
- [ ] FastAPI + Redis 缓存 + Docker Compose。
- [ ] 简单复核页面 + 反馈落库。
- [ ] 100 张海报评测集 + 随机抽检流程。
- [ ] 再决定是否上 antelopev2、域适配、批量队列。

---

## 13. 建议的代码结构

```text
facedec/
  app/
    main.py                  # FastAPI 入口
    api/
      galleries.py           # /v1/galleries
      identify.py            # /v1/identify
      jobs.py                # /v1/jobs
    core/
      config.py
      security.py            # SSRF、鉴权、限流
      logging.py
    pipeline/
      fetch.py               # 下载、校验、解码
      preprocess.py          # EXIF、sRGB、缩放、切片
      detect.py              # SCRFD
      align.py               # 5 点对齐
      quality.py             # 质量特征
      embed.py               # ArcFace
      gallery.py             # 选图、一致性、缓存
      match.py               # cosine、归一化
      calibrate.py           # 校准模型
      assign.py              # Hungarian + unknown
      orchestrator.py        # 串起整个流程
    adapters/
      baike.py               # 可选：百科数据适配器
    models/
      registry.py            # bundle 加载、校验、warmup
    schemas/
      api.py
      domain.py
    workers/
      tasks.py               # 异步任务
  eval/
    benchmark.py
    metrics.py
    calibration.py
    annotate/                # 标注工具/导入
  deploy/
    Dockerfile
    docker-compose.yml
    k8s/
  tests/
    unit/
    integration/
    golden/
  docs/
    design.md
```

建议的 baseline 配置：

```yaml
detector: scrfd_10g
recognizer: buffalo_l
det_size: 640
tile:
  enabled: true
  size: 1024
  overlap: 128
  min_long_side: 1600
face:
  min_detect_px: 24
  min_recognize_px: 48
  max_faces: 64
match:
  normalize: cast_cohort
  calibration: logistic
  assignment: hungarian_with_unknown
  allow_duplicate_cast: false
```

这份配置足以跑出第一版基线；后续每改一项都进消融实验，不要凭感觉调参。
