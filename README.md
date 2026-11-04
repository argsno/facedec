# facedec

海报演员识别的 **CPU-only 可行性验证** 工程。

输入一张海报 + 一组演员头像/剧照（调用方上传），输出海报中每张人脸对应的演员、原始相似度、状态（`accepted` / `review` / `unknown`）、未匹配人脸和未出现演员。完整生产方案见 [`docs/design.md`](docs/design.md)。

当前定位是研究/可行性验证，不是生产服务：

- 只跑 **InsightFace buffalo_l（SCRFD-10G + ArcFace R50）**，CPU 推理。
- 只有原始 cosine 分数，没有概率校准、per-identity 阈值、cohort normalization。
- 没有 Redis/队列/数据库/缓存；API 只做同步上传识别。
- 阈值是保守的占位值，必须用你自己的标注集校准。
- 不抓百度百科；头像由调用方上传，避免 ToS/反爬问题。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

首次运行会自动下载 `buffalo_l` 模型（约 300MB）到 `~/.insightface/models/buffalo_l`。网络慢时请预留时间。

## Cast JSON

`portrait_paths` 支持相对路径（相对于 cast JSON 所在目录）和绝对路径：

```json
{
  "name": "示例剧",
  "members": [
    {
      "cast_id": "c_01",
      "name": "演员A",
      "role": "角色A",
      "portrait_paths": ["portraits/a_01.jpg", "portraits/a_02.jpg"]
    },
    {
      "cast_id": "c_02",
      "name": "演员B",
      "portrait_paths": ["portraits/b_01.jpg"]
    }
  ]
}
```

注意：

- `cast_id` 必须唯一；展示名可能重名。
- 优先用剧中角色剧照，其次才是活动照/百科头像；前者和海报的域差异更小。
- 合照、低清、检测不到脸的头像会被标记并跳过，不要硬塞进 gallery。

## CLI

```bash
facedec \
  --poster poster.jpg \
  --cast cast.json \
  --out result.json \
  --debug debug.jpg \
  --faces-dir faces/
```

`--faces-dir` 会为每张检出的人脸保存裁剪图（`f_1.jpg`、`f_2.jpg`…），并在同目录写 `index.json`，把 `face_id`、bbox、Top-1 候选和裁剪图路径对应起来。调试图只画 bbox 和 cast_id；需要看名字时打开 `index.json` 或自己按 `face_id` 对照。

常用参数：

```text
--no-tile                 关闭切片（小图/调试用）
--tile-size 1024          切片大小
--tile-overlap 128        切片重叠
--tile-trigger 1600       长边超过该值时启用切片
--accept-threshold 0.45   自动接受阈值（原始 cosine，未校准）
--review-threshold 0.30   进入复核的阈值
--margin-threshold 0.08   Top-1 与最佳其他候选的最小差
--min-face-px 48          低于该像素的人脸不自动接受
```

输出示例：

```json
{
  "faces": [
    {
      "face_id": "f_1",
      "bbox": [820, 410, 1010, 640],
      "quality": {"score": 0.83, "size_px": 190, "blur": 620, "yaw": -8},
      "match": {
        "cast_id": "c_01",
        "name": "演员A",
        "score": 0.61,
        "margin": 0.12,
        "status": "accepted",
        "candidates": [
          {"cast_id": "c_01", "name": "演员A", "score": 0.61},
          {"cast_id": "c_07", "name": "演员G", "score": 0.49}
        ]
      },
      "confidence": "high"
    }
  ],
  "unknown_faces": [],
  "cast_not_found": [],
  "warnings": [],
  "timings_ms": {"gallery_ms": 190, "detect_embed_ms": 176, "total_ms": 370}
}
```

`score` 是原始 cosine，不是概率；不要把 0.61 展示成 61%。当前工程用 `accepted` / `review` / `unknown` 三档，而不是伪造百分比。

## 评测

`facedec-eval` 用一份人工标注的 manifest 计算检测召回、Top-1、自动接受精度/覆盖率/复核率。

`manifest.json` 格式：

```json
{
  "posters": [
    {
      "poster": "posters/example.jpg",
      "cast": "cast.json",
      "faces": [
        {"bbox": [820, 410, 1010, 640], "cast_id": "c_01"},
        {"bbox": [120, 900, 240, 1040], "cast_id": null}
      ]
    }
  ]
}
```

`cast_id: null` 表示这张脸是 unknown/路人，用来测开放集误接受。`poster` 和 `cast` 的相对路径相对于 manifest 所在目录。

```bash
facedec-eval --manifest manifest.json --out eval.json
```

输出包含：

- `detection_recall`：所有 GT 人脸中被检测到的比例；
- `top1_accuracy`：检测到且身份已知的人脸中，Top-1 正确的比例；
- `auto_accept_precision`：自动接受中正确的比例（精度优先阶段最关键）；
- `coverage`：自动接受的人脸 / 已知身份人脸；
- `review_rate` / `unknown_rate`：复核率和未知率；
- `unknown_gt_false_match_rate`：unknown/路人人脸被错误接受或送审的比例。

注意：manifest 必须独立人工标注，不能拿检测器输出当 GT；否则指标会虚高。`facedec-eval` 会在同一 manifest 内按 cast 复用 gallery；CLI 单次运行仍会重建 gallery。生产仍需要持久化 cache。

## FastAPI

```bash
uvicorn facedec.api:app --host 127.0.0.1 --port 8000
```

上传接口要求 cast JSON 用 `portrait_files` 引用上传文件名：

```bash
curl -X POST http://127.0.0.1:8000/v1/identify \
  -F "poster=@poster.jpg" \
  -F 'cast={"members":[{"cast_id":"c_01","name":"演员A","portrait_files":["a.jpg"]}]}' \
  -F "portraits=@portraits/a.jpg"
```

API 把上传文件写到临时目录，请求结束后删除；没有鉴权、限流、URL 抓取和持久化，不要直接暴露到公网。

## 已验证

用《蒸蒸日上》的一张官方海报 + 百度百科星图 92 位演员做过一次流程验证：

- 海报 5 张脸全部检出；
- 5 人 Top-1 全部正确；
- 4 人自动接受且全部正确，1 人（高叶）进入复核；
- 92 人 gallery 下没有把另外 87 位演员误接受；
- 5 人 gallery 约 1.2s，92 人 gallery 约 13s（CPU，gallery 构建占 12.6s）。

详细数据、分脸分数和局限见 [`docs/validation-zzrs.md`](docs/validation-zzrs.md)。这只是一张海报的流程验证，不是精度基准。

## CPU 性能（本机实测）

测试环境：Apple Silicon macOS，Python 3.12，`buffalo_l`，CPUExecutionProvider。

| 场景 | 实测 |
| --- | --- |
| 模型加载（已缓存） | ~0.4s |
| 512×512，单脸，检测+识别 | ~0.23s |
| CLI 端到端（单图、单演员 gallery） | ~0.37s |
| 2048×2048，切片检测+去重 | ~2.5s（测试用例） |

实际耗时取决于 CPU 核数、海报尺寸、切片数、人脸数和模型输入尺寸。CPU 做 10–20 张海报的可行性验证完全够用；不需要 GPU。

## 可行性验证流程

建议严格按这个顺序，不要先调模型：

1. 选 **10–20 张**目标海报，人工标所有人脸的 bbox 和身份（包括 unknown）。
2. Cast 头像用和线上一致的来源；保留低质量样本，不要只挑好图。
3. 跑 baseline，记录：
   - 检测召回（IoU≥0.5，人脸≥48px）
   - Top-1 正确率
   - 自动接受精度 / 覆盖率 / 复核率
   - 每张海报的耗时
4. 看错误案例，先分桶：检测漏脸、gallery 不可用、侧脸/小脸、域差异、相似演员。
5. 如果自动接受精度低于 97%，提高 `accept-threshold`；如果覆盖率太低，先修 gallery 和检测，不要急着换模型。
6. 扩到 **50 张**确认；再决定是否上 antelopev2、校准模型或轻量域适配。

建议的 go/no-go 线（可按业务调整）：

- 人脸≥48px 的检测召回 ≥ 90%；
- 自动接受精度 ≥ 97%；
- 自动接受覆盖率 ≥ 40–50%（精度优先阶段先接受较低覆盖）；
- 复核率可运营承受。

如果连清晰正脸都大量匹配失败，说明 gallery 或域差异有问题；如果清晰正脸可以、侧脸/小脸不行，再考虑难例升级和切片优化。

## 已知限制

- 没有做概率校准；阈值不能跨数据集直接复用。
- 没有做 per-identity 阈值、T-norm、双模型融合。
- Gallery 只用 `max cosine`，没有做质量加权或 pose-aware 融合。
- 海报风格不区分；手绘/动画海报效果不可靠。
- 不处理海报文字 OCR、角色名先验、跨海报 gallery 扩展。
- 不支持批量任务、队列、缓存、多租户、鉴权、SSRF 防护。
- 调试图标签只画 ASCII（cast_id/分数），中文名不渲染。
- InsightFace 官方预训练模型通常限制非商业研究用途；商用前核对许可。

## 目录

```text
facedec/            核心包（engine/gallery/matching/pipeline/cli/api）
tests/              单元测试 + 慢速模型冒烟测试
docs/design.md        完整设计方案
cast.example.json     cast 示例
manifest.example.json 评测 manifest 示例
pyproject.toml
```

## 测试

```bash
# 不需要模型
pytest -q -m "not slow"

# 需要模型（会加载 buffalo_l）
RUN_SLOW_TESTS=1 pytest -q -m slow
```
