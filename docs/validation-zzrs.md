# 单海报验证记录：《蒸蒸日上》

日期：2026-11-04

这是一次**流程验证**，不是精度基准。只用了 1 张海报，结论只能说明链路能跑通、CPU 速度可接受，不能说明跨海报泛化能力。

## 输入

- 海报：`https://bkimg.cdn.bcebos.com/pic/7e3e6709c93d70cf3bc7bdb37c81c600baa1cd11fd42...`（下载后 607×1080 JPEG）
- 演员数据：百度百科星图接口 `starmap/api/collectinfo`（`lemmaId=65590986`，`encodeRelId=8ebf3bd37dbe979085e0da81`）
- 星图共 92 条演员记录，下载到 89 张可用头像；3 条没有有效图片 URL
- 海报上实际出现 5 人，对应关系由海报顶部演员名和角色确认：

| 人脸 | 演员 | 角色 |
| --- | --- | --- |
| f_1 | 王皓 | 董北虎 |
| f_2 | 侯岩松 | 李国臣 |
| f_3 | 高叶 | 萧龙女 |
| f_4 | 张国强 | 萧军 |
| f_5 | 韩东君 | 李信 |

## 运行方式

5 人精简 gallery：

```bash
facedec --poster data/zzrs/poster_raw --cast data/zzrs/cast5.json \
  --out data/zzrs/result.json --debug data/zzrs/debug.jpg
```

92 人完整 gallery：

```bash
facedec --poster data/zzrs/poster_raw --cast data/zzrs/cast92.json \
  --out data/zzrs/result92.json --debug data/zzrs/debug92.jpg
```

评测：

```bash
facedec-eval --manifest data/zzrs/manifest92.json --out data/zzrs/eval92.json
```

## 结果

### 5 人 gallery

- 检测到 5 张脸，5 张全部 Top-1 正确。
- `accepted=4`，`review=1`，`unknown=0`。
- 自动接受精度 100%，覆盖率 80%，复核率 20%。

### 92 人 gallery

| 人脸 | 匹配 | 角色 | score | margin | 状态 |
| --- | --- | --- | --- | --- | --- |
| f_1 | 王皓 | 董北虎 | 0.5888 | 0.3378 | accepted |
| f_2 | 侯岩松 | 李国臣 | 0.5418 | 0.2742 | accepted |
| f_3 | 高叶 | 萧龙女 | 0.3346 | 0.0958 | review |
| f_4 | 张国强 | 萧军 | 0.5920 | 0.3953 | accepted |
| f_5 | 韩东君 | 李信 | 0.5168 | 0.2628 | accepted |

聚合指标：

| 指标 | 值 |
| --- | --- |
| 检测召回（5 张脸） | 1.0 |
| Top-1 准确率 | 1.0 |
| 自动接受数 / 正确数 | 4 / 4 |
| 自动接受精度 | 1.0 |
| 覆盖率 | 0.8 |
| 复核率 | 0.2 |
| unknown | 0 |
| 误接受 | 0 |
| 未出现演员 | 87（预期） |

## 观察

1. **链路可用**：星图接口能拿到演员名、角色、头像；海报下载、检测、对齐、embedding、Hungarian 分配、评测都能跑通。
2. **92 人 gallery 没有引入误匹配**：Top-1 仍然是正确的 5 人；87 个不在海报中的演员全部没有被错误接受。
3. **高叶是 review，不是 accepted**：score 0.3346，低于默认 `accept=0.45`。她的海报造型（短发、红马甲、笑容、场景光）与百科棚拍头像差异较大，但 Top-1 仍然正确，margin 0.0958。精度优先模式下进复核是预期行为，说明阈值在起作用。
4. **gallery 构建是 CPU 瓶颈**：92 人 × 1 张图，gallery 构建 12.6s；海报检测+识别只有 0.33s。生产环境必须缓存 gallery；可行性验证阶段可接受。
5. **3 个演员头像不可用**：接口返回了空/非法图片 URL。Gallery 构建把它们标成 `unusable`，不会让整个任务失败。这也说明生产环境必须把 gallery 质量状态返回给调用方。

## CPU 耗时

| 场景 | 耗时 |
| --- | --- |
| 5 人 gallery + 海报（CLI，含构建） | ~1.2–1.3s |
| 92 人 gallery + 海报（CLI，含构建） | ~12.9–13s |
| 92 人 gallery 构建 | ~12.6–13.7s |
| 海报检测+识别（暖 gallery） | ~0.33s |

`facedec-eval` 在同一个 manifest 内按 cast 复用 gallery；2 张海报的 92 人测试中，gallery 构建约 13.7s，暖缓存下每张海报约 0.34s。

环境：Apple Silicon macOS，Python 3.12，InsightFace `buffalo_l`，CPUExecutionProvider。

## 局限

- 只有 1 张海报、5 张清晰正脸；不能代表群像、小脸、侧脸、遮挡、浓妆、黑白、手绘海报。
- 本次 manifest 的 5 个 bbox 来自检测器输出并人工确认，不是独立标注；因此 `detection_recall=1.0` 只说明评测链路能对上，不能当作真实检测召回。真实评测必须独立标 bbox。
- 海报和头像都来自百度百科体系，域差异比“海报 vs 生活照”小；真实场景可能更难。
- 阈值没有校准；0.45/0.30/0.08 只是占位。高叶在 92 人 gallery 下 score 0.33，说明如果强行降低阈值到 0.33 可能提高覆盖率，但也可能引入误接受，不能凭单样本决定。
- 没有做 gallery 缓存、批量任务、并发和失败重试。
- 本次抓取只用于验证，没有把百度抓取逻辑写进代码；生产仍应由调用方上传头像或使用授权数据源。

## 下一步

1. 再收集 10–20 张同类型海报，人工标所有人脸和身份。
2. 用 `facedec-eval` 跑 precision–coverage 曲线，再决定阈值。
3. 观察高叶这类“造型差异大”的样本是普遍现象还是个案；如果是普遍现象，优先补充剧中剧照，而不是降阈值。
4. 如果 92 人 gallery 的 13s 成为瓶颈，先做 gallery embedding 缓存，再考虑并发/批处理。
