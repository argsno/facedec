# facedec

## 项目定位

- CPU-only 的海报演员识别**可行性验证**工程；生产方案见 `docs/design.md`。
- 研究用途；默认模型是 InsightFace `buffalo_l`（SCRFD-10G + ArcFace R50）。
- 不引入 GPU 依赖；不要为了“更先进”直接换大模型或上 Triton/向量数据库。

## 关键约束

- 精度优先：宁可不给结果，也不要自动给错人；`unknown` 和 `cast_not_found` 是合法输出。
- 原始 cosine **不是概率**；没有校准前，只输出 `accepted/review/unknown` 三档，不要展示伪造百分比。
- 头像由调用方上传；不要实现百度百科抓取。
- `data/` 下的本地验证图片/结果已 gitignore；不要把抓取的图片或头像提交到仓库。
- 阈值默认值只是占位，必须用标注集校准；改动阈值要记录依据。
- 人脸 embedding 按敏感数据处理；临时上传文件请求结束即删。

## 开发约定

- 依赖：`pip install -e ".[dev]"`。
- 快速测试：`pytest -q -m "not slow"`。
- 模型冒烟测试：`RUN_SLOW_TESTS=1 pytest -q -m slow`。
- 新增匹配逻辑时补单元测试；新增图像/模型行为时补慢速冒烟测试。
- 注释和文档用中文；代码标识符用英文。
