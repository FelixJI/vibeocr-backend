# 正式生产模型资产闭包调查

调查日期：2026-08-18。范围仅限当前 `codex/review-remediation` worktree 的真实生产调用、release locks，以及 PaddleOCR/PaddleX/MinerU 官方源码、官方文档和官方模型仓库元数据。

## 结论

当前不能生成可合入正式 Backend Release 的 `full` / `document_parsing` 生产模型资产闭包。

上游一手资料足以机器确定两类事实：锁定 Python 分发版本下的默认 pipeline 配置与可接受的本地目录参数；在选定某个模型仓库和不可变 commit 后，Hugging Face Hub 也能枚举该 snapshot 的文件、大小及 Git/LFS 元数据。但当前 Backend Release 没有确认并冻结以下输入：模型族/启用模块、每个仓库的不可变 commit、runtime 所需的精确文件子集，以及 Backend manifest 要求的每个文件 SHA-256。不能把 `main`、`master`、`latest`、分支名、模型仓库当前 HEAD 或不带 `revision=` 的 `snapshot_download()` 当作生产 pin。

另有两个必须先解决的契约冲突：

1. `PaddleOCRVL(device=...)` 在 PaddleOCR 3.7.0 中默认选择 `pipeline_version="v1.6"`，而 Backend 当前 UI 描述仍称“PaddleOCR-VL-1.5”。这是产品版本选择，不应由资产生成器猜测。
2. MinerU 3.4.4 的本地模式要求 `MINERU_MODEL_SOURCE=local`，且 `mineru.json` 的 `models-dir` 是含 `pipeline` / `vlm` 两个键的对象；调查开始时 Backend 写成单个字符串，并继续把远端 source id 注入 `MINERU_MODEL_SOURCE`。本轮已将 manifest 与 resolved binding 修正为 `pipeline` / `vlm` 双根，并在解析到本地资产时固定使用 `local`。

## 已锁定的软件事实

仓库没有根级 `uv.lock`；正式依赖事实来自四份 release requirements locks：

- CPU/full：`paddleocr==3.7.0`、`paddlex==3.7.2`、`paddlepaddle==3.3.1`、`mineru==3.4.4`。
- CUDA/full：公共 `win-x64-cu126` 同样锁定 `paddleocr==3.7.0`、`paddlex==3.7.2`、`mineru==3.4.4`；GPU supplement 将 `paddlepaddle-gpu` 固定到 3.3.1 CUDA 12.6 wheel。
- `packages/vibeocr-backend/pyproject.toml` 只给下界（`paddleocr[doc-parser]>=3.7.0`、`mineru>=3.4.3`），不能替代 release locks。

官方版本证据：[PaddleOCR v3.7.0 Release](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0)、[PaddleX v3.7.2 Release（页面给出 commit `ffb6490`）](https://github.com/PaddlePaddle/PaddleX/releases/tag/v3.7.2)、[MinerU `mineru-3.4.4-released` 源码引用](https://github.com/opendatalab/MinerU/tree/mineru-3.4.4-released)。最后一个是命名引用而非本文可证明的不可变 commit，不能直接充当模型 revision。

## 按 consumer 的闭包状态

| consumer | 可机器获取的官方事实 | repository / fixed revision / files / size / checksum 是否可确定 | 正确本地 binding | 状态 |
|---|---|---|---|---|
| `paddleocr` | PaddleOCR 3.7.0 未传 `lang`/`ocr_version` 时明确选择 `PP-OCRv6_medium_det` 与 `PP-OCRv6_medium_rec`；PaddleX 3.7.2 `OCR.yaml` 还声明 `PP-LCNet_x1_0_doc_ori`、`UVDoc`、`PP-LCNet_x1_0_textline_ori`。Backend 构造器没有在初始化时关闭这些默认模块。 | 官方 HF repository id 可按 `PaddlePaddle/<model_name>` 得到；当前 Backend 未固定各 repo commit。选定 commit 后可枚举文件和 size；HF LFS 文件有官方 LFS SHA-256，但普通 Git 文件元数据不是逐文件 SHA-256，不能直接满足 Backend 对所有文件的 `sha256` 字段。 | `doc_orientation_classify_model_dir`、`doc_unwarping_model_dir`、`text_detection_model_dir`、`textline_orientation_model_dir`、`text_recognition_model_dir`。 | **无法闭合**：需正式 revision、精确 runtime 文件集和逐文件 SHA-256 release 输入。 |
| `pp_structure` | PaddleX 3.7.2 `PP-StructureV3.yaml` 明确默认启用 table、formula、region，关闭 seal、chart、doc preprocessor；Backend predict 又把 doc orientation/unwarping、table、formula 设为真。官方 YAML 可枚举默认模型：`PP-DocLayout_plus-L`、`PP-DocBlockLayout`、`PP-LCNet_x1_0_doc_ori`、`UVDoc`、`PP-OCRv5_server_det`、`PP-LCNet_x1_0_textline_ori`、`PP-OCRv5_server_rec`、`PP-LCNet_x1_0_table_cls`、`SLANeXt_wired`、`SLANet_plus`、两种 RT-DETR table-cell 模型、`PP-FormulaNet_plus-L`；chart/seal 模型存在于配置但当前选项默认关闭。 | repository id 可从选定模型名映射到官方 repo，但“发布是否包含默认关闭的 chart/seal，以及是否覆盖所有请求可切换选项”是产品闭包定义；未决定前不存在唯一文件集合。revision/逐文件 SHA-256 同样未锁。 | `layout_detection_model_dir`、`region_detection_model_dir`、`doc_orientation_classify_model_dir`、`doc_unwarping_model_dir`、`text_detection_model_dir`、`textline_orientation_model_dir`、`text_recognition_model_dir`、`table_classification_model_dir`、`wired_table_structure_recognition_model_dir`、`wireless_table_structure_recognition_model_dir`、`wired_table_cells_detection_model_dir`、`wireless_table_cells_detection_model_dir`、`table_orientation_classify_model_dir`、`formula_recognition_model_dir`；若产品启用 chart/seal，还需相应 `chart_recognition_model_dir`、`seal_text_detection_model_dir`、`seal_text_recognition_model_dir`。 | **无法闭合**：模块闭包本身仍需 Owner 确认。 |
| `paddleocr_vl` | PaddleOCR 3.7.0 默认 `pipeline_version="v1.6"`；PaddleX 3.7.2 的 `PaddleOCR-VL-1.6.yaml` 使用 `PP-DocLayoutV3`、`PaddleOCR-VL-1.6-0.9B`、`PP-LCNet_x1_0_doc_ori`、`UVDoc`。Backend 文案则声称 1.5。 | 官方 repositories 可确定，但 1.5 与 1.6 是不同产品资产闭包；未作版本决策前 repository/model id 都不唯一。选定版本后仍需固定每个 repo commit、文件集合、size 与逐文件 SHA-256。 | 必须先显式传 `pipeline_version`；目录键为 `layout_detection_model_dir`、`vl_rec_model_dir`、`doc_orientation_classify_model_dir`、`doc_unwarping_model_dir`。 | **无法闭合**：明确的产品版本冲突。 |
| `mineru` | Backend/Protocol 当前默认 `hybrid-engine`，失败回退 `vlm-engine`、`pipeline`。MinerU 3.4.4 官方源码将 VLM repo 设为 `opendatalab/MinerU2.5-Pro-2605-1.2B`，pipeline repo 设为 `opendatalab/PDF-Extract-Kit-1.0`；pipeline 只按七个 `ModelPath` 子路径下载，VLM 下载整仓。 | repo id 与上游当前 allow-pattern 可读取；但 Backend 的运行契约允许三种 backend 与回退，因此生产闭包至少同时涉及 VLM 与 pipeline 两仓。上游 `snapshot_download()` 未传 revision，故是 floating；当前无固定 commit、正式文件清单或满足 Backend manifest 的逐文件 SHA-256。 | 不应作为 MinerU 构造 kwargs 注入。已设置 `MINERU_MODEL_SOURCE=local`，并生成 `MINERU_TOOLS_CONFIG_JSON` 指向 `{"models-dir":{"pipeline":"<root>","vlm":"<root>"}}`；当前产品保留三段回退，故 manifest 必须同时提供两根。 | **绑定契约已闭合，生产资产仍无法闭合**：两仓 revision、正式文件清单与 checksum 尚未锁定。 |

## 一手来源与可复现判断

- PaddleOCR 3.7.0 的 [OCR wrapper 源码](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.7.0/paddleocr/_pipelines/ocr.py) 给出默认 `PP-OCRv6_medium_det/rec` 和所有 `*_model_dir` 参数。
- PaddleOCR 3.7.0 的 [PPStructureV3 wrapper 源码](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.7.0/paddleocr/_pipelines/pp_structurev3.py) 给出完整本地目录参数面；[PaddleX 3.7.2 PP-StructureV3 配置](https://raw.githubusercontent.com/PaddlePaddle/PaddleX/v3.7.2/paddlex/configs/pipelines/PP-StructureV3.yaml) 给出默认模块开关与模型名。
- PaddleOCR 3.7.0 的 [PaddleOCRVL wrapper 源码](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.7.0/paddleocr/_pipelines/paddleocr_vl.py) 明确默认版本为 v1.6；[PaddleX 3.7.2 PaddleOCR-VL-1.6 配置](https://raw.githubusercontent.com/PaddlePaddle/PaddleX/v3.7.2/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml) 给出四个默认子模型。
- 官方模型存在性示例：[PaddlePaddle/PP-OCRv6_medium_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det)、[PaddlePaddle/PP-OCRv6_medium_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec)、[PaddlePaddle 官方模型索引](https://huggingface.co/PaddlePaddle/models)。这些页面的 `main` 只说明当前状态，不是 release pin。
- Hugging Face 官方 [`HfApi` / repository metadata 文档](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api) 说明可按 revision 列仓库树和文件元数据。只有在 Owner 先提供不可变 commit 后，才应抓取该 snapshot；LFS `oid` 可作为上游提供的 SHA-256，普通 Git blob 仍需由正式 release 输入生产者下载后计算并签署/绑定。
- MinerU 3.4.4 的 [ModelPath 官方源码](https://raw.githubusercontent.com/opendatalab/MinerU/mineru-3.4.4-released/mineru/utils/enum_class.py) 给出两仓 id 与 pipeline 子路径；[下载实现](https://raw.githubusercontent.com/opendatalab/MinerU/mineru-3.4.4-released/mineru/utils/models_download_utils.py) 显示调用 `snapshot_download(repo, ...)` 时没有 `revision`；[CLI 下载集合](https://raw.githubusercontent.com/opendatalab/MinerU/mineru-3.4.4-released/mineru/cli/models_download.py) 区分 pipeline 七项和 VLM 整仓；[配置读取实现](https://raw.githubusercontent.com/opendatalab/MinerU/mineru-3.4.4-released/mineru/utils/config_reader.py) 证明 `models-dir` 必须按 `repo_mode` 取键。
- MinerU 官方 3.4.4 源码页说明三种 backend 目标不同，并记录 3.3 将 VLM 升级到 `MinerU2.5-Pro-2605-1.2B`、3.4 将 pipeline OCR 升级为 PP-OCRv6：[官方源码/变更说明](https://github.com/opendatalab/MinerU/tree/mineru-3.4.4-released)。

## 产品 Owner / 正式 Release 必须提供的最小输入

1. 对四个 consumer 明确允许的 pipeline/模型版本与功能开关；特别确认 PaddleOCR-VL 1.5 还是 1.6，以及 PP-StructureV3 是否把默认关闭但 API 可开启的 chart/seal 纳入 `document_parsing`。
2. 明确 MinerU 是否保留 `hybrid-engine -> vlm-engine -> pipeline` 全回退；若保留，VLM 与 pipeline 两仓都必须进入闭包。
3. 为每个选定官方 repository 给出完整 40 位不可变 commit SHA；禁止 `main`、`master`、`latest`、分支名或省略 revision。
4. 由正式 release 输入生成步骤在该 commit 上记录 runtime 所需的精确相对路径、字节 size、每文件 SHA-256，并把来源 repository/revision 与 Backend release identity 一起签署/attest。普通 Git 文件的 SHA-256 是 Backend release producer 的权威摘要，不能冒充上游 registry 已发布的 checksum。
5. 按上述参数表生成一个 asset 对应一个真实模型目录的 binding；不要把多个 Paddle 子模型打成一个目录却只绑定单个 `*_model_dir`。
6. MinerU 双根本地配置与 `MINERU_MODEL_SOURCE=local` 已修正；资产输入就绪后仍须分别执行 CPU 与 CUDA 的禁网 full smoke，覆盖四个 consumer 和 MinerU 回退策略。

在这些输入到齐并由 release 构建验证之前，`document_parsing` 缺生产 `model-assets.json` 时 fail closed 是正确行为；不应从上游当前 HEAD 自动生成一个看似完整但会漂移的清单。
