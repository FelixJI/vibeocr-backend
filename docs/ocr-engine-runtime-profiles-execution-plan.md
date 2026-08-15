# OCR 引擎与离线运行时执行计划

> 状态:部分实施(2026-08-15,PR #43)。B1–B4 已完成并通过质量门与协议
> 一致性;B5 第一片已落地:新增 `win-x64-base` 离线 profile(RapidOCR +
> 显式 onnxruntime + pywinrt(winrt-runtime 及组件包) + 单套 opencv-python 的 hash lock)、manifest
> 三 profile 绑定(base 可选 runtime_pack + SHA-256)、installer 离线安装
> 路径(manifest 绑定 pack 时 `--no-index --find-links`,幂等解压,
> pack 缺失 fail closed)与 `scripts/build_runtime_pack.py`(确定性 zip,
> 下载即校验)。B5 剩余(full pack 附加下载/校验/移除流程、full 锁的
> OpenCV 去重)与 B6(正式 release)待后续工作包。本文只定义
> `vibeocr-backend` 仓库的工作包;不在 Backend 添加 UI。
>
> 许可决策记录(维护者拍板,2026-08-15):项目定位为开源、非商业分发。
> RapidOCR wheel/PP-OCRv6 ONNX 权重与 PyMuPDF(AGPL-3.0)按开源非商业口径
> 接受,B0.1/B0.2 不再作为实施阻塞;正式发布(B6)前仍需在 Release 资产中
> 完成 NOTICE/SPDX SBOM 记录,B0.3 的隔离机验证仍待执行。
>
> B5 第一片已真实验证(2026-08-15 本机):从 base hash lock 构建出
> 59-wheel/134 MiB 离线 pack(单 ORT、单 OpenCV、antlr sdist 离线 wheel 化),
> `--no-index --find-links` 全新目录安装成功,rapidocr/onnxruntime/winrt
> projection 全部可导入,并完成一次真实识别冒烟(`Hello VibeOCR 123`,
> score 0.96,bbox 归一化正确,模型来自 wheel 内置,零网络下载)。
> 附带发现并绕开两个生态事实:rapidocr 依赖的 omegaconf→antlr4-python3-runtime
> 4.9.3 只有 sdist;winsdk 是 scikit-build sdist——Windows adapter 改用带
> win_amd64 wheel 的 winrt-runtime + 组件包。
>
> 实施备注:Protocol 2.6.0 的 stdlib parser 尚不接受 wire `engine` 字段
> (OpenAPI schema 已允许);Backend 在 `supervisor/app.py` 的提交边界以
> `_extract_engine_selection` seam 严格解析并 fail closed,Protocol parser
> 修复后可移除该 seam。rapidocr 3.9.2 的 metadata 不声明 onnxruntime
> (运行时动态导入),因此 base lock 由本仓显式锁定 ORT(§4.3)。

## 1. 目标与边界

Backend 提供通用文本 OCR 的单一深模块，在同一请求/结果接口后封装三种 adapter：

- RapidOCR：全局默认，随基础离线运行时携带。
- Windows Media OCR：使用 Windows 自带能力，按系统语言包动态探测。
- PaddleOCR：可选重型组件，由用户显式准备。

同时把基础图片 OCR、批量 OCR、PDF 渲染与基础可搜索 PDF 所需依赖放入可离线携带的基础运行时。Classic/Next 只选择引擎和消费结果，不能重新嵌入推理实现。

## 2. 必须先通过的决策门

### B0.1 RapidOCR 体积与许可

已测基线（RapidOCR 3.9.2、CPython 3.13、win-x64）：

- wheel：27,275,208 B（约 26.01 MiB）。
- 安装后 RapidOCR tree：约 30.88 MiB。
- 随 wheel 模型：约 30.28 MiB。
- 加新增支持依赖后的保守安装增量：约 31.23 MiB。
- 现有 CPU lock 已含 ONNX Runtime 等主要依赖，因此不得打入第二份 ORT。

发布预算：RapidOCR 相关压缩资产增量不高于 35 MiB、安装增量不高于 45 MiB。超出任一值需重新评审，不得悄悄扩大 portable 基础包。

硬门禁：核实 RapidOCR 包、PP-OCRv6 ONNX 权重、字典/字体和全部再分发 NOTICE。Python 包的 Apache-2.0 不能自动代表模型权重许可。许可不清楚时不得进入正式 runtime pack。

### B0.2 PDF 依赖许可

当前 PDF 服务依赖 PyMuPDF/fitz。发布前必须记录选择：

1. 获得适用于本项目分发方式的商业/其他许可；或
2. 换用许可兼容的 PDF 实现并完成等价能力验证。

在该门禁关闭前，只能做技术验证，不能宣称基础包可正式分发或完全离线。

### B0.3 Windows OCR 可行性

在隔离的 Windows 10/11、Python 3.13、无全局开发环境机器上验证：

- WinRT projection/wheel 与冻结 runtime installer 的兼容性。
- 已装/未装语言包的探测和错误码。
- 行级 bbox、旋转、语言和置信度能否归一到稳定 OCRResult。
- PDF 隐形文字层所需几何信息是否足够。

未通过时 Windows engine 必须显示 `unavailable`，不能伪造成空识别结果。

## 3. 模块设计

### 3.1 稳定接口

在 application/core 边界建立与具体库无关的 `GeneralTextOcrEngine` 接口，最少负责：

- `descriptor()`：返回 ID、availability、reason、所需 component。
- `recognize_many(images, options, cancellation)`：统一单图/批量入口。
- 输出稳定 OCRResult：text、score、bbox/polygon、顺序与页关联。
- 生命周期、惰性初始化、取消和资源释放。

接口后放置三个 adapter：`RapidOcrAdapter`、`WindowsMediaOcrAdapter`、`PaddleOcrAdapter`。现有 Paddle pipeline parsing 可下沉或复用，但上层不再叫 `PaddleExecutor` 来代表所有通用 OCR。

### 3.2 Registry 与 resolver

`OcrEngineRegistry` 负责注册、探测和缓存；`OcrEngineResolver` 只按 Protocol 的显式 engine ID 选择。

规则：

- 请求缺省时选择 RapidOCR。
- 用户选择的引擎不可用时返回协议错误及可选 ID，不自动切到 Windows/Paddle。
- engine 仅适用于 `OCR` pipeline；其他 pipeline 保持现有执行器。
- adapter 异常映射到稳定 application error，不能把第三方异常/路径直接暴露给前端。
- engine 目录来自实际探针和已安装 component，不从静态依赖列表猜测。

### 3.3 PDF seam

`PdfOcrBackend.recognize_pages` 继续作为 PDF 编排边界。PDF 负责 render/mutate/text-layer/save，通用 OCR 只接收页图并返回规范化几何块。

基础离线验收范围：

- 打开本地 PDF、渲染页面。
- 用 RapidOCR 或可用的 Windows OCR 识别。
- 生成可搜索 PDF 并保存。
- 全流程禁网仍可运行。

复杂版面恢复、MinerU、公式、VLM 不属于基础离线承诺。

## 4. 依赖与运行时组件设计

### 4.1 组件层级

新增声明式 runtime component/profile，避免把所有重依赖塞进一个 CPU profile：

| 组件/档位 | 内容 | 获取策略 |
|---|---|---|
| `base-offline` | Backend/Protocol、RapidOCR+模型、单份 ORT、图片编解码、已批准的 PDF runtime、runtime host | 随 Classic/Next Portable 携带，禁网可安装 |
| `windows-ocr` | WinRT adapter；系统 OCR/语言包由 Windows 提供 | adapter 随 base，系统能力动态探测 |
| `full-cpu` | PaddleOCR、文档解析/MinerU 等 CPU 重依赖 | 用户显式选择后下载受 manifest 约束的 pack |
| `full-cu126` | CUDA/Paddle/Torch 等 GPU 闭包 | 用户显式选择后下载；与 CPU/base 隔离验证 |

`base-offline` 是必备闭包；`full-cpu` 与 `full-cu126` 是附加闭包，不改变 RapidOCR 默认值。

### 4.2 离线 pack

当前 installer 按 hash lock 从 PyPI 安装，单改 lock 不等于离线。需实现受 release manifest 管理的 runtime pack：

1. CI 从精确 lock 构建 wheel/model/PDF asset 闭包。
2. manifest 记录 component、平台、Python ABI、版本、size、hash、SBOM 和来源。
3. installer 使用 `--no-index --find-links <bound-pack>` 安装 base，禁止回退公网。
4. RapidOCR wheel 内模型或外置模型只能保留一份。
5. `full-*` pack 作为 Backend Release 的独立重型资产发布，由用户选择时下载，不塞进每次前端 nupkg 更新。
6. 下载的附加 pack 和已安装 runtime 均位于前端传入的 portable data root，不写系统 Python、用户 site-packages、注册表或服务。

### 4.3 冲突控制

- RapidOCR 显式锁定 ONNX Runtime；不能依赖 RapidOCR 的间接声明，因为其 metadata 不包含 ORT。
- 基础 pack 只保留一套 OpenCV 发行物，消除当前 `opencv-python`/`opencv-contrib-python` 重复风险。
- Paddle/Torch/CUDA DLL 不进入 base 进程搜索路径；保留并扩展现有 DLL 冲突 smoke。
- 各 component 的 lock 由 `.in` 和仓库脚本生成，禁止手改 hash lock。
- component resolver 必须检测磁盘空间、ABI/平台和互斥项，并给出结构化失败原因。

## 5. 分阶段工作包

### B1：消费正式 Protocol

更新 `release/protocol.lock.json` 及其生成/验证资产，接入 `OcrEngineId`、catalog 和错误码。必须消费正式 Protocol Release，不直接引用 Protocol worktree。

验收：现有 protocol conformance 加入 engine request/catalog fixture 并通过。

### B2：抽取通用引擎深模块

主要入口：

- `packages/vibeocr-backend/src/vibeocr/backend/application/ocr_facade.py`
- `packages/vibeocr-backend/src/vibeocr/backend/application/contracts.py`
- `packages/vibeocr-backend/src/vibeocr/backend/services/ocr_service.py`
- `packages/vibeocr-backend/src/vibeocr/backend/supervisor/composition.py`
- `packages/vibeocr-backend/src/vibeocr/backend/supervisor/inference/paddle_adapter.py`

先用现有 Paddle 实现新接口，保持结果 payload 与取消语义稳定；然后将 supervisor 的 `RECOGNITION` 路由改为通用 executor。不得同时改 PDF/UI。

验收：旧 Paddle adapter 测试迁移到引擎契约测试；单图、批量、取消、初始化失败均通过。

### B3：接入 RapidOCR 与 Windows adapter

RapidOCR：固定模型路径、禁用网络下载、映射 text/score/polygon、复用单份 ORT。Windows：探测 OS/API/language，映射行级几何，明确无 score 时的契约策略。

验收：golden 图片上输出结构稳定；禁网且清空用户缓存后 RapidOCR 首次识别成功；Windows 缺语言包返回 `ocr_engine_language_unavailable`。

### B4：接通 PDF 基础离线链路

在 `application/pdf_ocr_orchestrator.py` 保持 page render → engine → text layer → save 边界。补充多页、旋转页、空白页、取消和保存失败测试。

验收：在无网络、无系统 Python/全局包环境中，从便携 runtime pack 完成基础可搜索 PDF；提取文本和文字层位置满足 fixture 容差。

### B5：重构 profile/installer/manifest

主要入口：

- `packages/vibeocr-backend/runtime-profiles/`
- `packages/vibeocr-backend/src/vibeocr/backend/runtime_installer.py`
- `scripts/build_runtime_installer.py`
- `scripts/build_runtime_manifest.py`
- `.ci/project.json`

建立 base/full component manifest、离线安装路径、附加 pack 下载/校验/移除流程。开发阶段不保留旧 profile ID 的迁移逻辑，但要同步所有仓内消费者和测试。

验收：base 安装时网络被阻断仍成功；full pack 未选择时不会下载；选择后只下载对应 pack；重复 ensure 幂等；失败不破坏已有 base。

### B6：正式 release 与下游交接

按 `.ci/project.json` 执行 bootstrap、quality、Protocol E2E、真实 release build 和 asset smoke。最终 Release manifest 必须区分：

- 前端必须内嵌的 `base-offline` pack。
- GitHub Release 托管、用户按需下载的 `full-cpu`/`full-cu126` pack。

向 Classic/Next 提供正式 Backend Release、绑定 Protocol、component IDs、size、hash、catalog fixture 和离线验收样例。

## 6. 测试与验收矩阵

重点测试入口：

- `tests/application/test_ocr_facade.py`
- `tests/application/test_pdf_ocr_orchestrator.py`
- `tests/core/test_pipeline_ocr_parsing.py`
- `tests/core/test_pipelines.py`
- `tests/core/test_pipelines_metadata.py`
- `tests/supervisor/inference/test_paddle_adapter.py`
- `tests/supervisor/inference/test_paddle_executor.py`

必须新增/调整：

- 三 adapter 的共享 contract suite。
- engine resolver 无静默 fallback。
- catalog 与实际 component 状态一致。
- Rapid 模型路径、缺文件和 ORT load failure。
- Windows 可用/无 API/无语言包。
- base 禁网安装与首次 OCR/PDF smoke。
- full component 下载中断、hash 错误、空间不足和恢复。
- CPU/CUDA 原生 DLL 冲突 smoke。

仓库完整命令以 `.ci/project.json` 为准，至少执行其 bootstrap、quality、E2E、release build、release smoke；重型 profile 变更不能只跑 unit tests。

## 7. 最终验收标准

- [ ] RapidOCR 是 Backend 缺省引擎且存在于 `base-offline`。
- [ ] 三个 adapter 共享一个 application 接口和稳定 OCRResult。
- [ ] 不可用引擎返回结构化错误，Backend 不替用户切换。
- [ ] RapidOCR 和基础 PDF 在完全禁网、干净 Windows VM 中首次可用。
- [ ] RapidOCR 增量处于约定预算，ORT/OpenCV/模型无重复副本。
- [ ] Rapid 模型及 PDF runtime 许可、NOTICE、SPDX SBOM 全部通过。
- [ ] full CPU/CUDA 未选择时不进入 Portable，也不随每次应用更新重复下载。
- [ ] runtime 所有写入都受 portable data root 约束，无系统级侵入。
- [ ] 正式 Backend Release 与绑定 Protocol 通过完整 release smoke，供两个前端消费。

## 8. PR 边界

本仓形成一个独立 Draft PR，依赖 Protocol 正式资产；不得混入 Classic/Next UI。若工作量需要多个内部 commit，建议按“接口抽取 / adapters / runtime packs / PDF+release gate”组织，但对外交付仍保持一个可整体评审的 Backend PR。未经另行授权不合并、不发布。
