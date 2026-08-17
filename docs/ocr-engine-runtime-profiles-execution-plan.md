# OCR 引擎、Runtime Profiles 与可选组件执行计划

> 状态：选择面已实施（2026-08-17，Protocol 2.7.0）。PR #43 已完成三 OCR
> 引擎、`win-x64-base`、base runtime pack、runtime maintenance/profile
> 基础设施；本轮在正式 Protocol 2.7.0（2026-08-16 发布，attestation/
> manifest/hash 验证后经 `bind_component_releases` 重新绑定 lock）上落地
> `runtime.component-selection.v1` 与 `runtime.download-sources.v1`：
> `runtime_selection.py` 统一规范化（catalog 业务键唯一、`None`/`[]` 区分、
> 未知 id 与跨 accelerator fail closed）、health 携带两个 catalog、Settings
> 持久化 `download_source_ids`、maintenance ensure/retry 接入
> `install_component_ids` 与 source intent（requested/effective 贯通
> receipt/status/observe/SSE）、Runtime Host stdio 请求同步放行、marker
> 记录安装闭包（显式 base-only 装 base lock，范围变化触发重装）；retry
> command identity 也包含 component/source override，避免同 id 错误重放。
> package-index 已进入真实 pip 安装：默认使用仓库信任的清华 TUNA，官方
> PyPI 作为显式候选；运行时与 release pack 构建都会隔离宿主 pip/uv 源配置，
> hash lock 保持 source-neutral。CUDA 另有 `gpu_runtime` 精确 scope，不再把任意
> 非空选择无声扩大成完整 profile。
> `ocr.engine-selection.v1` 已在 PR #43 落地；**2.7.0 的 stdlib parser 仍不
> 接受 wire `engine` 字段（wire schema 已允许），`_extract_engine_selection`
> seam 保留至上游修复**。PaddleOCR 引擎的 `required_component` 修正为真实
> 可选组件 `document_parsing`（原占位 `full-cpu` 不在 component-selection
> 目录中）。待办：B6 的 model registry/模型资产消费及后续新增 feature 的精确
> lock，B0.3 隔离机验证，以及 B7 正式 Release 交接。

## 1. 当前事实与目标

已完成且继续保留的能力：

- 稳定 OCR wire id 为 `rapidocr`、`windows`、`paddleocr`；RapidOCR 是新配置的
  Backend 缺省引擎，不可用时不静默切换到另一引擎。
- `base-offline` 包含 RapidOCR、模型、单份 ONNX Runtime、图片/PDF 基础闭包和
  Runtime Host；Portable 在断网机器上可完成首次安装与基础 OCR/PDF。
- `profile_id` 与 accelerator 描述 Runtime 档位，现有 durable maintenance、
  journal、observe/SSE、cancel/retry、组件 repair 继续作为唯一安装状态机。
- Windows OCR adapter 随 base 提供，系统 OCR/语言包按 Windows 实际能力探测；
  PaddleOCR 与 MinerU 等重依赖不进入 base 进程搜索路径。

本轮目标是在不把推理实现移出 Backend 的前提下，补齐三个选择面：OCR 请求选择具体
引擎；用户按 feature/accelerator 选择可选重组件；用户选择 package index/model
registry，且每次安装操作拥有稳定的 source intent。

非目标：Protocol 不决定产品显示文案、镜像优先级和默认源；前端不解析依赖闭包；
不新增第二套安装事件或重建状态机；不把 full CPU/CUDA 闭包做成 Backend Release
资产。

## 2. 上游门槛与发布边界

- 开发门槛：锁定包含本文协议修订的 Protocol 源 fixture，生成 Python parser/DTO，
  通过 Backend conformance tests。
- 合并门槛：依赖约束和 component binding 指向包含新能力的正式 Protocol v2 minor
  Release（预期 2.7+，以实际正式版本为准）。
- 下游门槛：Backend 正式 Release 的 health/capability descriptor、manifest、
  component lock 和 base pack 同源且通过 attestation；Classic/Next 只消费该 Release。
- B0.1 RapidOCR 体积/许可和 B0.2 PDF 依赖许可已接受；正式发布前仍须在 NOTICE 与
  SPDX SBOM 中记录。B0.3 clean-machine Windows 隔离验证尚未完成。
- 实测 full CUDA 单个 torch wheel 约 2.44 GiB，超过 GitHub Release 单资产上限；
  加之 full 闭包与前端更新频率不同，full CPU/CUDA 统一改为 hash lock 驱动的在线
  安装，Backend Release 只发布 base pack。

## 3. Protocol 契约与 Backend 归属

| 能力/字段 | Backend 权威事实 | 约束 |
|---|---|---|
| `ocr.engine-selection.v1` / `ocr_engine_catalog` | Registry 汇总每个稳定引擎的 availability、base 包含性、reason、required component | catalog 不是产品默认；请求省略 `engine` 才使用 Backend 默认 |
| `PipelineSelection.engine` | job adapter 解析后交给 OCR resolver | 仅纯文本 `OCR` pipeline 有效；未知/不可用/需准备均 fail closed，不回退 |
| `runtime.component-selection.v1` / `component_variant_catalog` | 当前 Backend Release 的 manifest/component lock 生成 `feature_id + accelerator -> component_id` | `feature_id` 是能力族，不等同 OCR engine；MinerU 不能被命名为 OCR engine |
| `install_component_ids` | ensure/retry 的可选组件意图 | 省略=Backend 默认；`[]`=明确只保留 base；未知 id 返回 `RUNTIME_COMPONENT_UNKNOWN` |
| `runtime.download-sources.v1` / `download_source_catalog` | Backend 配置声明稳定 source id、开放 kind、endpoint | catalog 可有同 kind 多候选；单次选择每种 kind 至多一个；数组顺序无优先级；未知 id 返回 `DOWNLOAD_SOURCE_UNKNOWN` |
| Settings `download_source_ids` | 持久化用户默认偏好 | 省略=Backend 声明的默认源，不在 Protocol 写死“official” |
| maintenance `download_source_ids` | start/retry 显式覆盖并固化本次 operation source intent | 仅 ensure/retry 合法；省略时在开始瞬间快照当前 Settings/Backend default |
| `requested_*` / `effective_*` | durable operation status 回显规范化前后的组件与源 | observe、SSE、receipt、runtime status 必须一致，重启后可恢复 |

`DownloadSourceKind` 是开放 response string。Backend 可新增 kind，但旧前端只应展示已
理解的 kind，并原样保留未知值；不得因为未知 kind 导致整个 health 响应反序列化失败。

JSON Schema 的 `uniqueItems` 只能拒绝重复字符串/完全相同对象，不能证明 source id
跨 kind 唯一或 `(feature_id, accelerator)` 唯一。因此 catalog builder 必须以业务键做
conformance 校验，并在启动/构建测试中 fail closed。

## 4. 模块与稳定接口

### 4.1 OCR registry/resolver 深模块

保留一个 Backend-owned registry，隐藏 RapidOCR、Windows OCR、PaddleOCR adapter 的
导入、语言能力、模型准备与进程隔离差异。对外稳定接口只暴露：

- `catalog() -> OcrEngineCatalog`
- `resolve(engine_id, pipeline, languages) -> OcrEngineAdapter`
- `recognize(request) -> protocol result`

`packages/vibeocr-backend/src/vibeocr/backend/supervisor/app.py` 只做 HTTP adapter；正式
Protocol parser 可用后删除临时 `_extract_engine_selection`，不可长期维护手写协议镜像。

### 4.2 Runtime selection policy 深模块

新增 transport-neutral selection policy，作为 Settings、HTTP maintenance 与 Runtime
Host 三条 adapter 的共同 seam：

- 从 manifest/lock 构建并验证 component/source catalog。
- 规范化 profile、component 与 source intent。
- 区分 `None`（省略）和空 component list（base only）。
- 检查每种 source kind 至多一个，并将 endpoint 解析留在 Backend 内部。
- 输出不可变的 normalized intent，交给 durable operation store 持久化。

推荐落点为 `runtime_selection.py`（纯领域/校验）和已有
`runtime_control.py`/`runtime_installer.py`（编排/执行）。HTTP 路由、Runtime Host CLI
只调用该接口，不能各自重新解释省略、空集和 retry。

### 4.3 Settings 与 operation snapshot

`application/contracts.py`、`settings_facade.py` 和 supervisor `/v2/settings` 负责持久偏好；
`/v2/runtime/maintenance` 在事务开始时读取偏好并写入 operation intent。后续 Settings
更新不得改变正在运行或待 retry 的 operation。

retry 语义：

- selection 字段省略：复用 source operation 的 normalized intent。
- 显式 component/source 字段：重新按当前 catalog 验证并产生新 operation intent。
- cancel 不接受 selection 字段；inspect/repair 不接受 install/source selection。

不引入 catalog revision/hash。相同 Backend release identity、已验证 component lock、
manifest 和持久化 operation intent 已构成足够的一致性边界；Release 改变时由既有
source identity/compatibility gate 处理。

## 5. Runtime profile 与组件模型

| 层级 | 内容 | 交付/安装 |
|---|---|---|
| `base-offline` | Backend/Protocol、RapidOCR+模型、ORT、图片/PDF 基础 runtime、Windows adapter、Runtime Host | 唯一 Backend Release runtime pack；随 Classic/Next Portable 携带，`--no-index --find-links` 禁网安装 |
| `full-cpu` feature variants | PaddleOCR、MinerU/文档结构化 CPU 重依赖 | 用户确认后按当前 Backend release hash lock 在线安装 |
| `full-cu126` feature variants | CUDA/Paddle/Torch 等 GPU 闭包 | 用户确认且硬件预检通过后按 hash lock 在线安装 |

三个概念必须分开：`profile_id` 描述 Runtime/accelerator 档位和基础 ABI；`feature_id`
描述 UI/能力族，例如 `paddleocr`、`mineru`；`component_id` 是安装状态机接收的稳定 id，
依赖闭包由 Backend 解析。

full 组件不得在第一次 OCR 时懒下载；不得随每次前端更新重复下载；失败不得破坏已
可用 base。安装源 endpoint 不写入前端 lock，也不允许前端自行拼 `pip` 命令。

## 6. 分阶段工作包

### B0：发布前证据

- 完成 clean Windows x64、无缓存、禁网 base 安装与 RapidOCR/PDF smoke。
- 完成 Windows OCR 有/无语言包、服务不可用、权限失败的 fail-closed 验证。
- 更新 NOTICE/SPDX SBOM；核对 ORT/OpenCV/PDF 依赖只保留一份批准闭包。

### B1–B4：既有功能收口

- 将临时 engine parser 切换为正式 Protocol parser。
- 对 RapidOCR、Windows、PaddleOCR 补齐 catalog availability/reason/component 映射。
- 保持 PDF 文本层/结构化 pipeline 只通过稳定 OCR adapter seam 访问引擎。

### B5：selection policy 与 catalog

- 实现 component/source catalog builder 及业务键唯一性检查。
- Settings 读写 `download_source_ids`；未知 id/同 kind 多选 fail closed。
- maintenance ensure/retry 接入 optional components 和 source intent；持久化
  requested/effective 字段并贯通 receipt/status/observe/SSE。
- Runtime Host bootstrap/retry 复用同一 policy；显式空 component list保留到执行层。

### B6：在线 full 安装

- 已完成：当前 catalog 的 component/accelerator 闭包由 manifest scope 绑定权威 hash
  lock；`gpu_runtime` 可单独选择，`document_parsing` 在 CUDA 下自动包含 GPU 依赖。
- 已完成：package index 选择真实进入 pip `--index-url`；TUNA 是 release/runtime
  默认，官方 PyPI 为显式候选；lock 不嵌入 index 指令，宿主 pip/uv 配置被隔离。
- 下载、缓存、空间预检、取消、断点后的显式 retry、原子切换与失败回滚复用 durable
  maintenance，不做无依据的自动重试。
- 待完成：model registry 只用于模型资产并建立独立下载 Adapter；后续新增 feature 时
  同步增加精确 scope/lock，不回退为“任一非空选择即完整 profile”。

### B7：正式 Release 与三仓交接

- Release 资产只增加/保留 base pack、runtime manifest、component lock、identity、SBOM
  等声明资产；不得发布 full runtime pack。
- 正式 health fixture 必须声明三项能力及 catalog；下游 component resolver 验证最新
  Backend 兼容、required capabilities 和其绑定 Protocol。
- 向 Classic/Next 交付相同 fixture、失败码、clean-machine 证据与精确验证命令。

## 7. 测试与验收

单元/契约：

- 三引擎选择、非 OCR pipeline、未知/不可用/需准备/语言不可用均返回稳定错误。
- component catalog 的业务键 `feature_id + accelerator` 全局唯一、同 accelerator 内
  `component_id` 唯一（`document_parsing` 合法出现在 cpu 与 nvidia_cuda 两个
  variant，协议仅 MUST 约束业务键，比原文的“component id 全局唯一”更准确）；
  base 必备组件不进入可选目录。
- source id 全局唯一；catalog 允许同 kind 多候选，单次 selection 每 kind 至多一个；
  未知开放 kind 可序列化但不成为默认可选项。
- Settings omission、source selection、同 kind 冲突和未知 id；operation start 时的快照竞态。
- component `None`、`[]`、非空、unknown；ensure/retry 合法，inspect/repair/cancel 非法。
- retry 省略复用旧 intent、显式重选产生新 intent；重启恢复后 requested/effective 不变。

集成/E2E：

- 禁网 clean-machine 安装 base，首次 RapidOCR 与基础 PDF 成功且没有网络调用。
- full CPU/CUDA 未选择时零下载；选择后只下载 normalized closure；取消、下载中断、
  hash-lock 不匹配、空间不足时保持 base 可用。
- Settings 在安装期间改变不影响该 operation，后续 operation 才采用新默认。
- Classic/Next fixture 发出的相同 intent 得到相同 effective component/source 集合。
- `.ci/project.json` 的 quality、E2E、release build、release smoke 全部通过；正式门禁以
  GitHub PR `required` 为权威。

## 8. 完成标准与 PR 边界

- [ ] 正式 Protocol Release 已包含本文修订，Backend 不再使用临时 parser。
- [ ] base-offline 在无网 Windows 上可安装并完成 RapidOCR/PDF smoke。
- [ ] full CPU/CUDA 只通过显式 component selection 在线安装，不是 Release 资产。
- [ ] source/default/snapshot/retry 语义在 Settings、HTTP、Runtime Host 三条 seam 一致。
- [ ] catalog/status 对 Classic 与 Next 提供相同、无产品文案的机器契约。
- [ ] Backend 正式 Release、attestation、SBOM 与下游 E2E 全部通过。

按可独立验证的意图拆 PR：Protocol 正式依赖/parser → registry 收口 → selection policy
与 catalog → online full installer → clean-machine/release。不得在同一 PR 同时改协议语义、
放入未经验证的重依赖并重构 UI；不得通过跳过 E2E、放宽 capability gate 或回退旧
Backend 使门禁变绿。
