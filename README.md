<div align="center">

# VibeOCR Backend

**无 UI、本地优先的 OCR / PDF 运行时与 Supervisor 服务**

[![CI](https://github.com/FelixJI/vibeocr-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/FelixJI/vibeocr-backend/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/FelixJI/vibeocr-backend?display_name=tag)](https://github.com/FelixJI/vibeocr-backend/releases/latest)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Platform](https://img.shields.io/badge/Runtime-Windows%20x64-0078D4?logo=windows)](.ci/project.json)
[![License](https://img.shields.io/github/license/FelixJI/vibeocr-backend)](LICENSE)

[定位](#项目定位) · [架构](#架构) · [开发](#开发与验证) · [源码导读](docs/source-reading-guide.md) · [贡献](CONTRIBUTING.md)

</div>

VibeOCR Backend 是 Classic 与 Next 共用的本地计算组件：它通过 FastAPI/Uvicorn Supervisor 暴露
Protocol v2 API，管理 OCR/PDF job、模型调度、运行时安装与 CPU/CUDA 12.6 profile。

> [!IMPORTANT]
> 本仓库没有桌面 UI。普通用户应从 VibeOCR Classic 或 Next 的 Release 开始；前端通过绑定的
> VibeOCR Protocol 与已验证的 Backend Release 通信。

## 项目定位

Backend 负责：

- 本地 Supervisor 的启动、认证、ready/bootstrap 与健康检查；
- OCR、PDF、二维码等 job 的提交、观察、取消和结果交付；
- 推理 scheduler、executor 与 Paddle/MinerU 等 adapter 的编排；
- CPU/CUDA 运行时 profile 的安装、验证和资产身份绑定；
- 向前端提供稳定的 Protocol v2 HTTP 边界。

Backend 不负责桌面窗口、WebView 或用户工作流编排，这些职责属于 Classic/Next。

## 架构

```mermaid
flowchart LR
    UI["Classic / Next"] -->|"Protocol v2 HTTP"| API["FastAPI Supervisor"]
    API --> Module["SupervisorModule"]
    Module --> Jobs["JobRegistry"]
    Jobs --> Scheduler["Inference Scheduler"]
    Scheduler --> Executor["Executors"]
    Executor --> Paddle["Paddle adapters"]
    Executor --> MinerU["MinerU adapters"]
    API --> Runtime["Runtime installer / profiles"]
```

Supervisor 是进程与协议边界；application/services 负责用例编排；adapter 隔离具体引擎。PocketBase、
桌面 UI 或 Web 前端都不应成为本仓库的数据权威或展示职责。

## 一条真实请求链

提交 job 时，请求从 `supervisor/app.py` 的 `POST /v2/jobs` 进入，随后经过
`SupervisorModule.submit`、`JobRegistry` 与 executor，最终到达 composite/Paddle/MinerU adapter。
进度与结果通过 `/v2/jobs/{job_id}/observe` 获取，控制命令通过 `/v2/jobs/command` 提交。

这也是初学者最值得先读的纵向链，详见 [源码阅读指南](docs/source-reading-guide.md)。

## MinerU 4 配置与运行时边界

Backend 的 MinerU 依赖固定为正式 4.0.2，使用 `mineru.parser.api_server` 的
`/v1/health`、uploads → parse/jobs → files；取消任务使用 `DELETE /v1/parse/jobs/{id}`。
成功文件与失败文件分别交付，partial 不伪装为全批成功。输出下载并转换后清理本次创建的文件；
未确认取消终态时保留输入，交由上游过期机制处理。服务重启或资源不足不会切换 tier。

| 旧设置 | 新行为 |
| --- | --- |
| 缺省、hybrid-engine / medium | basic，保持产品默认 |
| hybrid-engine / high | standard |
| pipeline、vlm-engine、xhigh、关闭公式/表格、多语言 | 返回迁移错误，要求重新选择 |
| 0-based 起止页 | 转为 1-based 页范围；缺省全页，不截取前 10 页 |
| typed mineru 配置 | 保留 flash/basic/standard/advanced、ocr_mode、page_range 和单 language |

非 PDF 仅接受全文件，向上游省略 page_range。language 属服务级设置，不作为逐 job 参数；
不同语言的任务在服务重启、解析和下载期间串行隔离。原生 Office 文档可能由上游使用无模型提取，
不能据此宣称扫描 OCR 或 GPU 加速已验证。

结果正文从原生 middle 语义节点提取，不把 structured Markdown 当纯文本再包装；代码说明和脚注保留。
现有 VibeOCR block 模型将行内字体样式、链接和嵌套列表层级扁平化为可读文本；
原生 structured block 保存在 source 中，表格仍走既有结构化表格合同，不宣称原样保留全部 Office 排版。

新配置保存于 `MINERU_HOME/config.yaml`，也支持显式 `MINERU_CONFIG`；安装器为新版本提供独立
MinerU home，不覆盖旧 `mineru.json` 或旧模型。默认 `model.small_backend: onnx`、
`model.vlm.engine: llama-cpp`；cu126 profile 额外安装 `[torch]`，只有显式选择
`small_backend: torch` 才使用该小模型路径。默认不安装 `[full]` / LMDeploy。
模型继续由 MinerU 原生机制和已选模型源管理，不关闭 TLS 校验。

`ocr.mineru-config.v1` 目录分别报告 tier 的准备状态。包已安装、HTTP health 成功均不足以标记 ready；
当前进程必须实际完成该 tier 的准备解析。准备结果不跨 Supervisor 重启保留；失败不降为 flash。
客户端须先按正式 Protocol 公布的生命周期能力完成准备，再重新读取目录、构造 typed 请求。

Paddle 与 Base/RapidOCR/MinerU 使用不同解释器及 site-packages：Paddle 位于 runtime 的
`engines/paddle`，独立锁固定其 OpenCV contrib 和框架，主环境使用 OpenCV Python。
两个环境在同一未激活安装候选中构建并分别执行 `pip check`，任一失败不替换原有效 runtime；
Paddle 推理经私有子进程调用，主进程不导入 Paddle。完整独立组件选择和安装预览属于后续安装策略。

## 仓库地图

```text
packages/                         # 可发布 Python packages
└── vibeocr-backend/
    └── src/vibeocr/backend/
        ├── supervisor/           # FastAPI、job registry、scheduler、进程入口
        ├── application/          # 用例与 facade
        ├── services/             # OCR/PDF 服务编排
        └── adapters/             # Paddle、MinerU 等实现边界
scripts/
├── bootstrap-ci.ps1              # 锁定依赖与组件输入
├── check-quality.ps1             # 质量入口
└── automation.py                 # CI/发布稳定接口
tests/                            # 单元、协议、安装器与 smoke 测试
.ci/project.json                  # profile、门禁、构建与发布契约
```

实际 package 布局可能随模块拆分演进；定位入口时以 `pyproject.toml` 的 scripts 与 `rg --files` 为准。

## CLI 入口

| 命令 | Python 入口 | 用途 |
| --- | --- | --- |
| `vibeocr-supervisor` | `vibeocr.backend.supervisor.main:main` | 启动本地 Supervisor |
| `vibeocr-runtime-installer` | `vibeocr.backend.runtime_installer:main` | 安装与验证运行时 profile |

Supervisor 由前端按组件锁和 handshake 管理。除调试外，不建议绕过前端手工拼接启动参数。

## 开发与验证

需要 Windows、[uv](https://docs.astral.sh/uv/) 和仓库锁定的 Python：

```powershell
git clone https://github.com/FelixJI/vibeocr-backend.git
cd vibeocr-backend
uv venv --seed .venv
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
$env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
uv run --no-sync powershell -NoProfile -File scripts/bootstrap-ci.ps1
uv run --no-sync powershell -NoProfile -File scripts/check-quality.ps1
```

完整 CI 还会执行 Protocol conformance、release build、manifest 与 installer smoke。精确命令及 profile
输入以 [`.ci/project.json`](.ci/project.json) 为准。

> [!NOTE]
> 真实模型、CUDA、安装器与 frozen package 验证会下载或构建较大资产。普通逻辑修改先运行相邻单元测试；
> 只有涉及 profile、模型 adapter、打包或安装边界时才需要对应重型 smoke。

## Runtime 与 Protocol 边界

- Protocol capability 决定前端可用功能，不以版本号猜测行为。
- ready 表示 Supervisor 协议边界可用，不等于所有模型已经加载。
- 运行时 profile、manifest 与组件 identity 由发布自动化生成并验证。
- 本地开发不能通过 editable/path dependency 绕过已发布 Protocol/Backend 组件关系。

## 发布

正式 Release 由 CI/CD 生成并绑定源码 SHA、组件 identity、精确资产集合、SHA-256 与 SPDX SBOM。
版本只能通过仓库自动化更新；不要直接修改派生版本、tag 或 Release 资产。

## 参与贡献

先阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md) 与 [源码阅读指南](docs/source-reading-guide.md)。行为变化应在
相邻测试中覆盖成功与实际相关的失败/取消路径；提交使用 Conventional Commit。

## 许可证

本项目基于 [LICENSE](LICENSE) 中的条款发布。


## 在线依赖解析的超时诊断

解析阶段保留 30 分钟总上限，并在连续 5 分钟没有新的包状态或字节进展时停止。
网络连接/读取超时为 30 秒，连接重试和中断下载续传各最多 2 次；维护心跳只表示进程仍在运行，
不会重置无进展时限。持续进展允许超过空闲窗口，但仍受总时限约束。

`runtime.resolve_packages` 的活动事件和错误保留操作标识、来源 id、耗时、距最后活动的时间及脱敏尾部。
`total_timeout` 表示阶段总时限，`idle_timeout` 表示无有效进展，`exit_nonzero` 的尾部区分 pip 报告的网络错误或解析冲突。
取消和失败只清理该次操作的子进程树，不替换先前验证可用的运行环境；旧服务恢复不代表本次安装成功。

pip 的 `--dry-run --report` 也可能传输 wheel；本地受控 HTTP 回归已经证明这一点，但不能据此推断用户现场发生了重复下载。
未取得现场完整版本与日志时，应保留“当前实现可复现、原现场原因待确认”的边界。
