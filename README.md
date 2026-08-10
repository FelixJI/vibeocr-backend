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
