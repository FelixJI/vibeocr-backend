# AI Coding Playbook v4.0 — VibeOCR 仓库适配

## 使用入口与事实源

根 AGENTS.md 保留全部项目工程约束；.ai-flow/AGENTS.md 定义主体边界；project.json 保存长期设置。Issue/PR 是任务/结果单一事实源。本文件是长期工作规则，不是任何目标的实施计划。此安装是 v4.0 规则适配，不是原压缩包全量复制：不包含分散提示词、迁移器、安装报告或额外自动化脚本；必需交接格式直接写在下文。

## 规划与校准

网页先读现有代码、规则、Issues 和未合并 PR，复用同一目标的未完成任务，避免重复建单。每个 Task 给出范围/非目标、可观察 AC、依赖、风险、验证及回滚。使用 .github/ISSUE_TEMPLATE 中的 Goal/Task 模板。

Difficulty：L1 局部机械修改；L2 边界清晰的局部行为；L3 跨模块、需先定设计；L4 跨进程/状态机/复杂未知根因；L5 系统性高不确定工作。Risk：R0 文档；R1 可逆局部行为；R2 核心流程、兼容或持久状态；R3 安全权限、供应链信任、破坏性迁移或发布治理。两轴不要混用。

Execution mode 必须具体，例如 Codex investigation + implementation、Codex design → manual GLM implementation → Codex verification，或 GLM bounded implementation → Codex verification；不是只写 balanced。Codex 重新校准并记录原因，L1/L2 默认 GLM，L3 Codex-first，L4/L5 Codex。

## 授权范围与状态

PLANNED → READY → CODEX_WORKING，或 HANDOFF_READY → GLM_WORKING → GLM_DONE → CODEX_CHECKING → REVIEW_READY → REVIEWING → MERGE_READY → MERGED → ACCEPTED。

审阅整改用 CHANGES_REQUIRED / FIXING；真实阻碍用 WAITING_DEPENDENCY、WAITING_CI、REVIEW_BLOCKED、HUMAN_REQUIRED。这些状态不是自动调度器。每次写回 current_actor、next_actor、next_action、user_action_required、handoff_id（无交接为 none）。状态以实际动作和证据为准；同一主体同一授权任务跨内部 checkpoint 直接继续。

## 人工交接评论

Codex 向 GLM 的评论必须包含：Task 链接与唯一 handoff_id；明确实现范围和禁止修改项；仓库、分支、base/head、工作区写入所有权；技术决定与证据；输入/输出、AC、精确验证命令；已有/未做内容；失败上限、停止条件、返回 Codex 的动作。

只有确需跨工具时，Codex 给用户三句已代入真实编号的话：

- 施工：按本仓 AI Flow v4.0 执行 <Task URL> 的 <handoff_id>，只做交接范围，完成或受阻写回证据并给出返回 Codex 的一句话。
- 完工返回：按本仓 AI Flow v4.0 核验 <Task URL> 的 <handoff_id> 完工结果与 <PR URL>，继续技术验收及新的只读 reviewer 子代理审阅，到真实边界再停。
- 阻碍返回：按本仓 AI Flow v4.0 接管 <Task URL> 的 <handoff_id> 阻碍，先读失败证据并校准范围/实施者，不重复无效重试。

GLM 返回记录 changed files、真实 head、验证命令/结果、未完成项/风险、是否停止写入；不能自称 ACCEPTED。用户尚未粘贴启动时 next_actor 只是接收者，不能写 GLM_WORKING 或 Codex 已恢复。

## 审阅与合并

所有变更进入新的只读 Codex reviewer 子代理；独立核对 Issue AC、真实 PR diff、最新 base/head、代码、测试和副作用。写 reviewed_base_sha、reviewed_head_sha、结论与逐项阻碍；不用主执行者总结代替读取。审阅者不施工。新增提交后重新审阅，缺独立能力则 REVIEW_BLOCKED。

PASS 后核对同步 main 的 required、review conversations、范围及无意外版本/发布变更，才 MERGE_READY 并交用户。不能 Agent 自动合并。用户合并后核实实际 squash merge、main CI/CD 哨兵；发版单另循原工程流程。未完成真实安装/硬件验证不得以 CI 单元测试全绿替代 ACCEPTED。

## VibeOCR 跨仓约束

Protocol 是公开协议事实源；Backend 是依赖图、安装/检查/维修和运行时实现的事实源；Classic/Next 是独立桌面客户端。源码阅读可跨仓，构建/验收输入不得使用未发布邻仓源码或 editable 依赖。

协议扩展先证明既有能力不足，保持 v2 同 major 的可选响应和 capability 协商；Protocol 正式输入由 Backend lock 显式消费；两个前端依现有 CI 消费最新正式 Backend 及绑定 Protocol。不自动级联发版、不强制四仓同版本。不把恢复当前有效本地运行时混同于绕过最新正式组件 fail-closed 策略。

## 验证与卫生

以 .ci/project.json 和现有脚本为命令权威，先相关测试再完整 required 门禁。Python 使用 uv 管理的仓库环境；不改工具链/源来让本机通过，不手改锁或生成物。保留成功、失败、取消和超时中与风险直接相关的测试，不机械枚举所有组合。

提交前执行 git diff --cached --check，并逐项检查 git diff --cached --name-status；不得暂存敏感数据、机器配置、运行日志、下载资产、临时报告或本次计划文件。推送前检查真实提交范围与已有 hooks；此适配不增加新的脚本或工作流。缺 Windows/GPU/正式发布资产时写清阻塞和恢复条件，不虚报已验证。
