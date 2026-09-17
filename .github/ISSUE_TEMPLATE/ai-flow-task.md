---
name: AI Flow v4.0 Task
about: 必须显式填写 L/R、建议实施者与执行模式
title: "[Task][AI Flow v4.0] "
labels: ""
assignees: ""
---

## Goal / Background

上层 Goal、问题和现有代码/Issue/PR 证据：

## AI Flow Routing

- Difficulty: L? — 理由。
- Risk: R? — 理由。
- Recommended implementer: Codex / GLM / Codex-first → GLM
- Execution mode: 具体调查/施工/人工交接/验证安排。
- Routing rationale:
- Technical owner: Codex
- Review: fresh read-only Codex reviewer subagent
- Merge: human-only

Codex 必须根据实际工作区/代码/PR 校准，调整理由写回。

## Scope / Out of Scope

## Acceptance Criteria

- [ ] Given/When/Then 及可观察证据。
- [ ] 重要失败、取消/超时或恢复回归。
- [ ] 兼容与必须保留的不变量。

## Baseline / Dependencies

仓库/分支/base SHA（未核实写 UNVERIFIED）：
依赖 Issue/PR/正式发布、是否真正满足、重叠改动：

## Validation / Compatibility / Rollback

真实命令及 cwd、必要环境、失败证据；需要单独授权的操作：

## Control Pointer

status: PLANNED
current_actor:
next_actor:
next_action:
user_action_required:
handoff_id: none

GLM 完成/受阻主动给返回 Codex 的一句话；同一主体不要 self-handoff。审阅子代理由 Codex 内部启动。
