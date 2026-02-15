# Multi-Agent Git Workflow

## name: multi-agent-git-workflow

---

## description:

"面向多 AI agent 的 Git 协作协议。核心：同步优先，安全补偿，冲突暴露。适用于并发开发场景。"

---

## 核心原则

|原则|含义|违反后果|
|---|---|---|
|**同步优先**|每次操作前必须 pull|基于过时历史开发|
|**只能追加**|已 push 的历史禁止改写|破坏其他 agent 的工作|
|**冲突暴露**|发现冲突立即暂停并报告|数据丢失或覆盖|
|**原子提交**|一个任务一个 commit|回滚粒度过大|

---

# 标准工作流程（每个任务必须执行）

## Phase 1: 同步检查（SYNC）

```bash
# 1. 检查工作区状态
git status

# 2. 如果有未提交修改且不属于本次任务
→ 暂停，提示用户处理（stash 或 commit）

# 3. 拉取最新代码
git pull --rebase

# 4. 如果有冲突
→ 暂停，输出冲突文件列表，等待人工解决
```

**⚠️ CRITICAL**：没有执行完 Phase 1，不得进入 Phase 2。

---

## Phase 2: 分支创建（BRANCH）

### 分支创建规则

```bash
# ✅ 正确：基于当前最新分支
git checkout -b feat/task-name-{timestamp}

# ❌ 错误：未 pull 就创建分支
# ❌ 错误：每次都从 main 创建（除非明确要求）
```

### 分支命名格式

```
{type}/{task}-{agent-id}-{timestamp}
```

**示例**：

- `feat/user-login-agent1-20250214`
- `fix/api-timeout-agent2-1423`

**多 agent 特殊约定**：

- 包含 `agent-id` 避免分支名冲突
- 时间戳使用 `MMDDHHmm` 或递增序号

---

## Phase 3: 开发提交（COMMIT）

### Commit 规范

```
{type}({scope}): {description}

[agent-id] {optional details}
```

**示例**：

```
feat(auth): add login endpoint

[agent-1] implemented JWT-based authentication
```

### 提交时机

- 每个子任务完成后立即 commit
- 提交前运行必要的测试（如有）
- **禁止**积累多个不相关修改

---

## Phase 4: 推送同步（PUSH）

```bash
# 1. 推送前再次拉取（防止并发冲突）
git pull --rebase

# 2. 如果有冲突
→ 暂停，输出冲突详情

# 3. 推送
git push origin {branch-name}

# 4. 如果推送失败（被其他 agent 抢先）
→ 重新执行 pull --rebase，解决冲突后再推
```

**并发保护**：

- 使用 `--rebase` 保持线性历史
- 推送失败不要惊慌，这是正常的并发现象

---

## Phase 5: 完成报告（REPORT）

```markdown
✅ Task Completed

**Branch**: feat/user-login-agent1-20250214
**Commits**: 3
**Base**: main (commit: abc1234)
**Status**: Pushed to remote

**Files Changed**:
- src/auth/login.py (+45, -0)
- tests/test_login.py (+30, -0)

**Next Steps**:
人工审查后决定是否合并到 main。

**Review Checklist**:
- [ ] Run tests: pytest tests/
- [ ] Check diff: git diff main..{branch}
- [ ] Verify no conflicts with other agents' work
```

---

# 冲突处理协议

## 冲突类型判断

|场景|表现|Agent 行为|
|---|---|---|
|**pull 冲突**|`git pull` 报错|暂停，列出冲突文件|
|**push 冲突**|远端有新 commit|执行 `pull --rebase`，重新检查|
|**rebase 冲突**|rebase 过程中停止|暂停，输出当前 commit 和冲突文件|

## Agent 的标准输出

```markdown
⚠️ CONFLICT DETECTED

**Type**: Pull conflict
**Files**:
- src/database.py (CONFLICT)
- src/utils.py (CONFLICT)

**Current State**:
- Branch: feat/add-cache-agent2-0214
- Uncommitted changes: Yes
- Remote commits ahead: 2

**Required Action**:
Human must resolve conflicts manually.

**Commands to resolve**:
1. git status  # Check conflict markers
2. Edit conflicting files
3. git add {resolved-files}
4. git rebase --continue  # If during rebase
   OR
   git commit  # If during merge
```

**禁止 AI 做什么**：

- ❌ 自动选择 `--ours` 或 `--theirs`
- ❌ 尝试智能合并冲突
- ❌ 删除冲突标记后直接提交

---

# 历史操作决策（简化版）

## 快速判断矩阵

|是否已 push?|分支类型|允许操作|禁止操作|
|---|---|---|---|
|❌ No|-|reset, commit --amend, rebase -i|-|
|✅ Yes|私有分支|revert, 新 commit 修复|reset, rebase, --force|
|✅ Yes|共享分支|revert, cherry-pick|reset, rebase, --force, --force-with-lease|

## 核心规则

```
IF commit 已 push:
    → 只能通过新 commit 补偿（revert/新修复 commit）
ELSE:
    → 可以改写本地历史（reset/amend/rebase）
```

## 常见场景速查

|需求|是否 push|命令|
|---|---|---|
|修改上个 commit message|❌|`git commit --amend`|
|撤销上个 commit|❌|`git reset --soft HEAD^`|
|撤销已 push 的 commit|✅|`git revert HEAD`|
|把 commit 移到正确分支|✅|`git cherry-pick <hash>`|

---

# 多 Agent 并发场景

## Scenario 1: 同时修改不同文件

```
Agent A: 修改 auth.py → commit → push ✅
Agent B: 修改 db.py   → commit → push ✅
```

**结果**：自动合并成功

---

## Scenario 2: 修改同一文件不同区域

```
Agent A: 修改 config.py L1-10   → push ✅
Agent B: 修改 config.py L50-60  → pull --rebase → push ✅
```

**结果**：Git 自动合并

---

## Scenario 3: 修改同一文件相同区域（冲突）

```
Agent A: 修改 config.py L20 → push ✅
Agent B: 修改 config.py L20 → pull --rebase → ⚠️ CONFLICT
```

**Agent B 必须**：

1. 暂停操作
2. 输出冲突报告
3. 等待人工解决

---

## Scenario 4: 基于过时分支开发

```
Agent A: 在 main(commit 100) → 创建 feat-A
Agent B: 在 main(commit 100) → 创建 feat-B
         main 被更新到 commit 105
Agent A: feat-A 开发完成 → 合并到 main(105) ✅
Agent B: feat-B 开发完成 → 尝试合并 → ⚠️ CONFLICT
```

**防范措施**：

- 每次开始新任务前 `git pull`
- Push 前再次 `git pull --rebase`

---

# Agent 执行边界

## ✅ Agent 必须做

- [ ] 开始任务前执行完整的 Phase 1（同步检查）
- [ ] 每个子任务完成后立即 commit
- [ ] Push 前再次 pull --rebase
- [ ] 遇到冲突立即暂停并报告
- [ ] 完成后输出标准报告

## ❌ Agent 禁止做

- [ ] 跳过 `git pull`
- [ ] 在共享分支使用 `reset/rebase/--force`
- [ ] 自动解决冲突（选择 ours/theirs）
- [ ] 执行 merge 到 main/master
- [ ] 删除或覆盖其他 agent 的 commit

## ⚠️ 需要人工介入的场景

|场景|Agent 行为|
|---|---|
|任何形式的冲突|暂停，输出详细报告|
|工作区有未提交的非本任务修改|暂停，提示处理方式|
|Push 被拒绝（远端有新 commit）|执行 pull --rebase，如遇冲突则暂停|
|需要合并到主分支|输出审查建议，等待人工操作|
|无法确定分支是否被其他 agent 使用|暂停，询问用户|

---

# 检查清单（每次任务）

### 开始前

- [ ] `git status` 确认工作区干净
- [ ] `git pull --rebase` 同步最新代码
- [ ] 创建符合命名规范的分支

### 开发中

- [ ] 每个子任务一个 commit
- [ ] Commit message 包含 agent-id
- [ ] 不修改其他 agent 负责的文件（如可避免）

### 推送前

- [ ] 再次 `git pull --rebase`
- [ ] 确认无冲突
- [ ] Push 到远程

### 完成后

- [ ] 输出完整报告
- [ ] 未执行任何 merge 操作
- [ ] 遇到的冲突已全部报告

---

# 与单 Agent Workflow 的差异

|项目|单 Agent|多 Agent|
|---|---|---|
|同步频率|可选|**强制**（每次任务前后）|
|分支命名|可省略时间戳|**必须**包含 agent-id|
|冲突处理|较少见|**高频**，需完善协议|
|Push 策略|直接 push|**先 pull --rebase**|
|历史改写|私有分支可谨慎使用|**禁止**（即使私有）|
|Commit message|简洁即可|建议包含 agent-id|

---

# 故障恢复手册

## 问题 1: Push 失败

```bash
# 报错
! [rejected] feat/xxx -> feat/xxx (non-fast-forward)

# 解决
git pull --rebase
# 如无冲突
git push
# 如有冲突 → 暂停，报告人工
```

## 问题 2: 意外修改了已 push 的 commit

```bash
# ❌ 已执行：git commit --amend（在已 push 的 commit 上）

# ✅ 补救：创建新 commit
git reset --soft HEAD^  # 撤销 amend
git commit -m "fix: correct implementation"  # 新 commit
git push
```

## 问题 3: 多个 agent 创建了同名分支

```bash
# Agent A: feat/login
# Agent B: feat/login ← 推送失败

# Agent B 改名
git branch -m feat/login feat/login-agent2-0214
git push origin feat/login-agent2-0214
```

---

# 术语表

|术语|含义|
|---|---|
|**同步点**|执行 `git pull` 的时机|
|**私有分支**|仅一个 agent 使用的分支|
|**共享分支**|main/develop 或多 agent 协作的分支|
|**冲突暴露**|不自动解决，立即报告人工|
|**原子 commit**|一个 commit 只做一件事|
|**线性历史**|通过 rebase 保持无分叉的提交线|

---

## 版本

- **v1.0** - 初始版本（多 agent 协作优化）
- 上次更新：2025-02-14