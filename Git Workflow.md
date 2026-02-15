# Agentic Git-Worktree Collaboration Protocol (AGWP-v2)

## 1. System Topology & Definitions

### 1.1 Entities

- **Orchestrator ($O$):** System supervisor. Manages `main` branch, task dispatch, `worktree` lifecycle, and final integration.
    
- **Sub-Agent ($S_i$):** Task executor. Operates exclusively within an assigned `worktree`.
    

### 1.2 File System Structure

The repository strictly adheres to the following hierarchy:

Plaintext

```
${ROOT}/
├── main/                       # [Restricted] Orchestrator's workspace (HEAD -> main)
│   ├── .orchestrator.lock      # Global state registry (JSON)
│   └── src/                    # Production code
└── worktrees/                  # [Container] Isolated environments
    ├── task-${ID}-${AGENT}/    # [Sandboxed] Workspace for Agent S_i
    │   ├── .agent.meta         # Local task context (JSON)
    │   ├── .task.complete      # Handover signal (JSON)
    │   └── src/                # Development code (HEAD -> task/branch)
```

### 1.3 Data Schemas

Agents must parse and generate metadata strictly according to these schemas.

**Global Lock (`.orchestrator.lock`)**:

JSON

```
{
  "active_tasks": [{"id": "String", "agent_id": "String", "branch": "String", "status": "WIP|REVIEW"}],
  "last_sync": "ISO8601_Timestamp"
}
```

**Local Context (`.agent.meta`)**:

JSON

```
{
  "task_id": "String", "agent_id": "String", "target_branch": "String", "base_commit_hash": "String"
}
```

**Completion Signal (`.task.complete`)**:

JSON

```
{
  "task_id": "String", "commits_added": "Integer", "test_status": "PASS|FAIL", "ready_for_merge": true
}
```

---

## 2. Access Control List (ACL)

|**Action**|**Orchestrator (O)**|**Sub-Agent (Si​)**|
|---|---|---|
|**Branch: `main`**|**Read / Write / Merge**|**Read Only** (Must `fetch`/`rebase`)|
|**Branch: `task/*`**|Create / Delete|Read / Write / Push|
|**Dir: `worktrees/`**|Create / Prune|**Restricted** (Only inside assigned sub-dir)|
|**Command: `git push`**|`origin main`|`origin task/*` (**Strictly Forbidden**: `main`)|
|**Conflict Resolution**|Decision Maker|Reporter (Must pause & report)|

---

## 3. Finite State Machine (Workflow)

### Phase I: Dispatch (Orchestrator Action)

**Pre-condition:** `main` is clean and up-to-date.

1. **Branching:** Create branch `task/${TASK_ID}-${FEATURE}` from `main`.
    
2. **Isolation:** Execute `git worktree add worktrees/${DIR_NAME} ${BRANCH}`.
    
3. **Injection:** Write `.agent.meta` into the new worktree.
    
4. **Signal:** Invoke Sub-Agent $S_i$ with worktree path.
    

### Phase II: Development Loop (Sub-Agent Action)

**Context:** Inside `worktrees/${DIR_NAME}`.

1. **Init:** Read `.agent.meta`. Verify `HEAD` matches `target_branch`.
    
2. **Sync Check:** `git fetch origin`. IF `origin/main` > `HEAD`: execute **Rebase Protocol**.
    
3. **Execution:** Modify code. Create **Atomic Commits**.
    
4. **Push:** `git push origin ${BRANCH}` periodically.
    

### Phase III: Handover (Sub-Agent Action)

**Trigger:** Task Implementation & Local Tests passed.

1. **Final Rebase:** Mandatory `git pull --rebase origin main`.
    
    - _Exception:_ IF conflict occurs -> **Halt & Report**.
        
2. **Final Push:** `git push --force-with-lease origin ${BRANCH}`.
    
3. **Signaling:** Generate `.task.complete` file.
    
4. **Exit:** Terminate process. Return control to $O$.
    

### Phase IV: Integration (Orchestrator Action)

**Trigger:** Detection of `.task.complete` or Agent Report.

1. **Verification:** Enter worktree. Validate logic/tests. Check `git log main..HEAD`.
    
2. **Dry Run:** Attempt `git merge --no-commit --no-ff ${BRANCH}` in `main`.
    
    - _IF Conflict:_ Abort merge. Instruct $S_i$ to resolve (Go to Phase II).
        
3. **Commit:** Execute merge. Push `main`.
    
4. **Cleanup:** `git worktree remove ${DIR_NAME}`. Delete local branch.
    

---

## 4. Exception Handling Protocols

### Protocol: Rebase Conflict (Agent Side)

**Condition:** `git rebase origin/main` fails.

**Action:**

1. **DO NOT** attempt blind resolution (e.g., `--ours`/`--theirs`).
    
2. **Abort:** `git rebase --abort`.
    
3. **Report:** Return standardized error:
    
    Markdown
    
    ```
    [CONFLICT_DETECTED]
    File: <filepath>
    Base: <commit_hash>
    Incoming: <commit_hash>
    Status: PAUSED_FOR_HUMAN_OR_ORCHESTRATOR_INTERVENTION
    ```
    

### Protocol: Dirty State Recovery (Orchestrator Side)

**Condition:** Agent $S_i$ crashed or worktree is corrupted.

**Action:**

1. `git worktree prune`.
    
2. `git branch -D ${BRANCH}` (if task abandoned) OR `git worktree add ...` (to respawn).
    

---

## 5. Critical Instructions for LLM

1. **Self-Correction:** Before any `git` command, verify current directory (`pwd`) matches your Role scope.
    
2. **No Hallucination:** Do not invent git commands. Use standard `worktree` syntax.
    
3. **Atomic Thinking:** Treat the "Main Repo" and "Worktree" as physically separate machines sharing a `.git` database.
    
4. **Explicit Handoff:** Never leave a task "hanging". Must result in either `.task.complete` or `[ERROR_REPORT]`.