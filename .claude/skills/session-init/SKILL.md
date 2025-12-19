# Session Init

Initialize a new session with context loading and RAG health check.

## Description
Session Init - Initialize Session + RAG Health Check (project)

## Usage
/session-init

## Prompt

You are initializing a new development session for causal_inference_mastery.

### Step 1: RAG Services Health Check

Check that all context engineering services are running:

```bash
# Check RAG daemon
if [ -S /tmp/rag_daemon_${USER}.sock ]; then
    echo "RAG daemon:       OK /tmp/rag_daemon_${USER}.sock"
else
    echo "RAG daemon:       MISSING - run: systemctl --user start rag-daemon"
fi

# Check research-kb daemon
if [ -S /tmp/research_kb_daemon.sock ]; then
    echo "research-kb:      OK /tmp/research_kb_daemon.sock"
else
    echo "research-kb:      MISSING - run: systemctl --user start research-kb-daemon"
fi

# Check ProactiveContext (via lever_of_archimedes)
if pgrep -f "proactive_context" > /dev/null; then
    echo "ProactiveContext: OK"
else
    echo "ProactiveContext: NOT RUNNING (optional)"
fi
```

Report status in format:
```
=== RAG Services Status ===
RAG daemon:       [OK/MISSING]
research-kb:      [OK/MISSING]
ProactiveContext: [OK/NOT RUNNING]
```

### Step 2: Load Session Context

1. Read `CURRENT_WORK.md` for current session status
2. Identify:
   - Current session number
   - What was done last session
   - What's planned for this session
   - Any blockers or open issues

### Step 3: Check Git Status

```bash
git status --short
git log --oneline -5
```

Report:
- Uncommitted changes?
- Current branch
- Recent commits

### Step 4: Verify Test Suite Health

Quick smoke test:
```bash
# Python (fast subset)
pytest tests/test_rct/test_simple_ate.py -v -x --tb=short

# Julia (quick check)
julia --project -e "using CausalEstimators; println(\"Julia OK\")"
```

### Step 5: Session Summary

Provide a concise summary:
```
=== Session Initialized ===

Session: {N}
Status: {from CURRENT_WORK.md}
Branch: {current_branch}
Uncommitted: {yes/no}

RAG Services: {X}/3 available
Test Suite: {healthy/issues}

Ready to work on: {next task from CURRENT_WORK.md}
```

### Step 6: Recommendations

Based on context, suggest:
1. Files likely to be relevant
2. Tests to run after changes
3. Documentation to update

## Output Format
```
=== RAG Services Status ===
RAG daemon:       [status]
research-kb:      [status]
ProactiveContext: [status]

=== Session Context ===
Session: N
Last: {previous work}
Next: {planned work}

=== Git Status ===
Branch: {branch}
Uncommitted: {count} files
Recent: {last commit}

=== Test Health ===
Python: [OK/ISSUE]
Julia: [OK/ISSUE]

=== Ready ===
Focus: {main task}
Files: {likely files}
Tests: {relevant tests}
```
