# SessionOptimizer Quick Start Guides

5-minute guides for each Team Brain agent to start using SessionOptimizer.

---

## Table of Contents

1. [FORGE Quick Start](#forge-quick-start-orchestrator)
2. [ATLAS Quick Start](#atlas-quick-start-executor)
3. [BOLT Quick Start](#bolt-quick-start-free-executor)
4. [CLIO Quick Start](#clio-quick-start-ubuntu-cli)
5. [NEXUS Quick Start](#nexus-quick-start-ubuntu-cli)
6. [Universal Commands](#universal-commands)

---

## FORGE Quick Start (Orchestrator)

### Your Role
As the Team Brain orchestrator, you plan tasks and write specifications. SessionOptimizer helps you track if your planning sessions are efficient.

### 5-Minute Setup

```python
from sessionoptimizer import start_session, log_event, end_session, analyze_session, EventType

# 1. Start every planning session
session = start_session("FORGE", "Plan feature: user authentication")

# 2. Log key events during your session
log_event(session.id, EventType.CONTEXT_LOAD, 
          data={"source": "memory"}, tokens_input=1000)

log_event(session.id, EventType.THINKING, 
          data={"topic": "architecture"}, tokens_output=500)

log_event(session.id, EventType.AI_RESPONSE, 
          data={"type": "specification"}, tokens_output=2000)

# 3. End session and check efficiency
end_session(session.id)
report = analyze_session(session.id)

print(f"Planning efficiency: {report.efficiency_score:.1f}%")
if report.issues:
    print(f"Issues to address: {len(report.issues)}")
```

### What to Track

| Event | When to Log | Why |
|-------|-------------|-----|
| CONTEXT_LOAD | Loading memory/context | Track context overhead |
| THINKING | Extended reasoning | Monitor thinking/output ratio |
| AI_RESPONSE | Final specifications | Track spec verbosity |
| FILE_READ | Reading requirements | Detect repeated reads |

### Key Metrics for FORGE

- **Thinking ratio**: Should be <40% of total output
- **Repeated reads**: Avoid reading same file multiple times
- **Session duration**: Keep planning sessions under 60 minutes

### Quick Check

```python
# At end of each session
if report.efficiency_score < 80:
    print("Consider: Are your specs too verbose?")
    for issue in report.issues:
        print(f"  - {issue.suggestion}")
```

---

## ATLAS Quick Start (Executor)

### Your Role
You build tools and execute complex tasks. SessionOptimizer helps you identify inefficiencies in your development sessions.

### 5-Minute Setup

```python
from sessionoptimizer import start_session, log_event, end_session, analyze_session, EventType

# 1. Start every build session
session = start_session("ATLAS", "Build SessionOptimizer", "sonnet")

# 2. Log file operations
log_event(session.id, EventType.FILE_READ, 
          data={"file_path": "roadmap.md", "compressed": True},
          tokens_input=500)

# 3. Log searches
log_event(session.id, EventType.SEARCH, 
          data={"query": "session tracking patterns"},
          tokens_input=100, tokens_output=600)

# 4. Log code generation
log_event(session.id, EventType.AI_RESPONSE, 
          data={"type": "code"}, tokens_output=1500)

# 5. Log file writes
log_event(session.id, EventType.FILE_WRITE, 
          data={"file_path": "sessionoptimizer.py"},
          tokens_input=50)

# 6. Check efficiency at end
end_session(session.id)
report = analyze_session(session.id)

print(f"Build efficiency: {report.efficiency_score:.1f}%")
```

### What to Track

| Event | When to Log | Why |
|-------|-------------|-----|
| FILE_READ | Reading any file | Detect repeated reads |
| FILE_WRITE | Creating/updating files | Track output |
| SEARCH | Codebase searches | Detect repeated searches |
| AI_RESPONSE | Code generation | Track output tokens |
| ERROR | Test failures | Identify retry patterns |

### Key Metrics for ATLAS

- **Repeated file reads**: Cache files you read multiple times
- **Large files**: Mark `compressed: True` when using ContextCompressor
- **Error rate**: Keep below 20%, use ErrorRecovery for failures

### Integration with ContextCompressor

```python
# When reading large files, ALWAYS use ContextCompressor
log_event(session.id, EventType.FILE_READ,
          data={
              "file_path": "huge_file.json",
              "compressed": True,  # Important! Marks as compressed
              "original_tokens": 15000,
              "compressed_tokens": 3000
          },
          tokens_input=3000)  # Log compressed count
```

---

## BOLT Quick Start (FREE Executor)

### Your Role
You execute simple tasks using free Grok tokens. SessionOptimizer still tracks your efficiency (even though cost is $0).

### 5-Minute Setup

```python
from sessionoptimizer import start_session, log_event, end_session, analyze_session, EventType

# 1. Start session (note: model="grok" for free)
session = start_session("BOLT", "Execute file rename task", "grok")

# 2. Log tool calls
log_event(session.id, EventType.TOOL_CALL, 
          data={"tool": "file_rename", "files": 5},
          tokens_input=100)

# 3. Log any errors
log_event(session.id, EventType.ERROR, 
          data={"message": "Permission denied", "recoverable": True},
          tokens_input=50)

# 4. Log completion
log_event(session.id, EventType.AI_RESPONSE, 
          data={"type": "completion_report"}, tokens_output=200)

# 5. End and analyze
end_session(session.id)
report = analyze_session(session.id)

# Cost will be $0 for Grok, but efficiency still matters!
print(f"Task efficiency: {report.efficiency_score:.1f}%")
```

### What to Track

| Event | When to Log | Why |
|-------|-------------|-----|
| TOOL_CALL | Using any tool | Track tool usage |
| ERROR | Any failure | Identify error patterns |
| AI_RESPONSE | Task results | Track output |

### Key Metrics for BOLT

- **Error rate**: Keep below 20%
- **Retry patterns**: Use ErrorRecovery for repeated failures
- **Session time**: Even free sessions should be efficient

### Why Track Free Sessions?

1. **Time efficiency** - Inefficient sessions waste time
2. **Error patterns** - Identify tasks that should go to paid agents
3. **Team metrics** - Compare BOLT efficiency with paid agents

---

## CLIO Quick Start (Ubuntu CLI)

### Your Role
You handle CLI operations and deployments on Ubuntu. Track file operations and deployment efficiency.

### 5-Minute Setup

```python
from sessionoptimizer import start_session, log_event, end_session, analyze_session, EventType

# 1. Start deployment session
session = start_session("CLIO", "Deploy BCH update", "sonnet")

# 2. Log file operations
log_event(session.id, EventType.FILE_READ, 
          data={"file_path": "/etc/nginx/nginx.conf"},
          tokens_input=300)

log_event(session.id, EventType.FILE_WRITE, 
          data={"file_path": "/var/www/bch/config.json"},
          tokens_input=100)

# 3. Log commands executed
log_event(session.id, EventType.TOOL_CALL, 
          data={"tool": "systemctl", "command": "restart nginx"},
          tokens_input=50)

# 4. Log any errors
log_event(session.id, EventType.ERROR, 
          data={"message": "Service failed to start"},
          tokens_input=80)

# 5. End and review
end_session(session.id)
report = analyze_session(session.id)
print(f"Deployment efficiency: {report.efficiency_score:.1f}%")
```

### What to Track

| Event | When to Log | Why |
|-------|-------------|-----|
| FILE_READ | Reading config files | Detect repeated reads |
| FILE_WRITE | Updating configs | Track changes |
| TOOL_CALL | System commands | Monitor operations |
| ERROR | Any failures | Track deployment issues |

### Key Metrics for CLIO

- **Config reads**: Don't read the same config multiple times
- **Error rate**: Deployment errors are costly
- **Session time**: Deployments should be quick

---

## NEXUS Quick Start (Ubuntu CLI)

### Your Role
You handle analysis, reporting, and long-running tasks. Track analysis efficiency.

### 5-Minute Setup

```python
from sessionoptimizer import start_session, log_event, end_session, analyze_session, EventType

# 1. Start analysis session
session = start_session("NEXUS", "Analyze weekly logs", "sonnet")

# 2. Log data loads
log_event(session.id, EventType.FILE_READ, 
          data={"file_path": "/var/log/app.log", "compressed": True},
          tokens_input=2000)

# 3. Log searches/queries
log_event(session.id, EventType.SEARCH, 
          data={"query": "error patterns"},
          tokens_input=100, tokens_output=800)

# 4. Log report generation
log_event(session.id, EventType.AI_RESPONSE, 
          data={"type": "analysis_report"}, tokens_output=1500)

# 5. End and check
end_session(session.id)
report = analyze_session(session.id)
print(f"Analysis efficiency: {report.efficiency_score:.1f}%")
```

### What to Track

| Event | When to Log | Why |
|-------|-------------|-----|
| FILE_READ | Loading data files | Use compression! |
| SEARCH | Data queries | Detect repeated queries |
| AI_RESPONSE | Report generation | Track output tokens |
| THINKING | Complex analysis | Monitor reasoning overhead |

### Key Metrics for NEXUS

- **Large files**: ALWAYS use ContextCompressor for log files
- **Repeated queries**: Cache query results
- **Session duration**: Analysis sessions can be long, but should be productive

---

## Universal Commands

### CLI (All Agents)

```bash
# Start a session
sessionoptimizer start <AGENT> "<task>" --model <model>

# Log an event
sessionoptimizer log <session_id> FILE_READ --tokens-in 500 --data '{"file_path": "x.py"}'

# End a session
sessionoptimizer end <session_id>

# Get analysis
sessionoptimizer analyze <session_id>

# Check your stats
sessionoptimizer stats <AGENT> --days 7
```

### Python One-Liners

```python
from sessionoptimizer import start_session, end_session, log_event, analyze_session, EventType

# Start
s = start_session("AGENT", "Task")

# Log
log_event(s.id, EventType.FILE_READ, tokens_input=500)

# End + Analyze
end_session(s.id)
r = analyze_session(s.id)
print(f"Score: {r.efficiency_score}%")
```

### Check Team Stats

```bash
# See how all agents are doing
sessionoptimizer stats FORGE --days 7
sessionoptimizer stats ATLAS --days 7
sessionoptimizer stats BOLT --days 7
```

---

## Common Patterns

### Pattern 1: Full Session Wrapper

```python
from sessionoptimizer import start_session, end_session, log_event, analyze_session, EventType

def tracked_session(agent, task, work_function):
    """Run a function with full session tracking."""
    session = start_session(agent, task)
    try:
        result = work_function(session.id)
        return result
    except Exception as e:
        log_event(session.id, EventType.ERROR, data={"message": str(e)})
        raise
    finally:
        end_session(session.id)
        report = analyze_session(session.id)
        if report.efficiency_score < 70:
            print(f"Warning: Low efficiency ({report.efficiency_score:.1f}%)")
```

### Pattern 2: Context Manager

```python
from contextlib import contextmanager
from sessionoptimizer import start_session, end_session, analyze_session

@contextmanager
def session_tracker(agent, task, model="default"):
    """Context manager for session tracking."""
    session = start_session(agent, task, model)
    try:
        yield session
    finally:
        end_session(session.id)
        report = analyze_session(session.id)
        print(f"Session efficiency: {report.efficiency_score:.1f}%")

# Usage
with session_tracker("FORGE", "Plan feature") as session:
    # Your work here
    log_event(session.id, EventType.AI_RESPONSE, tokens_output=1000)
```

### Pattern 3: Decorator

```python
from functools import wraps
from sessionoptimizer import start_session, end_session, analyze_session

def track_session(agent, task_prefix=""):
    """Decorator to track function execution as a session."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            task = f"{task_prefix}{func.__name__}"
            session = start_session(agent, task)
            try:
                return func(*args, session_id=session.id, **kwargs)
            finally:
                end_session(session.id)
                analyze_session(session.id)
        return wrapper
    return decorator

# Usage
@track_session("ATLAS", "Build: ")
def build_tool(session_id=None):
    log_event(session_id, EventType.FILE_READ, tokens_input=500)
    # ... build logic ...
```

---

## Quick Reference Card

```
+----------------------------------------------------------+
|              SESSIONOPTIMIZER QUICK REFERENCE            |
+----------------------------------------------------------+
| START:    session = start_session("AGENT", "task")       |
| LOG:      log_event(sid, EventType.FILE_READ, tok_in=N)  |
| END:      end_session(session_id)                        |
| ANALYZE:  report = analyze_session(session_id)           |
+----------------------------------------------------------+
| EVENT TYPES: FILE_READ, FILE_WRITE, SEARCH, TOOL_CALL,   |
|              AI_RESPONSE, THINKING, ERROR, CONTEXT_LOAD  |
+----------------------------------------------------------+
| KEY METRICS:                                             |
|   - Efficiency Score: >80% is good, <70% needs work     |
|   - Thinking Ratio: Keep <40% of output                 |
|   - Error Rate: Keep <20%                               |
|   - Session Time: <60 minutes recommended               |
+----------------------------------------------------------+
| TIPS:                                                    |
|   - Mark compressed files: data={"compressed": True}     |
|   - Cache repeated reads                                 |
|   - Use ErrorRecovery for failures                       |
+----------------------------------------------------------+
```

---

## Need Help?

- **Full Documentation:** [README.md](README.md)
- **Examples:** [EXAMPLES.md](EXAMPLES.md)
- **Cheat Sheet:** [CHEAT_SHEET.txt](CHEAT_SHEET.txt)
- **Integration:** [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md)

---

**Q-MODE COMPLETE! Tool #18 of 18 - SessionOptimizer is ready for Team Brain!**
