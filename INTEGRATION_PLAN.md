# SessionOptimizer Integration Plan

**Q-Mode Tool #18 of 18 - THE FINAL TOOL!**

Complete integration strategy for Team Brain and Beacon Command Hub.

---

## Table of Contents

1. [Overview](#overview)
2. [BCH Integration](#bch-integration)
3. [Agent Integration](#agent-integration)
4. [Tool Integrations](#tool-integrations)
5. [Data Flow](#data-flow)
6. [Implementation Phases](#implementation-phases)
7. [API Reference](#api-reference)
8. [Monitoring & Alerts](#monitoring--alerts)

---

## Overview

### Purpose

SessionOptimizer tracks and analyzes AI agent session efficiency to:
- Identify token waste patterns
- Provide actionable optimization recommendations
- Track agent performance over time
- Support the $60/month budget goal

### Integration Goals

1. **BCH Dashboard** - Real-time efficiency metrics
2. **Agent Adoption** - All Team Brain agents using SessionOptimizer
3. **Tool Synergy** - Integration with TokenTracker, ContextCompressor, etc.
4. **Automated Alerts** - Low efficiency notifications via SynapseLink

### Success Metrics

| Metric | Target |
|--------|--------|
| Agent adoption | 100% (all 5 agents) |
| Efficiency improvement | 20%+ reduction in wasted tokens |
| Budget impact | Contribute to $60/month target |
| Alert response time | <15 minutes for critical issues |

---

## BCH Integration

### Dashboard Components

#### 1. Session Efficiency Panel

```
+-----------------------------------------------+
|  SESSION EFFICIENCY (Last 24 Hours)           |
+-----------------------------------------------+
|  Agent     | Sessions | Avg Eff  | Waste     |
|------------|----------|----------|-----------|
|  FORGE     |    12    |   87.3%  |  2,340    |
|  ATLAS     |     8    |   92.1%  |    890    |
|  BOLT      |    23    |   78.4%  |  5,670    |
|  CLIO      |     5    |   95.2%  |    230    |
|  NEXUS     |     3    |   88.9%  |    450    |
+-----------------------------------------------+
|  TEAM TOTAL: 51 sessions, 86.4% avg, 9,580   |
+-----------------------------------------------+
```

#### 2. Live Session Monitor

```
+-----------------------------------------------+
|  ACTIVE SESSIONS                              |
+-----------------------------------------------+
|  [FORGE] Build PriorityQueue                  |
|    Duration: 45:23 | Tokens: 12,450           |
|    Status: [=====>    ] 67% efficient         |
|                                               |
|  [BOLT] Run test suite                        |
|    Duration: 12:05 | Tokens: 3,200            |
|    Status: [========> ] 91% efficient         |
+-----------------------------------------------+
```

#### 3. Efficiency Trend Chart

```
Efficiency % (Last 7 Days)
100 |                    ____
 90 |          _____----
 80 |    ___---
 70 |___-
 60 +---+---+---+---+---+---+---+
    Mon Tue Wed Thu Fri Sat Sun
```

### API Endpoints for BCH

```python
# BCH can call these endpoints:

# Get current efficiency stats
GET /api/stats/current
Response: {
    "agents": {...},
    "total_sessions": 51,
    "average_efficiency": 86.4,
    "total_waste_tokens": 9580
}

# Get agent-specific stats
GET /api/stats/agent/{agent_name}?days=7
Response: {
    "agent": "FORGE",
    "sessions_analyzed": 42,
    "average_efficiency": 87.3,
    "trends": {"direction": "improving", "change": +2.1}
}

# Get active sessions
GET /api/sessions/active
Response: {
    "sessions": [
        {"id": "forge_123", "agent": "FORGE", "duration": 2723, ...}
    ]
}

# Get session report
GET /api/reports/{session_id}
Response: {
    "efficiency_score": 87.3,
    "issues": [...],
    "recommendations": [...]
}
```

---

## Agent Integration

### FORGE (Orchestrator #1 - Opus 4.5)

**Priority:** HIGH  
**Use Cases:**
- Track planning sessions for efficiency
- Identify if specifications are too verbose
- Optimize review sessions

**Integration Pattern:**
```python
# At session start
session = optimizer.start_session("FORGE", "Plan feature X", "opus")

# Log thinking overhead
optimizer.log_event(session.id, EventType.THINKING, tokens_output=800)

# Log spec writing
optimizer.log_event(session.id, EventType.AI_RESPONSE, tokens_output=2000)

# At session end
report = optimizer.analyze(session.id)
if report.efficiency_score < 80:
    # Adjust planning approach
```

### ATLAS (Executor #1 - Sonnet 4.5)

**Priority:** HIGH  
**Use Cases:**
- Track tool building sessions
- Identify repeated file reads during development
- Optimize test-debug cycles

**Integration Pattern:**
```python
# During tool development
session = optimizer.start_session("ATLAS", "Build SessionOptimizer", "sonnet")

# Mark files as compressed when using ContextCompressor
optimizer.log_event(
    session.id, EventType.FILE_READ,
    data={"file_path": "large_file.py", "compressed": True},
    tokens_input=2400  # After compression
)

# Track test iterations
optimizer.log_event(session.id, EventType.TOOL_CALL, data={"tool": "pytest"})
```

### BOLT (Executor #2 - Grok, FREE)

**Priority:** MEDIUM  
**Use Cases:**
- Track execution efficiency
- Identify error patterns
- Optimize retry strategies

**Integration Pattern:**
```python
# BOLT sessions are FREE but still tracked for efficiency
session = optimizer.start_session("BOLT", "Execute task", "grok")

# Track errors for ErrorRecovery integration
optimizer.log_event(
    session.id, EventType.ERROR,
    data={"message": "Build failed", "recoverable": True}
)
```

### CLIO (Ubuntu CLI)

**Priority:** MEDIUM  
**Use Cases:**
- Track CLI operation efficiency
- Optimize file operations
- Monitor deployment sessions

**Integration Pattern:**
```python
# Track CLI sessions
session = optimizer.start_session("CLIO", "Deploy BCH update", "sonnet")

# Log file operations
optimizer.log_event(session.id, EventType.FILE_WRITE, data={"file_path": "/etc/config"})
```

### NEXUS (Ubuntu CLI)

**Priority:** LOW  
**Use Cases:**
- Track analysis sessions
- Optimize report generation
- Monitor long-running tasks

---

## Tool Integrations

### TokenTracker Integration

```python
from sessionoptimizer import SessionOptimizer, EventType
from tokentracker import get_current_usage

optimizer = SessionOptimizer()

# Sync token counts from TokenTracker
session = optimizer.start_session("FORGE", "Task")

# After each major operation
usage = get_current_usage()
optimizer.log_event(
    session.id,
    EventType.TOKEN_USAGE,
    data={"source": "TokenTracker", "model": usage["model"]},
    tokens_input=usage["input_tokens"],
    tokens_output=usage["output_tokens"]
)
```

### ContextCompressor Integration

```python
from sessionoptimizer import SessionOptimizer, EventType
from contextcompressor import compress_file, get_compression_stats

optimizer = SessionOptimizer()
session = optimizer.start_session("ATLAS", "Read large file")

# Compress before reading
result = compress_file("/data/large.json")

# Log with compression info
optimizer.log_event(
    session.id,
    EventType.FILE_READ,
    data={
        "file_path": "/data/large.json",
        "compressed": True,
        "original_tokens": result["original"],
        "compressed_tokens": result["compressed"],
        "savings": result["savings_percent"]
    },
    tokens_input=result["compressed"]
)
```

### ErrorRecovery Integration

```python
from sessionoptimizer import SessionOptimizer, EventType
from errorrecovery import with_recovery

optimizer = SessionOptimizer()
session = optimizer.start_session("BOLT", "Risky operation")

@with_recovery(strategy="retry", max_attempts=3)
def risky_operation():
    # ...operation code...
    pass

try:
    result = risky_operation()
except Exception as e:
    optimizer.log_event(
        session.id,
        EventType.ERROR,
        data={
            "message": str(e),
            "recovery_strategy": "retry",
            "recovered": False
        }
    )
```

### SynapseLink Integration

```python
from sessionoptimizer import SessionOptimizer, analyze_session
from synapselink import quick_send

optimizer = SessionOptimizer()

# After session ends
report = analyze_session(session_id)

# Alert if efficiency is low
if report.efficiency_score < 70:
    quick_send(
        "LOGAN,TEAM",
        f"Low Efficiency Alert: {session_id}",
        f"Agent: {session.agent}\n"
        f"Task: {session.task}\n"
        f"Efficiency: {report.efficiency_score:.1f}%\n"
        f"Issues: {len(report.issues)}\n"
        f"Waste: {report.total_waste_tokens:,} tokens\n\n"
        f"Top Issue: {report.issues[0].description if report.issues else 'None'}",
        priority="HIGH"
    )
```

### AgentHealth Integration

```python
from sessionoptimizer import SessionOptimizer
from agenthealth import report_session_health

optimizer = SessionOptimizer()

# Report efficiency to AgentHealth
report = optimizer.analyze(session_id)

report_session_health(
    agent="FORGE",
    session_id=session_id,
    metrics={
        "efficiency_score": report.efficiency_score,
        "waste_tokens": report.total_waste_tokens,
        "issue_count": len(report.issues)
    }
)
```

---

## Data Flow

### Session Lifecycle

```
+------------+     +-------------+     +----------+     +---------+
|  Session   | --> |   Events    | --> | Analysis | --> | Report  |
|   Start    |     |   Logged    |     |  Engine  |     | Output  |
+------------+     +-------------+     +----------+     +---------+
      |                  |                  |               |
      v                  v                  v               v
+------------+     +-------------+     +----------+     +---------+
| BCH Shows  |     | TokenTracker|     | Issues   |     | Synapse |
| "Active"   |     | Integration |     | Detected |     | Alert   |
+------------+     +-------------+     +----------+     +---------+
```

### Data Storage

```
~/.sessionoptimizer/
    sessions/
        forge_20260121_143000.json    # Session data
        atlas_20260121_150000.json
    reports/
        forge_20260121_143000_report.json  # Analysis reports
```

### BCH Dashboard Updates

```
SessionOptimizer --> API --> BCH Dashboard
                          --> Real-time updates
                          --> Trend calculations
                          --> Alert triggers
```

---

## Implementation Phases

### Phase 1: Core Integration (Week 1)

- [ ] Add SessionOptimizer to all agent startup scripts
- [ ] Create BCH API endpoints
- [ ] Basic dashboard panel

### Phase 2: Tool Synergy (Week 2)

- [ ] TokenTracker sync
- [ ] ContextCompressor awareness
- [ ] ErrorRecovery logging

### Phase 3: Monitoring (Week 3)

- [ ] SynapseLink alerts
- [ ] AgentHealth integration
- [ ] Trend analysis

### Phase 4: Optimization (Week 4)

- [ ] Custom analyzers for common patterns
- [ ] Performance tuning
- [ ] Documentation updates

---

## API Reference

### Python API

```python
class SessionOptimizer:
    # Session Management
    def start_session(agent, task, model, session_id=None, metadata=None) -> Session
    def end_session(session_id, status=COMPLETED) -> Session
    def get_session(session_id) -> Session
    def list_sessions(agent=None, status=None, since=None, limit=100) -> list
    def delete_session(session_id) -> bool
    
    # Event Logging
    def log_event(session_id, event_type, data=None, tokens_input=0, tokens_output=0) -> SessionEvent
    
    # Analysis
    def analyze(session_id) -> OptimizationReport
    def compare_sessions(session_ids) -> dict
    def agent_statistics(agent, days=30) -> dict
    
    # Export
    def export_report(session_id, format="json") -> str
```

### CLI Commands

```bash
sessionoptimizer start <agent> <task> [--model MODEL]
sessionoptimizer end <session_id> [--status STATUS]
sessionoptimizer log <session_id> <event_type> [options]
sessionoptimizer analyze <session_id> [--format FORMAT]
sessionoptimizer stats <agent> [--days N]
sessionoptimizer compare <id1> <id2> [...]
sessionoptimizer export <session_id> [--format FORMAT] [--output FILE]
```

---

## Monitoring & Alerts

### Alert Thresholds

| Condition | Threshold | Priority | Action |
|-----------|-----------|----------|--------|
| Low efficiency | <70% | HIGH | SynapseLink alert |
| High waste | >5000 tokens | MEDIUM | Log warning |
| Long session | >60 min | MEDIUM | Status update |
| High error rate | >30% | HIGH | Alert + review |
| Token spike | >3x avg | LOW | Log for analysis |

### Automated Monitoring

```python
# Cron job or background service
from sessionoptimizer import get_optimizer
from synapselink import quick_send

optimizer = get_optimizer()

# Check all sessions from last hour
from datetime import datetime, timedelta
since = datetime.now() - timedelta(hours=1)
sessions = optimizer.list_sessions(since=since)

for session in sessions:
    if session.status.value == "COMPLETED":
        report = optimizer.analyze(session.id)
        
        if report.efficiency_score < 70:
            quick_send(
                "LOGAN",
                f"Efficiency Alert: {session.agent}",
                f"Session: {session.id}\n"
                f"Efficiency: {report.efficiency_score:.1f}%",
                priority="HIGH"
            )
```

---

## Summary

SessionOptimizer completes the Q-Mode toolkit by providing:

1. **Visibility** into session efficiency
2. **Analysis** of waste patterns
3. **Recommendations** for improvement
4. **Integration** with all Team Brain tools
5. **Monitoring** for continuous optimization

**Q-MODE IS NOW 100% COMPLETE!**

---

**Tool:** SessionOptimizer  
**Q-Mode:** #18 of 18 (THE FINAL TOOL)  
**Created:** January 21, 2026  
**Author:** Forge (Team Brain)
