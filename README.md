# SessionOptimizer

**AI Session Efficiency Analyzer for Team Brain**

Q-Mode Tool #18 of 18 (Tier 3: Advanced Capabilities)  
The FINAL tool completing the Q-Mode toolkit!

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](https://docs.python.org/3/library/index.html)
[![Tests: 54 Passing](https://img.shields.io/badge/tests-54%20passing-brightgreen.svg)](test_sessionoptimizer.py)

---

## The Problem

AI agent sessions waste tokens through:

- **Repeated file reads** - Reading the same file multiple times in one session
- **Redundant searches** - Running similar searches without caching results
- **Large uncompressed files** - Loading huge files without using ContextCompressor
- **Excessive thinking** - Too much reasoning overhead vs. actual output
- **Failed retries** - Retrying operations without fixing the root cause
- **Long sessions** - Sessions that drag on and lose focus
- **High error rates** - Too many failures indicating unclear tasks
- **Token spikes** - Unexplained jumps in token usage

**Without visibility, you can't optimize.** Token costs add up silently.

---

## The Solution

SessionOptimizer provides:

1. **Session Tracking** - Record every event, token, and duration
2. **8 Built-in Analyzers** - Detect specific inefficiency patterns
3. **Optimization Reports** - Actionable suggestions with estimated savings
4. **Agent Statistics** - Track efficiency over time per agent
5. **Session Comparison** - Compare sessions to find best practices
6. **Cost Estimation** - Know what each session actually costs

**Goal: 20%+ reduction in average tokens per task**

---

## Who Requested This

- **Requested by:** Q-Mode Strategic Roadmap (Atlas, January 17, 2026)
- **Built by:** Forge (Team Brain Orchestrator)
- **Purpose:** Tool #18 of 18 - Complete the Q-Mode toolkit
- **Impact:** Reduce token waste, identify inefficient patterns

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DonkRonk17/SessionOptimizer.git
cd SessionOptimizer

# No dependencies required - uses Python standard library only
python sessionoptimizer.py --help
```

### Basic Usage

#### Python API

```python
from sessionoptimizer import SessionOptimizer, EventType

# Create optimizer
optimizer = SessionOptimizer()

# Start tracking a session
session = optimizer.start_session(
    agent="FORGE",
    task="Build new feature",
    model="opus"
)

# Log events as they happen
optimizer.log_event(
    session.id,
    EventType.FILE_READ,
    data={"file_path": "/src/main.py"},
    tokens_input=500
)

optimizer.log_event(
    session.id,
    EventType.AI_RESPONSE,
    tokens_output=1500
)

# End session and analyze
optimizer.end_session(session.id)
report = optimizer.analyze(session.id)

print(f"Efficiency Score: {report.efficiency_score:.1f}%")
print(f"Issues Found: {len(report.issues)}")
print(f"Estimated Waste: {report.total_waste_tokens:,} tokens")
```

#### CLI

```bash
# Start a session
sessionoptimizer start FORGE "Build authentication system" --model opus

# Log events
sessionoptimizer log forge_20260121_143000 FILE_READ \
    --data '{"file_path": "/src/auth.py"}' \
    --tokens-in 500

# Analyze the session
sessionoptimizer analyze forge_20260121_143000

# View agent statistics
sessionoptimizer stats FORGE --days 7
```

---

## Features

### 8 Built-in Efficiency Analyzers

| Analyzer | Detects | Severity |
|----------|---------|----------|
| **RepeatedFileReadAnalyzer** | Same file read multiple times | MEDIUM-HIGH |
| **RepeatedSearchAnalyzer** | Similar searches executed repeatedly | MEDIUM |
| **LargeFileAnalyzer** | Large files loaded without compression | HIGH |
| **ExcessiveThinkingAnalyzer** | Thinking >40% of output tokens | MEDIUM |
| **FailedRetryAnalyzer** | Consecutive failed operations | HIGH |
| **LongSessionAnalyzer** | Sessions >60 minutes | MEDIUM-HIGH |
| **HighErrorRateAnalyzer** | Error rate >20% | MEDIUM-HIGH |
| **TokenSpikeAnalyzer** | Token usage >3x average | MEDIUM |

### Session Management

```python
# Start session with custom ID
session = optimizer.start_session(
    agent="ATLAS",
    task="Run test suite",
    model="sonnet",
    session_id="custom_id_123",
    metadata={"project": "BCH", "priority": "HIGH"}
)

# List sessions with filters
sessions = optimizer.list_sessions(
    agent="FORGE",
    status=SessionStatus.COMPLETED,
    since=datetime.now() - timedelta(days=7),
    limit=50
)

# Delete a session
optimizer.delete_session("session_id")
```

### Event Types

```python
from sessionoptimizer import EventType

# Available event types:
EventType.SESSION_START    # Session begins
EventType.SESSION_END      # Session ends
EventType.TOOL_CALL        # Tool/function call
EventType.TOKEN_USAGE      # Token usage record
EventType.FILE_READ        # File read operation
EventType.FILE_WRITE       # File write operation
EventType.SEARCH           # Search/grep operation
EventType.ERROR            # Error occurred
EventType.USER_MESSAGE     # User message received
EventType.AI_RESPONSE      # AI response generated
EventType.CONTEXT_LOAD     # Context loaded
EventType.THINKING         # Thinking/reasoning
EventType.CUSTOM           # Custom event type
```

### Optimization Reports

```python
report = optimizer.analyze(session.id)

# Report contents:
print(f"Session ID: {report.session_id}")
print(f"Generated: {report.generated_at}")
print(f"Efficiency Score: {report.efficiency_score:.1f}/100")
print(f"Total Waste Tokens: {report.total_waste_tokens:,}")
print(f"Total Waste Cost: ${report.total_waste_cost:.4f}")

# Issues found
for issue in report.issues:
    print(f"\n[{issue.severity.value}] {issue.issue_type.value}")
    print(f"  {issue.description}")
    print(f"  Suggestion: {issue.suggestion}")
    print(f"  Waste: {issue.estimated_waste_tokens:,} tokens")

# Token breakdown by event type
for event_type, tokens in report.token_breakdown.items():
    print(f"  {event_type}: {tokens:,} tokens")

# Recommendations
for rec in report.recommendations:
    print(f"  - {rec}")
```

### Agent Statistics

```python
stats = optimizer.agent_statistics("FORGE", days=30)

print(f"Agent: {stats['agent']}")
print(f"Sessions Analyzed: {stats['sessions_analyzed']}")
print(f"Total Tokens: {stats['total_tokens']:,}")
print(f"Total Cost: ${stats['total_cost']:.2f}")
print(f"Average Efficiency: {stats['average_efficiency']:.1f}%")
print(f"Total Waste Tokens: {stats['total_waste_tokens']:,}")

# Common issues
for issue in stats['common_issues']:
    print(f"  {issue['type']}: {issue['count']} occurrences")

# Trends
print(f"Trend: {stats['trends']['trend']}")
print(f"Change: {stats['trends']['change_percent']:+.1f}%")
```

### Session Comparison

```python
comparison = optimizer.compare_sessions([
    "session_1",
    "session_2",
    "session_3"
])

print("Session Comparison:")
for s in comparison["sessions"]:
    print(f"  {s['id']}: {s['efficiency_score']:.1f}% efficiency")

print(f"\nAverages:")
print(f"  Tokens: {comparison['averages']['tokens']:,.0f}")
print(f"  Cost: ${comparison['averages']['cost']:.4f}")
print(f"  Efficiency: {comparison['averages']['efficiency']:.1f}%")

print(f"\nBest Performer: {comparison['best_performer']}")
print(f"Worst Performer: {comparison['worst_performer']}")
```

### Export Reports

```python
# Export as JSON
json_output = optimizer.export_report(session.id, format="json")

# Export as Markdown
md_output = optimizer.export_report(session.id, format="markdown")

# Export as plain text
text_output = optimizer.export_report(session.id, format="text")

# Save to file
with open("report.md", "w") as f:
    f.write(md_output)
```

---

## CLI Reference

### Commands

```bash
# Session Management
sessionoptimizer start <agent> <task> [--model MODEL]
sessionoptimizer end <session_id> [--status STATUS]
sessionoptimizer list [--agent AGENT] [--status STATUS] [--limit N]
sessionoptimizer delete <session_id>

# Event Logging
sessionoptimizer log <session_id> <event_type> [options]
    --data JSON         Event data as JSON
    --tokens-in N       Input tokens used
    --tokens-out N      Output tokens used

# Analysis
sessionoptimizer analyze <session_id> [--format FORMAT]
sessionoptimizer stats <agent> [--days N]
sessionoptimizer compare <session_id1> <session_id2> [...]

# Export
sessionoptimizer export <session_id> [--format FORMAT] [--output FILE]

# Info
sessionoptimizer version
sessionoptimizer help
```

### Examples

```bash
# Start a session for Forge working on a tool
sessionoptimizer start FORGE "Build PriorityQueue" --model opus

# Log file read with token count
sessionoptimizer log forge_20260121_143000 FILE_READ \
    --data '{"file_path": "roadmap.md", "compressed": false}' \
    --tokens-in 2500

# Log a search operation
sessionoptimizer log forge_20260121_143000 SEARCH \
    --data '{"query": "how does priority queue work"}' \
    --tokens-in 100 --tokens-out 800

# Log an error
sessionoptimizer log forge_20260121_143000 ERROR \
    --data '{"message": "Test failed", "code": 1}'

# End the session
sessionoptimizer end forge_20260121_143000 --status COMPLETED

# Get full analysis
sessionoptimizer analyze forge_20260121_143000 --format markdown

# Check agent stats for the week
sessionoptimizer stats FORGE --days 7

# Compare today's sessions
sessionoptimizer compare forge_20260121_143000 forge_20260121_150000

# Export report to file
sessionoptimizer export forge_20260121_143000 --format markdown --output report.md
```

---

## Cost Model

SessionOptimizer tracks costs for different AI models:

| Model | Input ($/1K tokens) | Output ($/1K tokens) |
|-------|--------------------:|---------------------:|
| Opus 4.5 | $0.015 | $0.075 |
| Sonnet 4.5 | $0.003 | $0.015 |
| GPT-4 | $0.010 | $0.030 |
| Gemini | $0.00025 | $0.0005 |
| Grok (Cline) | $0.00 | $0.00 |

Example cost calculation:

```python
# Session with 10K input, 5K output on Opus
# Input: 10,000 / 1000 * $0.015 = $0.15
# Output: 5,000 / 1000 * $0.075 = $0.375
# Total: $0.525

session = Session(id="test", agent="FORGE", task="Test", model="opus")
session.total_tokens_input = 10000
session.total_tokens_output = 5000
print(f"Estimated cost: ${session.estimated_cost:.4f}")  # $0.5250
```

---

## Quick Functions

For simple usage without managing an optimizer instance:

```python
from sessionoptimizer import (
    start_session,
    end_session,
    log_event,
    analyze_session,
    get_agent_stats,
    EventType
)

# Start tracking
session = start_session("FORGE", "Quick task")

# Log events
log_event(session.id, EventType.FILE_READ, {"file": "main.py"}, tokens_input=500)
log_event(session.id, EventType.AI_RESPONSE, tokens_output=1500)

# End and analyze
end_session(session.id)
report = analyze_session(session.id)

print(f"Efficiency: {report.efficiency_score:.1f}%")

# Get agent stats
stats = get_agent_stats("FORGE", days=7)
print(f"Average efficiency: {stats['average_efficiency']:.1f}%")
```

---

## Custom Analyzers

Create custom analyzers for domain-specific efficiency patterns:

```python
from sessionoptimizer import (
    BaseAnalyzer,
    Session,
    EfficiencyIssue,
    IssueType,
    Severity,
    SessionOptimizer
)

class CustomApiCallAnalyzer(BaseAnalyzer):
    """Detect inefficient API call patterns."""
    
    def analyze(self, session: Session) -> list:
        issues = []
        
        # Custom analysis logic
        api_calls = [
            e for e in session.events 
            if e.data.get("type") == "api_call"
        ]
        
        if len(api_calls) > 10:
            issues.append(EfficiencyIssue(
                id=f"too_many_api_calls_{session.id[:8]}",
                issue_type=IssueType.INEFFICIENT_TOOL_USE,
                severity=Severity.MEDIUM,
                description=f"Session made {len(api_calls)} API calls",
                suggestion="Batch API calls or cache responses",
                estimated_waste_tokens=len(api_calls) * 100,
            ))
        
        return issues

# Use custom analyzer
optimizer = SessionOptimizer(
    analyzers=[
        CustomApiCallAnalyzer(),
        # ... include default analyzers too
    ]
)
```

---

## Integration with Team Brain

### With TokenTracker

```python
from sessionoptimizer import start_session, log_event, EventType

# Track session with token logging
session = start_session("FORGE", "Build feature")

# TokenTracker reports usage
log_event(
    session.id,
    EventType.TOKEN_USAGE,
    data={"source": "TokenTracker", "model": "opus"},
    tokens_input=1500,
    tokens_output=3000
)
```

### With ContextCompressor

```python
# Mark files as compressed to avoid false positives
log_event(
    session.id,
    EventType.FILE_READ,
    data={
        "file_path": "/large/file.json",
        "compressed": True,  # Mark as compressed
        "original_tokens": 15000,
        "compressed_tokens": 3000
    },
    tokens_input=3000
)
```

### With ErrorRecovery

```python
# Log errors and recovery attempts
log_event(
    session.id,
    EventType.ERROR,
    data={
        "message": "Connection timeout",
        "recovery_strategy": "retry",
        "recovered": True
    }
)
```

### With SynapseLink

```python
from synapselink import quick_send
from sessionoptimizer import analyze_session

# Analyze and report via Synapse
report = analyze_session(session_id)

if report.efficiency_score < 70:
    quick_send(
        "LOGAN,TEAM",
        f"Low Efficiency Alert: {session_id}",
        f"Score: {report.efficiency_score:.1f}%\n"
        f"Issues: {len(report.issues)}\n"
        f"Waste: {report.total_waste_tokens:,} tokens",
        priority="HIGH"
    )
```

---

## Storage

Sessions are stored in `~/.sessionoptimizer/` by default:

```
~/.sessionoptimizer/
    sessions/           # JSON session files
        session_001.json
        session_002.json
    reports/            # Generated reports
        session_001_report.json
```

Custom storage location:

```python
optimizer = SessionOptimizer(storage_path="/custom/path")
```

---

## API Reference

### SessionOptimizer

```python
class SessionOptimizer:
    def __init__(
        self,
        storage_path: str = None,     # Custom storage location
        analyzers: list = None,        # Custom analyzers
    )
    
    # Session Management
    def start_session(agent, task, model="default", session_id=None, metadata=None) -> Session
    def end_session(session_id, status=SessionStatus.COMPLETED) -> Session
    def get_session(session_id) -> Session
    def list_sessions(agent=None, status=None, since=None, limit=100) -> list
    def delete_session(session_id) -> bool
    
    # Event Logging
    def log_event(session_id, event_type, data=None, tokens_input=0, tokens_output=0, duration_ms=0) -> SessionEvent
    
    # Analysis
    def analyze(session_id) -> OptimizationReport
    def compare_sessions(session_ids) -> dict
    def agent_statistics(agent, days=30) -> dict
    
    # Export
    def export_report(session_id, format="json") -> str
    def import_session(data) -> Session
```

### Session

```python
@dataclass
class Session:
    id: str
    agent: str
    task: str
    status: SessionStatus = ACTIVE
    started_at: str
    ended_at: str
    events: list
    total_tokens_input: int
    total_tokens_output: int
    total_duration_ms: int
    model: str = "default"
    metadata: dict
    
    @property
    def total_tokens(self) -> int
    @property
    def estimated_cost(self) -> float
    @property
    def duration_seconds(self) -> float
```

### OptimizationReport

```python
@dataclass
class OptimizationReport:
    session_id: str
    generated_at: str
    issues: list[EfficiencyIssue]
    total_waste_tokens: int
    total_waste_cost: float
    efficiency_score: float  # 0-100
    recommendations: list[str]
    token_breakdown: dict[str, int]
    comparisons: dict
```

### EfficiencyIssue

```python
@dataclass
class EfficiencyIssue:
    id: str
    issue_type: IssueType
    severity: Severity
    description: str
    suggestion: str
    estimated_waste_tokens: int
    estimated_waste_cost: float
    events_involved: list[str]
    metadata: dict
```

---

## Testing

```bash
# Run all tests
python -m pytest test_sessionoptimizer.py -v

# Run with coverage
python -m pytest test_sessionoptimizer.py --cov=sessionoptimizer

# Run specific test class
python -m pytest test_sessionoptimizer.py::TestSessionOptimizer -v
```

**Test Coverage:** 54 tests covering:
- Session basics (creation, events, serialization)
- All 8 analyzers
- Optimizer operations
- Statistics and comparison
- Export/import
- Edge cases and error handling
- Cost calculation

---

## Requirements

- **Python:** 3.8+
- **Dependencies:** None (Python standard library only)
- **Storage:** ~1KB per session (JSON)

---

## Q-Mode Completion Status

SessionOptimizer is **Tool #18 of 18** - the FINAL Q-Mode tool!

| Tier | Tools | Status |
|------|-------|--------|
| Tier 1: Critical Foundation | 8 tools | COMPLETE |
| Tier 2: Workflow Enhancement | 6 tools | COMPLETE |
| Tier 3: Advanced Capabilities | 4 tools | COMPLETE |

**Q-MODE IS NOW 100% COMPLETE!**

---

## Credits

- **Created by:** Forge (Team Brain Orchestrator)
- **Requested by:** Q-Mode Strategic Roadmap
- **Architecture:** Atlas (Sonnet 4.5)
- **For:** Logan Smith / Metaphy LLC
- **Part of:** Team Brain AI Collaboration Framework

---

## License

MIT License - See [LICENSE](LICENSE) file.

Copyright (c) 2026 Logan Smith / Metaphy LLC

---

## Related Q-Mode Tools

| Tool | Purpose |
|------|---------|
| TokenTracker | Track token costs in real-time |
| ContextCompressor | Reduce large file tokens by 50-90% |
| ErrorRecovery | Auto-recover from common failures |
| AgentHealth | Monitor agent efficiency |
| SynapseLink | AI-to-AI messaging |

---

## Changelog

### v1.0.0 (January 21, 2026)

- Initial release
- 8 built-in efficiency analyzers
- Session tracking and management
- Optimization reports with recommendations
- Agent statistics and comparison
- Export to JSON, Markdown, Text
- CLI interface
- 54 tests passing
- Zero dependencies

---

**SessionOptimizer - Completing the Q-Mode Vision**

*"What gets measured gets optimized."*
