# SessionOptimizer Examples

Comprehensive examples for AI Session Efficiency Analysis.

---

## Table of Contents

1. [Basic Session Tracking](#example-1-basic-session-tracking)
2. [Logging Different Event Types](#example-2-logging-different-event-types)
3. [Detecting Repeated File Reads](#example-3-detecting-repeated-file-reads)
4. [Analyzing Large File Usage](#example-4-analyzing-large-file-usage)
5. [Error Pattern Detection](#example-5-error-pattern-detection)
6. [Session Comparison](#example-6-session-comparison)
7. [Agent Statistics Dashboard](#example-7-agent-statistics-dashboard)
8. [Export Reports](#example-8-export-reports)
9. [Custom Analyzer](#example-9-custom-analyzer)
10. [Team Brain Integration](#example-10-team-brain-integration)

---

## Example 1: Basic Session Tracking

Track a complete session from start to finish.

```python
#!/usr/bin/env python3
"""Example 1: Basic Session Tracking"""

from sessionoptimizer import SessionOptimizer, EventType

def main():
    # Create optimizer
    optimizer = SessionOptimizer()
    
    # Start a session
    session = optimizer.start_session(
        agent="FORGE",
        task="Build authentication module",
        model="opus"
    )
    print(f"Session started: {session.id}")
    
    # Log some events
    optimizer.log_event(
        session.id,
        EventType.FILE_READ,
        data={"file_path": "/src/auth/login.py"},
        tokens_input=800
    )
    
    optimizer.log_event(
        session.id,
        EventType.SEARCH,
        data={"query": "JWT token validation"},
        tokens_input=100,
        tokens_output=500
    )
    
    optimizer.log_event(
        session.id,
        EventType.AI_RESPONSE,
        data={"type": "code_generation"},
        tokens_output=2000
    )
    
    optimizer.log_event(
        session.id,
        EventType.FILE_WRITE,
        data={"file_path": "/src/auth/token.py"},
        tokens_input=50
    )
    
    # End session
    session = optimizer.end_session(session.id)
    print(f"Session ended: {session.status.value}")
    
    # Analyze
    report = optimizer.analyze(session.id)
    
    print(f"\n=== Analysis Results ===")
    print(f"Efficiency Score: {report.efficiency_score:.1f}/100")
    print(f"Total Tokens: {session.total_tokens:,}")
    print(f"Estimated Cost: ${session.estimated_cost:.4f}")
    print(f"Issues Found: {len(report.issues)}")


if __name__ == "__main__":
    main()
```

**Output:**
```
Session started: forge_20260121_143000
Session ended: COMPLETED

=== Analysis Results ===
Efficiency Score: 100.0/100
Total Tokens: 3,450
Estimated Cost: $0.1620
Issues Found: 0
```

---

## Example 2: Logging Different Event Types

Demonstrate all available event types.

```python
#!/usr/bin/env python3
"""Example 2: Logging Different Event Types"""

from sessionoptimizer import SessionOptimizer, EventType
import json

def main():
    optimizer = SessionOptimizer()
    session = optimizer.start_session("ATLAS", "Event type demo", "sonnet")
    
    # Tool call
    optimizer.log_event(
        session.id,
        EventType.TOOL_CALL,
        data={"tool": "grep", "args": ["pattern", "file.py"]},
        tokens_input=50,
        tokens_output=200,
        duration_ms=150
    )
    
    # File operations
    optimizer.log_event(
        session.id,
        EventType.FILE_READ,
        data={"file_path": "/src/main.py", "lines": 500},
        tokens_input=1500
    )
    
    optimizer.log_event(
        session.id,
        EventType.FILE_WRITE,
        data={"file_path": "/src/new_file.py", "action": "create"},
        tokens_input=100
    )
    
    # Search
    optimizer.log_event(
        session.id,
        EventType.SEARCH,
        data={"query": "authentication middleware", "results": 5},
        tokens_input=80,
        tokens_output=400
    )
    
    # User message and AI response
    optimizer.log_event(
        session.id,
        EventType.USER_MESSAGE,
        data={"content": "Add error handling"},
        tokens_input=50
    )
    
    optimizer.log_event(
        session.id,
        EventType.AI_RESPONSE,
        data={"type": "explanation"},
        tokens_output=800
    )
    
    # Thinking
    optimizer.log_event(
        session.id,
        EventType.THINKING,
        data={"topic": "architecture decision"},
        tokens_output=500
    )
    
    # Error
    optimizer.log_event(
        session.id,
        EventType.ERROR,
        data={"message": "File not found", "recoverable": True},
        tokens_input=20
    )
    
    # Context load
    optimizer.log_event(
        session.id,
        EventType.CONTEXT_LOAD,
        data={"source": "memory", "context_type": "project_structure"},
        tokens_input=1000
    )
    
    # Custom event
    optimizer.log_event(
        session.id,
        EventType.CUSTOM,
        data={"custom_field": "custom_value", "action": "special_operation"},
        tokens_input=100,
        tokens_output=200
    )
    
    optimizer.end_session(session.id)
    
    # Show token breakdown
    report = optimizer.analyze(session.id)
    
    print("Token Breakdown by Event Type:")
    print("-" * 40)
    for event_type, tokens in sorted(report.token_breakdown.items()):
        if not event_type.startswith("TOTAL"):
            print(f"  {event_type:<20} {tokens:>8,} tokens")
    
    print("-" * 40)
    print(f"  {'TOTAL':<20} {report.token_breakdown['TOTAL']:>8,} tokens")


if __name__ == "__main__":
    main()
```

**Output:**
```
Token Breakdown by Event Type:
----------------------------------------
  AI_RESPONSE                  800 tokens
  CONTEXT_LOAD               1,000 tokens
  CUSTOM                       300 tokens
  ERROR                         20 tokens
  FILE_READ                  1,500 tokens
  FILE_WRITE                   100 tokens
  SEARCH                       480 tokens
  SESSION_START                  0 tokens
  THINKING                     500 tokens
  TOOL_CALL                    250 tokens
  USER_MESSAGE                  50 tokens
----------------------------------------
  TOTAL                      5,000 tokens
```

---

## Example 3: Detecting Repeated File Reads

Show how the analyzer detects inefficient patterns.

```python
#!/usr/bin/env python3
"""Example 3: Detecting Repeated File Reads"""

from sessionoptimizer import SessionOptimizer, EventType

def main():
    optimizer = SessionOptimizer()
    session = optimizer.start_session("FORGE", "Inefficient file access", "opus")
    
    # Read the same file multiple times (inefficient!)
    for i in range(4):
        optimizer.log_event(
            session.id,
            EventType.FILE_READ,
            data={"file_path": "/src/config.json"},
            tokens_input=500
        )
        print(f"Read #{i+1}: config.json (500 tokens)")
    
    # This is also repeated
    for i in range(3):
        optimizer.log_event(
            session.id,
            EventType.FILE_READ,
            data={"file_path": "/src/utils.py"},
            tokens_input=300
        )
        print(f"Read #{i+1}: utils.py (300 tokens)")
    
    optimizer.end_session(session.id)
    report = optimizer.analyze(session.id)
    
    print(f"\n=== Efficiency Analysis ===")
    print(f"Score: {report.efficiency_score:.1f}/100")
    print(f"Total Tokens: {session.total_tokens:,}")
    print(f"Wasted Tokens: {report.total_waste_tokens:,}")
    
    print(f"\n=== Issues Found ({len(report.issues)}) ===")
    for issue in report.issues:
        print(f"\n[{issue.severity.value}] {issue.issue_type.value}")
        print(f"  {issue.description}")
        print(f"  Suggestion: {issue.suggestion}")
        print(f"  Waste: {issue.estimated_waste_tokens:,} tokens")
    
    print(f"\n=== Recommendations ===")
    for rec in report.recommendations:
        print(f"  - {rec}")


if __name__ == "__main__":
    main()
```

**Output:**
```
Read #1: config.json (500 tokens)
Read #2: config.json (500 tokens)
Read #3: config.json (500 tokens)
Read #4: config.json (500 tokens)
Read #1: utils.py (300 tokens)
Read #2: utils.py (300 tokens)
Read #3: utils.py (300 tokens)

=== Efficiency Analysis ===
Score: 53.8/100
Total Tokens: 2,900
Wasted Tokens: 1,500

=== Issues Found (2) ===

[HIGH] REPEATED_FILE_READ
  File 'config.json' read 4 times
  Suggestion: Cache file content after first read or use ContextCompressor
  Waste: 1,500 tokens

[MEDIUM] REPEATED_FILE_READ
  File 'utils.py' read 3 times
  Suggestion: Cache file content after first read or use ContextCompressor
  Waste: 600 tokens

=== Recommendations ===
  - [HIGH PRIORITY] Cache file content after first read or use ContextCompressor
  - [TIP] Consider using MemoryBridge for cross-session file caching
```

---

## Example 4: Analyzing Large File Usage

Detect large files that could be compressed.

```python
#!/usr/bin/env python3
"""Example 4: Analyzing Large File Usage"""

from sessionoptimizer import SessionOptimizer, EventType

def main():
    optimizer = SessionOptimizer()
    session = optimizer.start_session("ATLAS", "Large file analysis", "sonnet")
    
    # Large file WITHOUT compression (inefficient)
    optimizer.log_event(
        session.id,
        EventType.FILE_READ,
        data={
            "file_path": "/data/huge_dataset.json",
            "compressed": False,
            "file_size_kb": 500
        },
        tokens_input=15000  # 15K tokens - very large!
    )
    print("Read huge_dataset.json: 15,000 tokens (uncompressed)")
    
    # Large file WITH compression (efficient)
    optimizer.log_event(
        session.id,
        EventType.FILE_READ,
        data={
            "file_path": "/data/another_large_file.json",
            "compressed": True,
            "original_tokens": 12000,
            "compressed_tokens": 2400
        },
        tokens_input=2400  # Only 2.4K after compression
    )
    print("Read another_large_file.json: 2,400 tokens (compressed from 12,000)")
    
    # Small file - no issue
    optimizer.log_event(
        session.id,
        EventType.FILE_READ,
        data={"file_path": "/src/config.py", "compressed": False},
        tokens_input=500
    )
    print("Read config.py: 500 tokens (small file, compression not needed)")
    
    optimizer.end_session(session.id)
    report = optimizer.analyze(session.id)
    
    print(f"\n=== Analysis ===")
    print(f"Efficiency Score: {report.efficiency_score:.1f}/100")
    print(f"Total Tokens Used: {session.total_tokens:,}")
    
    for issue in report.issues:
        if issue.issue_type.value == "LARGE_FILE_UNCOMPRESSED":
            print(f"\n[{issue.severity.value}] Large Uncompressed File Detected")
            print(f"  {issue.description}")
            print(f"  Potential Savings: ~{issue.estimated_waste_tokens:,} tokens")
            print(f"  Suggestion: {issue.suggestion}")


if __name__ == "__main__":
    main()
```

**Output:**
```
Read huge_dataset.json: 15,000 tokens (uncompressed)
Read another_large_file.json: 2,400 tokens (compressed from 12,000)
Read config.py: 500 tokens (small file, compression not needed)

=== Analysis ===
Efficiency Score: 49.7/100
Total Tokens Used: 17,900

[HIGH] Large Uncompressed File Detected
  Large file (15,000 tokens) read without compression
  Potential Savings: ~9,000 tokens
  Suggestion: Use ContextCompressor before loading large files (50-90% savings)
```

---

## Example 5: Error Pattern Detection

Detect sessions with high error rates or repeated failures.

```python
#!/usr/bin/env python3
"""Example 5: Error Pattern Detection"""

from sessionoptimizer import SessionOptimizer, EventType

def main():
    optimizer = SessionOptimizer()
    session = optimizer.start_session("BOLT", "Error-prone session", "grok")
    
    # Simulate a session with many errors
    events = [
        ("TOOL_CALL", {"tool": "run_tests"}, 100, 50, None),
        ("ERROR", {"message": "Test failed"}, 50, 0, None),
        ("ERROR", {"message": "Test failed again"}, 50, 0, None),
        ("ERROR", {"message": "Still failing"}, 50, 0, None),
        ("TOOL_CALL", {"tool": "fix_code"}, 100, 200, None),
        ("ERROR", {"message": "Syntax error"}, 50, 0, None),
        ("ERROR", {"message": "Import error"}, 50, 0, None),
        ("TOOL_CALL", {"tool": "run_tests"}, 100, 50, None),
        ("AI_RESPONSE", {"type": "solution"}, 0, 500, None),
        ("TOOL_CALL", {"tool": "apply_fix"}, 100, 100, None),
    ]
    
    for event_type_str, data, tok_in, tok_out, _ in events:
        event_type = getattr(EventType, event_type_str)
        optimizer.log_event(session.id, event_type, data, tok_in, tok_out)
        if event_type_str == "ERROR":
            print(f"ERROR: {data['message']}")
    
    optimizer.end_session(session.id)
    report = optimizer.analyze(session.id)
    
    print(f"\n=== Session Analysis ===")
    print(f"Efficiency Score: {report.efficiency_score:.1f}/100")
    print(f"Issues Found: {len(report.issues)}")
    
    # Find error-related issues
    for issue in report.issues:
        if "ERROR" in issue.issue_type.value or "RETRY" in issue.issue_type.value:
            print(f"\n[{issue.severity.value}] {issue.issue_type.value}")
            print(f"  {issue.description}")
            print(f"  Suggestion: {issue.suggestion}")


if __name__ == "__main__":
    main()
```

**Output:**
```
ERROR: Test failed
ERROR: Test failed again
ERROR: Still failing
ERROR: Syntax error
ERROR: Import error

=== Session Analysis ===
Efficiency Score: 62.5/100
Issues Found: 2

[HIGH] FAILED_RETRY
  Operation failed 3 times consecutively
  Suggestion: Use ErrorRecovery tool for automatic fallback strategies

[HIGH] HIGH_ERROR_RATE
  Error rate: 50.0% (5/10 events)
  Suggestion: Review task clarity, use ErrorRecovery, check environment setup
```

---

## Example 6: Session Comparison

Compare efficiency across multiple sessions.

```python
#!/usr/bin/env python3
"""Example 6: Session Comparison"""

from sessionoptimizer import SessionOptimizer, EventType
import time

def create_sample_session(optimizer, agent, task, model, efficiency_level):
    """Create a sample session with controlled efficiency."""
    session = optimizer.start_session(
        agent, task, model,
        session_id=f"{agent.lower()}_{int(time.time() * 1000)}"
    )
    
    # Base events
    optimizer.log_event(session.id, EventType.FILE_READ, 
                        data={"file_path": "main.py"}, tokens_input=500)
    optimizer.log_event(session.id, EventType.AI_RESPONSE, tokens_output=1000)
    
    # Add inefficiency based on level
    if efficiency_level == "low":
        # Lots of repeated reads
        for _ in range(5):
            optimizer.log_event(session.id, EventType.FILE_READ,
                              data={"file_path": "main.py"}, tokens_input=500)
    elif efficiency_level == "medium":
        # Some repeated reads
        for _ in range(2):
            optimizer.log_event(session.id, EventType.FILE_READ,
                              data={"file_path": "main.py"}, tokens_input=500)
    # High efficiency = no extra waste
    
    optimizer.end_session(session.id)
    return session.id

def main():
    optimizer = SessionOptimizer()
    
    # Create sessions with different efficiency levels
    sessions = [
        create_sample_session(optimizer, "FORGE", "Task A", "opus", "high"),
        create_sample_session(optimizer, "FORGE", "Task B", "opus", "medium"),
        create_sample_session(optimizer, "FORGE", "Task C", "opus", "low"),
    ]
    
    print("Created 3 sessions with varying efficiency levels\n")
    
    # Compare them
    comparison = optimizer.compare_sessions(sessions)
    
    print("=== Session Comparison ===")
    print(f"{'ID':<30} {'Tokens':>10} {'Efficiency':>12}")
    print("-" * 55)
    
    for s in comparison["sessions"]:
        print(f"{s['id']:<30} {s['total_tokens']:>10,} {s['efficiency_score']:>11.1f}%")
    
    print("\n=== Averages ===")
    avgs = comparison["averages"]
    print(f"  Average Tokens:     {avgs['tokens']:,.0f}")
    print(f"  Average Cost:       ${avgs['cost']:.4f}")
    print(f"  Average Efficiency: {avgs['efficiency']:.1f}%")
    
    print(f"\n=== Verdict ===")
    print(f"  Best Performer:  {comparison['best_performer']}")
    print(f"  Worst Performer: {comparison['worst_performer']}")


if __name__ == "__main__":
    main()
```

**Output:**
```
Created 3 sessions with varying efficiency levels

=== Session Comparison ===
ID                             Tokens   Efficiency
-------------------------------------------------------
forge_1737451234567                1,500      100.0%
forge_1737451234568                2,500       60.0%
forge_1737451234569                4,000       25.0%

=== Averages ===
  Average Tokens:     2,667
  Average Cost:       $0.2000
  Average Efficiency: 61.7%

=== Verdict ===
  Best Performer:  forge_1737451234567
  Worst Performer: forge_1737451234569
```

---

## Example 7: Agent Statistics Dashboard

Generate agent performance statistics over time.

```python
#!/usr/bin/env python3
"""Example 7: Agent Statistics Dashboard"""

from sessionoptimizer import SessionOptimizer, EventType
import random

def simulate_historical_sessions(optimizer, agent, days=7, sessions_per_day=3):
    """Simulate historical session data."""
    session_ids = []
    
    for day in range(days):
        for s in range(sessions_per_day):
            session_id = f"{agent.lower()}_day{day}_session{s}"
            session = optimizer.start_session(
                agent,
                f"Task {day*sessions_per_day + s}",
                "opus",
                session_id=session_id
            )
            
            # Random token usage
            tokens = random.randint(1000, 5000)
            optimizer.log_event(session.id, EventType.FILE_READ, 
                              tokens_input=tokens)
            
            # Random inefficiency
            if random.random() < 0.3:  # 30% chance of repeated read
                optimizer.log_event(session.id, EventType.FILE_READ,
                                  data={"file_path": "repeated.py"},
                                  tokens_input=500)
                optimizer.log_event(session.id, EventType.FILE_READ,
                                  data={"file_path": "repeated.py"},
                                  tokens_input=500)
            
            optimizer.end_session(session.id)
            session_ids.append(session_id)
    
    return session_ids

def main():
    optimizer = SessionOptimizer()
    
    # Simulate 7 days of sessions for FORGE
    print("Simulating 21 historical sessions for FORGE...")
    simulate_historical_sessions(optimizer, "FORGE", days=7, sessions_per_day=3)
    
    # Get statistics
    stats = optimizer.agent_statistics("FORGE", days=30)
    
    print("\n" + "=" * 50)
    print(f"  FORGE PERFORMANCE DASHBOARD (Last 30 Days)")
    print("=" * 50)
    
    print(f"\n  Sessions Analyzed:   {stats['sessions_analyzed']}")
    print(f"  Total Tokens:        {stats['total_tokens']:,}")
    print(f"  Total Cost:          ${stats['total_cost']:.2f}")
    print(f"  Average Efficiency:  {stats['average_efficiency']:.1f}%")
    print(f"  Total Waste:         {stats['total_waste_tokens']:,} tokens")
    print(f"  Waste Cost:          ${stats['total_waste_cost']:.4f}")
    
    if stats.get('common_issues'):
        print(f"\n  Most Common Issues:")
        for issue in stats['common_issues']:
            print(f"    - {issue['type']}: {issue['count']} occurrences")
    
    if stats.get('trends'):
        trends = stats['trends']
        print(f"\n  Efficiency Trend: {trends['trend'].upper()}")
        if 'change_percent' in trends:
            arrow = "^" if trends['change_percent'] > 0 else "v"
            print(f"    {arrow} {abs(trends['change_percent']):.1f}% change")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
```

**Output:**
```
Simulating 21 historical sessions for FORGE...

==================================================
  FORGE PERFORMANCE DASHBOARD (Last 30 Days)
==================================================

  Sessions Analyzed:   21
  Total Tokens:        56,234
  Total Cost:          $4.22
  Average Efficiency:  78.3%
  Total Waste:         6,500 tokens
  Waste Cost:          $0.4875

  Most Common Issues:
    - REPEATED_FILE_READ: 7 occurrences

  Efficiency Trend: STABLE
    ^ 2.3% change

==================================================
```

---

## Example 8: Export Reports

Export optimization reports in different formats.

```python
#!/usr/bin/env python3
"""Example 8: Export Reports"""

from sessionoptimizer import SessionOptimizer, EventType
import json

def main():
    optimizer = SessionOptimizer()
    
    # Create a session with some issues
    session = optimizer.start_session("FORGE", "Export demo", "opus")
    
    # Add events that will trigger issues
    for i in range(3):
        optimizer.log_event(session.id, EventType.FILE_READ,
                          data={"file_path": "/src/repeated.py"},
                          tokens_input=500)
    
    optimizer.log_event(session.id, EventType.AI_RESPONSE, tokens_output=1500)
    optimizer.end_session(session.id)
    
    # Export as JSON
    json_report = optimizer.export_report(session.id, format="json")
    print("=== JSON Export ===")
    data = json.loads(json_report)
    print(f"Session: {data['session_id']}")
    print(f"Efficiency: {data['efficiency_score']:.1f}%")
    print(f"Issues: {len(data['issues'])}")
    
    # Export as Markdown
    md_report = optimizer.export_report(session.id, format="markdown")
    print("\n=== Markdown Export (first 30 lines) ===")
    for line in md_report.split('\n')[:30]:
        print(line)
    
    # Export as plain text
    text_report = optimizer.export_report(session.id, format="text")
    print("\n=== Text Export (first 20 lines) ===")
    for line in text_report.split('\n')[:20]:
        print(line)
    
    # Save to files
    with open("report.json", "w") as f:
        f.write(json_report)
    print("\nSaved: report.json")
    
    with open("report.md", "w") as f:
        f.write(md_report)
    print("Saved: report.md")
    
    with open("report.txt", "w") as f:
        f.write(text_report)
    print("Saved: report.txt")


if __name__ == "__main__":
    main()
```

**Output:**
```
=== JSON Export ===
Session: forge_20260121_143000
Efficiency: 66.7%
Issues: 1

=== Markdown Export (first 30 lines) ===
# Session Optimization Report

**Session ID:** forge_20260121_143000
**Generated:** 2026-01-21T14:30:05
**Agent:** FORGE
**Task:** Export demo
**Model:** opus
**Duration:** 0.0 seconds

## Summary

- **Efficiency Score:** 66.7/100
- **Total Waste Tokens:** 1,000
- **Estimated Waste Cost:** $0.0150
- **Issues Found:** 1

## Token Breakdown

| Category | Tokens |
|----------|--------|
...

=== Text Export (first 20 lines) ===
============================================================
SESSION OPTIMIZATION REPORT
============================================================

Session ID: forge_20260121_143000
Generated:  2026-01-21T14:30:05
Agent:      FORGE
Task:       Export demo
Model:      opus

----------------------------------------
SUMMARY
----------------------------------------
Efficiency Score:    66.7/100
Total Waste Tokens:  1,000
Estimated Waste:     $0.0150
Issues Found:        1

Saved: report.json
Saved: report.md
Saved: report.txt
```

---

## Example 9: Custom Analyzer

Create a custom analyzer for domain-specific patterns.

```python
#!/usr/bin/env python3
"""Example 9: Custom Analyzer"""

from sessionoptimizer import (
    SessionOptimizer,
    BaseAnalyzer,
    Session,
    EfficiencyIssue,
    IssueType,
    Severity,
    EventType,
    # Include default analyzers
    RepeatedFileReadAnalyzer,
    RepeatedSearchAnalyzer,
    LargeFileAnalyzer,
)


class GitOperationAnalyzer(BaseAnalyzer):
    """
    Custom analyzer for Git operation patterns.
    Detects inefficient Git usage like multiple status checks.
    """
    
    def analyze(self, session: Session) -> list:
        issues = []
        
        # Find all git status calls
        git_status_events = [
            e for e in session.events
            if e.event_type == EventType.TOOL_CALL
            and e.data.get("tool") == "git"
            and e.data.get("command") == "status"
        ]
        
        if len(git_status_events) > 3:
            waste = sum(e.tokens_input + e.tokens_output 
                       for e in git_status_events[3:])
            
            issues.append(EfficiencyIssue(
                id=f"excessive_git_status_{session.id[:8]}",
                issue_type=IssueType.INEFFICIENT_TOOL_USE,
                severity=Severity.LOW,
                description=f"'git status' called {len(git_status_events)} times",
                suggestion="Use GitFlow tool for efficient status tracking",
                estimated_waste_tokens=waste,
                events_involved=[e.id for e in git_status_events],
            ))
        
        return issues


class ApiCallBatchingAnalyzer(BaseAnalyzer):
    """
    Custom analyzer for API call patterns.
    Detects when multiple small API calls could be batched.
    """
    
    BATCH_THRESHOLD = 5
    
    def analyze(self, session: Session) -> list:
        issues = []
        
        # Find sequential API calls to same endpoint
        api_calls = [
            e for e in session.events
            if e.event_type == EventType.CUSTOM
            and e.data.get("type") == "api_call"
        ]
        
        # Group by endpoint
        from collections import defaultdict
        endpoint_calls = defaultdict(list)
        for call in api_calls:
            endpoint = call.data.get("endpoint", "unknown")
            endpoint_calls[endpoint].append(call)
        
        for endpoint, calls in endpoint_calls.items():
            if len(calls) >= self.BATCH_THRESHOLD:
                waste = sum(c.tokens_input for c in calls) * 0.3  # 30% overhead
                
                issues.append(EfficiencyIssue(
                    id=f"unbatched_api_{endpoint[:20]}",
                    issue_type=IssueType.INEFFICIENT_TOOL_USE,
                    severity=Severity.MEDIUM,
                    description=f"{len(calls)} sequential calls to '{endpoint}'",
                    suggestion=f"Batch API calls to {endpoint} for efficiency",
                    estimated_waste_tokens=int(waste),
                    metadata={"endpoint": endpoint, "call_count": len(calls)}
                ))
        
        return issues


def main():
    # Create optimizer with custom + default analyzers
    optimizer = SessionOptimizer(
        analyzers=[
            # Default analyzers
            RepeatedFileReadAnalyzer(),
            RepeatedSearchAnalyzer(),
            LargeFileAnalyzer(),
            # Custom analyzers
            GitOperationAnalyzer(),
            ApiCallBatchingAnalyzer(),
        ]
    )
    
    # Create session with patterns our custom analyzers detect
    session = optimizer.start_session("FORGE", "Custom analyzer demo", "opus")
    
    # Multiple git status calls
    for i in range(5):
        optimizer.log_event(
            session.id,
            EventType.TOOL_CALL,
            data={"tool": "git", "command": "status"},
            tokens_input=50,
            tokens_output=100
        )
    
    # Multiple API calls to same endpoint
    for i in range(7):
        optimizer.log_event(
            session.id,
            EventType.CUSTOM,
            data={
                "type": "api_call",
                "endpoint": "/api/users",
                "method": "GET"
            },
            tokens_input=100
        )
    
    optimizer.end_session(session.id)
    report = optimizer.analyze(session.id)
    
    print("=== Custom Analyzer Results ===")
    print(f"Efficiency Score: {report.efficiency_score:.1f}%")
    print(f"Issues Found: {len(report.issues)}")
    
    for issue in report.issues:
        print(f"\n[{issue.severity.value}] {issue.issue_type.value}")
        print(f"  {issue.description}")
        print(f"  Suggestion: {issue.suggestion}")


if __name__ == "__main__":
    main()
```

**Output:**
```
=== Custom Analyzer Results ===
Efficiency Score: 85.3%
Issues Found: 2

[LOW] INEFFICIENT_TOOL_USE
  'git status' called 5 times
  Suggestion: Use GitFlow tool for efficient status tracking

[MEDIUM] INEFFICIENT_TOOL_USE
  7 sequential calls to '/api/users'
  Suggestion: Batch API calls to /api/users for efficiency
```

---

## Example 10: Team Brain Integration

Full integration with Team Brain tools.

```python
#!/usr/bin/env python3
"""Example 10: Team Brain Integration"""

from sessionoptimizer import (
    SessionOptimizer,
    EventType,
    SessionStatus
)

# Mock imports for demonstration
# In real usage, import from actual tools:
# from synapselink import quick_send
# from tokentracker import log_session
# from contextcompressor import compress


def mock_quick_send(to, subject, content, priority="NORMAL"):
    """Mock SynapseLink function."""
    print(f"[SYNAPSE] To: {to}")
    print(f"[SYNAPSE] Subject: {subject}")
    print(f"[SYNAPSE] Priority: {priority}")


def main():
    optimizer = SessionOptimizer()
    
    # Start session for FORGE
    session = optimizer.start_session(
        agent="FORGE",
        task="Build TeamBrain integration demo",
        model="opus",
        metadata={
            "project": "SessionOptimizer",
            "priority": "HIGH",
            "requested_by": "Logan"
        }
    )
    
    print(f"Session started: {session.id}")
    
    # Simulate a realistic workflow
    
    # 1. Load context (integrate with ContextCompressor)
    optimizer.log_event(
        session.id,
        EventType.CONTEXT_LOAD,
        data={
            "source": "ContextCompressor",
            "original_tokens": 5000,
            "compressed_tokens": 1000,
            "compression_ratio": "80%"
        },
        tokens_input=1000
    )
    print("Loaded compressed context (5000 -> 1000 tokens)")
    
    # 2. Read files
    optimizer.log_event(
        session.id,
        EventType.FILE_READ,
        data={"file_path": "roadmap.md", "compressed": True},
        tokens_input=500
    )
    
    # 3. Search codebase
    optimizer.log_event(
        session.id,
        EventType.SEARCH,
        data={"query": "session tracking implementation"},
        tokens_input=100,
        tokens_output=600
    )
    
    # 4. AI thinking and response
    optimizer.log_event(
        session.id,
        EventType.THINKING,
        data={"topic": "architecture design"},
        tokens_output=800
    )
    
    optimizer.log_event(
        session.id,
        EventType.AI_RESPONSE,
        data={"type": "code_implementation"},
        tokens_output=2000
    )
    
    # 5. Write code
    optimizer.log_event(
        session.id,
        EventType.FILE_WRITE,
        data={"file_path": "sessionoptimizer.py"},
        tokens_input=100
    )
    
    # End session
    optimizer.end_session(session.id, SessionStatus.COMPLETED)
    
    # Analyze
    report = optimizer.analyze(session.id)
    
    print(f"\n=== Session Complete ===")
    print(f"Total Tokens: {session.total_tokens:,}")
    print(f"Estimated Cost: ${session.estimated_cost:.4f}")
    print(f"Efficiency Score: {report.efficiency_score:.1f}%")
    
    # Integrate with TokenTracker (mock)
    print(f"\n[TokenTracker] Logged: {session.total_tokens} tokens, ${session.estimated_cost:.4f}")
    
    # Report via SynapseLink if efficiency is low
    if report.efficiency_score < 80:
        mock_quick_send(
            "LOGAN,TEAM",
            f"Efficiency Alert: {session.id}",
            f"Session efficiency is {report.efficiency_score:.1f}%\n"
            f"Issues found: {len(report.issues)}\n"
            f"Estimated waste: {report.total_waste_tokens:,} tokens",
            priority="HIGH"
        )
    else:
        print(f"\n[STATUS] Session completed efficiently - no alerts needed")
    
    # Generate recommendations for Team Brain
    print(f"\n=== Team Brain Recommendations ===")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    if not report.recommendations:
        print("  No recommendations - session was efficient!")


if __name__ == "__main__":
    main()
```

**Output:**
```
Session started: forge_20260121_143000
Loaded compressed context (5000 -> 1000 tokens)

=== Session Complete ===
Total Tokens: 5,100
Estimated Cost: $0.2325
Efficiency Score: 100.0%

[TokenTracker] Logged: 5100 tokens, $0.2325

[STATUS] Session completed efficiently - no alerts needed

=== Team Brain Recommendations ===
  No recommendations - session was efficient!
```

---

## Summary

These 10 examples demonstrate:

1. **Basic tracking** - Start, log events, end, analyze
2. **Event types** - All available event types and their use
3. **Repeated reads** - Detection and recommendations
4. **Large files** - Compression awareness
5. **Error patterns** - Failed retry and high error rate detection
6. **Comparison** - Multi-session efficiency comparison
7. **Statistics** - Agent performance over time
8. **Export** - JSON, Markdown, and text reports
9. **Custom analyzers** - Extend with domain-specific patterns
10. **Integration** - Full Team Brain workflow

For more information, see the [README](README.md) and [CHEAT_SHEET](CHEAT_SHEET.txt).

---

**SessionOptimizer - What gets measured gets optimized.**
