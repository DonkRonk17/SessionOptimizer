# SessionOptimizer Integration Examples

Copy-paste-ready code for integrating SessionOptimizer with Team Brain tools.

---

## Table of Contents

1. [TokenTracker Integration](#tokentracker-integration)
2. [ContextCompressor Integration](#contextcompressor-integration)
3. [ErrorRecovery Integration](#errorrecovery-integration)
4. [SynapseLink Integration](#synapselink-integration)
5. [AgentHealth Integration](#agenthealth-integration)
6. [CollabSession Integration](#collabsession-integration)
7. [PriorityQueue Integration](#priorityqueue-integration)
8. [KnowledgeSync Integration](#knowledgesync-integration)
9. [BCH Dashboard Integration](#bch-dashboard-integration)
10. [Complete Workflow Example](#complete-workflow-example)

---

## TokenTracker Integration

### Sync Token Usage

```python
#!/usr/bin/env python3
"""SessionOptimizer + TokenTracker Integration"""

from sessionoptimizer import SessionOptimizer, EventType

# Mock TokenTracker import (replace with real import)
# from tokentracker import TokenTracker, get_current_session_usage

class TokenTrackerMock:
    """Mock for demonstration"""
    def __init__(self):
        self.sessions = {}
    
    def get_session_usage(self, session_id):
        return {
            "input_tokens": 2500,
            "output_tokens": 3500,
            "model": "opus",
            "cost": 0.30
        }

tokentracker = TokenTrackerMock()
optimizer = SessionOptimizer()


def start_tracked_session(agent, task, model="opus"):
    """Start a session tracked by both tools."""
    session = optimizer.start_session(agent, task, model)
    
    # TokenTracker would also start tracking here
    # tokentracker.start_session(session.id, model)
    
    return session


def sync_token_usage(session_id):
    """Sync token counts from TokenTracker to SessionOptimizer."""
    usage = tokentracker.get_session_usage(session_id)
    
    optimizer.log_event(
        session_id,
        EventType.TOKEN_USAGE,
        data={
            "source": "TokenTracker",
            "model": usage["model"],
            "cost": usage["cost"]
        },
        tokens_input=usage["input_tokens"],
        tokens_output=usage["output_tokens"]
    )
    
    return usage


def end_tracked_session(session_id):
    """End session and sync final token counts."""
    # Sync final usage
    usage = sync_token_usage(session_id)
    
    # End SessionOptimizer tracking
    session = optimizer.end_session(session_id)
    
    # Analyze efficiency
    report = optimizer.analyze(session_id)
    
    return {
        "session": session,
        "report": report,
        "token_usage": usage
    }


# Example usage
if __name__ == "__main__":
    session = start_tracked_session("FORGE", "Plan authentication module")
    
    # Do work...
    optimizer.log_event(session.id, EventType.AI_RESPONSE, tokens_output=1000)
    
    # Sync periodically
    sync_token_usage(session.id)
    
    # End and get full report
    result = end_tracked_session(session.id)
    
    print(f"Efficiency: {result['report'].efficiency_score:.1f}%")
    print(f"Cost: ${result['token_usage']['cost']:.4f}")
```

---

## ContextCompressor Integration

### Track Compression Savings

```python
#!/usr/bin/env python3
"""SessionOptimizer + ContextCompressor Integration"""

from sessionoptimizer import SessionOptimizer, EventType
from pathlib import Path

# Mock ContextCompressor (replace with real import)
# from contextcompressor import compress_file, compress_text

class ContextCompressorMock:
    """Mock for demonstration"""
    def compress_file(self, path):
        # Simulate 70% compression
        original = 15000
        compressed = 4500
        return {
            "original_tokens": original,
            "compressed_tokens": compressed,
            "compressed_text": "...",
            "savings_percent": 70
        }

compressor = ContextCompressorMock()
optimizer = SessionOptimizer()


def read_file_compressed(session_id, file_path):
    """Read a file with compression and track in SessionOptimizer."""
    
    # Compress the file
    result = compressor.compress_file(file_path)
    
    # Log the read with compression info
    optimizer.log_event(
        session_id,
        EventType.FILE_READ,
        data={
            "file_path": str(file_path),
            "compressed": True,
            "original_tokens": result["original_tokens"],
            "compressed_tokens": result["compressed_tokens"],
            "savings_percent": result["savings_percent"]
        },
        tokens_input=result["compressed_tokens"]
    )
    
    return result["compressed_text"]


def track_compression_stats(session_id):
    """Get compression stats for a session."""
    session = optimizer.get_session(session_id)
    
    total_original = 0
    total_compressed = 0
    
    for event in session.events:
        if event.event_type == EventType.FILE_READ:
            if event.data.get("compressed"):
                total_original += event.data.get("original_tokens", 0)
                total_compressed += event.data.get("compressed_tokens", 0)
    
    if total_original > 0:
        savings_percent = ((total_original - total_compressed) / total_original) * 100
    else:
        savings_percent = 0
    
    return {
        "total_original": total_original,
        "total_compressed": total_compressed,
        "tokens_saved": total_original - total_compressed,
        "savings_percent": savings_percent
    }


# Example usage
if __name__ == "__main__":
    session = optimizer.start_session("ATLAS", "Read large codebase", "sonnet")
    
    # Read multiple files with compression
    read_file_compressed(session.id, "/src/huge_module.py")
    read_file_compressed(session.id, "/data/config.json")
    read_file_compressed(session.id, "/docs/api_reference.md")
    
    # Get compression stats
    stats = track_compression_stats(session.id)
    print(f"Tokens saved: {stats['tokens_saved']:,}")
    print(f"Savings: {stats['savings_percent']:.1f}%")
    
    optimizer.end_session(session.id)
    report = optimizer.analyze(session.id)
    
    # No "large file uncompressed" issues because we used compression!
    print(f"Efficiency: {report.efficiency_score:.1f}%")
```

---

## ErrorRecovery Integration

### Log Errors with Recovery Info

```python
#!/usr/bin/env python3
"""SessionOptimizer + ErrorRecovery Integration"""

from sessionoptimizer import SessionOptimizer, EventType
from functools import wraps

# Mock ErrorRecovery (replace with real import)
# from errorrecovery import with_recovery, RecoveryStrategy

optimizer = SessionOptimizer()


def log_error_with_recovery(session_id, error, strategy, recovered):
    """Log an error with recovery information."""
    optimizer.log_event(
        session_id,
        EventType.ERROR,
        data={
            "message": str(error),
            "error_type": type(error).__name__,
            "recovery_strategy": strategy,
            "recovered": recovered
        },
        tokens_input=50  # Estimate for error handling
    )


def with_session_tracking(session_id):
    """Decorator to track errors in SessionOptimizer."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                log_error_with_recovery(
                    session_id,
                    e,
                    strategy="none",
                    recovered=False
                )
                raise
        return wrapper
    return decorator


def retry_with_tracking(session_id, max_attempts=3):
    """Decorator for retry with SessionOptimizer tracking."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            last_error = None
            
            while attempts < max_attempts:
                try:
                    result = func(*args, **kwargs)
                    if attempts > 0:
                        # Log successful recovery
                        log_error_with_recovery(
                            session_id,
                            last_error,
                            strategy="retry",
                            recovered=True
                        )
                    return result
                except Exception as e:
                    last_error = e
                    attempts += 1
                    
                    if attempts < max_attempts:
                        # Log retry attempt
                        optimizer.log_event(
                            session_id,
                            EventType.CUSTOM,
                            data={
                                "type": "retry_attempt",
                                "attempt": attempts,
                                "max_attempts": max_attempts,
                                "error": str(e)
                            }
                        )
            
            # All retries failed
            log_error_with_recovery(
                session_id,
                last_error,
                strategy="retry",
                recovered=False
            )
            raise last_error
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    session = optimizer.start_session("BOLT", "Risky file operations", "grok")
    
    attempt_count = 0
    
    @retry_with_tracking(session.id, max_attempts=3)
    def flaky_operation():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Network timeout")
        return "Success!"
    
    try:
        result = flaky_operation()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    optimizer.end_session(session.id)
    report = optimizer.analyze(session.id)
    
    print(f"\nEfficiency: {report.efficiency_score:.1f}%")
    print(f"Issues found: {len(report.issues)}")
```

---

## SynapseLink Integration

### Send Efficiency Alerts

```python
#!/usr/bin/env python3
"""SessionOptimizer + SynapseLink Integration"""

from sessionoptimizer import SessionOptimizer, EventType, analyze_session
import json
from pathlib import Path
from datetime import datetime

# Mock SynapseLink (replace with real import)
# from synapselink import quick_send

def mock_quick_send(to, subject, content, priority="NORMAL"):
    """Mock SynapseLink function for demonstration."""
    message = {
        "to": to,
        "subject": subject,
        "content": content,
        "priority": priority,
        "timestamp": datetime.now().isoformat()
    }
    print(f"\n[SYNAPSE MESSAGE]")
    print(f"To: {to}")
    print(f"Subject: {subject}")
    print(f"Priority: {priority}")
    print(f"Content:\n{content}")
    return message

quick_send = mock_quick_send
optimizer = SessionOptimizer()


def send_efficiency_alert(session_id, threshold=70):
    """Send Synapse alert if efficiency is below threshold."""
    session = optimizer.get_session(session_id)
    report = optimizer.analyze(session_id)
    
    if report.efficiency_score < threshold:
        # Build alert content
        content = f"""Session Efficiency Alert

Agent: {session.agent}
Task: {session.task}
Session ID: {session_id}

Efficiency Score: {report.efficiency_score:.1f}% (threshold: {threshold}%)
Issues Found: {len(report.issues)}
Wasted Tokens: {report.total_waste_tokens:,}
Estimated Waste Cost: ${report.total_waste_cost:.4f}

Top Issues:"""
        
        for i, issue in enumerate(report.issues[:3], 1):
            content += f"\n  {i}. [{issue.severity.value}] {issue.description}"
        
        content += "\n\nRecommendations:"
        for rec in report.recommendations[:3]:
            content += f"\n  - {rec}"
        
        # Send via SynapseLink
        quick_send(
            "LOGAN,TEAM",
            f"Low Efficiency: {session.agent} - {report.efficiency_score:.0f}%",
            content,
            priority="HIGH" if report.efficiency_score < 50 else "MEDIUM"
        )
        
        return True
    
    return False


def send_session_summary(session_id):
    """Send session summary via Synapse."""
    session = optimizer.get_session(session_id)
    report = optimizer.analyze(session_id)
    
    content = f"""Session Complete

Agent: {session.agent}
Task: {session.task}
Model: {session.model}

Results:
- Efficiency: {report.efficiency_score:.1f}%
- Total Tokens: {session.total_tokens:,}
- Estimated Cost: ${session.estimated_cost:.4f}
- Issues: {len(report.issues)}

Token Breakdown:
- Input: {session.total_tokens_input:,}
- Output: {session.total_tokens_output:,}
"""
    
    quick_send(
        "TEAM",
        f"Session Complete: {session.agent} - {session.task[:30]}",
        content,
        priority="LOW"
    )


# Example usage
if __name__ == "__main__":
    session = optimizer.start_session("FORGE", "Plan inefficient task", "opus")
    
    # Simulate inefficient session (repeated file reads)
    for i in range(4):
        optimizer.log_event(
            session.id,
            EventType.FILE_READ,
            data={"file_path": "/src/same_file.py"},
            tokens_input=500
        )
    
    optimizer.end_session(session.id)
    
    # Check and alert if needed
    alert_sent = send_efficiency_alert(session.id, threshold=70)
    
    if not alert_sent:
        send_session_summary(session.id)
```

---

## AgentHealth Integration

### Report Session Health

```python
#!/usr/bin/env python3
"""SessionOptimizer + AgentHealth Integration"""

from sessionoptimizer import SessionOptimizer, EventType
from datetime import datetime

# Mock AgentHealth (replace with real import)
# from agenthealth import AgentHealth, report_health

class AgentHealthMock:
    """Mock for demonstration."""
    def report_session(self, agent, metrics):
        print(f"\n[AGENT HEALTH] {agent}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

agent_health = AgentHealthMock()
optimizer = SessionOptimizer()


def report_session_to_health(session_id):
    """Report session metrics to AgentHealth."""
    session = optimizer.get_session(session_id)
    report = optimizer.analyze(session_id)
    
    # Calculate additional metrics
    error_events = [e for e in session.events if e.event_type == EventType.ERROR]
    error_rate = len(error_events) / len(session.events) if session.events else 0
    
    metrics = {
        "session_id": session_id,
        "efficiency_score": report.efficiency_score,
        "total_tokens": session.total_tokens,
        "waste_tokens": report.total_waste_tokens,
        "issue_count": len(report.issues),
        "error_rate": error_rate * 100,
        "duration_seconds": session.duration_seconds,
        "estimated_cost": session.estimated_cost,
        "status": session.status.value,
        "timestamp": datetime.now().isoformat()
    }
    
    agent_health.report_session(session.agent, metrics)
    
    return metrics


def get_health_summary(agent, days=7):
    """Get health summary combining SessionOptimizer and AgentHealth data."""
    stats = optimizer.agent_statistics(agent, days)
    
    summary = {
        "agent": agent,
        "period_days": days,
        "sessions": stats.get("sessions_analyzed", 0),
        "avg_efficiency": stats.get("average_efficiency", 0),
        "total_waste": stats.get("total_waste_tokens", 0),
        "trend": stats.get("trends", {}).get("trend", "unknown"),
        "common_issues": stats.get("common_issues", [])
    }
    
    # Health status based on efficiency
    if summary["avg_efficiency"] >= 85:
        summary["health_status"] = "HEALTHY"
    elif summary["avg_efficiency"] >= 70:
        summary["health_status"] = "WARNING"
    else:
        summary["health_status"] = "CRITICAL"
    
    return summary


# Example usage
if __name__ == "__main__":
    # Simulate several sessions
    for i in range(3):
        session = optimizer.start_session("ATLAS", f"Task {i}", "sonnet",
                                         session_id=f"health_demo_{i}")
        optimizer.log_event(session.id, EventType.FILE_READ, tokens_input=500)
        optimizer.log_event(session.id, EventType.AI_RESPONSE, tokens_output=1000)
        optimizer.end_session(session.id)
        
        # Report to AgentHealth
        report_session_to_health(session.id)
    
    # Get health summary
    summary = get_health_summary("ATLAS", days=7)
    
    print(f"\n=== ATLAS Health Summary ===")
    print(f"Status: {summary['health_status']}")
    print(f"Avg Efficiency: {summary['avg_efficiency']:.1f}%")
    print(f"Trend: {summary['trend']}")
```

---

## CollabSession Integration

### Track Multi-Agent Sessions

```python
#!/usr/bin/env python3
"""SessionOptimizer + CollabSession Integration"""

from sessionoptimizer import SessionOptimizer, EventType
from datetime import datetime

# Mock CollabSession (replace with real import)
# from collabsession import CollabSession, start_collab

optimizer = SessionOptimizer()


class CollabSessionTracker:
    """Track collaborative sessions across agents."""
    
    def __init__(self, collab_id, task):
        self.collab_id = collab_id
        self.task = task
        self.agent_sessions = {}
    
    def add_agent(self, agent, role):
        """Add an agent to the collaboration with SessionOptimizer tracking."""
        session_id = f"{self.collab_id}_{agent.lower()}"
        session = optimizer.start_session(
            agent,
            f"[Collab:{self.collab_id}] {self.task} ({role})",
            metadata={
                "collab_id": self.collab_id,
                "role": role
            }
        )
        self.agent_sessions[agent] = session
        return session
    
    def log_agent_event(self, agent, event_type, **kwargs):
        """Log event for a specific agent in the collab."""
        session = self.agent_sessions.get(agent)
        if session:
            optimizer.log_event(session.id, event_type, **kwargs)
    
    def log_handoff(self, from_agent, to_agent, context):
        """Log a handoff between agents."""
        # Log on source agent
        self.log_agent_event(
            from_agent,
            EventType.CUSTOM,
            data={
                "type": "handoff_out",
                "to_agent": to_agent,
                "context_size": len(str(context))
            }
        )
        
        # Log on destination agent
        self.log_agent_event(
            to_agent,
            EventType.CUSTOM,
            data={
                "type": "handoff_in",
                "from_agent": from_agent,
                "context_size": len(str(context))
            },
            tokens_input=len(str(context)) // 4  # Rough estimate
        )
    
    def end_collaboration(self):
        """End all agent sessions and get combined report."""
        reports = {}
        total_tokens = 0
        total_waste = 0
        
        for agent, session in self.agent_sessions.items():
            optimizer.end_session(session.id)
            report = optimizer.analyze(session.id)
            reports[agent] = report
            total_tokens += session.total_tokens
            total_waste += report.total_waste_tokens
        
        # Calculate combined efficiency
        if total_tokens > 0:
            combined_efficiency = 100 - (total_waste / total_tokens * 100)
        else:
            combined_efficiency = 100
        
        return {
            "collab_id": self.collab_id,
            "agent_reports": reports,
            "total_tokens": total_tokens,
            "total_waste": total_waste,
            "combined_efficiency": combined_efficiency
        }


# Example usage
if __name__ == "__main__":
    # Create collaborative session
    collab = CollabSessionTracker("build_feature_001", "Build authentication module")
    
    # Add agents with roles
    forge_session = collab.add_agent("FORGE", "planner")
    atlas_session = collab.add_agent("ATLAS", "builder")
    bolt_session = collab.add_agent("BOLT", "tester")
    
    # FORGE plans
    collab.log_agent_event("FORGE", EventType.THINKING, tokens_output=500)
    collab.log_agent_event("FORGE", EventType.AI_RESPONSE, 
                           data={"type": "specification"}, tokens_output=1500)
    
    # Handoff to ATLAS
    collab.log_handoff("FORGE", "ATLAS", {"spec": "auth module spec..."})
    
    # ATLAS builds
    collab.log_agent_event("ATLAS", EventType.FILE_READ, tokens_input=500)
    collab.log_agent_event("ATLAS", EventType.FILE_WRITE, tokens_output=2000)
    
    # Handoff to BOLT
    collab.log_handoff("ATLAS", "BOLT", {"code": "built code..."})
    
    # BOLT tests
    collab.log_agent_event("BOLT", EventType.TOOL_CALL, 
                           data={"tool": "pytest"}, tokens_input=100)
    
    # End collaboration
    result = collab.end_collaboration()
    
    print(f"\n=== Collaboration Summary ===")
    print(f"Collab ID: {result['collab_id']}")
    print(f"Total Tokens: {result['total_tokens']:,}")
    print(f"Combined Efficiency: {result['combined_efficiency']:.1f}%")
    
    print(f"\nAgent Breakdown:")
    for agent, report in result["agent_reports"].items():
        print(f"  {agent}: {report.efficiency_score:.1f}%")
```

---

## PriorityQueue Integration

### Track Task Execution Efficiency

```python
#!/usr/bin/env python3
"""SessionOptimizer + PriorityQueue Integration"""

from sessionoptimizer import SessionOptimizer, EventType

# Mock PriorityQueue (replace with real import)
# from priorityqueue import PriorityQueue, Task

optimizer = SessionOptimizer()


class TaskTracker:
    """Track task execution with SessionOptimizer."""
    
    def __init__(self):
        self.task_sessions = {}
    
    def start_task(self, task_id, task_title, agent, priority):
        """Start tracking a task from PriorityQueue."""
        session = optimizer.start_session(
            agent,
            f"[Task:{task_id}] {task_title}",
            metadata={
                "task_id": task_id,
                "priority": priority,
                "source": "PriorityQueue"
            }
        )
        self.task_sessions[task_id] = session
        
        # Log task start
        optimizer.log_event(
            session.id,
            EventType.CUSTOM,
            data={
                "type": "task_start",
                "task_id": task_id,
                "priority": priority
            }
        )
        
        return session
    
    def complete_task(self, task_id, success=True):
        """Complete a task and analyze efficiency."""
        session = self.task_sessions.get(task_id)
        if not session:
            return None
        
        # Log completion
        optimizer.log_event(
            session.id,
            EventType.CUSTOM,
            data={
                "type": "task_complete",
                "success": success
            }
        )
        
        optimizer.end_session(session.id)
        report = optimizer.analyze(session.id)
        
        return {
            "task_id": task_id,
            "efficiency": report.efficiency_score,
            "tokens_used": session.total_tokens,
            "issues": len(report.issues),
            "success": success
        }
    
    def get_task_efficiency_stats(self):
        """Get efficiency stats across all tracked tasks."""
        completed = []
        
        for task_id, session in self.task_sessions.items():
            if session.status.value == "COMPLETED":
                report = optimizer.analyze(session.id)
                completed.append({
                    "task_id": task_id,
                    "efficiency": report.efficiency_score,
                    "tokens": session.total_tokens
                })
        
        if not completed:
            return {"message": "No completed tasks"}
        
        avg_efficiency = sum(t["efficiency"] for t in completed) / len(completed)
        total_tokens = sum(t["tokens"] for t in completed)
        
        return {
            "completed_tasks": len(completed),
            "average_efficiency": avg_efficiency,
            "total_tokens": total_tokens,
            "tasks": completed
        }


# Example usage
if __name__ == "__main__":
    tracker = TaskTracker()
    
    # Simulate tasks from PriorityQueue
    tasks = [
        ("task_001", "Fix login bug", "BOLT", 90),
        ("task_002", "Write API docs", "ATLAS", 70),
        ("task_003", "Review PR #123", "FORGE", 80),
    ]
    
    for task_id, title, agent, priority in tasks:
        print(f"\nStarting: {title}")
        session = tracker.start_task(task_id, title, agent, priority)
        
        # Simulate work
        optimizer.log_event(session.id, EventType.FILE_READ, tokens_input=300)
        optimizer.log_event(session.id, EventType.AI_RESPONSE, tokens_output=800)
        
        # Complete task
        result = tracker.complete_task(task_id)
        print(f"  Efficiency: {result['efficiency']:.1f}%")
        print(f"  Tokens: {result['tokens_used']:,}")
    
    # Get overall stats
    stats = tracker.get_task_efficiency_stats()
    print(f"\n=== Task Efficiency Summary ===")
    print(f"Completed: {stats['completed_tasks']}")
    print(f"Avg Efficiency: {stats['average_efficiency']:.1f}%")
    print(f"Total Tokens: {stats['total_tokens']:,}")
```

---

## KnowledgeSync Integration

### Track Knowledge Operations

```python
#!/usr/bin/env python3
"""SessionOptimizer + KnowledgeSync Integration"""

from sessionoptimizer import SessionOptimizer, EventType

optimizer = SessionOptimizer()


def log_knowledge_load(session_id, knowledge_id, topic, tokens):
    """Log loading knowledge from KnowledgeSync."""
    optimizer.log_event(
        session_id,
        EventType.CONTEXT_LOAD,
        data={
            "source": "KnowledgeSync",
            "knowledge_id": knowledge_id,
            "topic": topic
        },
        tokens_input=tokens
    )


def log_knowledge_save(session_id, topic, tokens):
    """Log saving knowledge to KnowledgeSync."""
    optimizer.log_event(
        session_id,
        EventType.CUSTOM,
        data={
            "type": "knowledge_save",
            "destination": "KnowledgeSync",
            "topic": topic
        },
        tokens_output=tokens
    )


def log_knowledge_query(session_id, query, results_count, tokens_in, tokens_out):
    """Log querying the knowledge base."""
    optimizer.log_event(
        session_id,
        EventType.SEARCH,
        data={
            "source": "KnowledgeSync",
            "query": query,
            "results_count": results_count
        },
        tokens_input=tokens_in,
        tokens_output=tokens_out
    )


# Example usage
if __name__ == "__main__":
    session = optimizer.start_session("FORGE", "Use knowledge base", "opus")
    
    # Load existing knowledge
    log_knowledge_load(session.id, "kb_001", "authentication patterns", 800)
    
    # Query knowledge
    log_knowledge_query(session.id, "JWT best practices", 5, 100, 500)
    
    # Generate new knowledge
    optimizer.log_event(session.id, EventType.AI_RESPONSE, 
                       data={"type": "analysis"}, tokens_output=1200)
    
    # Save to knowledge base
    log_knowledge_save(session.id, "JWT implementation guide", 600)
    
    optimizer.end_session(session.id)
    report = optimizer.analyze(session.id)
    
    print(f"Knowledge session efficiency: {report.efficiency_score:.1f}%")
```

---

## BCH Dashboard Integration

### API Endpoints for BCH

```python
#!/usr/bin/env python3
"""SessionOptimizer API for BCH Dashboard"""

from sessionoptimizer import SessionOptimizer, SessionStatus
from datetime import datetime, timedelta
import json

optimizer = SessionOptimizer()


def api_get_current_stats():
    """GET /api/stats/current - Current efficiency stats."""
    agents = ["FORGE", "ATLAS", "BOLT", "CLIO", "NEXUS"]
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "agents": {},
        "totals": {
            "sessions": 0,
            "tokens": 0,
            "waste": 0,
            "cost": 0
        }
    }
    
    for agent in agents:
        stats = optimizer.agent_statistics(agent, days=1)
        
        if stats.get("sessions_analyzed", 0) > 0:
            result["agents"][agent] = {
                "sessions": stats["sessions_analyzed"],
                "efficiency": stats["average_efficiency"],
                "waste_tokens": stats["total_waste_tokens"],
                "total_tokens": stats["total_tokens"],
                "cost": stats["total_cost"]
            }
            
            result["totals"]["sessions"] += stats["sessions_analyzed"]
            result["totals"]["tokens"] += stats["total_tokens"]
            result["totals"]["waste"] += stats["total_waste_tokens"]
            result["totals"]["cost"] += stats["total_cost"]
    
    # Calculate team average
    if result["totals"]["tokens"] > 0:
        waste_ratio = result["totals"]["waste"] / result["totals"]["tokens"]
        result["totals"]["efficiency"] = 100 - (waste_ratio * 100)
    else:
        result["totals"]["efficiency"] = 100
    
    return result


def api_get_active_sessions():
    """GET /api/sessions/active - Currently active sessions."""
    sessions = optimizer.list_sessions(status=SessionStatus.ACTIVE, limit=50)
    
    result = []
    for session in sessions:
        result.append({
            "id": session.id,
            "agent": session.agent,
            "task": session.task,
            "model": session.model,
            "started_at": session.started_at,
            "tokens": session.total_tokens,
            "duration_seconds": session.duration_seconds
        })
    
    return {"active_sessions": result, "count": len(result)}


def api_get_efficiency_trend(days=7):
    """GET /api/stats/trend - Efficiency trend over time."""
    # Group sessions by day
    since = datetime.now() - timedelta(days=days)
    sessions = optimizer.list_sessions(since=since, limit=1000)
    
    daily_stats = {}
    
    for session in sessions:
        date = session.started_at[:10]  # YYYY-MM-DD
        
        if date not in daily_stats:
            daily_stats[date] = {"sessions": 0, "total_tokens": 0, "waste_tokens": 0}
        
        report = optimizer.analyze(session.id)
        daily_stats[date]["sessions"] += 1
        daily_stats[date]["total_tokens"] += session.total_tokens
        daily_stats[date]["waste_tokens"] += report.total_waste_tokens
    
    # Calculate daily efficiency
    trend = []
    for date in sorted(daily_stats.keys()):
        stats = daily_stats[date]
        if stats["total_tokens"] > 0:
            efficiency = 100 - (stats["waste_tokens"] / stats["total_tokens"] * 100)
        else:
            efficiency = 100
        
        trend.append({
            "date": date,
            "efficiency": round(efficiency, 1),
            "sessions": stats["sessions"],
            "tokens": stats["total_tokens"]
        })
    
    return {"trend": trend, "days": days}


# Example: simulate BCH API calls
if __name__ == "__main__":
    # Create some sample sessions
    for i in range(3):
        session = optimizer.start_session("FORGE", f"BCH demo {i}", "opus",
                                         session_id=f"bch_demo_{i}")
        optimizer.log_event(session.id, EventType.FILE_READ, tokens_input=500)
        optimizer.log_event(session.id, EventType.AI_RESPONSE, tokens_output=1000)
        optimizer.end_session(session.id)
    
    print("=== Current Stats ===")
    print(json.dumps(api_get_current_stats(), indent=2))
    
    print("\n=== Active Sessions ===")
    print(json.dumps(api_get_active_sessions(), indent=2))
    
    print("\n=== Efficiency Trend ===")
    print(json.dumps(api_get_efficiency_trend(days=7), indent=2))
```

---

## Complete Workflow Example

### Full Team Brain Session

```python
#!/usr/bin/env python3
"""Complete SessionOptimizer Workflow with All Integrations"""

from sessionoptimizer import SessionOptimizer, EventType, SessionStatus
from datetime import datetime

optimizer = SessionOptimizer()


def complete_team_brain_session():
    """
    Complete workflow demonstrating all integrations:
    1. Start session
    2. Log events with various tools
    3. Track efficiency
    4. Send alerts
    5. Report to AgentHealth
    """
    
    # === PHASE 1: Session Start ===
    print("=" * 50)
    print("PHASE 1: Starting Session")
    print("=" * 50)
    
    session = optimizer.start_session(
        "FORGE",
        "Build complete feature with Team Brain",
        "opus",
        metadata={
            "project": "BCH",
            "priority": "HIGH",
            "requested_by": "Logan"
        }
    )
    print(f"Session started: {session.id}")
    
    # === PHASE 2: Context Loading (ContextCompressor) ===
    print("\n" + "=" * 50)
    print("PHASE 2: Loading Context (ContextCompressor)")
    print("=" * 50)
    
    # Simulate compressed context load
    optimizer.log_event(
        session.id,
        EventType.CONTEXT_LOAD,
        data={
            "source": "ContextCompressor",
            "original_tokens": 10000,
            "compressed_tokens": 2500,
            "savings": "75%"
        },
        tokens_input=2500
    )
    print("Loaded compressed context: 10K -> 2.5K tokens (75% savings)")
    
    # === PHASE 3: Knowledge Query (KnowledgeSync) ===
    print("\n" + "=" * 50)
    print("PHASE 3: Knowledge Query (KnowledgeSync)")
    print("=" * 50)
    
    optimizer.log_event(
        session.id,
        EventType.SEARCH,
        data={
            "source": "KnowledgeSync",
            "query": "authentication implementation patterns",
            "results": 5
        },
        tokens_input=100,
        tokens_output=800
    )
    print("Queried knowledge base: 5 results")
    
    # === PHASE 4: Planning (FORGE role) ===
    print("\n" + "=" * 50)
    print("PHASE 4: Planning")
    print("=" * 50)
    
    optimizer.log_event(
        session.id,
        EventType.THINKING,
        data={"topic": "architecture design"},
        tokens_output=600
    )
    
    optimizer.log_event(
        session.id,
        EventType.AI_RESPONSE,
        data={"type": "specification"},
        tokens_output=2000
    )
    print("Generated specification: 2000 tokens")
    
    # === PHASE 5: Execution Tracking ===
    print("\n" + "=" * 50)
    print("PHASE 5: Tracking Execution")
    print("=" * 50)
    
    # File operations
    optimizer.log_event(
        session.id,
        EventType.FILE_READ,
        data={"file_path": "src/auth.py", "compressed": True},
        tokens_input=500
    )
    
    optimizer.log_event(
        session.id,
        EventType.FILE_WRITE,
        data={"file_path": "src/auth_v2.py"},
        tokens_output=1500
    )
    print("File operations logged")
    
    # === PHASE 6: Error Handling (ErrorRecovery) ===
    print("\n" + "=" * 50)
    print("PHASE 6: Error Handling")
    print("=" * 50)
    
    # Simulate an error that was recovered
    optimizer.log_event(
        session.id,
        EventType.ERROR,
        data={
            "message": "Test failed",
            "recovery_strategy": "retry",
            "recovered": True
        },
        tokens_input=50
    )
    print("Error occurred and recovered via retry")
    
    # === PHASE 7: Token Sync (TokenTracker) ===
    print("\n" + "=" * 50)
    print("PHASE 7: Token Sync")
    print("=" * 50)
    
    optimizer.log_event(
        session.id,
        EventType.TOKEN_USAGE,
        data={
            "source": "TokenTracker",
            "sync_point": "mid_session"
        }
    )
    print("Synced with TokenTracker")
    
    # === PHASE 8: Session End & Analysis ===
    print("\n" + "=" * 50)
    print("PHASE 8: Session End & Analysis")
    print("=" * 50)
    
    optimizer.end_session(session.id)
    report = optimizer.analyze(session.id)
    
    print(f"\nSession Results:")
    print(f"  Efficiency Score: {report.efficiency_score:.1f}%")
    print(f"  Total Tokens: {session.total_tokens:,}")
    print(f"  Estimated Cost: ${session.estimated_cost:.4f}")
    print(f"  Issues Found: {len(report.issues)}")
    
    # === PHASE 9: Reporting (SynapseLink) ===
    print("\n" + "=" * 50)
    print("PHASE 9: Team Notification")
    print("=" * 50)
    
    if report.efficiency_score < 80:
        print("[ALERT] Sending low efficiency alert via SynapseLink")
        # quick_send("TEAM", "Low Efficiency", ...)
    else:
        print("[OK] Session completed efficiently - no alerts needed")
    
    # === PHASE 10: Health Reporting (AgentHealth) ===
    print("\n" + "=" * 50)
    print("PHASE 10: Health Reporting")
    print("=" * 50)
    
    health_metrics = {
        "session_id": session.id,
        "efficiency": report.efficiency_score,
        "tokens": session.total_tokens,
        "issues": len(report.issues),
        "cost": session.estimated_cost
    }
    print(f"Reported to AgentHealth: {health_metrics}")
    
    # === SUMMARY ===
    print("\n" + "=" * 50)
    print("COMPLETE SESSION SUMMARY")
    print("=" * 50)
    
    print(f"""
Session: {session.id}
Agent: {session.agent}
Task: {session.task}
Model: {session.model}

Efficiency: {report.efficiency_score:.1f}%
Total Tokens: {session.total_tokens:,}
Cost: ${session.estimated_cost:.4f}
Waste: {report.total_waste_tokens:,} tokens

Token Breakdown:
  - Input: {session.total_tokens_input:,}
  - Output: {session.total_tokens_output:,}

Integrations Used:
  - ContextCompressor (75% savings)
  - KnowledgeSync (5 results)
  - TokenTracker (synced)
  - ErrorRecovery (1 recovery)
  - AgentHealth (reported)
""")
    
    if report.recommendations:
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    return {
        "session_id": session.id,
        "efficiency": report.efficiency_score,
        "total_tokens": session.total_tokens,
        "cost": session.estimated_cost
    }


if __name__ == "__main__":
    result = complete_team_brain_session()
    print(f"\nFinal Result: {result}")
```

---

## Summary

These integration examples show how SessionOptimizer connects with:

1. **TokenTracker** - Sync token counts
2. **ContextCompressor** - Track compression savings
3. **ErrorRecovery** - Log errors and recovery
4. **SynapseLink** - Send efficiency alerts
5. **AgentHealth** - Report session health
6. **CollabSession** - Track multi-agent work
7. **PriorityQueue** - Track task execution
8. **KnowledgeSync** - Log knowledge operations
9. **BCH Dashboard** - Provide API endpoints
10. **Complete Workflow** - All integrations together

**Q-MODE IS NOW 100% COMPLETE! 18/18 TOOLS BUILT!**

---

**Tool:** SessionOptimizer  
**Q-Mode:** #18 of 18 (THE FINAL TOOL)  
**Created:** January 21, 2026  
**Author:** Forge (Team Brain)
