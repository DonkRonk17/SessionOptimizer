#!/usr/bin/env python3
"""
SessionOptimizer - AI Session Efficiency Analyzer
Q-Mode Tool #18 of 18 (Tier 3: Advanced Capabilities)

Analyze and optimize agent session efficiency:
- Session replay and analysis
- Token usage breakdown
- Inefficiency detection
- Optimization suggestions

Goal: 20%+ reduction in average tokens per task

Part of Team Brain toolkit for Logan Smith / Metaphy LLC
"""

import json
import os
import re
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
from pathlib import Path
from collections import defaultdict
import statistics


__version__ = "1.0.0"
__author__ = "Forge (Team Brain)"


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class SessionStatus(Enum):
    """Session lifecycle status."""
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


class EventType(Enum):
    """Types of session events."""
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    TOOL_CALL = "TOOL_CALL"
    TOKEN_USAGE = "TOKEN_USAGE"
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    SEARCH = "SEARCH"
    ERROR = "ERROR"
    USER_MESSAGE = "USER_MESSAGE"
    AI_RESPONSE = "AI_RESPONSE"
    CONTEXT_LOAD = "CONTEXT_LOAD"
    THINKING = "THINKING"
    CUSTOM = "CUSTOM"


class IssueType(Enum):
    """Types of efficiency issues."""
    REPEATED_FILE_READ = "REPEATED_FILE_READ"
    REPEATED_SEARCH = "REPEATED_SEARCH"
    LARGE_FILE_UNCOMPRESSED = "LARGE_FILE_UNCOMPRESSED"
    EXCESSIVE_THINKING = "EXCESSIVE_THINKING"
    UNUSED_CONTEXT = "UNUSED_CONTEXT"
    FAILED_RETRY = "FAILED_RETRY"
    CIRCULAR_SEARCH = "CIRCULAR_SEARCH"
    LONG_SESSION = "LONG_SESSION"
    HIGH_ERROR_RATE = "HIGH_ERROR_RATE"
    INEFFICIENT_TOOL_USE = "INEFFICIENT_TOOL_USE"
    TOKEN_SPIKE = "TOKEN_SPIKE"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"


class Severity(Enum):
    """Issue severity levels."""
    LOW = "LOW"           # Minor optimization opportunity
    MEDIUM = "MEDIUM"     # Noticeable waste, should fix
    HIGH = "HIGH"         # Significant waste, fix soon
    CRITICAL = "CRITICAL" # Major waste, fix immediately


# Token cost estimates (per 1K tokens)
TOKEN_COSTS = {
    "opus": {"input": 0.015, "output": 0.075},
    "sonnet": {"input": 0.003, "output": 0.015},
    "grok": {"input": 0.0, "output": 0.0},  # Free via Cline
    "gpt4": {"input": 0.01, "output": 0.03},
    "gemini": {"input": 0.00025, "output": 0.0005},
    "default": {"input": 0.01, "output": 0.03},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SessionEvent:
    """Single event within a session."""
    id: str
    timestamp: str
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    tokens_input: int = 0
    tokens_output: int = 0
    duration_ms: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["event_type"] = self.event_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SessionEvent":
        """Create from dictionary."""
        data = data.copy()
        data["event_type"] = EventType(data["event_type"])
        return cls(**data)


@dataclass
class EfficiencyIssue:
    """Detected efficiency issue."""
    id: str
    issue_type: IssueType
    severity: Severity
    description: str
    suggestion: str
    estimated_waste_tokens: int = 0
    estimated_waste_cost: float = 0.0
    events_involved: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result["issue_type"] = self.issue_type.value
        result["severity"] = self.severity.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EfficiencyIssue":
        """Create from dictionary."""
        data = data.copy()
        data["issue_type"] = IssueType(data["issue_type"])
        data["severity"] = Severity(data["severity"])
        return cls(**data)


@dataclass
class Session:
    """Complete agent session record."""
    id: str
    agent: str
    task: str
    status: SessionStatus = SessionStatus.ACTIVE
    started_at: str = ""
    ended_at: str = ""
    events: List[SessionEvent] = field(default_factory=list)
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_duration_ms: int = 0
    model: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
    
    def add_event(self, event: SessionEvent):
        """Add event and update totals."""
        self.events.append(event)
        self.total_tokens_input += event.tokens_input
        self.total_tokens_output += event.tokens_output
        self.total_duration_ms += event.duration_ms
    
    def end(self, status: SessionStatus = SessionStatus.COMPLETED):
        """End the session."""
        self.status = status
        self.ended_at = datetime.now().isoformat()
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_tokens_input + self.total_tokens_output
    
    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD."""
        costs = TOKEN_COSTS.get(self.model.lower(), TOKEN_COSTS["default"])
        input_cost = (self.total_tokens_input / 1000) * costs["input"]
        output_cost = (self.total_tokens_output / 1000) * costs["output"]
        return input_cost + output_cost
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.total_duration_ms / 1000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent": self.agent,
            "task": self.task,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "events": [e.to_dict() for e in self.events],
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_duration_ms": self.total_duration_ms,
            "model": self.model,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = SessionStatus(data["status"])
        data["events"] = [SessionEvent.from_dict(e) for e in data.get("events", [])]
        return cls(**data)


@dataclass
class OptimizationReport:
    """Complete optimization analysis report."""
    session_id: str
    generated_at: str
    issues: List[EfficiencyIssue] = field(default_factory=list)
    total_waste_tokens: int = 0
    total_waste_cost: float = 0.0
    efficiency_score: float = 100.0
    recommendations: List[str] = field(default_factory=list)
    token_breakdown: Dict[str, int] = field(default_factory=dict)
    comparisons: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "generated_at": self.generated_at,
            "issues": [i.to_dict() for i in self.issues],
            "total_waste_tokens": self.total_waste_tokens,
            "total_waste_cost": self.total_waste_cost,
            "efficiency_score": self.efficiency_score,
            "recommendations": self.recommendations,
            "token_breakdown": self.token_breakdown,
            "comparisons": self.comparisons,
        }


# =============================================================================
# ANALYZERS
# =============================================================================

class BaseAnalyzer:
    """Base class for session analyzers."""
    
    def analyze(self, session: Session) -> List[EfficiencyIssue]:
        """Analyze session and return issues found."""
        raise NotImplementedError


class RepeatedFileReadAnalyzer(BaseAnalyzer):
    """Detect repeated reads of same file."""
    
    def analyze(self, session: Session) -> List[EfficiencyIssue]:
        issues = []
        file_reads: Dict[str, List[SessionEvent]] = defaultdict(list)
        
        for event in session.events:
            if event.event_type == EventType.FILE_READ:
                file_path = event.data.get("file_path", "")
                if file_path:
                    file_reads[file_path].append(event)
        
        for file_path, events in file_reads.items():
            if len(events) >= 2:
                waste = sum(e.tokens_input for e in events[1:])
                cost = (waste / 1000) * TOKEN_COSTS.get(
                    session.model.lower(), TOKEN_COSTS["default"]
                )["input"]
                
                issues.append(EfficiencyIssue(
                    id=f"repeated_read_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    issue_type=IssueType.REPEATED_FILE_READ,
                    severity=Severity.MEDIUM if len(events) == 2 else Severity.HIGH,
                    description=f"File '{Path(file_path).name}' read {len(events)} times",
                    suggestion="Cache file content after first read or use ContextCompressor",
                    estimated_waste_tokens=waste,
                    estimated_waste_cost=cost,
                    events_involved=[e.id for e in events],
                    metadata={"file_path": file_path, "read_count": len(events)}
                ))
        
        return issues


class RepeatedSearchAnalyzer(BaseAnalyzer):
    """Detect repeated similar searches."""
    
    def analyze(self, session: Session) -> List[EfficiencyIssue]:
        issues = []
        searches: Dict[str, List[SessionEvent]] = defaultdict(list)
        
        for event in session.events:
            if event.event_type == EventType.SEARCH:
                query = event.data.get("query", "").lower().strip()
                # Normalize query for comparison
                normalized = re.sub(r'\s+', ' ', query)
                if normalized:
                    searches[normalized].append(event)
        
        for query, events in searches.items():
            if len(events) >= 2:
                waste = sum(e.tokens_input + e.tokens_output for e in events[1:])
                cost = (waste / 1000) * TOKEN_COSTS.get(
                    session.model.lower(), TOKEN_COSTS["default"]
                )["input"]
                
                issues.append(EfficiencyIssue(
                    id=f"repeated_search_{hashlib.md5(query.encode()).hexdigest()[:8]}",
                    issue_type=IssueType.REPEATED_SEARCH,
                    severity=Severity.MEDIUM,
                    description=f"Similar search executed {len(events)} times: '{query[:50]}...'",
                    suggestion="Cache search results or refine initial search query",
                    estimated_waste_tokens=waste,
                    estimated_waste_cost=cost,
                    events_involved=[e.id for e in events],
                    metadata={"query": query, "search_count": len(events)}
                ))
        
        return issues


class LargeFileAnalyzer(BaseAnalyzer):
    """Detect large files read without compression."""
    
    LARGE_FILE_THRESHOLD = 10000  # tokens
    
    def analyze(self, session: Session) -> List[EfficiencyIssue]:
        issues = []
        
        for event in session.events:
            if event.event_type == EventType.FILE_READ:
                tokens = event.tokens_input
                compressed = event.data.get("compressed", False)
                
                if tokens >= self.LARGE_FILE_THRESHOLD and not compressed:
                    potential_savings = int(tokens * 0.6)  # 60% savings typical
                    cost = (potential_savings / 1000) * TOKEN_COSTS.get(
                        session.model.lower(), TOKEN_COSTS["default"]
                    )["input"]
                    
                    issues.append(EfficiencyIssue(
                        id=f"large_file_{event.id}",
                        issue_type=IssueType.LARGE_FILE_UNCOMPRESSED,
                        severity=Severity.HIGH,
                        description=f"Large file ({tokens:,} tokens) read without compression",
                        suggestion="Use ContextCompressor before loading large files (50-90% savings)",
                        estimated_waste_tokens=potential_savings,
                        estimated_waste_cost=cost,
                        events_involved=[event.id],
                        metadata={
                            "file_path": event.data.get("file_path", ""),
                            "tokens": tokens,
                            "potential_savings_percent": 60
                        }
                    ))
        
        return issues


class ExcessiveThinkingAnalyzer(BaseAnalyzer):
    """Detect excessive thinking/reasoning overhead."""
    
    THINKING_THRESHOLD_PERCENT = 40  # More than 40% thinking = excessive
    
    def analyze(self, session: Session) -> List[EfficiencyIssue]:
        issues = []
        
        thinking_tokens = sum(
            e.tokens_output for e in session.events 
            if e.event_type == EventType.THINKING
        )
        total_output = session.total_tokens_output
        
        if total_output > 0:
            thinking_percent = (thinking_tokens / total_output) * 100
            
            if thinking_percent > self.THINKING_THRESHOLD_PERCENT:
                excess = thinking_tokens - int(total_output * 0.3)  # 30% is acceptable
                cost = (excess / 1000) * TOKEN_COSTS.get(
                    session.model.lower(), TOKEN_COSTS["default"]
                )["output"]
                
                issues.append(EfficiencyIssue(
                    id=f"excessive_thinking_{session.id[:8]}",
                    issue_type=IssueType.EXCESSIVE_THINKING,
                    severity=Severity.MEDIUM,
                    description=f"Thinking tokens are {thinking_percent:.1f}% of output (threshold: {self.THINKING_THRESHOLD_PERCENT}%)",
                    suggestion="Use more direct approaches, provide clearer task specifications",
                    estimated_waste_tokens=max(0, excess),
                    estimated_waste_cost=max(0, cost),
                    events_involved=[e.id for e in session.events if e.event_type == EventType.THINKING],
                    metadata={
                        "thinking_tokens": thinking_tokens,
                        "total_output": total_output,
                        "thinking_percent": thinking_percent
                    }
                ))
        
        return issues


class FailedRetryAnalyzer(BaseAnalyzer):
    """Detect failed operations that were retried without fixing the cause."""
    
    def analyze(self, session: Session) -> List[EfficiencyIssue]:
        issues = []
        error_sequences: List[List[SessionEvent]] = []
        current_sequence: List[SessionEvent] = []
        
        for event in session.events:
            if event.event_type == EventType.ERROR:
                current_sequence.append(event)
            elif current_sequence:
                if len(current_sequence) >= 2:
                    error_sequences.append(current_sequence)
                current_sequence = []
        
        if len(current_sequence) >= 2:
            error_sequences.append(current_sequence)
        
        for sequence in error_sequences:
            waste = sum(e.tokens_input + e.tokens_output for e in sequence[1:])
            cost = (waste / 1000) * TOKEN_COSTS.get(
                session.model.lower(), TOKEN_COSTS["default"]
            )["input"]
            
            issues.append(EfficiencyIssue(
                id=f"failed_retry_{sequence[0].id}",
                issue_type=IssueType.FAILED_RETRY,
                severity=Severity.HIGH,
                description=f"Operation failed {len(sequence)} times consecutively",
                suggestion="Use ErrorRecovery tool for automatic fallback strategies",
                estimated_waste_tokens=waste,
                estimated_waste_cost=cost,
                events_involved=[e.id for e in sequence],
                metadata={"retry_count": len(sequence)}
            ))
        
        return issues


class LongSessionAnalyzer(BaseAnalyzer):
    """Detect sessions that run too long."""
    
    LONG_SESSION_MINUTES = 60
    
    def analyze(self, session: Session) -> List[EfficiencyIssue]:
        issues = []
        
        duration_minutes = session.duration_seconds / 60
        
        if duration_minutes > self.LONG_SESSION_MINUTES:
            # Estimate overhead: 10% token waste per 30 mins over threshold
            excess_minutes = duration_minutes - self.LONG_SESSION_MINUTES
            waste_factor = min(0.3, (excess_minutes / 30) * 0.1)
            waste_tokens = int(session.total_tokens * waste_factor)
            
            issues.append(EfficiencyIssue(
                id=f"long_session_{session.id[:8]}",
                issue_type=IssueType.LONG_SESSION,
                severity=Severity.MEDIUM if duration_minutes < 120 else Severity.HIGH,
                description=f"Session duration: {duration_minutes:.0f} minutes (threshold: {self.LONG_SESSION_MINUTES})",
                suggestion="Break into smaller, focused tasks. Use CollabSession for handoffs.",
                estimated_waste_tokens=waste_tokens,
                estimated_waste_cost=(waste_tokens / 1000) * 0.01,
                events_involved=[],
                metadata={
                    "duration_minutes": duration_minutes,
                    "threshold_minutes": self.LONG_SESSION_MINUTES
                }
            ))
        
        return issues


class HighErrorRateAnalyzer(BaseAnalyzer):
    """Detect sessions with high error rates."""
    
    ERROR_THRESHOLD_PERCENT = 20
    
    def analyze(self, session: Session) -> List[EfficiencyIssue]:
        issues = []
        
        error_count = sum(1 for e in session.events if e.event_type == EventType.ERROR)
        total_events = len(session.events)
        
        if total_events > 5 and error_count > 0:
            error_rate = (error_count / total_events) * 100
            
            if error_rate > self.ERROR_THRESHOLD_PERCENT:
                waste = sum(
                    e.tokens_input + e.tokens_output 
                    for e in session.events 
                    if e.event_type == EventType.ERROR
                )
                
                issues.append(EfficiencyIssue(
                    id=f"high_errors_{session.id[:8]}",
                    issue_type=IssueType.HIGH_ERROR_RATE,
                    severity=Severity.HIGH if error_rate > 40 else Severity.MEDIUM,
                    description=f"Error rate: {error_rate:.1f}% ({error_count}/{total_events} events)",
                    suggestion="Review task clarity, use ErrorRecovery, check environment setup",
                    estimated_waste_tokens=waste,
                    estimated_waste_cost=(waste / 1000) * 0.01,
                    events_involved=[e.id for e in session.events if e.event_type == EventType.ERROR],
                    metadata={
                        "error_count": error_count,
                        "total_events": total_events,
                        "error_rate": error_rate
                    }
                ))
        
        return issues


class TokenSpikeAnalyzer(BaseAnalyzer):
    """Detect sudden spikes in token usage."""
    
    SPIKE_MULTIPLIER = 3.0  # Event uses 3x the average
    
    def analyze(self, session: Session) -> List[EfficiencyIssue]:
        issues = []
        
        token_events = [
            (e, e.tokens_input + e.tokens_output) 
            for e in session.events 
            if (e.tokens_input + e.tokens_output) > 0
        ]
        
        if len(token_events) < 3:
            return issues
        
        tokens = [t for _, t in token_events]
        avg_tokens = statistics.mean(tokens)
        
        for event, event_tokens in token_events:
            if event_tokens > avg_tokens * self.SPIKE_MULTIPLIER:
                waste = event_tokens - int(avg_tokens * 1.5)  # 1.5x is acceptable
                
                issues.append(EfficiencyIssue(
                    id=f"token_spike_{event.id}",
                    issue_type=IssueType.TOKEN_SPIKE,
                    severity=Severity.MEDIUM,
                    description=f"Token spike: {event_tokens:,} tokens (avg: {avg_tokens:.0f})",
                    suggestion="Investigate cause - large context, verbose output, or inefficient query",
                    estimated_waste_tokens=max(0, waste),
                    estimated_waste_cost=(max(0, waste) / 1000) * 0.01,
                    events_involved=[event.id],
                    metadata={
                        "event_tokens": event_tokens,
                        "average_tokens": avg_tokens,
                        "multiplier": event_tokens / avg_tokens
                    }
                ))
        
        return issues


# =============================================================================
# SESSION OPTIMIZER
# =============================================================================

class SessionOptimizer:
    """
    Main session optimizer class.
    
    Analyzes sessions for efficiency issues and provides optimization suggestions.
    """
    
    DEFAULT_ANALYZERS: List[BaseAnalyzer] = [
        RepeatedFileReadAnalyzer(),
        RepeatedSearchAnalyzer(),
        LargeFileAnalyzer(),
        ExcessiveThinkingAnalyzer(),
        FailedRetryAnalyzer(),
        LongSessionAnalyzer(),
        HighErrorRateAnalyzer(),
        TokenSpikeAnalyzer(),
    ]
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        analyzers: Optional[List[BaseAnalyzer]] = None,
    ):
        """
        Initialize SessionOptimizer.
        
        Args:
            storage_path: Path to store session data (default: ~/.sessionoptimizer)
            analyzers: Custom analyzers to use (default: all built-in)
        """
        self.storage_path = Path(storage_path or os.path.expanduser("~/.sessionoptimizer"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.sessions_dir = self.storage_path / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.storage_path / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.analyzers = analyzers or self.DEFAULT_ANALYZERS
        
        # In-memory session cache
        self._sessions: Dict[str, Session] = {}
        self._load_sessions()
    
    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------
    
    def _load_sessions(self):
        """Load sessions from storage."""
        for file in self.sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = Session.from_dict(data)
                    self._sessions[session.id] = session
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    
    def _save_session(self, session: Session):
        """Save session to storage."""
        file_path = self.sessions_dir / f"{session.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2)
    
    def start_session(
        self,
        agent: str,
        task: str,
        model: str = "default",
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Session:
        """
        Start a new session for tracking.
        
        Args:
            agent: Agent name (e.g., "FORGE", "ATLAS")
            task: Task description
            model: Model being used (e.g., "opus", "sonnet")
            session_id: Optional custom ID
            metadata: Additional metadata
        
        Returns:
            New Session object
        """
        session_id = session_id or f"{agent.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = Session(
            id=session_id,
            agent=agent.upper(),
            task=task,
            model=model.lower(),
            metadata=metadata or {},
        )
        
        # Add start event
        session.add_event(SessionEvent(
            id=f"{session_id}_start",
            timestamp=datetime.now().isoformat(),
            event_type=EventType.SESSION_START,
            data={"agent": agent, "task": task, "model": model},
        ))
        
        self._sessions[session_id] = session
        self._save_session(session)
        
        return session
    
    def end_session(
        self,
        session_id: str,
        status: SessionStatus = SessionStatus.COMPLETED,
    ) -> Optional[Session]:
        """
        End a session and mark it complete.
        
        Args:
            session_id: Session to end
            status: Final status
        
        Returns:
            Updated Session or None if not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        session.add_event(SessionEvent(
            id=f"{session_id}_end",
            timestamp=datetime.now().isoformat(),
            event_type=EventType.SESSION_END,
            data={"status": status.value},
        ))
        
        session.end(status)
        self._save_session(session)
        
        return session
    
    def log_event(
        self,
        session_id: str,
        event_type: EventType,
        data: Optional[Dict] = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        duration_ms: int = 0,
    ) -> Optional[SessionEvent]:
        """
        Log an event to a session.
        
        Args:
            session_id: Target session
            event_type: Type of event
            data: Event data
            tokens_input: Input tokens used
            tokens_output: Output tokens used
            duration_ms: Duration in milliseconds
        
        Returns:
            Created SessionEvent or None if session not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        event = SessionEvent(
            id=f"{session_id}_{len(session.events)}_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            data=data or {},
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            duration_ms=duration_ms,
        )
        
        session.add_event(event)
        self._save_session(session)
        
        return event
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def list_sessions(
        self,
        agent: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Session]:
        """
        List sessions with optional filtering.
        
        Args:
            agent: Filter by agent name
            status: Filter by status
            since: Only sessions after this time
            limit: Maximum results
        
        Returns:
            List of matching sessions
        """
        sessions = list(self._sessions.values())
        
        if agent:
            sessions = [s for s in sessions if s.agent.upper() == agent.upper()]
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        if since:
            since_str = since.isoformat()
            sessions = [s for s in sessions if s.started_at >= since_str]
        
        # Sort by start time, newest first
        sessions.sort(key=lambda s: s.started_at, reverse=True)
        
        return sessions[:limit]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id not in self._sessions:
            return False
        
        del self._sessions[session_id]
        
        file_path = self.sessions_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
        
        return True
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    
    def analyze(self, session_id: str) -> Optional[OptimizationReport]:
        """
        Analyze a session and generate optimization report.
        
        Args:
            session_id: Session to analyze
        
        Returns:
            OptimizationReport or None if session not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        # Run all analyzers
        all_issues: List[EfficiencyIssue] = []
        for analyzer in self.analyzers:
            try:
                issues = analyzer.analyze(session)
                all_issues.extend(issues)
            except Exception:
                continue
        
        # Calculate totals
        total_waste_tokens = sum(i.estimated_waste_tokens for i in all_issues)
        total_waste_cost = sum(i.estimated_waste_cost for i in all_issues)
        
        # Calculate efficiency score (0-100)
        if session.total_tokens > 0:
            waste_percent = (total_waste_tokens / session.total_tokens) * 100
            efficiency_score = max(0, 100 - waste_percent)
        else:
            efficiency_score = 100.0
        
        # Generate token breakdown
        token_breakdown = self._calculate_token_breakdown(session)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues)
        
        # Create report
        report = OptimizationReport(
            session_id=session_id,
            generated_at=datetime.now().isoformat(),
            issues=all_issues,
            total_waste_tokens=total_waste_tokens,
            total_waste_cost=total_waste_cost,
            efficiency_score=efficiency_score,
            recommendations=recommendations,
            token_breakdown=token_breakdown,
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _calculate_token_breakdown(self, session: Session) -> Dict[str, int]:
        """Calculate token usage by event type."""
        breakdown: Dict[str, int] = defaultdict(int)
        
        for event in session.events:
            key = event.event_type.value
            breakdown[key] += event.tokens_input + event.tokens_output
        
        # Add totals
        breakdown["TOTAL_INPUT"] = session.total_tokens_input
        breakdown["TOTAL_OUTPUT"] = session.total_tokens_output
        breakdown["TOTAL"] = session.total_tokens
        
        return dict(breakdown)
    
    def _generate_recommendations(self, issues: List[EfficiencyIssue]) -> List[str]:
        """Generate prioritized recommendations from issues."""
        recommendations: List[str] = []
        
        # Group by type and sort by severity
        issue_types = defaultdict(list)
        for issue in issues:
            issue_types[issue.issue_type].append(issue)
        
        # Priority order
        priority_order = [
            (Severity.CRITICAL, "CRITICAL"),
            (Severity.HIGH, "HIGH PRIORITY"),
            (Severity.MEDIUM, "RECOMMENDED"),
            (Severity.LOW, "OPTIONAL"),
        ]
        
        for severity, label in priority_order:
            severe_issues = [
                i for i in issues if i.severity == severity
            ]
            
            if severe_issues:
                # Group similar suggestions
                seen_suggestions: Set[str] = set()
                for issue in severe_issues:
                    if issue.suggestion not in seen_suggestions:
                        recommendations.append(f"[{label}] {issue.suggestion}")
                        seen_suggestions.add(issue.suggestion)
        
        # Add general recommendations based on patterns
        if IssueType.REPEATED_FILE_READ in issue_types:
            recommendations.append(
                "[TIP] Consider using MemoryBridge for cross-session file caching"
            )
        
        if IssueType.LARGE_FILE_UNCOMPRESSED in issue_types:
            recommendations.append(
                "[TIP] ContextCompressor can reduce large files by 50-90%"
            )
        
        if IssueType.HIGH_ERROR_RATE in issue_types or IssueType.FAILED_RETRY in issue_types:
            recommendations.append(
                "[TIP] ErrorRecovery tool provides automatic retry with fallback strategies"
            )
        
        return recommendations
    
    def _save_report(self, report: OptimizationReport):
        """Save report to storage."""
        file_path = self.reports_dir / f"{report.session_id}_report.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
    
    # -------------------------------------------------------------------------
    # Comparative Analysis
    # -------------------------------------------------------------------------
    
    def compare_sessions(
        self,
        session_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple sessions for efficiency patterns.
        
        Args:
            session_ids: Sessions to compare
        
        Returns:
            Comparison data
        """
        sessions = [self._sessions.get(sid) for sid in session_ids if sid in self._sessions]
        
        if len(sessions) < 2:
            return {"error": "Need at least 2 sessions to compare"}
        
        comparison = {
            "sessions": [],
            "averages": {},
            "best_performer": None,
            "worst_performer": None,
            "recommendations": [],
        }
        
        metrics = []
        for session in sessions:
            report = self.analyze(session.id)
            session_data = {
                "id": session.id,
                "agent": session.agent,
                "task": session.task[:50],
                "total_tokens": session.total_tokens,
                "cost": session.estimated_cost,
                "efficiency_score": report.efficiency_score if report else 100,
                "issue_count": len(report.issues) if report else 0,
            }
            comparison["sessions"].append(session_data)
            metrics.append(session_data)
        
        # Calculate averages
        comparison["averages"] = {
            "tokens": statistics.mean([m["total_tokens"] for m in metrics]),
            "cost": statistics.mean([m["cost"] for m in metrics]),
            "efficiency": statistics.mean([m["efficiency_score"] for m in metrics]),
            "issues": statistics.mean([m["issue_count"] for m in metrics]),
        }
        
        # Find best/worst
        sorted_by_efficiency = sorted(metrics, key=lambda m: m["efficiency_score"], reverse=True)
        comparison["best_performer"] = sorted_by_efficiency[0]["id"]
        comparison["worst_performer"] = sorted_by_efficiency[-1]["id"]
        
        return comparison
    
    def agent_statistics(
        self,
        agent: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get statistics for an agent over time.
        
        Args:
            agent: Agent name
            days: Number of days to analyze
        
        Returns:
            Agent statistics
        """
        since = datetime.now() - timedelta(days=days)
        sessions = self.list_sessions(agent=agent, since=since, limit=1000)
        
        if not sessions:
            return {"agent": agent, "sessions": 0, "message": "No sessions found"}
        
        # Analyze all sessions
        reports = []
        for session in sessions:
            report = self.analyze(session.id)
            if report:
                reports.append((session, report))
        
        if not reports:
            return {"agent": agent, "sessions": len(sessions), "message": "No analyzable sessions"}
        
        stats = {
            "agent": agent,
            "period_days": days,
            "sessions_analyzed": len(reports),
            "total_tokens": sum(s.total_tokens for s, _ in reports),
            "total_cost": sum(s.estimated_cost for s, _ in reports),
            "average_efficiency": statistics.mean([r.efficiency_score for _, r in reports]),
            "total_waste_tokens": sum(r.total_waste_tokens for _, r in reports),
            "total_waste_cost": sum(r.total_waste_cost for _, r in reports),
            "common_issues": self._get_common_issues(reports),
            "trends": self._calculate_trends(reports),
        }
        
        return stats
    
    def _get_common_issues(
        self,
        reports: List[Tuple[Session, OptimizationReport]],
    ) -> List[Dict[str, Any]]:
        """Get most common issues across reports."""
        issue_counts: Dict[IssueType, int] = defaultdict(int)
        
        for _, report in reports:
            for issue in report.issues:
                issue_counts[issue.issue_type] += 1
        
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"type": issue_type.value, "count": count}
            for issue_type, count in sorted_issues[:5]
        ]
    
    def _calculate_trends(
        self,
        reports: List[Tuple[Session, OptimizationReport]],
    ) -> Dict[str, str]:
        """Calculate efficiency trends."""
        if len(reports) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by time
        sorted_reports = sorted(reports, key=lambda x: x[0].started_at)
        
        # Compare first half vs second half
        mid = len(sorted_reports) // 2
        first_half = sorted_reports[:mid]
        second_half = sorted_reports[mid:]
        
        first_avg = statistics.mean([r.efficiency_score for _, r in first_half])
        second_avg = statistics.mean([r.efficiency_score for _, r in second_half])
        
        change = second_avg - first_avg
        
        if change > 5:
            trend = "improving"
        elif change < -5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change_percent": round(change, 1),
            "first_period_avg": round(first_avg, 1),
            "second_period_avg": round(second_avg, 1),
        }
    
    # -------------------------------------------------------------------------
    # Export & Import
    # -------------------------------------------------------------------------
    
    def export_report(
        self,
        session_id: str,
        format: str = "json",
    ) -> Optional[str]:
        """
        Export optimization report.
        
        Args:
            session_id: Session to export report for
            format: Export format ("json", "markdown", "text")
        
        Returns:
            Exported content or None
        """
        report = self.analyze(session_id)
        if not report:
            return None
        
        session = self._sessions.get(session_id)
        
        if format == "json":
            return json.dumps(report.to_dict(), indent=2)
        
        elif format == "markdown":
            return self._report_to_markdown(session, report)
        
        elif format == "text":
            return self._report_to_text(session, report)
        
        return None
    
    def _report_to_markdown(
        self,
        session: Optional[Session],
        report: OptimizationReport,
    ) -> str:
        """Convert report to markdown."""
        lines = [
            "# Session Optimization Report",
            "",
            f"**Session ID:** {report.session_id}",
            f"**Generated:** {report.generated_at}",
        ]
        
        if session:
            lines.extend([
                f"**Agent:** {session.agent}",
                f"**Task:** {session.task}",
                f"**Model:** {session.model}",
                f"**Duration:** {session.duration_seconds:.1f} seconds",
            ])
        
        lines.extend([
            "",
            "## Summary",
            "",
            f"- **Efficiency Score:** {report.efficiency_score:.1f}/100",
            f"- **Total Waste Tokens:** {report.total_waste_tokens:,}",
            f"- **Estimated Waste Cost:** ${report.total_waste_cost:.4f}",
            f"- **Issues Found:** {len(report.issues)}",
            "",
            "## Token Breakdown",
            "",
            "| Category | Tokens |",
            "|----------|--------|",
        ])
        
        for category, tokens in sorted(report.token_breakdown.items()):
            lines.append(f"| {category} | {tokens:,} |")
        
        if report.issues:
            lines.extend([
                "",
                "## Issues Detected",
                "",
            ])
            
            for issue in sorted(report.issues, key=lambda x: x.severity.value):
                lines.extend([
                    f"### {issue.issue_type.value} [{issue.severity.value}]",
                    "",
                    f"**Description:** {issue.description}",
                    f"**Suggestion:** {issue.suggestion}",
                    f"**Estimated Waste:** {issue.estimated_waste_tokens:,} tokens (${issue.estimated_waste_cost:.4f})",
                    "",
                ])
        
        if report.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for rec in report.recommendations:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def _report_to_text(
        self,
        session: Optional[Session],
        report: OptimizationReport,
    ) -> str:
        """Convert report to plain text."""
        lines = [
            "=" * 60,
            "SESSION OPTIMIZATION REPORT",
            "=" * 60,
            "",
            f"Session ID: {report.session_id}",
            f"Generated:  {report.generated_at}",
        ]
        
        if session:
            lines.extend([
                f"Agent:      {session.agent}",
                f"Task:       {session.task}",
                f"Model:      {session.model}",
            ])
        
        lines.extend([
            "",
            "-" * 40,
            "SUMMARY",
            "-" * 40,
            f"Efficiency Score:    {report.efficiency_score:.1f}/100",
            f"Total Waste Tokens:  {report.total_waste_tokens:,}",
            f"Estimated Waste:     ${report.total_waste_cost:.4f}",
            f"Issues Found:        {len(report.issues)}",
        ])
        
        if report.issues:
            lines.extend([
                "",
                "-" * 40,
                "ISSUES",
                "-" * 40,
            ])
            
            for i, issue in enumerate(report.issues, 1):
                lines.extend([
                    f"",
                    f"{i}. [{issue.severity.value}] {issue.issue_type.value}",
                    f"   {issue.description}",
                    f"   Suggestion: {issue.suggestion}",
                ])
        
        if report.recommendations:
            lines.extend([
                "",
                "-" * 40,
                "RECOMMENDATIONS",
                "-" * 40,
            ])
            for rec in report.recommendations:
                lines.append(f"  * {rec}")
        
        return "\n".join(lines)
    
    def import_session(self, data: Dict) -> Optional[Session]:
        """
        Import a session from dictionary data.
        
        Args:
            data: Session data dictionary
        
        Returns:
            Imported Session or None on error
        """
        try:
            session = Session.from_dict(data)
            self._sessions[session.id] = session
            self._save_session(session)
            return session
        except (KeyError, ValueError):
            return None


# =============================================================================
# QUICK FUNCTIONS
# =============================================================================

# Global instance
_default_optimizer: Optional[SessionOptimizer] = None


def get_optimizer(storage_path: Optional[str] = None) -> SessionOptimizer:
    """Get or create default optimizer instance."""
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = SessionOptimizer(storage_path)
    return _default_optimizer


def start_session(
    agent: str,
    task: str,
    model: str = "default",
) -> Session:
    """Quick function to start a session."""
    return get_optimizer().start_session(agent, task, model)


def end_session(session_id: str) -> Optional[Session]:
    """Quick function to end a session."""
    return get_optimizer().end_session(session_id)


def log_event(
    session_id: str,
    event_type: EventType,
    data: Optional[Dict] = None,
    tokens_input: int = 0,
    tokens_output: int = 0,
) -> Optional[SessionEvent]:
    """Quick function to log an event."""
    return get_optimizer().log_event(
        session_id, event_type, data, tokens_input, tokens_output
    )


def analyze_session(session_id: str) -> Optional[OptimizationReport]:
    """Quick function to analyze a session."""
    return get_optimizer().analyze(session_id)


def get_agent_stats(agent: str, days: int = 30) -> Dict[str, Any]:
    """Quick function to get agent statistics."""
    return get_optimizer().agent_statistics(agent, days)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def cli():
    """Command-line interface."""
    import sys
    
    args = sys.argv[1:]
    
    if not args or args[0] in ["-h", "--help", "help"]:
        print_help()
        return
    
    command = args[0]
    
    if command == "start":
        cli_start(args[1:])
    elif command == "end":
        cli_end(args[1:])
    elif command == "log":
        cli_log(args[1:])
    elif command == "analyze":
        cli_analyze(args[1:])
    elif command == "list":
        cli_list(args[1:])
    elif command == "stats":
        cli_stats(args[1:])
    elif command == "compare":
        cli_compare(args[1:])
    elif command == "export":
        cli_export(args[1:])
    elif command == "delete":
        cli_delete(args[1:])
    elif command == "version":
        print(f"SessionOptimizer v{__version__}")
    else:
        print(f"Unknown command: {command}")
        print("Use 'sessionoptimizer help' for usage")


def print_help():
    """Print CLI help."""
    help_text = f"""
SessionOptimizer v{__version__} - AI Session Efficiency Analyzer
Q-Mode Tool #18 of 18 (Tier 3: Advanced Capabilities)

USAGE:
    sessionoptimizer <command> [options]

COMMANDS:
    start       Start a new session
    end         End an active session
    log         Log an event to a session
    analyze     Analyze a session for efficiency issues
    list        List sessions
    stats       Show agent statistics
    compare     Compare multiple sessions
    export      Export optimization report
    delete      Delete a session
    version     Show version
    help        Show this help

START SESSION:
    sessionoptimizer start <agent> <task> [--model MODEL]
    
    Examples:
        sessionoptimizer start FORGE "Build new tool"
        sessionoptimizer start ATLAS "Run tests" --model sonnet

END SESSION:
    sessionoptimizer end <session_id> [--status STATUS]
    
    Status: COMPLETED (default), FAILED, ABANDONED
    
    Example:
        sessionoptimizer end forge_20260121_143000 --status COMPLETED

LOG EVENT:
    sessionoptimizer log <session_id> <event_type> [options]
    
    Event types: TOOL_CALL, FILE_READ, FILE_WRITE, SEARCH, ERROR, 
                 TOKEN_USAGE, USER_MESSAGE, AI_RESPONSE, THINKING
    
    Options:
        --data JSON         Event data as JSON
        --tokens-in N       Input tokens used
        --tokens-out N      Output tokens used
    
    Examples:
        sessionoptimizer log forge_20260121 FILE_READ --data '{{"file_path": "src/main.py"}}' --tokens-in 500
        sessionoptimizer log forge_20260121 ERROR --data '{{"message": "File not found"}}'

ANALYZE:
    sessionoptimizer analyze <session_id> [--format FORMAT]
    
    Format: text (default), json, markdown
    
    Example:
        sessionoptimizer analyze forge_20260121_143000 --format markdown

LIST SESSIONS:
    sessionoptimizer list [options]
    
    Options:
        --agent AGENT       Filter by agent
        --status STATUS     Filter by status
        --limit N           Maximum results (default: 20)
    
    Example:
        sessionoptimizer list --agent FORGE --status COMPLETED --limit 10

AGENT STATS:
    sessionoptimizer stats <agent> [--days N]
    
    Example:
        sessionoptimizer stats FORGE --days 7

COMPARE:
    sessionoptimizer compare <session_id1> <session_id2> [...]
    
    Example:
        sessionoptimizer compare forge_20260121 forge_20260120 forge_20260119

EXPORT:
    sessionoptimizer export <session_id> [--format FORMAT] [--output FILE]
    
    Format: json, markdown, text
    
    Example:
        sessionoptimizer export forge_20260121 --format markdown --output report.md

DELETE:
    sessionoptimizer delete <session_id>
    
    Example:
        sessionoptimizer delete forge_20260121_143000

INTEGRATION:
    Python API:
        from sessionoptimizer import SessionOptimizer, start_session, analyze_session
        
        # Start tracking
        session = start_session("FORGE", "Build feature X")
        
        # Log events
        log_event(session.id, EventType.FILE_READ, {{"file_path": "main.py"}}, tokens_input=500)
        
        # Analyze
        report = analyze_session(session.id)
        print(f"Efficiency: {{report.efficiency_score}}%")

For more examples, see: https://github.com/DonkRonk17/SessionOptimizer
"""
    print(help_text)


def cli_start(args: List[str]):
    """Handle start command."""
    if len(args) < 2:
        print("Usage: sessionoptimizer start <agent> <task> [--model MODEL]")
        return
    
    agent = args[0]
    task = args[1]
    model = "default"
    
    for i, arg in enumerate(args):
        if arg == "--model" and i + 1 < len(args):
            model = args[i + 1]
    
    optimizer = get_optimizer()
    session = optimizer.start_session(agent, task, model)
    
    print(f"Session started: {session.id}")
    print(f"Agent: {session.agent}")
    print(f"Task: {session.task}")
    print(f"Model: {session.model}")


def cli_end(args: List[str]):
    """Handle end command."""
    if not args:
        print("Usage: sessionoptimizer end <session_id> [--status STATUS]")
        return
    
    session_id = args[0]
    status = SessionStatus.COMPLETED
    
    for i, arg in enumerate(args):
        if arg == "--status" and i + 1 < len(args):
            try:
                status = SessionStatus(args[i + 1].upper())
            except ValueError:
                print(f"Invalid status: {args[i + 1]}")
                return
    
    optimizer = get_optimizer()
    session = optimizer.end_session(session_id, status)
    
    if session:
        print(f"Session ended: {session_id}")
        print(f"Status: {session.status.value}")
        print(f"Total tokens: {session.total_tokens:,}")
        print(f"Estimated cost: ${session.estimated_cost:.4f}")
    else:
        print(f"Session not found: {session_id}")


def cli_log(args: List[str]):
    """Handle log command."""
    if len(args) < 2:
        print("Usage: sessionoptimizer log <session_id> <event_type> [options]")
        return
    
    session_id = args[0]
    
    try:
        event_type = EventType(args[1].upper())
    except ValueError:
        print(f"Invalid event type: {args[1]}")
        print("Valid types: " + ", ".join(e.value for e in EventType))
        return
    
    data = {}
    tokens_input = 0
    tokens_output = 0
    
    i = 2
    while i < len(args):
        if args[i] == "--data" and i + 1 < len(args):
            try:
                data = json.loads(args[i + 1])
            except json.JSONDecodeError:
                print(f"Invalid JSON: {args[i + 1]}")
                return
            i += 2
        elif args[i] == "--tokens-in" and i + 1 < len(args):
            tokens_input = int(args[i + 1])
            i += 2
        elif args[i] == "--tokens-out" and i + 1 < len(args):
            tokens_output = int(args[i + 1])
            i += 2
        else:
            i += 1
    
    optimizer = get_optimizer()
    event = optimizer.log_event(session_id, event_type, data, tokens_input, tokens_output)
    
    if event:
        print(f"Event logged: {event.id}")
        print(f"Type: {event.event_type.value}")
        if tokens_input or tokens_output:
            print(f"Tokens: {tokens_input} in / {tokens_output} out")
    else:
        print(f"Session not found: {session_id}")


def cli_analyze(args: List[str]):
    """Handle analyze command."""
    if not args:
        print("Usage: sessionoptimizer analyze <session_id> [--format FORMAT]")
        return
    
    session_id = args[0]
    format_type = "text"
    
    for i, arg in enumerate(args):
        if arg == "--format" and i + 1 < len(args):
            format_type = args[i + 1].lower()
    
    optimizer = get_optimizer()
    output = optimizer.export_report(session_id, format_type)
    
    if output:
        print(output)
    else:
        print(f"Session not found or analysis failed: {session_id}")


def cli_list(args: List[str]):
    """Handle list command."""
    agent = None
    status = None
    limit = 20
    
    i = 0
    while i < len(args):
        if args[i] == "--agent" and i + 1 < len(args):
            agent = args[i + 1]
            i += 2
        elif args[i] == "--status" and i + 1 < len(args):
            try:
                status = SessionStatus(args[i + 1].upper())
            except ValueError:
                print(f"Invalid status: {args[i + 1]}")
                return
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        else:
            i += 1
    
    optimizer = get_optimizer()
    sessions = optimizer.list_sessions(agent=agent, status=status, limit=limit)
    
    if not sessions:
        print("No sessions found")
        return
    
    print(f"{'ID':<30} {'AGENT':<10} {'STATUS':<12} {'TOKENS':>10} {'COST':>10}")
    print("-" * 75)
    
    for session in sessions:
        print(
            f"{session.id:<30} {session.agent:<10} {session.status.value:<12} "
            f"{session.total_tokens:>10,} ${session.estimated_cost:>9.4f}"
        )


def cli_stats(args: List[str]):
    """Handle stats command."""
    if not args:
        print("Usage: sessionoptimizer stats <agent> [--days N]")
        return
    
    agent = args[0]
    days = 30
    
    for i, arg in enumerate(args):
        if arg == "--days" and i + 1 < len(args):
            days = int(args[i + 1])
    
    optimizer = get_optimizer()
    stats = optimizer.agent_statistics(agent, days)
    
    print(f"\n=== {agent} Statistics ({days} days) ===\n")
    print(f"Sessions Analyzed:   {stats.get('sessions_analyzed', 0)}")
    print(f"Total Tokens:        {stats.get('total_tokens', 0):,}")
    print(f"Total Cost:          ${stats.get('total_cost', 0):.2f}")
    print(f"Average Efficiency:  {stats.get('average_efficiency', 0):.1f}%")
    print(f"Total Waste Tokens:  {stats.get('total_waste_tokens', 0):,}")
    print(f"Total Waste Cost:    ${stats.get('total_waste_cost', 0):.4f}")
    
    if "common_issues" in stats and stats["common_issues"]:
        print("\nMost Common Issues:")
        for issue in stats["common_issues"]:
            print(f"  - {issue['type']}: {issue['count']} occurrences")
    
    if "trends" in stats:
        trends = stats["trends"]
        print(f"\nTrend: {trends.get('trend', 'N/A').upper()}")
        if "change_percent" in trends:
            print(f"  Change: {trends['change_percent']:+.1f}%")


def cli_compare(args: List[str]):
    """Handle compare command."""
    if len(args) < 2:
        print("Usage: sessionoptimizer compare <session_id1> <session_id2> [...]")
        return
    
    optimizer = get_optimizer()
    comparison = optimizer.compare_sessions(args)
    
    if "error" in comparison:
        print(comparison["error"])
        return
    
    print("\n=== Session Comparison ===\n")
    print(f"{'ID':<30} {'AGENT':<10} {'TOKENS':>10} {'EFFICIENCY':>12}")
    print("-" * 65)
    
    for s in comparison["sessions"]:
        print(
            f"{s['id']:<30} {s['agent']:<10} "
            f"{s['total_tokens']:>10,} {s['efficiency_score']:>11.1f}%"
        )
    
    print("\nAverages:")
    avgs = comparison["averages"]
    print(f"  Tokens:     {avgs['tokens']:,.0f}")
    print(f"  Cost:       ${avgs['cost']:.4f}")
    print(f"  Efficiency: {avgs['efficiency']:.1f}%")
    
    print(f"\nBest Performer:  {comparison['best_performer']}")
    print(f"Worst Performer: {comparison['worst_performer']}")


def cli_export(args: List[str]):
    """Handle export command."""
    if not args:
        print("Usage: sessionoptimizer export <session_id> [--format FORMAT] [--output FILE]")
        return
    
    session_id = args[0]
    format_type = "json"
    output_file = None
    
    i = 1
    while i < len(args):
        if args[i] == "--format" and i + 1 < len(args):
            format_type = args[i + 1].lower()
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        else:
            i += 1
    
    optimizer = get_optimizer()
    output = optimizer.export_report(session_id, format_type)
    
    if not output:
        print(f"Session not found: {session_id}")
        return
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report exported to: {output_file}")
    else:
        print(output)


def cli_delete(args: List[str]):
    """Handle delete command."""
    if not args:
        print("Usage: sessionoptimizer delete <session_id>")
        return
    
    session_id = args[0]
    
    optimizer = get_optimizer()
    if optimizer.delete_session(session_id):
        print(f"Session deleted: {session_id}")
    else:
        print(f"Session not found: {session_id}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    cli()
