#!/usr/bin/env python3
"""
Comprehensive Test Suite for SessionOptimizer
Q-Mode Tool #18 of 18 (Tier 3: Advanced Capabilities)

Tests cover:
- Session management (start, end, log events)
- Event tracking
- Efficiency analyzers (8 types)
- Optimization reports
- Token breakdown
- Agent statistics
- Session comparison
- Import/Export
- Edge cases
"""

import unittest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from sessionoptimizer import (
    SessionOptimizer,
    Session,
    SessionEvent,
    SessionStatus,
    EventType,
    IssueType,
    Severity,
    EfficiencyIssue,
    OptimizationReport,
    # Analyzers
    RepeatedFileReadAnalyzer,
    RepeatedSearchAnalyzer,
    LargeFileAnalyzer,
    ExcessiveThinkingAnalyzer,
    FailedRetryAnalyzer,
    LongSessionAnalyzer,
    HighErrorRateAnalyzer,
    TokenSpikeAnalyzer,
    # Quick functions
    start_session,
    end_session,
    log_event,
    analyze_session,
    get_agent_stats,
    get_optimizer,
)


class TestSessionBasics(unittest.TestCase):
    """Test Session data class basics."""
    
    def test_create_session(self):
        """Test basic session creation."""
        session = Session(
            id="test_001",
            agent="FORGE",
            task="Test task",
        )
        
        self.assertEqual(session.id, "test_001")
        self.assertEqual(session.agent, "FORGE")
        self.assertEqual(session.task, "Test task")
        self.assertEqual(session.status, SessionStatus.ACTIVE)
        self.assertIsNotNone(session.started_at)
    
    def test_session_add_event(self):
        """Test adding events to session."""
        session = Session(id="test_002", agent="ATLAS", task="Build tool")
        
        event = SessionEvent(
            id="event_001",
            timestamp=datetime.now().isoformat(),
            event_type=EventType.FILE_READ,
            tokens_input=500,
            tokens_output=0,
        )
        
        session.add_event(event)
        
        self.assertEqual(len(session.events), 1)
        self.assertEqual(session.total_tokens_input, 500)
    
    def test_session_total_tokens(self):
        """Test total token calculation."""
        session = Session(id="test_003", agent="FORGE", task="Analyze")
        
        session.add_event(SessionEvent(
            id="e1", timestamp="", event_type=EventType.FILE_READ,
            tokens_input=1000, tokens_output=0
        ))
        session.add_event(SessionEvent(
            id="e2", timestamp="", event_type=EventType.AI_RESPONSE,
            tokens_input=0, tokens_output=500
        ))
        
        self.assertEqual(session.total_tokens, 1500)
        self.assertEqual(session.total_tokens_input, 1000)
        self.assertEqual(session.total_tokens_output, 500)
    
    def test_session_end(self):
        """Test ending a session."""
        session = Session(id="test_004", agent="BOLT", task="Execute")
        session.end(SessionStatus.COMPLETED)
        
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertIsNotNone(session.ended_at)
    
    def test_session_to_dict(self):
        """Test session serialization."""
        session = Session(
            id="test_005",
            agent="FORGE",
            task="Serialize test",
            model="opus"
        )
        
        data = session.to_dict()
        
        self.assertEqual(data["id"], "test_005")
        self.assertEqual(data["agent"], "FORGE")
        self.assertEqual(data["model"], "opus")
        self.assertIn("status", data)
    
    def test_session_from_dict(self):
        """Test session deserialization."""
        data = {
            "id": "test_006",
            "agent": "ATLAS",
            "task": "Deserialize test",
            "status": "COMPLETED",
            "started_at": "2026-01-21T10:00:00",
            "ended_at": "2026-01-21T11:00:00",
            "events": [],
            "total_tokens_input": 1000,
            "total_tokens_output": 500,
            "total_duration_ms": 3600000,
            "model": "sonnet",
            "metadata": {},
        }
        
        session = Session.from_dict(data)
        
        self.assertEqual(session.id, "test_006")
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertEqual(session.model, "sonnet")


class TestSessionEvent(unittest.TestCase):
    """Test SessionEvent data class."""
    
    def test_create_event(self):
        """Test event creation."""
        event = SessionEvent(
            id="evt_001",
            timestamp=datetime.now().isoformat(),
            event_type=EventType.TOOL_CALL,
            data={"tool": "grep"},
            tokens_input=100,
            tokens_output=200,
        )
        
        self.assertEqual(event.id, "evt_001")
        self.assertEqual(event.event_type, EventType.TOOL_CALL)
        self.assertEqual(event.data["tool"], "grep")
    
    def test_event_to_dict(self):
        """Test event serialization."""
        event = SessionEvent(
            id="evt_002",
            timestamp="2026-01-21T12:00:00",
            event_type=EventType.FILE_WRITE,
            tokens_input=50,
        )
        
        data = event.to_dict()
        
        self.assertEqual(data["event_type"], "FILE_WRITE")
        self.assertEqual(data["tokens_input"], 50)
    
    def test_event_from_dict(self):
        """Test event deserialization."""
        data = {
            "id": "evt_003",
            "timestamp": "2026-01-21T12:00:00",
            "event_type": "SEARCH",
            "data": {"query": "test"},
            "tokens_input": 100,
            "tokens_output": 300,
            "duration_ms": 500,
        }
        
        event = SessionEvent.from_dict(data)
        
        self.assertEqual(event.event_type, EventType.SEARCH)
        self.assertEqual(event.data["query"], "test")


class TestEfficiencyIssue(unittest.TestCase):
    """Test EfficiencyIssue data class."""
    
    def test_create_issue(self):
        """Test issue creation."""
        issue = EfficiencyIssue(
            id="issue_001",
            issue_type=IssueType.REPEATED_FILE_READ,
            severity=Severity.MEDIUM,
            description="File read 3 times",
            suggestion="Use caching",
            estimated_waste_tokens=1000,
        )
        
        self.assertEqual(issue.issue_type, IssueType.REPEATED_FILE_READ)
        self.assertEqual(issue.severity, Severity.MEDIUM)
        self.assertEqual(issue.estimated_waste_tokens, 1000)
    
    def test_issue_to_dict(self):
        """Test issue serialization."""
        issue = EfficiencyIssue(
            id="issue_002",
            issue_type=IssueType.LARGE_FILE_UNCOMPRESSED,
            severity=Severity.HIGH,
            description="Large file",
            suggestion="Compress",
        )
        
        data = issue.to_dict()
        
        self.assertEqual(data["issue_type"], "LARGE_FILE_UNCOMPRESSED")
        self.assertEqual(data["severity"], "HIGH")


class TestRepeatedFileReadAnalyzer(unittest.TestCase):
    """Test RepeatedFileReadAnalyzer."""
    
    def test_detect_repeated_reads(self):
        """Test detection of repeated file reads."""
        session = Session(id="test", agent="FORGE", task="Test")
        
        # Add same file read 3 times
        for i in range(3):
            session.add_event(SessionEvent(
                id=f"read_{i}",
                timestamp="",
                event_type=EventType.FILE_READ,
                data={"file_path": "/src/main.py"},
                tokens_input=500,
            ))
        
        analyzer = RepeatedFileReadAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, IssueType.REPEATED_FILE_READ)
        self.assertEqual(issues[0].metadata["read_count"], 3)
        self.assertEqual(issues[0].estimated_waste_tokens, 1000)  # 2x500
    
    def test_no_repeated_reads(self):
        """Test no issues for unique file reads."""
        session = Session(id="test", agent="FORGE", task="Test")
        
        # Different files
        for i in range(3):
            session.add_event(SessionEvent(
                id=f"read_{i}",
                timestamp="",
                event_type=EventType.FILE_READ,
                data={"file_path": f"/src/file{i}.py"},
                tokens_input=500,
            ))
        
        analyzer = RepeatedFileReadAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 0)


class TestRepeatedSearchAnalyzer(unittest.TestCase):
    """Test RepeatedSearchAnalyzer."""
    
    def test_detect_repeated_searches(self):
        """Test detection of repeated searches."""
        session = Session(id="test", agent="ATLAS", task="Test")
        
        # Same search 3 times
        for i in range(3):
            session.add_event(SessionEvent(
                id=f"search_{i}",
                timestamp="",
                event_type=EventType.SEARCH,
                data={"query": "find user authentication"},
                tokens_input=100,
                tokens_output=500,
            ))
        
        analyzer = RepeatedSearchAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, IssueType.REPEATED_SEARCH)


class TestLargeFileAnalyzer(unittest.TestCase):
    """Test LargeFileAnalyzer."""
    
    def test_detect_large_uncompressed(self):
        """Test detection of large uncompressed files."""
        session = Session(id="test", agent="FORGE", task="Test")
        
        session.add_event(SessionEvent(
            id="read_large",
            timestamp="",
            event_type=EventType.FILE_READ,
            data={"file_path": "/src/huge.json", "compressed": False},
            tokens_input=15000,
        ))
        
        analyzer = LargeFileAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, IssueType.LARGE_FILE_UNCOMPRESSED)
        self.assertGreater(issues[0].estimated_waste_tokens, 0)
    
    def test_no_issue_compressed(self):
        """Test no issue for compressed large files."""
        session = Session(id="test", agent="FORGE", task="Test")
        
        session.add_event(SessionEvent(
            id="read_large",
            timestamp="",
            event_type=EventType.FILE_READ,
            data={"file_path": "/src/huge.json", "compressed": True},
            tokens_input=15000,
        ))
        
        analyzer = LargeFileAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 0)


class TestExcessiveThinkingAnalyzer(unittest.TestCase):
    """Test ExcessiveThinkingAnalyzer."""
    
    def test_detect_excessive_thinking(self):
        """Test detection of excessive thinking tokens."""
        session = Session(id="test", agent="FORGE", task="Test")
        
        # 60% thinking tokens (over 40% threshold)
        session.add_event(SessionEvent(
            id="think_1",
            timestamp="",
            event_type=EventType.THINKING,
            tokens_output=600,
        ))
        session.add_event(SessionEvent(
            id="response_1",
            timestamp="",
            event_type=EventType.AI_RESPONSE,
            tokens_output=400,
        ))
        
        analyzer = ExcessiveThinkingAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, IssueType.EXCESSIVE_THINKING)


class TestFailedRetryAnalyzer(unittest.TestCase):
    """Test FailedRetryAnalyzer."""
    
    def test_detect_consecutive_errors(self):
        """Test detection of consecutive failed retries."""
        session = Session(id="test", agent="BOLT", task="Test")
        
        # 3 consecutive errors
        for i in range(3):
            session.add_event(SessionEvent(
                id=f"error_{i}",
                timestamp="",
                event_type=EventType.ERROR,
                data={"message": "Connection failed"},
                tokens_input=100,
                tokens_output=50,
            ))
        
        analyzer = FailedRetryAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, IssueType.FAILED_RETRY)


class TestLongSessionAnalyzer(unittest.TestCase):
    """Test LongSessionAnalyzer."""
    
    def test_detect_long_session(self):
        """Test detection of long sessions."""
        session = Session(id="test", agent="FORGE", task="Test")
        session.total_duration_ms = 90 * 60 * 1000  # 90 minutes
        
        analyzer = LongSessionAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, IssueType.LONG_SESSION)
    
    def test_no_issue_short_session(self):
        """Test no issue for short sessions."""
        session = Session(id="test", agent="FORGE", task="Test")
        session.total_duration_ms = 30 * 60 * 1000  # 30 minutes
        
        analyzer = LongSessionAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 0)


class TestHighErrorRateAnalyzer(unittest.TestCase):
    """Test HighErrorRateAnalyzer."""
    
    def test_detect_high_error_rate(self):
        """Test detection of high error rate."""
        session = Session(id="test", agent="BOLT", task="Test")
        
        # 5 errors out of 10 events = 50%
        for i in range(5):
            session.add_event(SessionEvent(
                id=f"error_{i}",
                timestamp="",
                event_type=EventType.ERROR,
                tokens_input=50,
            ))
            session.add_event(SessionEvent(
                id=f"ok_{i}",
                timestamp="",
                event_type=EventType.TOOL_CALL,
                tokens_input=100,
            ))
        
        analyzer = HighErrorRateAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, IssueType.HIGH_ERROR_RATE)


class TestTokenSpikeAnalyzer(unittest.TestCase):
    """Test TokenSpikeAnalyzer."""
    
    def test_detect_token_spike(self):
        """Test detection of token spikes."""
        session = Session(id="test", agent="FORGE", task="Test")
        
        # Normal events (~100 tokens)
        for i in range(5):
            session.add_event(SessionEvent(
                id=f"normal_{i}",
                timestamp="",
                event_type=EventType.TOOL_CALL,
                tokens_input=100,
            ))
        
        # Spike event (1000 tokens = 10x average)
        session.add_event(SessionEvent(
            id="spike",
            timestamp="",
            event_type=EventType.FILE_READ,
            tokens_input=1000,
        ))
        
        analyzer = TokenSpikeAnalyzer()
        issues = analyzer.analyze(session)
        
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, IssueType.TOKEN_SPIKE)


class TestSessionOptimizer(unittest.TestCase):
    """Test SessionOptimizer main class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = SessionOptimizer(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_start_session(self):
        """Test starting a new session."""
        session = self.optimizer.start_session(
            agent="FORGE",
            task="Build new feature",
            model="opus"
        )
        
        self.assertIsNotNone(session.id)
        self.assertEqual(session.agent, "FORGE")
        self.assertEqual(session.model, "opus")
        self.assertEqual(session.status, SessionStatus.ACTIVE)
    
    def test_end_session(self):
        """Test ending a session."""
        session = self.optimizer.start_session("ATLAS", "Test task")
        
        ended = self.optimizer.end_session(session.id)
        
        self.assertEqual(ended.status, SessionStatus.COMPLETED)
        self.assertIsNotNone(ended.ended_at)
    
    def test_log_event(self):
        """Test logging an event."""
        session = self.optimizer.start_session("FORGE", "Build")
        
        event = self.optimizer.log_event(
            session.id,
            EventType.FILE_READ,
            data={"file_path": "/src/main.py"},
            tokens_input=500,
        )
        
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EventType.FILE_READ)
        
        # Check session updated
        updated = self.optimizer.get_session(session.id)
        self.assertEqual(updated.total_tokens_input, 500)
    
    def test_list_sessions(self):
        """Test listing sessions."""
        # Clear any existing sessions first
        for sid in list(self.optimizer._sessions.keys()):
            self.optimizer.delete_session(sid)
        
        # Use explicit unique session IDs to avoid timestamp collisions
        self.optimizer.start_session("FORGE", "Task 1", session_id="list_test_1")
        self.optimizer.start_session("ATLAS", "Task 2", session_id="list_test_2")
        self.optimizer.start_session("FORGE", "Task 3", session_id="list_test_3")
        
        all_sessions = self.optimizer.list_sessions()
        self.assertEqual(len(all_sessions), 3)
        
        forge_sessions = self.optimizer.list_sessions(agent="FORGE")
        self.assertEqual(len(forge_sessions), 2)
    
    def test_analyze_session(self):
        """Test session analysis."""
        session = self.optimizer.start_session("FORGE", "Build tool", "opus")
        
        # Add some events including inefficiencies
        for i in range(3):
            self.optimizer.log_event(
                session.id,
                EventType.FILE_READ,
                data={"file_path": "/src/same_file.py"},
                tokens_input=500,
            )
        
        report = self.optimizer.analyze(session.id)
        
        self.assertIsNotNone(report)
        self.assertIsInstance(report, OptimizationReport)
        self.assertGreater(len(report.issues), 0)
        self.assertLessEqual(report.efficiency_score, 100)
    
    def test_delete_session(self):
        """Test deleting a session."""
        session = self.optimizer.start_session("FORGE", "Delete me")
        session_id = session.id
        
        result = self.optimizer.delete_session(session_id)
        
        self.assertTrue(result)
        self.assertIsNone(self.optimizer.get_session(session_id))
    
    def test_persistence(self):
        """Test session persistence to disk."""
        session = self.optimizer.start_session("FORGE", "Persist test")
        session_id = session.id
        
        # Create new optimizer with same storage
        new_optimizer = SessionOptimizer(storage_path=self.temp_dir)
        
        loaded = new_optimizer.get_session(session_id)
        
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.agent, "FORGE")


class TestAgentStatistics(unittest.TestCase):
    """Test agent statistics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = SessionOptimizer(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_agent_statistics(self):
        """Test agent statistics generation."""
        # Clear any existing sessions first
        for sid in list(self.optimizer._sessions.keys()):
            self.optimizer.delete_session(sid)
        
        # Create multiple sessions with unique IDs
        for i in range(3):
            session = self.optimizer.start_session(
                "FORGE", f"Task {i}", "opus", 
                session_id=f"stats_test_{i}"
            )
            self.optimizer.log_event(
                session.id,
                EventType.FILE_READ,
                tokens_input=1000,
            )
            self.optimizer.end_session(session.id)
        
        stats = self.optimizer.agent_statistics("FORGE", days=30)
        
        self.assertEqual(stats["agent"], "FORGE")
        self.assertEqual(stats["sessions_analyzed"], 3)
        self.assertGreater(stats["total_tokens"], 0)


class TestSessionComparison(unittest.TestCase):
    """Test session comparison functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = SessionOptimizer(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compare_sessions(self):
        """Test comparing multiple sessions."""
        session1 = self.optimizer.start_session("FORGE", "Task A")
        self.optimizer.log_event(session1.id, EventType.FILE_READ, tokens_input=1000)
        
        session2 = self.optimizer.start_session("FORGE", "Task B")
        self.optimizer.log_event(session2.id, EventType.FILE_READ, tokens_input=2000)
        
        comparison = self.optimizer.compare_sessions([session1.id, session2.id])
        
        self.assertIn("sessions", comparison)
        self.assertIn("averages", comparison)
        self.assertIn("best_performer", comparison)
        self.assertEqual(len(comparison["sessions"]), 2)
    
    def test_compare_insufficient_sessions(self):
        """Test comparison with insufficient sessions."""
        session = self.optimizer.start_session("FORGE", "Only one")
        
        result = self.optimizer.compare_sessions([session.id])
        
        self.assertIn("error", result)


class TestExportImport(unittest.TestCase):
    """Test export and import functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = SessionOptimizer(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_json(self):
        """Test JSON export."""
        session = self.optimizer.start_session("FORGE", "Export test")
        self.optimizer.log_event(session.id, EventType.FILE_READ, tokens_input=500)
        
        output = self.optimizer.export_report(session.id, format="json")
        
        self.assertIsNotNone(output)
        data = json.loads(output)
        self.assertEqual(data["session_id"], session.id)
    
    def test_export_markdown(self):
        """Test markdown export."""
        session = self.optimizer.start_session("FORGE", "Export MD test")
        self.optimizer.log_event(session.id, EventType.FILE_READ, tokens_input=500)
        
        output = self.optimizer.export_report(session.id, format="markdown")
        
        self.assertIsNotNone(output)
        self.assertIn("# Session Optimization Report", output)
        self.assertIn("Efficiency Score", output)
    
    def test_export_text(self):
        """Test text export."""
        session = self.optimizer.start_session("ATLAS", "Export text test")
        
        output = self.optimizer.export_report(session.id, format="text")
        
        self.assertIsNotNone(output)
        self.assertIn("SESSION OPTIMIZATION REPORT", output)
    
    def test_import_session(self):
        """Test importing a session."""
        data = {
            "id": "imported_session_001",
            "agent": "NEXUS",
            "task": "Imported task",
            "status": "COMPLETED",
            "started_at": "2026-01-21T10:00:00",
            "ended_at": "2026-01-21T11:00:00",
            "events": [],
            "total_tokens_input": 5000,
            "total_tokens_output": 2000,
            "total_duration_ms": 3600000,
            "model": "grok",
            "metadata": {},
        }
        
        session = self.optimizer.import_session(data)
        
        self.assertIsNotNone(session)
        self.assertEqual(session.id, "imported_session_001")
        self.assertEqual(session.agent, "NEXUS")
        
        # Verify it's stored
        loaded = self.optimizer.get_session("imported_session_001")
        self.assertIsNotNone(loaded)


class TestQuickFunctions(unittest.TestCase):
    """Test quick/convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Reset global optimizer
        import sessionoptimizer
        sessionoptimizer._default_optimizer = SessionOptimizer(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_quick_start_session(self):
        """Test quick start_session function."""
        session = start_session("FORGE", "Quick test")
        
        self.assertIsNotNone(session)
        self.assertEqual(session.agent, "FORGE")
    
    def test_quick_end_session(self):
        """Test quick end_session function."""
        session = start_session("ATLAS", "End test")
        ended = end_session(session.id)
        
        self.assertEqual(ended.status, SessionStatus.COMPLETED)
    
    def test_quick_log_event(self):
        """Test quick log_event function."""
        session = start_session("BOLT", "Log test")
        event = log_event(session.id, EventType.TOOL_CALL, {"tool": "grep"}, 100, 200)
        
        self.assertIsNotNone(event)
        self.assertEqual(event.tokens_input, 100)
    
    def test_quick_analyze(self):
        """Test quick analyze_session function."""
        session = start_session("FORGE", "Analyze test")
        log_event(session.id, EventType.FILE_READ, {"file_path": "test.py"}, 500)
        
        report = analyze_session(session.id)
        
        self.assertIsNotNone(report)
        self.assertIsInstance(report, OptimizationReport)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = SessionOptimizer(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_nonexistent_session(self):
        """Test getting a session that doesn't exist."""
        session = self.optimizer.get_session("nonexistent_id")
        self.assertIsNone(session)
    
    def test_end_nonexistent_session(self):
        """Test ending a session that doesn't exist."""
        result = self.optimizer.end_session("nonexistent_id")
        self.assertIsNone(result)
    
    def test_log_to_nonexistent_session(self):
        """Test logging to a session that doesn't exist."""
        event = self.optimizer.log_event("nonexistent", EventType.FILE_READ)
        self.assertIsNone(event)
    
    def test_analyze_nonexistent_session(self):
        """Test analyzing a session that doesn't exist."""
        report = self.optimizer.analyze("nonexistent")
        self.assertIsNone(report)
    
    def test_delete_nonexistent_session(self):
        """Test deleting a session that doesn't exist."""
        result = self.optimizer.delete_session("nonexistent")
        self.assertFalse(result)
    
    def test_empty_session_analysis(self):
        """Test analyzing a session with no events."""
        session = self.optimizer.start_session("FORGE", "Empty")
        report = self.optimizer.analyze(session.id)
        
        self.assertIsNotNone(report)
        self.assertEqual(report.efficiency_score, 100.0)  # Perfect score for no waste
    
    def test_export_nonexistent_session(self):
        """Test exporting a report for nonexistent session."""
        output = self.optimizer.export_report("nonexistent")
        self.assertIsNone(output)
    
    def test_session_with_all_event_types(self):
        """Test session with all event types."""
        session = self.optimizer.start_session("FORGE", "All events")
        
        for event_type in EventType:
            self.optimizer.log_event(session.id, event_type, tokens_input=100)
        
        report = self.optimizer.analyze(session.id)
        
        self.assertIsNotNone(report)
        self.assertIn("TOTAL", report.token_breakdown)
    
    def test_import_invalid_data(self):
        """Test importing invalid session data."""
        invalid_data = {"missing": "required_fields"}
        result = self.optimizer.import_session(invalid_data)
        self.assertIsNone(result)


class TestTokenBreakdown(unittest.TestCase):
    """Test token breakdown calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = SessionOptimizer(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_token_breakdown_by_type(self):
        """Test token breakdown is calculated by event type."""
        session = self.optimizer.start_session("FORGE", "Breakdown test")
        
        self.optimizer.log_event(session.id, EventType.FILE_READ, tokens_input=1000)
        self.optimizer.log_event(session.id, EventType.FILE_READ, tokens_input=500)
        self.optimizer.log_event(session.id, EventType.SEARCH, tokens_input=200, tokens_output=800)
        self.optimizer.log_event(session.id, EventType.AI_RESPONSE, tokens_output=1500)
        
        report = self.optimizer.analyze(session.id)
        
        self.assertEqual(report.token_breakdown["FILE_READ"], 1500)
        self.assertEqual(report.token_breakdown["SEARCH"], 1000)
        self.assertEqual(report.token_breakdown["AI_RESPONSE"], 1500)
        self.assertEqual(report.token_breakdown["TOTAL"], 4000)


class TestRecommendations(unittest.TestCase):
    """Test recommendation generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = SessionOptimizer(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_recommendations_for_repeated_reads(self):
        """Test recommendations are generated for repeated file reads."""
        session = self.optimizer.start_session("FORGE", "Recommendations test")
        
        # Trigger repeated file read issue
        for i in range(3):
            self.optimizer.log_event(
                session.id,
                EventType.FILE_READ,
                data={"file_path": "/same/file.py"},
                tokens_input=500,
            )
        
        report = self.optimizer.analyze(session.id)
        
        self.assertGreater(len(report.recommendations), 0)
        # Should recommend MemoryBridge or caching
        has_caching_rec = any("cache" in r.lower() or "MemoryBridge" in r for r in report.recommendations)
        self.assertTrue(has_caching_rec)


class TestCostCalculation(unittest.TestCase):
    """Test cost calculation for different models."""
    
    def test_opus_cost(self):
        """Test cost calculation for Opus model."""
        session = Session(id="test", agent="FORGE", task="Test", model="opus")
        session.total_tokens_input = 10000
        session.total_tokens_output = 5000
        
        # Opus: $0.015/1K input, $0.075/1K output
        expected = (10000/1000 * 0.015) + (5000/1000 * 0.075)  # 0.15 + 0.375 = 0.525
        
        self.assertAlmostEqual(session.estimated_cost, expected, places=4)
    
    def test_sonnet_cost(self):
        """Test cost calculation for Sonnet model."""
        session = Session(id="test", agent="ATLAS", task="Test", model="sonnet")
        session.total_tokens_input = 10000
        session.total_tokens_output = 5000
        
        # Sonnet: $0.003/1K input, $0.015/1K output
        expected = (10000/1000 * 0.003) + (5000/1000 * 0.015)  # 0.03 + 0.075 = 0.105
        
        self.assertAlmostEqual(session.estimated_cost, expected, places=4)
    
    def test_grok_free(self):
        """Test cost calculation for Grok (free)."""
        session = Session(id="test", agent="BOLT", task="Test", model="grok")
        session.total_tokens_input = 10000
        session.total_tokens_output = 5000
        
        self.assertEqual(session.estimated_cost, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
