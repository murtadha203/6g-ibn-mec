"""
Supervisor Agent: Rule-Based Arbiter for Multi-Agent Coordination.
Coordinates handover and offloading decisions to prevent destructive interference.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    """Priority levels for agent actions."""
    CRITICAL = 3    # Outage imminent or deadline < 10ms
    HIGH = 2        # Poor signal or deadline < 50ms
    NORMAL = 1      # Standard operation
    LOW = 0         # Can be delayed


@dataclass
class AgentRequest:
    """A request from an agent to take action."""
    agent_type: str           # "ho" or "mec"
    action: Any               # Cell ID for HO, target for MEC
    priority: Priority
    context: Dict[str, Any]   # Supporting information


class SupervisorAgent:
    """
    Rule-based arbiter that coordinates HO and MEC agents.
    
    Core Rules:
    1. NEVER handover during active transfer (MEC lock)
    2. NEVER start transfer if handover imminent (HO lock)
    3. Priority override: CRITICAL always wins
    4. Cooldown: Minimum gap between actions
    
    Usage:
        supervisor = SupervisorAgent()
        ho_action = supervisor.approve_ho_action(ho_proposed_action, context)
        mec_action = supervisor.approve_mec_action(mec_proposed_action, context, task)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Thresholds
        self.outage_rsrp_dbm = config.get("outage_rsrp_dbm", -100.0)
        self.poor_signal_rsrp_dbm = config.get("poor_signal_rsrp_dbm", -95.0)
        self.critical_deadline_s = config.get("critical_deadline_s", 0.010)  # 10ms
        self.urgent_deadline_s = config.get("urgent_deadline_s", 0.050)      # 50ms
        
        # Cooldowns - TUNED: Less aggressive blocking
        self.ho_cooldown_s = config.get("ho_cooldown_s", 0.2)      # 200ms between HOs (was 500ms)
        self.mec_lock_duration_s = config.get("mec_lock_duration_s", 0.02)  # 20ms lock (was 100ms)
        
        # State tracking
        self.last_ho_time_s = -10.0          # Time of last handover
        self.active_transfer_start_s = None  # When current MEC transfer started
        self.active_transfer_deadline_s = None
        self.pending_ho_blocked = False      # Track if we blocked a needed HO
        
        # Statistics
        self.stats = {
            "ho_approved": 0,
            "ho_blocked": 0,
            "mec_approved": 0,
            "mec_delayed": 0,
            "conflicts_resolved": 0
        }
    
    def reset(self):
        """Reset state for new episode."""
        self.last_ho_time_s = -10.0
        self.active_transfer_start_s = None
        self.active_transfer_deadline_s = None
        self.pending_ho_blocked = False
    
    # Handover Approval
    
    def approve_ho_action(
        self, 
        proposed_cell: int, 
        current_cell: int,
        context: Dict[str, Any]
    ) -> int:
        """
        Approve or modify HO agent's proposed action.
        
        Rules:
        1. If no change proposed, always approve
        2. If MEC transfer active, block unless signal critical
        3. If cooldown not expired, block unless signal critical
        4. If signal critical (outage imminent), always approve
        
        Returns:
            Approved cell ID (may be current_cell if blocked)
        """
        time_s = context.get("time_s", 0.0)
        serving_rsrp = context.get("serving_rsrp_dbm", -80.0)
        
        # No change requested - approve
        if proposed_cell == current_cell:
            return current_cell
        
        # Check priority
        is_critical = serving_rsrp < self.outage_rsrp_dbm
        is_poor = serving_rsrp < self.poor_signal_rsrp_dbm
        
        # Rule 1: Critical signal always wins
        if is_critical:
            self._complete_ho(time_s)
            return proposed_cell
        
        # Rule 2: Block during active MEC transfer
        if self._is_transfer_active(time_s):
            self.stats["ho_blocked"] += 1
            self.pending_ho_blocked = True
            return current_cell  # Stay in current cell
        
        # Rule 3: Cooldown check (unless poor signal)
        if not is_poor and (time_s - self.last_ho_time_s) < self.ho_cooldown_s:
            self.stats["ho_blocked"] += 1
            return current_cell
        
        # Approve handover
        self._complete_ho(time_s)
        return proposed_cell
    
    def _complete_ho(self, time_s: float):
        """Record completed handover."""
        self.last_ho_time_s = time_s
        self.pending_ho_blocked = False
        self.stats["ho_approved"] += 1
    
    # MEC Approval
    
    def approve_mec_action(
        self,
        proposed_target: str,
        context: Dict[str, Any],
        task: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Approve or modify MEC agent's proposed action.
        
        Rules:
        1. If handover was just blocked, prefer local (network unstable)
        2. If handover imminent (poor signal), prefer local
        3. Otherwise, approve proposed action
        
        Returns:
            Approved offload target ("local", "edge", "cloud")
        """
        time_s = context.get("time_s", 0.0)
        serving_rsrp = context.get("serving_rsrp_dbm", -80.0)
        
        # Check if handover is imminent
        ho_imminent = serving_rsrp < self.poor_signal_rsrp_dbm or self.pending_ho_blocked
        
        # Determine task urgency
        deadline_s = 1.0  # Default
        if task:
            deadline_s = task.get("deadline_s", 1.0)
        
        is_urgent = deadline_s < self.urgent_deadline_s
        is_critical = deadline_s < self.critical_deadline_s
        
        # Rule 1: If network unstable and task not critical, use local
        if ho_imminent and not is_critical:
            if proposed_target in ["edge", "cloud"]:
                self.stats["mec_delayed"] += 1
                self.stats["conflicts_resolved"] += 1
                return "local"  # Safer choice during instability
        
        # Rule 2: Lock the network for this transfer
        if proposed_target in ["edge", "cloud"]:
            self._start_transfer(time_s, deadline_s)
        
        self.stats["mec_approved"] += 1
        return proposed_target
    
    def _start_transfer(self, time_s: float, deadline_s: float):
        """Mark start of network transfer."""
        self.active_transfer_start_s = time_s
        # Transfer window is min of deadline or lock duration
        self.active_transfer_deadline_s = time_s + min(deadline_s, self.mec_lock_duration_s)
    
    def _is_transfer_active(self, time_s: float) -> bool:
        """Check if a network transfer is currently active."""
        if self.active_transfer_start_s is None:
            return False
        if self.active_transfer_deadline_s is None:
            return False
        return time_s < self.active_transfer_deadline_s
    
    def complete_transfer(self):
        """Mark transfer as complete (call after task execution)."""
        self.active_transfer_start_s = None
        self.active_transfer_deadline_s = None
    
    # Utilities
    
    def get_stats(self) -> Dict[str, int]:
        """Return coordination statistics."""
        return self.stats.copy()
    
    def __repr__(self):
        return (f"SupervisorAgent(ho_approved={self.stats['ho_approved']}, "
                f"ho_blocked={self.stats['ho_blocked']}, "
                f"conflicts_resolved={self.stats['conflicts_resolved']})")


