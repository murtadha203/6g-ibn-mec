from enum import Enum, auto

class ControlMode(Enum):
    """
    Strategic Modes of Operation.
    The LLM selects one of these modes based on the Symbolic State.
    """
    BALANCED = "BALANCED"   # Default
    SURVIVAL = "SURVIVAL"   # Max Reliability (High Penalty for Drops)
    GREEN = "GREEN"         # Max Efficiency (High Penalty for Energy)
    
class TrafficState(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    CONGESTED = "CONGESTED"
    URLLC_SURGE = "URLLC_SURGE" # High priority traffic spike

class ReliabilityState(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING" # Approaching violation
    DANGER = "DANGER"   # Active violations

class EnergyState(Enum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    CRITICAL = "CRITICAL"

class MobilityState(Enum):
    STATIC = "STATIC"
    MODERATE = "MODERATE"
    HIGH_VELOCITY = "HIGH_VELOCITY"
