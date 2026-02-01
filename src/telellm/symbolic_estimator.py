from .schemas import TrafficState, ReliabilityState, EnergyState, MobilityState
from typing import Dict

class SymbolicEstimator:
    """
    The 'Neuro-Symbolic Interface Layer'.
    Converts continuous noisy KPIs into discrete symbolic states for the LLM.
    """
    
    def estimate(self, metrics: dict) -> dict:
        """
        Input: metrics dict from Simulation (e.g., {'avg_latency': 45.3, 'rlf_rate': 0.02, ...})
        Output: Dict of Symbolic States
        """
        return {
            "traffic": self._estimate_traffic(metrics),
            "reliability": self._estimate_reliability(metrics),
            "energy": self._estimate_energy(metrics),
            "mobility": self._estimate_mobility(metrics)
        }

    def _estimate_traffic(self, m):
        # Thresholds could be dynamic, but purely symbolic for now
        # avg_load: 0.0 - 1.0 (normalized)
        load = m.get('avg_load', 0.0)
        arrival_rate = m.get('arrival_rate', 0.0)
        
        if arrival_rate > 50.0: # Arbitrary high surge
            return TrafficState.URLLC_SURGE
        if load > 0.8:
            return TrafficState.CONGESTED
        if load < 0.2:
            return TrafficState.LOW
        return TrafficState.NORMAL

    def _estimate_reliability(self, metrics: Dict[str, float]) -> ReliabilityState:
        # Check explicit RSRP floor (Leading Indicator for Cell Failure)
        if metrics.get('rsrp', -80.0) < -110.0:
            return ReliabilityState.DANGER

        if metrics.get('rlf_rate', 0.0) > 0.05:
            return ReliabilityState.DANGER
        if metrics.get('avg_latency_ms', 0.0) > 50.0:
            return ReliabilityState.WARNING
        return ReliabilityState.SAFE

    def _estimate_energy(self, m):
        batt = m.get('ue_battery_percent', 100.0)
        
        if batt < 20.0:
            return EnergyState.CRITICAL
        if batt < 50.0:
            return EnergyState.LOW
        return EnergyState.NORMAL

    def _estimate_mobility(self, m):
        vel = m.get('avg_velocity_kmh', 0.0)
        
        if vel > 60.0:
            return MobilityState.HIGH_VELOCITY
        if vel > 10.0:
            return MobilityState.MODERATE
        return MobilityState.STATIC
