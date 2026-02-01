"""
Network Simulation Environment for 5G/6G Mobility and MEC Offloading.
Provides a multi-agent environment for handovers and computational offloading decisions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


# # Utility functions

# In[2]:


def dbm_to_watts(dbm: float) -> float: 
    return 10 ** ((dbm - 30) / 10.0)


# In[3]:


def watts_to_dbm(watts: float) -> float:
    if watts <= 0:
        return -200.0
    return 10 * math.log10(watts) + 30


# In[4]:


def path_loss_db(distance_m: float,
                 pl0_db: float = 43.0, # Friis reference loss for 3.5 GHz 
                 path_loss_exp: float = 3.5) -> float:
    """
    PL(d) = PL0 + 10 * n * log10(d / d0)
    """
    d = max(distance_m, 1.0)
    return pl0_db + 10 * path_loss_exp * math.log10(d)


# In[5]:


def shannon_capacity_hz(bandwidth_hz: float, sinr_linear: float) -> float:
    """C = B * log2(1 + SINR)"""
    return bandwidth_hz * math.log2(1.0 + sinr_linear)


# # Data models

# In[6]:


@dataclass
class BaseStation:
    id: int
    x: float
    y: float
    tx_power_dbm: float = 46.0  # Transmission power in dBm (~40W)
    noise_dbm: float = -93.0   # Thermal noise floor
    # Based on 3GPP TS 38.104
    bandwidth_mhz: float = 200.0  # Configured bandwidth (MHz)
    load_factor: float = 0.0  # Cell congestion level (0.0 to 1.0)
    
    # Handover Control Parameters (3GPP-compliant)
    handover_margin_db: float = 3.0  # Hysteresis for handover decisions
    time_to_trigger_s: float = 0.16  # Condition persistence duration (seconds)


    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


# In[7]:


@dataclass
class UserEquipment:
    x: float
    y: float
    speed_mps: float
    direction_rad: float
    local_cpu_ghz: float = 0.5  # Reference local processing capability
    battery_joules: float = 1000.0  # Initial battery capacity

    def step(self, dt_s: float, area_size: Tuple[float, float]) -> None:
        """Move UE with simple constant velocity model and reflecting borders."""
        dx = self.speed_mps * math.cos(self.direction_rad) * dt_s
        dy = self.speed_mps * math.sin(self.direction_rad) * dt_s

        new_x = self.x + dx
        new_y = self.y + dy
        max_x, max_y = area_size

        # Reflect from borders
        if new_x < 0 or new_x > max_x:
            self.direction_rad = math.pi - self.direction_rad
            new_x = max(0, min(max_x, new_x))
        if new_y < 0 or new_y > max_y:
            self.direction_rad = -self.direction_rad
            new_y = max(0, min(max_y, new_y))

        self.x = new_x
        self.y = new_y



# In[8]:


@dataclass
class Task:
    id: int
    arrival_time_s: float
    data_size_bits: float
    cpu_cycles: float
    deadline_s: float
    service_type: str


# In[9]:


@dataclass
class ServiceProfile:
    name: str
    latency_budget_s: float
    energy_weight: float
    latency_weight: float
    task_data_bits_mean: float
    task_cpu_cycles_mean: float
    task_interarrival_s: float


# In[ ]:


SERVICE_PROFILES: Dict[str, ServiceProfile] = {
    "VR": ServiceProfile(
        name="VR",  # eMBB (Enhanced Mobile Broadband)
        latency_budget_s=0.3,
        energy_weight=0.2,
        latency_weight=0.8,
        task_data_bits_mean=15e6,
        task_cpu_cycles_mean=1e9,
        task_interarrival_s=0.2
    ),
    "EV": ServiceProfile(
        name="EV",  # URLLC (Ultra-Reliable Low-Latency)
        latency_budget_s=0.35,
        energy_weight=0.1,
        latency_weight=0.9, 
        task_data_bits_mean=0.1e6,
        task_cpu_cycles_mean=0.5e9,
        task_interarrival_s=1.0
    ),
    "IoT": ServiceProfile(
        name="IoT",  # mMTC (Massive Machine Type Communication)
        latency_budget_s=2.0,
        energy_weight=0.9,
        latency_weight=0.1,
        task_data_bits_mean=0.05e6,
        task_cpu_cycles_mean=0.05e9,
        task_interarrival_s=5.0
    ),
}

# In[11]:


@dataclass
class MecServer:
    id: int
    attached_bs: int
    cpu_ghz: float = 500.0 # Computational capacity


# Mobility Models

class ChaoticRandomWaypoint:
    """
    Random Waypoint mobility with variable speed.
    
    Speed Distribution:
      - 0 m/s (30%): Loitering
      - 1 m/s (20%): Walking
      - 5 m/s (30%): Jogging/Cycling  
      - 15 m/s (15%): Driving
      - 25 m/s (5%): Highway
    """
    
    def __init__(self, bounds: Dict[str, float], rng: random.Random, min_speed: float = 0.0):
        self.bounds = bounds
        self.rng = rng
        self.min_speed = min_speed
        self.target: Optional[Tuple[float, float]] = None
        self.current_speed: float = 0.0
        self.steps_until_change: int = 0
        
    def step(self, ue, dt: float) -> None:
        """Move UE toward current target at current speed."""
        if self.steps_until_change <= 0:
            self._randomize_behavior(ue)
        
        if self.target and self.current_speed > 0:
            dx = self.target[0] - ue.x
            dy = self.target[1] - ue.y
            dist = math.hypot(dx, dy)
            
            if dist > 1.0:
                ue.vx = (dx / dist) * self.current_speed
                ue.vy = (dy / dist) * self.current_speed
                ue.x += ue.vx * dt
                ue.y += ue.vy * dt
            else:
                self._randomize_behavior(ue)
        else:
             # Even if speed is 0, we track time
             ue.vx = 0
             ue.vy = 0
        
        self.steps_until_change -= 1
    
    def _randomize_behavior(self, ue) -> None:
        """Pick new target and speed."""
        self.target = (
            self.rng.uniform(self.bounds['x_min'], self.bounds['x_max']),
            self.rng.uniform(self.bounds['y_min'], self.bounds['y_max'])
        )
        
        speed_profiles = [(0.0, 0.3), (1.0, 0.2), (5.0, 0.3), (15.0, 0.15), (25.0, 0.05)]
        
        # Filter by min_speed
        valid = [s for s in speed_profiles if s[0] >= self.min_speed]
        
        if not valid:
            # Fallback if min_speed too high
            valid = [(self.min_speed, 1.0)]
            
        speeds = [s[0] for s in valid]
        weights = [s[1] for s in valid]
        
        self.current_speed = self.rng.choices(speeds, weights=weights)[0]
        self.steps_until_change = self.rng.randint(50, 200)


class NetworkSimulation:

    def __init__(
        self,
        num_cells: int = 7,  # 1 center + 6 neighbors
        isd_range: Tuple[float, float] = (400, 600),  # Inter-Site Distance range
        dt_s: float = 0.01,
        seed: Optional[int] = None,
        mobility_min_speed: float = 0.0, 
    ) -> None:
        self.num_cells = num_cells
        self.isd_range = isd_range
        self.current_isd: float = 0.0
        self.map_bounds: Dict[str, float] = {}
        self.mobility: Optional[ChaoticRandomWaypoint] = None
        self.mobility_min_speed = mobility_min_speed
        self.dt_s = dt_s
        self.rng = random.Random(seed)
        
        # Measurement noise standard deviation (dB)
        self.measurement_noise_std_db: float = 2.5 

        self.base_stations: List[BaseStation] = []
        self.mec_servers: Dict[int, MecServer] = {}
        self.ue: Optional[UserEquipment] = None
        self.current_time_s: float = 0.0
        self.service_weights: Dict[str, float] = {"VR": 1.0}  # Default to VR only
        self.service_profiles: List[ServiceProfile] = [SERVICE_PROFILES["VR"]]
        self.serving_cell_id: int = 0
        self.task_counter: int = 0
        self.time_until_next_task_s: float = 0.0
        self.handover_history: List[float] = []  # Timestamps of handover events
        self.trace: List[Dict[str, Any]] = []
        
        # [PHASE 17] Handover State Tracking for TTT enforcement
        self.pending_handover_target: Optional[int] = None
        self.pending_handover_timer: float = 0.0
        self.pending_handover_rsrp_delta: float = 0.0
        
        # Curriculum learning phase
        self.curriculum_phase: int = 1
        
        # Physics Remediation
        # Correlated Shadowing State
        self.current_shadowing: Dict[int, float] = {}
        self.shadowing_corr_dist: float = 20.0  # Correlation distance (meters)
        self.shadowing_std_db: float = 6.0      # Shadowing standard deviation
        
        # Adversarial Injection State
        self.anomaly_traffic_multiplier: float = 1.0
        self.anomaly_cell_failure_id: Optional[int] = None
        self.anomaly_cell_failure_dbm: float = -120.0
        
        # 2. RLF State
        self.rlf_timer_s: float = 0.0           # Timer for out-of-sync
        self.is_rlf_active: bool = False        # RLF Flag
        self.rlf_penalty_flag: bool = False     # Flag to apply penalty in step()
        
        # Constants
        self.RSRP_RLF_THRESH_DBM = -110.0      # Out-of-sync threshold
        self.T310_TIMER_S = 0.2                 # Time to trigger RLF (200ms)
        self.RLF_RECOVERY_TIME_S = 1.0          # Delay penalty for RLF
        
        # Topology and MEC servers are now initialized in reset(), not here

    # ------------------------ initialization -----------------------------


    def _generate_hex_topology(self, isd: float) -> List[BaseStation]:
        """Generate 7-cell hexagonal grid with given Inter-Site Distance."""
        towers = []
        load = self._sample_curriculum_load(True)
        towers.append(BaseStation(id=0, x=0.0, y=0.0, load_factor=load))
        
        for i, angle_deg in enumerate([0, 60, 120, 180, 240, 300], start=1):
            angle_rad = math.radians(angle_deg)
            x = isd * math.cos(angle_rad)
            y = isd * math.sin(angle_rad)
            load = self._sample_curriculum_load(False)
            towers.append(BaseStation(id=i, x=x, y=y, load_factor=load))
        
        return towers
    
    def _sample_curriculum_load(self, is_center: bool = False) -> float:
        """Sample load factor based on curriculum phase."""
        if not is_center:
            return 0.0
        if self.curriculum_phase == 1:
            return random.uniform(0.0, 0.99)
        elif self.curriculum_phase == 2:
            rand = random.random()
            if rand < 0.6:
                return random.uniform(0.75, 0.95)
            elif rand < 0.85:
                return random.uniform(0.5, 0.75)
            else:
                return random.choice([0.0, 0.3, 0.99])
        else:
            loads = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]
            weights = [0.1, 0.15, 0.25, 0.25, 0.15, 0.1]
            return random.choices(loads, weights=weights)[0]
    
    def _init_cells_and_mec(self) -> None:
        """Place base stations in a simple line for now, each with a MEC server."""
        max_x, max_y = self.area_size
        spacing = max_x / (self.num_cells + 1)

        self.base_stations.clear()
        self.mec_servers.clear()

        for i in range(self.num_cells):
            bs_x = spacing * (i + 1)
            bs_y = max_y / 2.0
            
            # [PHASE 18] CURRICULUM LEARNING with 3-Phase Awareness
            # Phase 1 (Epochs 1-100): Exploration - random loads
            # Phase 2 (Epochs 101-350): Learning Zone - 80% on 0.5-0.95
            # Phase 3 (Epochs 351-500): Fine Tuning - balanced distribution
            if i == 0:
                if self.curriculum_phase == 1:
                    # Exploration: Full random range
                    load = random.uniform(0.0, 0.99)
                elif self.curriculum_phase == 2:
                    # Phase 2: Learning Zone (High Congestion Focus)
                    # Prioritize loads where greedy policies fail (0.75-0.95)
                    rand = random.random()
                    if rand < 0.6:
                        # Critical Congestion Zone
                        load = random.uniform(0.75, 0.95)
                    elif rand < 0.85:
                        # Moderate Congestion
                        load = random.uniform(0.5, 0.75)
                    else:
                        # Edge Cases
                        load = random.choice([0.0, 0.3, 0.99])
                else:
                    # Fine Tuning: Balanced distribution
                    loads = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]
                    weights = [0.1, 0.15, 0.25, 0.25, 0.15, 0.1]
                    load = random.choices(loads, weights=weights)[0]
                # Log load factor if verbose logging is enabled
                # print(f"[Curriculum P{self.curriculum_phase}] Cell 0 Load: {load:.2f}")
            else:
                load = 0.0
            
            bs = BaseStation(id=i, x=bs_x, y=bs_y, load_factor=load)
            self.base_stations.append(bs)
            self.mec_servers[i] = MecServer(id=i, attached_bs=i)

        # Intent-Based Networking (IBN) state
        self.intent_weights: Dict[str, float] = {'throughput': 0.34, 'latency': 0.33, 'energy': 0.33}
        
    def set_intent(self, throughput: float, latency: float, energy: float) -> None:
        """Explicitly set the User Intent Vector for the current episode."""
        total = throughput + latency + energy
        if total <= 0: total = 1.0 # Prevent div by zero
        self.intent_weights = {
            'throughput': throughput / total,
            'latency': latency / total,
            'energy': energy / total
        }

    def set_load_factor(self, cell_id: int, load_factor: float) -> None:
        """Allow external control of cell congestion (e.g., for stress testing)."""
        if 0 <= cell_id < len(self.base_stations):
            self.base_stations[cell_id].load_factor = load_factor
            print(f"[Sim] Cell {cell_id} Load Factor set to {load_factor:.2f}")
    
    def set_curriculum_phase(self, phase: int) -> None:
        """Set the curriculum learning phase for training.
        
        Args:
            phase: 1 = Exploration (random loads, high epsilon)
                   2 = Learning Zone (80% on 0.5-0.95, medium epsilon)
                   3 = Fine Tuning (balanced distribution, low epsilon)
        """
        if phase in [1, 2, 3]:
            self.curriculum_phase = phase
        else:
            print(f"[Warning] Invalid curriculum phase {phase}, keeping {self.curriculum_phase}")

    # ------------------------ public API ---------------------------------

    def reset(
        self,
        service_type: Optional[str] = None,
        service_weights: Optional[Dict[str, float]] = None,
        isd_range: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = 907,
        mobility_min_speed: Optional[float] = None,
        intent_weights: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """Reset simulation state and return initial context."""
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)

        # Intent logic
        if intent_weights is not None:
            self.set_intent(*intent_weights)
        else:
            # Default to random intent (Dirichlet distribution)
            w = np.random.dirichlet((1.0, 1.0, 1.0))
            self.set_intent(w[0], w[1], w[2])

        # Handle service profile configuration
        if service_weights is not None and service_type is not None:
            raise ValueError("Cannot specify both service_type and service_weights")
        
        if service_weights is not None:
            # Validate all service types exist
            for svc in service_weights.keys():
                if svc not in SERVICE_PROFILES:
                    raise ValueError(f"Unknown service_type '{svc}'")
            
            # Normalize weights to sum to 1.0
            total_weight = sum(service_weights.values())
            if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
                 # Auto-normalize or raise? Normalizing is safer
                 factor = 1.0 / total_weight
                 service_weights = {k: v*factor for k,v in service_weights.items()}
            self.service_weights = service_weights
            self.service_profiles = [SERVICE_PROFILES[k] for k in service_weights.keys()]
            
        elif service_type is not None:
            if service_type not in SERVICE_PROFILES:
                raise ValueError(f"Unknown service_type '{service_type}'")
            self.service_weights = {service_type: 1.0}
            self.service_profiles = [SERVICE_PROFILES[service_type]]
            
        # Defaults if none provided
        if not self.service_weights:
            self.service_weights = {"VR": 1.0}
            self.service_profiles = [SERVICE_PROFILES["VR"]]

        self.ue = None
        
        # Reset Base Station loads
        for bs in self.base_stations:
            # Randomize load based on Curriculum Phase
            if self.curriculum_phase == 1:
                # Exploration: Uniform random loads
                bs.load_factor = self.rng.random()
            elif self.curriculum_phase == 2:
                # Learning: 80% heavy (0.5-0.95), 20% light
                if self.rng.random() < 0.8:
                    bs.load_factor = self.rng.uniform(0.5, 0.95)
                else:
                    bs.load_factor = self.rng.uniform(0.0, 0.5)
            else:
                # Fine Tuning: Balanced
                bs.load_factor = self.rng.beta(2, 2)
            
            # Apply Anomaly Injection
            if self.anomaly_traffic_multiplier != 1.0:
                bs.load_factor = min(1.0, bs.load_factor * self.anomaly_traffic_multiplier)
            
            # Apply Cell Failure (Power check done in radio state)
            bs.is_active = True
            if self.anomaly_cell_failure_id == bs.id:
                # We handle this by reducing TX power in compute_radio_state, 
                # but we can also mark it here if useful.
                pass

        self.current_time_s = 0.0
        self.task_counter = 0
        self.time_until_next_task_s = self._sample_interarrival()
        self.handover_history = []  # Reset handover history
        
        # Reset Shadows & RLF
        self.current_shadowing = {}
        for i in range(self.num_cells):
            self.current_shadowing[i] = self.rng.normalvariate(0, self.shadowing_std_db)
            
        self.rlf_timer_s = 0.0
        self.is_rlf_active = False
        self.rlf_penalty_flag = False
        
        self.trace = []

        # Dynamic topology generation
        if isd_range is None:
            isd_range = self.isd_range
        self.current_isd = self.rng.uniform(isd_range[0], isd_range[1])
        self.base_stations = self._generate_hex_topology(self.current_isd)
        
        # Dynamic map bounds
        x_coords = [bs.x for bs in self.base_stations]
        y_coords = [bs.y for bs in self.base_stations]
        margin = self.current_isd * 0.2
        self.map_bounds = {
            'x_min': min(x_coords) - margin,
            'x_max': max(x_coords) - margin,
            'y_min': min(y_coords) - margin,
            'y_max': max(y_coords) + margin
        }
        
        # Initialize MEC servers
        self.mec_servers = {bs.id: MecServer(id=bs.id, attached_bs=bs.id) 
                           for bs in self.base_stations}
        
        # Spawn UE with coverage
        self.ue = self._spawn_ue_with_coverage()
        
        # Initialize mobility
        if mobility_min_speed is not None:
            self.mobility_min_speed = mobility_min_speed
        self.mobility = ChaoticRandomWaypoint(self.map_bounds, self.rng, min_speed=self.mobility_min_speed)

        # Initial serving cell: max RSRP
        radio_state = self._compute_radio_state()
        self.serving_cell_id = int(np.argmax(radio_state["rsrp_dbm"]))


        return self.get_context()
    
    def _spawn_ue_with_coverage(self, min_rsrp: float = -120.0) -> UserEquipment:
        """Spawn UE at random position with coverage guarantee."""
        for _ in range(100):
            x = self.rng.uniform(self.map_bounds['x_min'], self.map_bounds['x_max'])
            y = self.rng.uniform(self.map_bounds['y_min'], self.map_bounds['y_max'])
            
            for cell in self.base_stations:
                dist = cell.distance_to(x, y)
                pl_db = path_loss_db(dist)
                rsrp = cell.tx_power_dbm - pl_db
                
                if rsrp >= min_rsrp:
                    return UserEquipment(x=x, y=y, speed_mps=0.0, 
                                        direction_rad=0.0, battery_joules=1000.0)
        
        # Fallback to center
        return UserEquipment(x=0.0, y=0.0, speed_mps=0.0, 
                            direction_rad=0.0, battery_joules=1000.0)

    # ------------------------ radio model --------------------------------

    def _update_shadowing(self, distance_moved: float) -> None:
        """
        Update shadowing values using Gudmundson's Model (AR-1 Process).
        S(d + delta) = rho * S(d) + sqrt(1 - rho^2) * N(0, sigma)
        rho = exp(-delta_d / d_corr)
        """
        if distance_moved <= 0:
            return

        rho = math.exp(-distance_moved / self.shadowing_corr_dist)
        scaling = math.sqrt(1.0 - rho**2) * self.shadowing_std_db
        
        for i in range(self.num_cells):
            # Check if initialized
            if i not in self.current_shadowing:
                self.current_shadowing[i] = self.rng.normalvariate(0, self.shadowing_std_db)
                continue
                
            old_s = self.current_shadowing[i]
            innovation = self.rng.normalvariate(0, 1) # Standard normal
            # AR-1 update
            new_s = rho * old_s + scaling * innovation
            self.current_shadowing[i] = new_s

    def _compute_radio_state(self) -> Dict[str, Any]:
        """Compute distances, RSRP, SINR and throughput per cell."""
        if self.ue is None:
            raise RuntimeError("Simulation not reset. Call reset() first.")

        rsrp_dbm_list: List[float] = []
        sinr_db_list: List[float] = []
        throughput_bps_list: List[float] = []

        ue_x, ue_y = self.ue.x, self.ue.y
        
        # Doppler Penalty Calculation
        # Simple linear degradation: 0.2 dB per m/s approx
        # At 30 m/s -> 6 dB loss
        doppler_loss_db = 0.2 * self.ue.speed_mps

        for bs in self.base_stations:
            d = bs.distance_to(ue_x, ue_y)
            pl_db = path_loss_db(d)
            
            # Apply Correlated Shadowing
            shadowing = self.current_shadowing.get(bs.id, 0.0)
            
            # Apply Measurement Noise
            # Independent Gaussian noise per step/cell to simulate fast fading/measurement error
            noise = self.rng.normalvariate(0, self.measurement_noise_std_db)
            
            rx_power_dbm = bs.tx_power_dbm - pl_db + shadowing + noise
            
            # Adversarial Cell Failure
            if self.anomaly_cell_failure_id == bs.id:
                rx_power_dbm = self.anomaly_cell_failure_dbm
                
            rsrp_dbm_list.append(rx_power_dbm)

        # For simplicity, treat interference as coming from all non-serving cells when computing SINR for each candidate cell
        for i, bs in enumerate(self.base_stations):
            signal_w = dbm_to_watts(rsrp_dbm_list[i])
            interference_w = 0.0
            for j, _ in enumerate(self.base_stations):
                if j == i:
                    continue
                interference_w += dbm_to_watts(rsrp_dbm_list[j])

            noise_w = dbm_to_watts(bs.noise_dbm)
            
            # Apply Doppler Penalty to SINR (effectively increasing noise/interf)
            # We apply it by reducing signal power effectively in the ratio, or just subtract dB later.
            # Let's subtract dB from the final SINR for simplicity.
            
            sinr_linear = signal_w / (interference_w + noise_w + 1e-15)
            sinr_db = 10 * math.log10(max(sinr_linear, 1e-15))
            
            # Apply Doppler penalty
            sinr_db -= doppler_loss_db
            
            # Re-linearize for Capacity calculation
            sinr_linear_adj = 10**(sinr_db / 10.0)
            
            sinr_db_list.append(sinr_db)

            bandwidth_hz = bs.bandwidth_mhz * 1e6
            capacity_bps = shannon_capacity_hz(bandwidth_hz, sinr_linear_adj)
            
            # Apply load factor to reduce effective throughput under congestion
            # If load_factor = 0.9, only 10% of capacity is available
            available_capacity = capacity_bps * (1.0 - bs.load_factor)
            
            # RLF Effect: If RLF is active, throughput is ZERO for serving cell
            # Only applied if this cell is the serving one? 
            # Actually RLF is a state of the UE. If RLF is active, no data moves.
            if self.is_rlf_active:
                available_capacity = 0.0
                
            throughput_bps_list.append(available_capacity)

        return {
            "rsrp_dbm": rsrp_dbm_list,
            "sinr_db": sinr_db_list,
            "throughput_bps": throughput_bps_list,
        }

    # ------------------------ task model ---------------------------------

    def _sample_interarrival(self) -> float:
        """Sample time until next task using exponential distribution."""
        # 1. Calculate weighted Arrival Rate (Lambda)
        weighted_lambda = sum(
            self.service_weights[profile.name] * (1.0 / profile.task_interarrival_s)
            for profile in self.service_profiles
        )
        
        # Traffic Surge Injection
        weighted_lambda *= self.anomaly_traffic_multiplier
        
        if weighted_lambda <= 0:
            return 1.0 # Fallback
            
        avg_interarrival = 1.0 / weighted_lambda
        
        return max(np.random.exponential(avg_interarrival), 0.01)

    def _maybe_generate_task(self) -> Optional[Task]:
        if self.time_until_next_task_s > 0:
            return None

        self.task_counter += 1

        # Randomly select a service profile based on weights
        profile_names = list(self.service_weights.keys())
        profile_weights = list(self.service_weights.values())
        selected_profile_name = np.random.choice(profile_names, p=profile_weights)
        selected_profile = SERVICE_PROFILES[selected_profile_name]

        
        base_mean = selected_profile.task_data_bits_mean * 3.0
        
        data_size = max(
            np.random.normal(base_mean, 0.3 * base_mean),
            1e4,
        )
        
        cpu_cycles = max(
            np.random.normal(selected_profile.task_cpu_cycles_mean,
                             0.3 * selected_profile.task_cpu_cycles_mean),
            1e7,
        )

        deadline = self.current_time_s + selected_profile.latency_budget_s

        task = Task(
            id=self.task_counter,
            arrival_time_s=self.current_time_s,
            data_size_bits=data_size,
            cpu_cycles=cpu_cycles,
            deadline_s=deadline,
            service_type=selected_profile.name,
        )

        # Schedule next task
        self.time_until_next_task_s = self._sample_interarrival()
        return task
    
    def will_task_arrive(self) -> bool:
        """
        Check if a task will arrive in the NEXT step (before calling step()).
        
        CRITICAL: This must account for dt_s time advancement!
        If time_until_next_task <= dt_s, then after step() subtracts dt_s,
        the task will generate.
        """
        return self.time_until_next_task_s <= self.dt_s

    # ------------------------ context & decisions ------------------------

    def get_context(self) -> Dict[str, Any]:
        """Return the observable context for decision-making agents."""
        if self.ue is None:
            raise RuntimeError("Simulation not reset. Call reset() first.")

        radio_state = self._compute_radio_state()
        serving_idx = self.serving_cell_id
        rsrp_serving = radio_state["rsrp_dbm"][serving_idx]
        sinr_serving = radio_state["sinr_db"][serving_idx]
        throughput_serving = radio_state["throughput_bps"][serving_idx]

        # Compute weighted averages for service profile parameters
        avg_latency_budget = sum(
            self.service_weights[profile.name] * profile.latency_budget_s
            for profile in self.service_profiles
        )
        avg_latency_weight = sum(
            self.service_weights[profile.name] * profile.latency_weight
            for profile in self.service_profiles
        )
        avg_energy_weight = sum(
            self.service_weights[profile.name] * profile.energy_weight
            for profile in self.service_profiles
        )

        context = {
            "time_s": self.current_time_s,
            "service_weights": self.service_weights,  # NEW: provide service mix info
            "latency_budget_s": avg_latency_budget,
            "ue_position": (self.ue.x, self.ue.y),
            "ue_speed_mps": self.ue.speed_mps,
            "ue_battery_joules": self.ue.battery_joules,  # For ContextAgent battery monitoring
            "serving_cell_id": self.serving_cell_id,
            "handover_history": self.handover_history,  # For ping-pong detection
            "rsrp_dbm": radio_state["rsrp_dbm"],
            "sinr_db": radio_state["sinr_db"],
            "throughput_bps": radio_state["throughput_bps"],
            "serving_rsrp_dbm": rsrp_serving,
            "serving_sinr_db": sinr_serving,
            "serving_throughput_bps": throughput_serving,
            "current_isd": self.current_isd,
            "map_bounds": self.map_bounds,
            "bs_positions": [(bs.x, bs.y) for bs in self.base_stations],
            "user_pref": {
                "latency_weight": avg_latency_weight,
                "energy_weight": avg_energy_weight,
            },
            "intent_weights": self.intent_weights,
        }
        return context

    # ------------------------ task evaluation ----------------------------

    def _evaluate_task_decision(
        self,
        task: Task,
        offload_target: str,
        serving_throughput_bps: float,
    ) -> Dict[str, Any]:
        """Evaluate latency / energy for a single task given offload target."""
        ue = self.ue
        if ue is None:
            raise RuntimeError("Simulation not reset. Call reset() first.")

        # Energy model parameters
        tx_power_w = 2.5
        cpu_power_w = 0.05

        # Local processing
        if offload_target == "local":
            exec_time_s = task.cpu_cycles / (ue.local_cpu_ghz * 1e9)
            total_latency_s = exec_time_s
            energy_j = exec_time_s * cpu_power_w

        # Edge processing via serving cell
        elif offload_target == "edge":
            tx_rate_bps = max(serving_throughput_bps, 1e3)
            tx_time_s = task.data_size_bits / tx_rate_bps
            mec = self.mec_servers[self.serving_cell_id]
            exec_time_s = task.cpu_cycles / (mec.cpu_ghz * 1e9)
            total_latency_s = tx_time_s + exec_time_s
            energy_j = tx_time_s * tx_power_w

        # "Cloud" with extra backhaul latency
        elif offload_target == "cloud":
            tx_rate_bps = max(serving_throughput_bps, 1e3)
            tx_time_s = task.data_size_bits / tx_rate_bps
            backhaul_s = 0.02  # 20 ms backbone delay (toy)
            cloud_cpu_ghz = 1000.0
            exec_time_s = task.cpu_cycles / (cloud_cpu_ghz * 1e9)
            total_latency_s = tx_time_s + backhaul_s + exec_time_s
            energy_j = tx_time_s * tx_power_w

        else:
            raise ValueError(f"Unknown offload_target '{offload_target}'")

        deadline_met = total_latency_s <= (task.deadline_s - task.arrival_time_s)

        # Update UE battery
        ue.battery_joules -= energy_j
        
        # Get serving cell load factor and RSRP for context-aware rewards
        serving_cell = self.base_stations[self.serving_cell_id]
        cell_congestion = serving_cell.load_factor
        rsrp_serving = self._compute_radio_state()["rsrp_dbm"][self.serving_cell_id]

        return {
            "task_id": task.id,
            "offload_target": offload_target,
            "latency_s": total_latency_s,
            "deadline_s": task.deadline_s - task.arrival_time_s,
            "deadline_met": deadline_met,
            "energy_j": energy_j,
            # Context for intelligent reward calculation
            "cell_congestion": cell_congestion,
            "rsrp_dbm": rsrp_serving,
        }

    # ------------------------ baseline controller ------------------------

    def baseline_controller(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple rule-based controller for both HO and offloading.

        - Handover to the cell with highest RSRP if it's better than the
          current one by > hysteresis_db.
        - Offload locally for small tasks / bad channel, edge otherwise.
        - Cloud is unused in this simple baseline.
        """
        rsrp_list = context["rsrp_dbm"]
        serving_idx = context["serving_cell_id"] 
        serving_rsrp = rsrp_list[serving_idx]
        best_idx = int(np.argmax(rsrp_list))
        best_rsrp = rsrp_list[best_idx]

        hysteresis_db = 3.0 #at least 3 dB better to trigger HO
        target_cell = serving_idx
        if best_idx != serving_idx and (best_rsrp - serving_rsrp) > hysteresis_db: 
            target_cell = best_idx 

        # Offloading rule: here we simply look at serving throughput and latency budget.
        thr = context["serving_throughput_bps"]
        latency_budget_s = context["latency_budget_s"]

        if latency_budget_s < 0.07 and thr > 100e6: 
            offload_target = "edge"
        elif latency_budget_s < 0.2 and thr > 2e6: 
            offload_target = "edge"
        else:
            offload_target = "local"

        return {
            "handover_target": target_cell,
            "offload_target": offload_target,
        }

    # ------------------------ environment step ---------------------------

    def step(
        self,
        decision: Any,  # For HO agent
        mec_callback: Optional[Callable[[Task, Dict[str, Any]], str]] = None, # For MEC agent
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Advance the simulation by one time step (dt_s).

        Args:
            decision:
                For HO agent: An integer action representing target cell ID.
                For Baseline: A dict containing 'handover_target' and 'offload_target'.
                If None, calls baseline_controller.
            
            mec_callback:
                A function that takes (Task, Context) and returns 'local', 'edge', or 'cloud'.
                This is called ONLY if a new task is generated in this step.
                This ensures the agent decides for the EXACT task that will execute (no RNG desync).

        Returns:
            next_context, info

        info contains any generated task metrics and misc diagnostics.
        """
        if self.ue is None:
            raise RuntimeError("Simulation not reset. Call reset() first.")

        # Use baseline controller if none provided
        if decision is None:
            decision = self.baseline_controller(self.get_context())

        # Extract handover control parameters
        # 1) Apply handover decision (if any)
        if isinstance(decision, dict):
            ho_target = decision.get("handover_target", self.serving_cell_id)
            # NEW: Extract dynamic HOM/TTT from context-aware decision
            ho_margin_db = decision.get("handover_margin_db", 3.0)
            ttt_s = decision.get("time_to_trigger_s", 0.16)
        else:
            # Assume it's an integer/numpy int action index
            ho_target = int(decision)
            ho_margin_db = 3.0  # Default values
            ttt_s = 0.16
            
        # Validate handover target
        if not isinstance(ho_target, int) or ho_target < 0 or ho_target >= self.num_cells:
            ho_target = self.serving_cell_id
        
        # Handover condition enforcement logic
        current_cell = self.serving_cell_id
        handover_executed = False
        
        if ho_target != current_cell:
            # Compute radio state to check RSRP advantage
            radio_state = self._compute_radio_state()
            rsrp_serving = radio_state["rsrp_dbm"][current_cell]
            rsrp_target = radio_state["rsrp_dbm"][ho_target]
            rsrp_delta = rsrp_target - rsrp_serving
            
            # Check HOM: Does target cell beat serving by required margin?
            if rsrp_delta > ho_margin_db:
                # RSRP condition met - start/continue TTT timer
                if self.pending_handover_target == ho_target:
                    # Same target - increment timer
                    self.pending_handover_timer += self.dt_s
                else:
                    # New target - reset timer
                    self.pending_handover_target = ho_target
                    self.pending_handover_timer = 0.0
                    self.pending_handover_rsrp_delta = rsrp_delta
                
                # Check TTT: Has condition persisted long enough?
                if self.pending_handover_timer >= ttt_s:
                    # Execute handover
                    self.serving_cell_id = ho_target
                    self.handover_history.append(self.current_time_s)
                    handover_executed = True
                    
                    # Reset pending state
                    self.pending_handover_target = None
                    self.pending_handover_timer = 0.0
                    self.pending_handover_rsrp_delta = 0.0
            else:
                # RSRP advantage lost - cancel pending HO
                self.pending_handover_target = None
                self.pending_handover_timer = 0.0
                self.pending_handover_rsrp_delta = 0.0
        else:
            # Decision is to stay on current cell - cancel any pending HO
            self.pending_handover_target = None
            self.pending_handover_timer = 0.0
            self.pending_handover_rsrp_delta = 0.0

        # 2) Move UE using Chaotic Random Waypoint
        old_x, old_y = self.ue.x, self.ue.y
        self.mobility.step(self.ue, self.dt_s)
        dist_moved = math.hypot(self.ue.x - old_x, self.ue.y - old_y)
        
        # Update Shadowing
        self._update_shadowing(dist_moved)

        # 3) Time & task arrival
        self.current_time_s += self.dt_s
        self.time_until_next_task_s -= self.dt_s

        new_task = self._maybe_generate_task()

        # 4) Compute new radio state
        radio_state = self._compute_radio_state()
        serving_thr = radio_state["throughput_bps"][self.serving_cell_id]

        # 5) Evaluate offloading decision if there's a task
        task_info: Optional[Dict[str, Any]] = None
        if new_task is not None:
            # CRITICAL: Use callback if provided to get agent's decision for THIS exact task
            # This prevents RNG desync where agent sees different task than what executes
            if not self.is_rlf_active:
                if mec_callback is not None:
                    # Get current context before task execution
                    temp_context = self.get_context()
                    
                    # [EXECUTION GUARD] Prevent MEC Offloading during Handover (Hard Lock)
                    # If we just switched cells, the link is unstable. Force LOCAL processing.
                    if handover_executed:
                        # print("[Sim] Guard: Handover active. Forcing LOCAL execution.")
                        offload_target = "local"
                    else:
                        offload_target = mec_callback(new_task, temp_context)
                else:
                    # Fall back to decision dict or default
                    if handover_executed:
                        offload_target = "local"
                    else:
                        offload_target = decision.get("offload_target", "local") if isinstance(decision, dict) else "local"
                
                if offload_target is None:
                    offload_target = "local"
                
                task_info = self._evaluate_task_decision(new_task, offload_target, serving_thr)
            else:
                # RLF Active: Task Fails immediately
                task_info = {
                    "task_id": new_task.id,
                    "offload_target": "none",
                    "latency_s": self.RLF_RECOVERY_TIME_S, # Penalty latency
                    "deadline_s": new_task.deadline_s - new_task.arrival_time_s,
                    "deadline_met": False,
                    "energy_j": 0.0,
                    "cell_congestion": 0.0,
                    "rsrp_dbm": -120.0
                }

        # 6) RLF Monitor & Penalty Injection
        # Check Serving Cell RSRP
        current_rsrp = radio_state["rsrp_dbm"][self.serving_cell_id]
        
        if current_rsrp < self.RSRP_RLF_THRESH_DBM:
            self.rlf_timer_s += self.dt_s
            if self.rlf_timer_s >= self.T310_TIMER_S:
                if not self.is_rlf_active:
                    print(f"[Sim] RLF TRIGGERED! Cell {self.serving_cell_id} RSRP {current_rsrp:.1f} < {self.RSRP_RLF_THRESH_DBM}")
                    self.is_rlf_active = True
                    self.rlf_penalty_flag = True # Signal to external agent to penalize
        else:
            # Signal recovered?
            if self.is_rlf_active and current_rsrp > (self.RSRP_RLF_THRESH_DBM + 3.0):
                # Hysteresis for recovery
                self.is_rlf_active = False
                self.rlf_timer_s = 0.0
                print(f"[Sim] RLF RECOVERED. Cell {self.serving_cell_id} RSRP {current_rsrp:.1f}")
            elif not self.is_rlf_active:
                self.rlf_timer_s = 0.0

        # RLF forces UE to stay put or reconnect? 
        # For simplicity, if RLF active, we stay on cell but get 0 throughput.
        # Ideally, RLF should force a "Re-establishment" to strongest cell after delay.
        if self.is_rlf_active:
            # Force switch to best cell after recovery time?
            # Or just let HO agent figure it out (it observes bad RSRP)
            pass

        # 7) Build context & log
        context = self.get_context()
        
        # Inject RLF penalty into info for Reward Function
        info: Dict[str, Any] = {
            "decision": decision,
            "new_task": new_task is not None,
            "task_info": task_info,
            "rlf_penalty": self.rlf_penalty_flag
        }
        
        # Consumed the flag
        self.rlf_penalty_flag = False

        log_entry = {
            "time_s": self.current_time_s,
            "context": context,
            "decision": decision,
            "task_info": task_info,
        }
        self.trace.append(log_entry)

        return context, info

    # ------------------------ rollouts -----------------------------------

    def run_episode(
        self,
        num_steps: int,
        controller_fn: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Run a full episode and return the collected trace.

        controller_fn: callable taking context and returning a decision dict.
                       If None, uses baseline_controller.
        """
        self.trace = []
        for _ in range(num_steps):
            ctx = self.get_context()
            if controller_fn is None:
                decision = None
            else:
                decision = controller_fn(ctx)
            self.step(decision)
        return self.trace

    # ------------------------ ADVERSARIAL INJECTION API ------------------
    
    def inject_traffic_surge(self, multiplier: float = 5.0):
        """Simulate high traffic conditions."""
        self.anomaly_traffic_multiplier = multiplier
        print(f"[Sim] ⚠ INJECTING TRAFFIC SURGE: {multiplier}x Arrival Rate ⚠")
        
    def inject_battery_drop(self, target_level_percent: float = 10.0):
        """Simulate battery depletion."""
        if self.ue:
            target_joules = 1000.0 * (target_level_percent / 100.0)
            self.ue.battery_joules = target_joules
            print(f"[Sim] ⚠ INJECTING BATTERY DROP: {target_level_percent}% ({target_joules}J) ⚠")
            
    def inject_cell_failure(self, cell_id: int):
        """Simulate equipment failure."""
        self.anomaly_cell_failure_id = cell_id
        print(f"[Sim] ⚠ INJECTING CELL FAILURE: ID {cell_id} -> -120dBm ⚠")
        
    def clear_anomalies(self):
        """Reset all adversarial conditions."""
        self.anomaly_traffic_multiplier = 1.0
        self.anomaly_cell_failure_id = None
        print(f"[Sim] Anomalies Cleared.")





# In[14]:


if __name__ == "__main__":
    sim = NetworkSimulation(num_cells=3, area_size=(1000.0, 1000.0), dt_s=0.01, seed=907)
    ctx = sim.reset(service_type="VR")
    print("Initial context:")
    print({k: v for k, v in ctx.items() if k not in ("rsrp_dbm", "sinr_db", "throughput_bps")})

    trace = sim.run_episode(num_steps=5000)
    num_tasks = sum(1 for t in trace if t["task_info"] is not None)
    print(f"Ran 5000 steps, tasks generated: {num_tasks}")
