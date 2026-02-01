"""
Final System Evaluation: Integrated Performance Benchmark.
Analyzes the performance of synergistic MEC and HO agents under stress and polymorphic conditions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# Path Setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.simulation import NetworkSimulation
from src.agents.mec_agent_ppo import MECAgentPPO
from src.agents.ho_agent_ppo import HOAgentPPO

# Configuration
MEC_CHECKPOINT = os.path.join(project_root, "models", "mec_policy.pth")
HO_CHECKPOINT = os.path.join(project_root, "models", "ho_policy.pth")

RESULTS_DIR = os.path.join(project_root, "data")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Simulation Parameters
DT_S = 0.1
STEPS = 500 # 50s per episode
N_EPISODES = 50 # Per group/scenario

def run_joint_episode(mec_net_state, ho_net_state, mec_intent, ho_intent, 
                      use_mec_agent=True, use_ho_agent=True, 
                      speed=20.0, load=50.0, seed_offset=0):
    
    seed = 8888 + seed_offset
    sim = NetworkSimulation(num_cells=7, seed=seed, dt_s=DT_S, 
                            mobility_min_speed=speed)
    
    # Custom Load: Inject a high-load service profile
    # The 'load' parameter (tasks/sec) translates to interarrival = 1/load
    # However, sim resets profiles in reset(). We must pass it to reset.
    from src.simulation import ServiceProfile
    stress_profile = ServiceProfile(
        name="STRESS",
        latency_budget_s=0.3,
        energy_weight=0.5,
        latency_weight=0.5,
        task_data_bits_mean=0.1e6,
        task_cpu_cycles_mean=0.2e9,
        task_interarrival_s=1.0/max(1, load)
    )
    
    # Inject into global profiles so reset can find it
    import src.simulation
    src.simulation.SERVICE_PROFILES["STRESS"] = stress_profile
    
    # Combined Intent
    sim.set_intent(*ho_intent) 
    
    # Reset with our stress profile
    context = sim.reset(service_type="STRESS", isd_range=(500, 500), intent_weights=ho_intent)
    
    # Load Agents if needed
    mec_agent = None
    if use_mec_agent and mec_net_state:
        mec_agent = MECAgentPPO(agent_id="eval_mec")
        mec_agent.network.load_state_dict(mec_net_state)
        mec_agent.network.eval()
        
    ho_agent = None
    if use_ho_agent and ho_net_state:
        ho_agent = HOAgentPPO(agent_id="eval_ho")
        ho_agent.network.load_state_dict(ho_net_state)
        ho_agent.network.eval()
        
    stats = {
        'tasks_total': 0,
        'tasks_success': 0,
        'handovers': 0,
        'sinr_sum': 0.0,
        'energy_sum': 0.0,
        'steps': 0
    }
    
    prev_cell = context['serving_cell_id']
    
    # Multi-agent decision logic
    def mec_callback(task, ctx):
        if mec_agent:
            obs = mec_agent.get_observation(ctx)
            action = mec_agent.select_action(obs, context=ctx)
            return mec_agent.action_map.get(action, "local")
        return "local" 
    
    for _ in range(STEPS):
        # 1. Get current Context
        ctx = sim.get_context()
        
        # 2. Ask HO Agent for Radio Decision
        ho_decision = None
        if ho_agent:
            obs_ho = ho_agent.get_observation(ctx)
            ho_action, _, _ = ho_agent.select_action_with_info(obs_ho)
            ho_decision = int(ho_action)
        else:
            # Baseline: Let the sim's internal greedy baseline handle it
            # To do that, we pass None as decision to step(), 
            # but we want to separate HO and MEC. 
            # If we pass None to step, it runs greedy HO.
            pass
            
        # 3. Ask MEC Agent for Compute Decision
        # This is handled dynamically by mec_callback during the simulation.step()
        
        # Execute simulation step
        next_context, info = sim.step(decision=ho_decision, mec_callback=mec_callback)
        
        # Metric Capture
        # We only capture metrics ONCE per step.
        task_info = info.get('task_info')
        if task_info:
            stats['tasks_total'] += 1
            if task_info.get('deadline_met'):
                stats['tasks_success'] += 1
                
        curr_cell = next_context['serving_cell_id']
        if curr_cell != prev_cell:
            stats['handovers'] += 1
            
        # Ensure we use next_context for SINR/Energy to reflect the result of the move
        stats['sinr_sum'] += next_context['serving_sinr_db']
        stats['energy_sum'] += next_context.get('total_energy_j', 0.1) 
        stats['steps'] += 1
        
        prev_cell = curr_cell
        context = next_context
        # (Removed duplicated block that was causing double-counting)
        
    return {
        'SuccessRate': stats['tasks_success'] / max(1, stats['tasks_total']),
        'Handovers': stats['handovers'],
        'AvgSINR': stats['sinr_sum'] / stats['steps'],
        'TotalEnergy': stats['energy_sum']
    }

def exp_1_synergy(mec_state, ho_state):
    print("\n>>> Performance Evaluation: System Synergy (Ablation)...")
    configs = [
        ("Grp 0: Baseline", False, False),
        ("Grp 1: MEC Only", True, False),
        ("Grp 2: HO Only", False, True),
        ("Grp 3: Full IBN", True, True)
    ]
    
    # High Stress: Speed=20m/s, Load=50
    intent_ho = (0.5, 0.5, 0.0) # Balanced Throughput/Stability
    intent_mec = (0.5, 0.5, 0.0)
    
    data = []
    with Parallel(n_jobs=-1) as parallel:
        for name, use_mec, use_ho in configs:
            results = parallel(delayed(run_joint_episode)(
                mec_state, ho_state, intent_mec, intent_ho, use_mec, use_ho, speed=20.0, load=50.0, seed_offset=i
            ) for i in tqdm(range(N_EPISODES), desc=name))
            
            df_res = pd.DataFrame(results)
            data.append({
                'Combination': name,
                'SuccessRate': df_res['SuccessRate'].mean(),
                'Handovers': df_res['Handovers'].mean(),
                'SINR': df_res['AvgSINR'].mean()
            })
            
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(RESULTS_DIR, "synergy_analysis.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Combination', y='SuccessRate', palette='coolwarm')
    plt.title('Integrated System Reliability: Ablation Study (Speed=20m/s, Load=50)')
    plt.ylabel('Service Success Rate')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "reliability_benchmark.png"))
    plt.close()
    
    print(df.to_string(index=False))

def exp_2_polymorphic(mec_state, ho_state):
    print("\n>>> Performance Evaluation: Polymorphic Intent-Shifting...")
    
    # Mode A: Eco-Safe (MEC Latency/Energy, HO Stability)
    # Mode B: Max-Power (MEC Speed/Latency, HO Throughput)
    
    scenarios = [
        ("Eco-Safe", (0.2, 0.2, 0.6), (0.0, 1.0, 0.0)), # HO: Stability, MEC: Energy
        ("Max-Power", (0.8, 0.1, 0.1), (1.0, 0.0, 0.0)) # HO: Throughput, MEC: Latency
    ]
    
    data = []
    with Parallel(n_jobs=-1) as parallel:
        for name, mec_int, ho_int in scenarios:
            results = parallel(delayed(run_joint_episode)(
                mec_state, ho_state, mec_int, ho_int, True, True, speed=10.0, load=30.0, seed_offset=i
            ) for i in tqdm(range(N_EPISODES), desc=name))
            
            df_res = pd.DataFrame(results)
            data.append({
                'Mode': name,
                'AvgSINR': df_res['AvgSINR'].mean(),
                'Energy (J)': df_res['TotalEnergy'].mean(),
                'SuccessRate': df_res['SuccessRate'].mean()
            })
            
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(RESULTS_DIR, "polymorphism_analysis.csv"), index=False)
    print(df.to_string(index=False))

def main():
    print("=== STARTING INTEGRATED SYSTEM BENCHMARK ===")
    
    # Load Weights
    mec_agent = MECAgentPPO()
    ho_agent = HOAgentPPO()
    
    try:
        mec_agent.load(MEC_CHECKPOINT)
        ho_agent.load(HO_CHECKPOINT)
        print(f"MEC Loaded: {MEC_CHECKPOINT}")
        print(f"HO Loaded: {HO_CHECKPOINT}")
    except Exception as e:
        print(f"Warning: Loading failed ({e}). Check your checkpoint paths.")
        return

    mec_state = {k: v.cpu() for k, v in mec_agent.network.state_dict().items()}
    ho_state = {k: v.cpu() for k, v in ho_agent.network.state_dict().items()}
    
    exp_1_synergy(mec_state, ho_state)
    exp_2_polymorphic(mec_state, ho_state)
    
    print("\n>>> ALL FINALE BENCHMARKS COMPLETE.")

if __name__ == "__main__":
    main()
