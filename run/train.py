"""
Cooperative MARL Training for Integrated MEC and HO.
Implements a shared reward architecture to align HO decisions with MEC system state.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
import json
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.simulation import NetworkSimulation, SERVICE_PROFILES, ServiceProfile
from src.agents.mec_agent_ppo import MECAgentPPO
from src.agents.ho_agent_ppo import HOAgentPPO

# Hardened Mode: Enforce strict latency constraints
for p in SERVICE_PROFILES.values():
    p.latency_budget_s = 0.3
    p.task_interarrival_s = 0.5 # Steady load

class MecTrainingWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.reset()
        
    def reset(self):
        self.last_obs = None
        self.last_act = None
        self.last_logprob = None
        self.last_val = None
        self.occured = False

    def callback(self, task, context):
        task_dict = {
            "data_size_bits": task.data_size_bits,
            "cpu_cycles": task.cpu_cycles,
            "deadline_s": task.deadline_s,
            "service_type": task.service_type
        }
        obs = self.agent.get_observation(context, task=task_dict)
        act, logprob, val = self.agent.select_action_with_info(obs, context)
        act_name = self.agent.action_map.get(act, "local")
        
        self.last_obs = obs
        self.last_act = act
        self.last_logprob = logprob
        self.last_val = val
        self.occured = True
        return act_name

def calculate_mec_reward(metrics, intent_vector, context):
    if not metrics['deadline_met']:
        return -2.0 # Harsh penalty for unity
    
    # Normalization
    thr_bps = context.get('serving_throughput_bps', 0.0)
    MAX_THR = 1e9 
    MAX_LAT = 0.3 # Hardened
    MAX_ENG = 1.0 
    
    g_thr = min(thr_bps / MAX_THR, 1.0)
    lat_val = metrics.get('latency_s', 0.3)
    g_lat = 1.0 - min(lat_val / MAX_LAT, 1.0)
    eng_val = metrics.get('energy_j', 1.0)
    g_eng = 1.0 - min(eng_val / MAX_ENG, 1.0)
    
    w_thr, w_lat, w_eng = intent_vector
    weighted_score = (w_thr * g_thr) + (w_lat * g_lat) + (w_eng * g_eng)
    return 1.0 + weighted_score

def calculate_ho_reward(context, prev_context, intent_vector):
    # Normalized SINR (Goodness)
    sinr = context['serving_sinr_db']
    g_thr = np.clip((sinr + 10.0) / 40.0, 0.0, 1.0)
    
    # Stability
    is_ho = (context['serving_cell_id'] != prev_context['serving_cell_id'])
    g_lat = 0.0 if is_ho else 1.0
    
    # Coverage
    rsrp = context['serving_rsrp_dbm']
    g_eng = np.clip((rsrp + 120.0) / 70.0, 0.0, 1.0)
    
    w_thr, w_lat, w_eng = intent_vector
    weighted_score = (w_thr * g_thr) + (w_lat * g_lat) + (w_eng * g_eng)
    
    # Penalty for HO
    penalty = 0.0
    if is_ho:
        penalty = (2.0 * w_lat) + 0.1
        
    return weighted_score - penalty

def run_marl_episode(episode_idx, mec_state, ho_state, steps_per_episode):
    seed = int(episode_idx + 80000)
    
    # Share the same intent across both agents for unity
    intent_mode = episode_idx % 3
    if intent_mode == 0: intent = (0.8, 0.1, 0.1) # Throughput Focus
    elif intent_mode == 1: intent = (0.1, 0.8, 0.1) # Stability Focus
    else: intent = (0.1, 0.1, 0.8) # Coverage/Energy Focus

    sim = NetworkSimulation(num_cells=7, seed=seed, dt_s=0.1)
    sim.set_intent(*intent)
    
    # Init Agents
    mec_agent = MECAgentPPO(agent_id="mec_worker", num_actions=3)
    mec_agent.network.load_state_dict(mec_state)
    mec_agent.network.eval()
    
    ho_agent = HOAgentPPO(agent_id="ho_worker")
    ho_agent.network.load_state_dict(ho_state)
    ho_agent.network.eval()
    
    wrapper = MecTrainingWrapper(mec_agent)
    context = sim.reset(isd_range=(500, 1000), intent_weights=intent)
    wrapper.reset()
    
    mec_rollout = {'obs':[], 'act':[], 'logprob':[], 'rew':[], 'val':[], 'done':[]}
    ho_rollout = {'obs':[], 'act':[], 'logprob':[], 'rew':[], 'val':[], 'done':[]}
    
    stats = {'reward_mec': 0, 'reward_ho': 0, 'tasks': 0, 'successes': 0, 'handovers': 0}
    
    for step in range(steps_per_episode):
        prev_context = context
        
        # 1. HO Decision
        obs_ho = ho_agent.get_observation(context)
        act_ho, logp_ho, val_ho = ho_agent.select_action_with_info(obs_ho)
        
        # 2. MEC Step via Callback
        next_context, info = sim.step(decision=int(act_ho), mec_callback=wrapper.callback)
        
        # 3. Process MEC Rollout
        if wrapper.occured and info.get('task_info'):
            metrics = info['task_info']
            rew_mec = calculate_mec_reward(metrics, intent, context)
            
            mec_rollout['obs'].append(wrapper.last_obs)
            mec_rollout['act'].append(wrapper.last_act)
            mec_rollout['logprob'].append(wrapper.last_logprob)
            mec_rollout['rew'].append(rew_mec)
            mec_rollout['val'].append(wrapper.last_val)
            mec_rollout['done'].append(False)
            
            stats['reward_mec'] += rew_mec
            stats['tasks'] += 1
            if metrics['deadline_met']:
                stats['successes'] += 1

        # 4. Process HO Rollout
        rew_ho = calculate_ho_reward(next_context, prev_context, intent)
        
        # Cooperative Reinforcement: Apply penalty if task generation fails
        if info.get('tasks_generated', 0) > 0 and info.get('tasks_success', 0) == 0:
            rew_ho -= 5.0 
            
        ho_rollout['obs'].append(obs_ho)
        ho_rollout['act'].append(act_ho)
        ho_rollout['logprob'].append(logp_ho)
        ho_rollout['rew'].append(rew_ho)
        ho_rollout['val'].append(val_ho)
        ho_rollout['done'].append(False)
        
        stats['reward_ho'] += rew_ho
        if next_context['serving_cell_id'] != prev_context['serving_cell_id']:
            stats['handovers'] += 1
            
        context = next_context
        
    # Mark Dones
    if mec_rollout['done']: mec_rollout['done'][-1] = True
    if ho_rollout['done']: ho_rollout['done'][-1] = True
    
    return {
        'mec_rollout': mec_rollout,
        'ho_rollout': ho_rollout,
        'stats': {
            'ep': episode_idx,
            'sr': stats['successes'] / max(1, stats['tasks']),
            'ho': stats['handovers'],
            'rew_mec': stats['reward_mec'],
            'rew_ho': stats['reward_ho']
        }
    }

def train():
    MAX_EPISODES = 10000
    STEPS_PER_EPISODE = 50
    BATCH_SIZE = 2048
    LR = 1e-5 # Ultra-low for synergistic fine-tuning
    
    MEC_PATH = os.path.join(project_root, "models", "mec_policy.pth")
    HO_PATH = os.path.join(project_root, "models", "ho_policy.pth")
    
    # Initialize
    mec_marl = MECAgentPPO(agent_id="marl_mec", lr=LR)
    ho_marl = HOAgentPPO(agent_id="marl_ho", lr=LR)
    
    # Warm-Start
    if os.path.exists(MEC_PATH):
        mec_marl.load(MEC_PATH)
        print(f"MEC Warm-Start: {MEC_PATH}")
    if os.path.exists(HO_PATH):
        ho_marl.load(HO_PATH)
        print(f"HO Warm-Start: {HO_PATH}")

    # Loggers
    stats_history = []
    best_sr = 0.0
    
    mec_buffer = {'obs':[], 'act':[], 'logprob':[], 'rew':[], 'val':[], 'done':[]}
    ho_buffer = {'obs':[], 'act':[], 'logprob':[], 'rew':[], 'val':[], 'done':[]}
    
    pbar = tqdm(total=MAX_EPISODES)
    current_ep = 0
    
    with Parallel(n_jobs=-1) as parallel:
        while current_ep < MAX_EPISODES:
            n_batch = cpu_count()
            if current_ep + n_batch > MAX_EPISODES:
                n_batch = MAX_EPISODES - current_ep
                
            task_ids = range(current_ep, current_ep + n_batch)
            mec_state = {k: v.cpu() for k, v in mec_marl.network.state_dict().items()}
            ho_state = {k: v.cpu() for k, v in ho_marl.network.state_dict().items()}
            
            results = parallel(
                delayed(run_marl_episode)(idx, mec_state, ho_state, STEPS_PER_EPISODE)
                for idx in task_ids
            )
            
            for res in results:
                # Store rollouts
                for k in mec_buffer: mec_buffer[k].extend(res['mec_rollout'][k])
                for k in ho_buffer: ho_buffer[k].extend(res['ho_rollout'][k])
                stats_history.append(res['stats'])
            
            # Update MEC
            if len(mec_buffer['obs']) >= BATCH_SIZE:
                mec_marl.update(mec_buffer)
                mec_buffer = {'obs':[], 'act':[], 'logprob':[], 'rew':[], 'val':[], 'done':[]}
            
            # Update HO
            if len(ho_buffer['obs']) >= BATCH_SIZE:
                ho_marl.update(ho_buffer)
                ho_buffer = {'obs':[], 'act':[], 'logprob':[], 'rew':[], 'val':[], 'done':[]}
                
            # Best & Save
            if len(stats_history) >= 100:
                avg_sr = np.mean([s['sr'] for s in stats_history[-100:]])
                if avg_sr > best_sr:
                    best_sr = avg_sr
                    mec_marl.save(os.path.join(project_root, "models", "mec_policy_best.pth"))
                    ho_marl.save(os.path.join(project_root, "models", "ho_policy_best.pth"))

            if current_ep % 500 == 0:
                pd.DataFrame(stats_history).to_csv(os.path.join(project_root, "data", "training_metrics.csv"), index=False)
                
            current_ep += n_batch
            pbar.update(n_batch)
            ls = stats_history[-1]
            pbar.set_description(f"SR={ls['sr']:.2f} | HO={ls['ho']} | R_HO={ls['rew_ho']:.1f}")

    # Final Save
    mec_marl.save(os.path.join(project_root, "models", "mec_policy.pth"))
    ho_marl.save(os.path.join(project_root, "models", "ho_policy.pth"))
    print("\n>>> Integrated MARL Training Complete.")

if __name__ == "__main__":
    train()
