from typing import TypedDict, Dict, Any

# Context Weights type for multi-agent coordination
ContextWeights = TypedDict('ContextWeights', {
    'alpha': float,          # Latency weight
    'beta': float,           # Energy weight
    'gamma': float,          # Stability weight
    'handover_margin_db': float,      # Dynamic HOM parameter
    'time_to_trigger_s': float,       # Dynamic TTT parameter
    'ho_hints': dict,        # Handover agent hints
    'mec_hints': dict        # MEC agent hints
}, total=False)

class SystemState(TypedDict):
    epoch: int                          # Current macro step
    max_epochs: int                     # Total epochs to run
    context_weights: ContextWeights     # weights from LLM
    logs: str                           # Summary from last batch simulation
    total_steps: int                    # Steps per epoch (micro loop size)
    history: list                       # List of Dicts containing epoch metrics
    is_done: bool                       # Termination flag
