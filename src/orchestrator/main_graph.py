from langgraph.graph import StateGraph, END
from .state import SystemState
from .nodes import context_node, simulation_node

def should_continue(state: SystemState) -> str:
    """Decide whether to continue or end."""
    if state["is_done"]:
        return "end"
    return "continue"

def create_training_graph() -> StateGraph:
    """Create the LangGraph workflow."""
    
    workflow = StateGraph(SystemState)
    
    # Add nodes
    workflow.add_node("context_analyzer", context_node)
    workflow.add_node("simulation_runner", simulation_node)
    
    
    workflow.set_entry_point("simulation_runner")
    
    # Add edges
    workflow.add_edge("context_analyzer", "simulation_runner")
    
    # PATH C: CONDITIONAL EDGE (System 2 Trigger)
    # Decisions:
    # 1. Continue to Orchestrator (if needs_intervention=True)
    # 2. Loop back to Simulation (if needs_intervention=False) to skip LLM
    # 3. End (if is_done=True)
    
    def route_simulation_step(state: SystemState):
        intervene = state.get("needs_intervention", False)
        done = state["is_done"]
        
        if done:
            return "end"
        if intervene:
            return "context_analyzer"
        return "simulation_runner"

    workflow.add_conditional_edges(
        "simulation_runner",
        route_simulation_step,
        {
            "context_analyzer": "context_analyzer",
            "simulation_runner": "simulation_runner",
            "end": END
        }
    )
    
    return workflow.compile()

def run_training(max_epochs: int = 50, steps_per_epoch: int = 5000):
    """Run the graph with configurable parameters."""
    graph = create_training_graph()
    
    initial_state: SystemState = {
        "epoch": 0,
        "max_epochs": max_epochs,
        "context_weights": {
            "alpha": 0.33,
            "beta": 0.33,
            "gamma": 0.34,
            "ho_hints": {},
            "mec_hints": {}
        },
        "logs": "Initial state",
        "total_steps": steps_per_epoch,
        "history": [],
        "is_done": False
    }
    
    print(f">>> Starting LangGraph Training (Epochs={max_epochs}, Steps={steps_per_epoch}) <<<")
    final_state = graph.invoke(initial_state)
    print("\n>>> Training Complete <<<")
    print(f"Final Weights: {final_state['context_weights']}")
    return final_state

if __name__ == "__main__":
    run_training(max_epochs=3, steps_per_epoch=1000)
