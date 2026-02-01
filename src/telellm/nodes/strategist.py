
import json
import os
from typing import Dict, Any
from ..schemas import ControlMode, ReliabilityState, TrafficState, EnergyState, MobilityState
from ..prompts import TELELLM_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

class StrategistNode:
    """
    Tier 2: Strategic Orchestrator (The Brain).
    Uses an LLM to reason about Symbolic State and select Control Mode.
    Falls back to a Rule-Based Expert System if no LLM is available.
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = None
        self.chain = None
        
        # Attempt to initialize LLM
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import JsonOutputParser
            
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", TELELLM_SYSTEM_PROMPT),
                    ("user", USER_PROMPT_TEMPLATE)
                ])
                
                self.chain = prompt | self.llm | JsonOutputParser()
            else:
                print(f"[Strategist] OPENAI_API_KEY not found. Using Rule-Based Fallback.")
                
        except ImportError:
            print(f"[Strategist] LangChain/OpenAI not installed. Using Rule-Based Fallback.")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Strategist Logic.
        
        Args:
            state: Dictionary containing 'symbolic_state' (and other context).
            
        Returns:
            Dict update with 'proposed_mode' and 'reasoning'.
        """
        sym_state = state.get("symbolic_state", {})
        current_mode = state.get("current_mode", ControlMode.BALANCED.value)
        
        
        if self.chain:
            try:
                response = self.chain.invoke({
                    "traffic": sym_state.get("traffic", "NORMAL"),
                    "reliability": sym_state.get("reliability", "SAFE"),
                    "energy": sym_state.get("energy", "NORMAL"),
                    "mobility": sym_state.get("mobility", "MODERATE"),
                    "current_mode": current_mode
                })
                
                mode_str = response.get("mode", "BALANCED").upper()
                # Validate mode string
                try:
                    mode_enum = ControlMode(mode_str)
                except ValueError:
                    mode_enum = ControlMode.BALANCED
                    
                return {
                    "proposed_mode": mode_enum,
                    "reasoning": response.get("reasoning", "LLM Decision")
                }
            except Exception as e:
                print(f"[Strategist] LLM Inference Error: {e}. Reverting to Rule-Based Logic.")
        
        # 2. Fallback: Rule-Based Expert System (The "Safety Net")
        return self._rule_based_logic(sym_state)

    def _rule_based_logic(self, state: Dict) -> Dict:
        """Deterministic fallback logic."""
        traffic = state.get("traffic", TrafficState.NORMAL)
        reliability = state.get("reliability", ReliabilityState.SAFE)
        energy = state.get("energy", EnergyState.NORMAL)
        mobility = state.get("mobility", MobilityState.MODERATE)

        # 1. Safety/Reliability is paramount
        if reliability == ReliabilityState.DANGER:
             return {"proposed_mode": ControlMode.SURVIVAL, "reasoning": "Rule-based: Reliability DANGER detected."}

        # 2. Energy Criticality overrides Traffic
        if energy == EnergyState.CRITICAL:
             return {"proposed_mode": ControlMode.GREEN, "reasoning": "Rule-based: Battery CRITICAL (<20%)."}

        # 3. Traffic Logic
        if traffic == TrafficState.URLLC_SURGE:
            return {"proposed_mode": ControlMode.SURVIVAL, "reasoning": "Rule-based: Traffic Surge."}
        
        if traffic == TrafficState.CONGESTED:
             return {"proposed_mode": ControlMode.BALANCED, "reasoning": "Rule-based: Congestion managed."}

        # Default
        return {"proposed_mode": ControlMode.BALANCED, "reasoning": "Rule-based: Conditions normal."}
