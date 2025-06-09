# Databricks notebook source
# MAGIC %md
# MAGIC # SkyLink Navigator - Main ATC Agent
# MAGIC
# MAGIC This notebook implements the main SkyLink Navigator agent workflow using LangGraph.
# MAGIC Sub-agents are developed separately and will be integrated later.
# MAGIC
# MAGIC ## Workflow Overview:
# MAGIC ```
# MAGIC START → ATC_MAIN_AGENT → [Sub-Agents] → ATC_SYNTHESIZE → END
# MAGIC                    ↓
# MAGIC          ┌─────────────────────┐
# MAGIC          │   Conditional       │
# MAGIC          │   Routing to:       │
# MAGIC          │   • GeoTracker      │
# MAGIC          │   • Scheduler       │
# MAGIC          │   • Weather         │
# MAGIC          │   • CommsAgent      │
# MAGIC          └─────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC %pip install langgraph langchain pydantic

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Required Libraries

# COMMAND ----------

from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
import asyncio
from enum import Enum
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Core Data Models and States (FIXED)

# COMMAND ----------

def update_subagent_data(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer function for merging sub-agent responses"""
    if not left:
        return right
    if not right:
        return left
    
    # Merge dictionaries, with right taking precedence
    result = left.copy()
    result.update(right)
    return result

class PilotRequestType(Enum):
    """Types of pilot requests that SkyLink can handle"""
    IFR_CLEARANCE = "ifr_clearance"        # Instrument Flight Rules clearance requests
    WEATHER_REQUEST = "weather_request"    # Weather information requests  
    TRAFFIC_ADVISORY = "traffic_advisory"  # Traffic separation and advisory
    EMERGENCY = "emergency"                # Emergency situations (Mayday, Pan-Pan)
    GENERAL_INQUIRY = "general_inquiry"    # General ATC communications

class ATCState(TypedDict):
    """
    FIXED: Main state management for ATC operations using TypedDict
    
    Processing Phases:
    - ANALYZING: Initial request analysis
    - ROUTING: Determining which sub-agents to engage  
    - PROCESSING: Sub-agents working (handled externally)
    - SYNTHESIZING: Combining responses into final ATC communication
    """
    # Core message flow - FIXED: Use Annotated for proper message handling
    messages: Annotated[List[BaseMessage], lambda x, y: x + y if isinstance(y, list) else x + [y]]
    
    # Request identification  
    pilot_callsign: Optional[str]
    request_type: Optional[str]
    priority_level: str
    
    # Processing state
    current_phase: str
    required_agents: List[str]
    
    # Sub-agent data - FIXED: Use Annotated to handle multiple updates
    subagent_responses: Annotated[Dict[str, Any], update_subagent_data]
    
    # Final output
    final_response: Optional[str]
    confidence_score: float

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main ATC Agent Class

# COMMAND ----------

class SkyLinkNavigator:
    """
    Main Air Traffic Control Agent
    
    Central coordinator that:
    1. Analyzes pilot requests
    2. Routes to appropriate sub-agents (external)
    3. Synthesizes final ATC responses
    """
    
    def __init__(self):
        self.graph = None
        # Sub-agents are developed separately - these are just references
        self.available_subagents = {
            "geo_tracker": "Position and trajectory monitoring",
            "scheduler": "Clearances and slot management", 
            "weather": "Weather data and alerts",
            "comms_agent": "Communication analysis"
        }
        self._build_atc_workflow()
    
    def _build_atc_workflow(self):
        """
        Build the main ATC workflow using LangGraph - FIXED VERSION
        """
        # FIXED: Use the proper ATCState TypedDict
        workflow = StateGraph(ATCState)
        
        # Main workflow nodes
        workflow.add_node("atc_main_agent", self._atc_main_coordinator)
        workflow.add_node("route_to_subagents", self._route_to_subagents) 
        workflow.add_node("atc_synthesize", self._synthesize_atc_response)
        workflow.add_node("emergency_priority", self._emergency_handler)
        
        # Set entry point
        workflow.set_entry_point("atc_main_agent")
        
        # Main routing logic
        workflow.add_conditional_edges(
            "atc_main_agent",
            self._decide_routing,
            {
                "emergency": "emergency_priority",
                "normal": "route_to_subagents",
                "synthesize": "atc_synthesize"
            }
        )
        
        # Route back to synthesis
        workflow.add_edge("route_to_subagents", "atc_synthesize")
        workflow.add_edge("emergency_priority", "atc_synthesize")
        workflow.add_edge("atc_synthesize", END)
        
        self.graph = workflow.compile()
        print("✅ ATC Main Workflow compiled successfully (FIXED)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main ATC Coordinator

# COMMAND ----------

def _atc_main_coordinator(self, state: ATCState) -> Dict[str, Any]:
    """
    Main ATC Agent - FIXED: Returns only updated fields to avoid state conflicts
    """
    print("🎙️ ATC Main Agent: Processing pilot request...")
    
    messages = state.get("messages", [])
    if not messages:
        return {
            "messages": [HumanMessage(content="SkyLink Navigator ready. Please state your request.")],
            "pilot_callsign": None,
            "request_type": None,
            "priority_level": "NORMAL",
            "current_phase": "ANALYZING",
            "required_agents": [],
            "subagent_responses": {},
            "final_response": None,
            "confidence_score": 0.0
        }
        
    latest_message = messages[-1].content.lower()
    print(f"📨 Processing: {latest_message}")
    
    # Extract pilot callsign (improved pattern matching)
    callsign = None
    words = latest_message.split()
    for word in words:
        if (len(word) >= 3 and word.isalnum()) or \
           (len(word) >= 4 and any(c.isdigit() for c in word)):
            callsign = word.upper()
            break
    
    # Classify request type and determine required sub-agents
    if "mayday" in latest_message or "emergency" in latest_message or "pan pan" in latest_message:
        request_type = PilotRequestType.EMERGENCY.value
        priority = "EMERGENCY"
        agents = ["geo_tracker", "scheduler", "weather", "comms_agent"]
        
    elif "clearance" in latest_message or "ifr" in latest_message:
        request_type = PilotRequestType.IFR_CLEARANCE.value
        priority = "NORMAL"
        agents = ["scheduler", "comms_agent"]
        
    elif "weather" in latest_message or "metar" in latest_message:
        request_type = PilotRequestType.WEATHER_REQUEST.value
        priority = "NORMAL"
        agents = ["weather"]
        
    elif "traffic" in latest_message or "advisory" in latest_message:
        request_type = PilotRequestType.TRAFFIC_ADVISORY.value
        priority = "NORMAL"
        agents = ["geo_tracker", "comms_agent"]
        
    else:
        request_type = PilotRequestType.GENERAL_INQUIRY.value
        priority = "NORMAL"
        agents = ["comms_agent"]
    
    print(f"🎯 Request Type: {request_type}")
    print(f"👥 Required Sub-Agents: {agents}")
    
    # FIXED: Return only the fields that need updating
    return {
        "pilot_callsign": callsign or "Aircraft",
        "request_type": request_type,
        "priority_level": priority,
        "current_phase": "ROUTING",
        "required_agents": agents
    }

def _decide_routing(self, state: Dict[str, Any]) -> str:
    """
    Decide routing path based on request priority
    """
    if state.get("priority_level") == "EMERGENCY":
        return "emergency"
    elif state.get("required_agents"):
        return "normal" 
    else:
        return "synthesize"

# Add methods to SkyLinkNavigator
SkyLinkNavigator._atc_main_coordinator = _atc_main_coordinator
SkyLinkNavigator._decide_routing = _decide_routing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sub-Agent Routing (Placeholder)

# COMMAND ----------

def _route_to_subagents(self, state: ATCState) -> Dict[str, Any]:
    """
    Route requests to sub-agents - FIXED: Returns only subagent_responses
    """
    print("🔀 Routing to sub-agents...")
    
    required_agents = state.get("required_agents", [])
    callsign = state.get("pilot_callsign", "Aircraft")
    
    # Placeholder responses (will be replaced by actual sub-agent calls)
    mock_responses = {}
    
    for agent in required_agents:
        if agent == "geo_tracker":
            mock_responses[agent] = {
                "position": "10 miles southeast of KSEA", 
                "altitude": "5000 feet",
                "status": "ON_COURSE"
            }
        elif agent == "scheduler":
            mock_responses[agent] = {
                "clearance": f"{callsign} cleared to destination via filed route",
                "squawk": "1234"
            }
        elif agent == "weather":
            mock_responses[agent] = {
                "conditions": "VFR, winds 270 at 10 knots, visibility 10 miles"
            }
        elif agent == "comms_agent":
            mock_responses[agent] = {
                "intent_confidence": 0.9,
                "communication_quality": "CLEAR"
            }
    
    print(f"📡 Sub-agents called: {required_agents}")
    
    # FIXED: Return only the subagent_responses field
    return {"subagent_responses": mock_responses}

def _emergency_handler(self, state: ATCState) -> Dict[str, Any]:
    """
    Emergency priority handler - FIXED: Returns only needed fields
    """
    print("🚨 EMERGENCY: Activating priority protocols...")
    
    callsign = state.get("pilot_callsign", "Aircraft")
    
    # Emergency response (simplified)
    emergency_response = {
        "emergency": {
            "status": "DECLARED",
            "immediate_guidance": f"{callsign}, roger emergency. Squawk 7700. Turn heading 090, descend and maintain 3000 feet.",
            "services_notified": ["ATC Tower", "Emergency Services", "Fire Rescue"]
        }
    }
    
    # FIXED: Return only the needed fields
    return {"subagent_responses": emergency_response}

# Add methods to SkyLinkNavigator
SkyLinkNavigator._route_to_subagents = _route_to_subagents
SkyLinkNavigator._emergency_handler = _emergency_handler

# COMMAND ----------

# MAGIC %md
# MAGIC ## Response Synthesis

# COMMAND ----------

def _synthesize_atc_response(self, state: ATCState) -> Dict[str, Any]:
    """
    Synthesize sub-agent responses - FIXED: Returns only needed fields
    """
    print("🎙️ Synthesizing final ATC response...")
    
    callsign = state.get("pilot_callsign", "Aircraft")
    request_type = state.get("request_type", "")
    responses = state.get("subagent_responses", {})
    
    # Generate response based on request type
    if "emergency" in request_type or state.get("priority_level") == "EMERGENCY":
        emergency_data = responses.get("emergency", {})
        final_response = emergency_data.get("immediate_guidance", 
                                          f"{callsign}, emergency assistance provided.")
        confidence = 1.0
        
    elif "ifr_clearance" in request_type:
        scheduler_data = responses.get("scheduler", {})
        clearance = scheduler_data.get("clearance", f"{callsign} cleared as filed")
        squawk = scheduler_data.get("squawk", "1200")
        final_response = f"{callsign}, {clearance}. Squawk {squawk}. Contact departure 121.9."
        confidence = 0.9
        
    elif "weather_request" in request_type:
        weather_data = responses.get("weather", {})
        conditions = weather_data.get("conditions", "Weather information unavailable")
        final_response = f"{callsign}, current conditions: {conditions}."
        confidence = 0.85
        
    elif "traffic_advisory" in request_type:
        geo_data = responses.get("geo_tracker", {})
        position = geo_data.get("position", "position unknown")
        final_response = f"{callsign}, no reported traffic. Continue approach."
        confidence = 0.8
        
    else:
        final_response = f"{callsign}, SkyLink Navigator. Go ahead with your request."
        confidence = 0.7
    
    print(f"📻 Response ready (confidence: {confidence*100:.0f}%)")
    
    # FIXED: Return only the fields that need updating
    return {
        "final_response": final_response,
        "confidence_score": confidence,
        "current_phase": "RESPONDING"
    }

async def process_pilot_request(self, message: str) -> Dict[str, Any]:
    """
    Main entry point for processing pilot requests - FIXED
    """
    print(f"\n🎙️ SkyLink Navigator: '{message}'")
    
    # FIXED: Proper initial state with all required fields
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "pilot_callsign": None,
        "request_type": None,
        "priority_level": "NORMAL",
        "current_phase": "ANALYZING",
        "required_agents": [],
        "subagent_responses": {},
        "final_response": None,
        "confidence_score": 0.0
    }
    
    try:
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "response": result.get("final_response", "SkyLink Navigator error - please repeat."),
            "callsign": result.get("pilot_callsign", "Unknown"),
            "request_type": result.get("request_type", "unknown"),
            "confidence": result.get("confidence_score", 0.0),
            "required_agents": result.get("required_agents", [])
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "response": "SkyLink Navigator technical difficulties. Please repeat request.",
            "error": str(e)
        }

# Add methods to SkyLinkNavigator
SkyLinkNavigator._synthesize_atc_response = _synthesize_atc_response
SkyLinkNavigator.process_pilot_request = process_pilot_request

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize and Test

# COMMAND ----------

# Initialize SkyLink Navigator
print("🚀 Initializing SkyLink Navigator Main Agent...")
navigator = SkyLinkNavigator()
print("✅ Main ATC Agent ready!")
print(f"🔧 Sub-agent interfaces: {list(navigator.available_subagents.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Graph Visualization (FIXED)

# COMMAND ----------

# FIXED: Proper graph visualization with error handling
try:
    from IPython.display import Image, display
    print("🎨 Generating workflow visualization...")
    
    # This should work now without state errors
    mermaid_png = navigator.graph.get_graph().draw_mermaid_png()
    display(Image(mermaid_png))
    print("✅ Graph visualization successful!")
    
except Exception as e:
    print(f"❌ Visualization error: {e}")
    
    # Fallback text representation
    print("\n📋 Text-based workflow:")
    print("START → atc_main_agent → [emergency/normal] → atc_synthesize → END")

# MAGIC %md
# MAGIC ## Test Main Workflow

# COMMAND ----------

async def test_main_workflow():
    """Test the main ATC workflow"""
    
    test_cases = [
        "SkyLink, Delta 123 requesting IFR clearance to Seattle",
        "United 456, request current weather", 
        "American 789, Mayday engine failure!",
        "Southwest 321, requesting traffic advisory"
    ]
    
    print("🛫 Testing Main ATC Workflow...")
    print("=" * 50)
    
    for i, request in enumerate(test_cases, 1):
        print(f"\n📻 Test {i}: {request}")
        
        result = await navigator.process_pilot_request(request)
        print(f"📡 ATC: {result['response']}")
        print(f"📊 Sub-agents needed: {result.get('required_agents', [])}")
        print("-" * 30)

# FIXED: Run in async context
async def run_tests():
    await test_main_workflow()

# Run the tests
import asyncio
asyncio.run(run_tests())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Testing

# COMMAND ----------

# FIXED: Test with custom message in async context
async def run_interactive_test():
    test_message = "SkyLink, this is Cessna N123AB requesting taxi clearance"
    
    print(f"📻 Pilot: {test_message}")
    result = await navigator.process_pilot_request(test_message)
    print(f"📡 ATC: {result['response']}")
    print(f"🔧 Sub-agents required: {result.get('required_agents', [])}")

# Run the interactive test
asyncio.run(run_interactive_test())

# COMMAND ----------

# MAGIC %md  
# MAGIC ## Workflow Summary
# MAGIC
# MAGIC ### ✅ Main ATC Agent Complete:
# MAGIC - Request analysis and classification
# MAGIC - Pilot callsign extraction  
# MAGIC - Sub-agent routing logic
# MAGIC - Professional ATC response synthesis
# MAGIC - Emergency priority handling
# MAGIC
# MAGIC ### 🔌 Sub-Agent Integration Points:
# MAGIC - **GeoTracker**: Position/trajectory data
# MAGIC - **Scheduler**: Clearances and slots
# MAGIC - **Weather**: Meteorological data
# MAGIC - **CommsAgent**: Communication analysis
# MAGIC
# MAGIC ### 📋 Next Steps:
# MAGIC 1. Integrate actual sub-agents when ready
# MAGIC 2. Replace mock responses with real sub-agent calls
# MAGIC 3. Add error handling for sub-agent failures
# MAGIC 4. Implement sub-agent timeout handling
