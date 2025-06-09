# Databricks notebook source
# MAGIC %md
# MAGIC # SkyLink Navigator - Main ATC Agent with Tools
# MAGIC
# MAGIC This notebook implements the main SkyLink Navigator agent that has access to 4 tools and uses Claude Sonnet for final response generation.
# MAGIC
# MAGIC ## Workflow Overview:
# MAGIC ```
# MAGIC START → ATC_MAIN_AGENT → Claude Sonnet Response → END
# MAGIC            ↓
# MAGIC    ┌─────────────────┐
# MAGIC    │     TOOLS       │
# MAGIC    │  • GeoTracker   │
# MAGIC    │  • Scheduler    │
# MAGIC    │  • Weather      │
# MAGIC    │  • CommsAgent   │
# MAGIC    └─────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Required Libraries

# COMMAND ----------

from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import tool
import asyncio
from enum import Enum
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Tools for ATC Operations

# COMMAND ----------

@tool
def geo_tracker_tool(pilot_callsign: str, request_context: str) -> Dict[str, Any]:
    """
    GeoTracker Tool - Provides aircraft position, trajectory, and conflict detection
    
    Args:
        pilot_callsign: Aircraft identification
        request_context: Context of the pilot's request
    
    Returns:
        Dict containing position, altitude, trajectory data
    """
    print(f"🛰️ GeoTracker: Analyzing position for {pilot_callsign}")
    
    # Mock position data (will be replaced with real ADS-B data)
    return {
        "tool": "geo_tracker",
        "callsign": pilot_callsign,
        "current_position": {
            "latitude": 47.4502,
            "longitude": -122.3088,
            "altitude_ft": 5000,
            "ground_speed_kts": 180
        },
        "trajectory_status": "ON_COURSE",
        "distance_to_destination": "15 nautical miles",
        "flight_phase": "APPROACH",
        "conflicts": [],
        "runway_distance": "8 miles from runway 16R"
    }

@tool  
def scheduler_tool(pilot_callsign: str, request_context: str) -> Dict[str, Any]:
    """
    Scheduler Tool - Manages clearances, slots, and gate assignments
    
    Args:
        pilot_callsign: Aircraft identification  
        request_context: Context of the pilot's request
        
    Returns:
        Dict containing clearance, slot, and scheduling data
    """
    print(f"📋 Scheduler: Processing clearance for {pilot_callsign}")
    
    # Determine clearance type based on context
    if "clearance" in request_context.lower() or "ifr" in request_context.lower():
        return {
            "tool": "scheduler",
            "callsign": pilot_callsign,
            "clearance_issued": True,
            "clearance_text": f"{pilot_callsign} cleared to destination airport via filed route, maintain 5000 feet",
            "squawk_code": "1234",
            "departure_runway": "16R",
            "departure_frequency": "121.9",
            "estimated_departure": "15 minutes"
        }
    elif "taxi" in request_context.lower():
        return {
            "tool": "scheduler", 
            "callsign": pilot_callsign,
            "taxi_clearance": f"{pilot_callsign} taxi to runway 16R via taxiway Alpha, hold short of runway",
            "ground_frequency": "121.7",
            "expected_delay": "5 minutes"
        }
    else:
        return {
            "tool": "scheduler",
            "callsign": pilot_callsign,
            "status": "standing_by",
            "next_available_slot": "10 minutes"
        }

@tool
def weather_tool(pilot_callsign: str, request_context: str) -> Dict[str, Any]:
    """
    Weather Tool - Provides meteorological data and alerts
    
    Args:
        pilot_callsign: Aircraft identification
        request_context: Context of the pilot's request
        
    Returns:
        Dict containing weather conditions and alerts
    """
    print(f"🌤️ Weather: Gathering conditions for {pilot_callsign}")
    
    return {
        "tool": "weather",
        "callsign": pilot_callsign,
        "current_conditions": {
            "visibility": "10 statute miles",
            "ceiling": "Few clouds at 3000 feet",
            "wind": "270 degrees at 8 knots",
            "temperature": "15°C",
            "altimeter": "30.12 inches Hg"
        },
        "weather_alerts": [],
        "conditions_summary": "VFR conditions prevail",
        "runway_conditions": "Dry, no contamination",
        "forecast": "Conditions expected to remain VFR for next 2 hours"
    }

@tool
def comms_agent_tool(pilot_callsign: str, request_context: str) -> Dict[str, Any]:
    """
    CommsAgent Tool - Analyzes communication quality and intent
    
    Args:
        pilot_callsign: Aircraft identification
        request_context: Context of the pilot's request
        
    Returns:
        Dict containing communication analysis
    """
    print(f"📞 CommsAgent: Analyzing communication from {pilot_callsign}")
    
    # Analyze request type and quality
    intent_confidence = 0.95
    if "mayday" in request_context.lower() or "emergency" in request_context.lower():
        intent_confidence = 0.99
        communication_type = "EMERGENCY"
    elif "clearance" in request_context.lower():
        communication_type = "CLEARANCE_REQUEST"
    elif "weather" in request_context.lower():
        communication_type = "WEATHER_REQUEST"
    else:
        communication_type = "GENERAL_REQUEST"
    
    return {
        "tool": "comms_agent",
        "callsign": pilot_callsign,
        "intent_confidence": intent_confidence,
        "communication_type": communication_type,
        "message_completeness": "COMPLETE",
        "phraseology_compliance": "STANDARD",
        "requires_readback": communication_type == "CLEARANCE_REQUEST",
        "priority_level": "EMERGENCY" if communication_type == "EMERGENCY" else "NORMAL"
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## State Management for Tool-Based Workflow

# COMMAND ----------

class ATCState(TypedDict):
    """
    State management for tool-based ATC workflow
    """
    # Input and identification
    messages: Annotated[List[BaseMessage], lambda x, y: x + y if isinstance(y, list) else x + [y]]
    pilot_callsign: Optional[str]
    pilot_request: str
    
    # Tool outputs
    tool_results: Dict[str, Any]
    tools_called: List[str]
    
    # Final response
    atc_response: Optional[str]
    confidence_score: float
    next_actions: List[str]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main ATC Agent with Tools

# COMMAND ----------

class SkyLinkNavigator:
    """
    Main ATC Agent with integrated tools and Claude Sonnet response generation
    """
    
    def __init__(self):
        self.graph = None
        self.tools = [geo_tracker_tool, scheduler_tool, weather_tool, comms_agent_tool]
        self.tool_node = ToolNode(self.tools)
        self._build_atc_workflow()
    
    def _build_atc_workflow(self):
        """
        Build streamlined ATC workflow: Input → Tool Analysis → Claude Response → End
        """
        workflow = StateGraph(ATCState)
        
        # Main workflow nodes
        workflow.add_node("atc_agent", self._atc_main_agent)
        workflow.add_node("claude_response", self._generate_claude_response)
        
        # Set entry point
        workflow.set_entry_point("atc_agent")
        
        # Simple linear flow
        workflow.add_edge("atc_agent", "claude_response")
        workflow.add_edge("claude_response", END)
        
        self.graph = workflow.compile()
        print("✅ ATC Tool-Based Workflow compiled successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ATC Agent Implementation

# COMMAND ----------

def _atc_main_agent(self, state: ATCState) -> Dict[str, Any]:
    """
    Main ATC Agent - Analyzes request and calls appropriate tools
    """
    print("🎙️ ATC Main Agent: Processing pilot request with tools...")
    
    messages = state.get("messages", [])
    if not messages:
        return {
            "pilot_request": "No request received",
            "tool_results": {},
            "tools_called": []
        }
    
    # Extract pilot request
    pilot_request = messages[-1].content
    print(f"📨 Pilot Request: {pilot_request}")
    
    # Extract callsign
    words = pilot_request.split()
    callsign = None
    for word in words:
        if len(word) >= 3 and (word.isalnum() or any(c.isdigit() for c in word)):
            callsign = word.upper()
            break
    
    callsign = callsign or "Aircraft"
    
    # Determine which tools to call based on request content
    tools_to_call = self._determine_required_tools(pilot_request)
    print(f"🔧 Tools to call: {tools_to_call}")
    
    # Call the tools
    tool_results = {}
    for tool_name in tools_to_call:
        if tool_name == "geo_tracker":
            result = geo_tracker_tool.invoke({"pilot_callsign": callsign, "request_context": pilot_request})
        elif tool_name == "scheduler":
            result = scheduler_tool.invoke({"pilot_callsign": callsign, "request_context": pilot_request})
        elif tool_name == "weather":
            result = weather_tool.invoke({"pilot_callsign": callsign, "request_context": pilot_request})
        elif tool_name == "comms_agent":
            result = comms_agent_tool.invoke({"pilot_callsign": callsign, "request_context": pilot_request})
        
        tool_results[tool_name] = result
    
    return {
        "pilot_callsign": callsign,
        "pilot_request": pilot_request,
        "tool_results": tool_results,
        "tools_called": tools_to_call
    }

def _determine_required_tools(self, request: str) -> List[str]:
    """
    Determine which tools are needed based on pilot request
    """
    request_lower = request.lower()
    tools_needed = []
    
    # Always call comms_agent for communication analysis
    tools_needed.append("comms_agent")
    
    # Emergency - call all tools
    if "mayday" in request_lower or "emergency" in request_lower or "pan pan" in request_lower:
        return ["geo_tracker", "scheduler", "weather", "comms_agent"]
    
    # Clearance requests
    if "clearance" in request_lower or "ifr" in request_lower or "taxi" in request_lower:
        tools_needed.extend(["scheduler", "geo_tracker"])
    
    # Weather requests
    if "weather" in request_lower or "conditions" in request_lower or "metar" in request_lower:
        tools_needed.append("weather")
    
    # Traffic/position requests
    if "traffic" in request_lower or "position" in request_lower or "advisory" in request_lower:
        tools_needed.extend(["geo_tracker"])
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(tools_needed))

def _generate_claude_response(self, state: ATCState) -> Dict[str, Any]:
    """
    Generate final ATC response using Claude Sonnet (placeholder - will integrate with Databricks)
    """
    print("🤖 Generating Claude Sonnet response...")
    
    callsign = state.get("pilot_callsign", "Aircraft")
    request = state.get("pilot_request", "")
    tool_results = state.get("tool_results", {})
    
    # Create context for Claude
    context = f"""
    Pilot: {callsign}
    Request: {request}
    
    Tool Results:
    """
    
    for tool_name, result in tool_results.items():
        context += f"\n{tool_name}: {json.dumps(result, indent=2)}"
    
    # Placeholder for Claude Sonnet call (will be replaced with actual Databricks model serving)
    # For now, generate basic ATC response based on tool results
    atc_response = self._generate_mock_atc_response(callsign, request, tool_results)
    
    # Determine next actions
    next_actions = self._determine_next_actions(tool_results)
    
    return {
        "atc_response": atc_response,
        "confidence_score": 0.9,
        "next_actions": next_actions
    }

def _generate_mock_atc_response(self, callsign: str, request: str, tool_results: Dict[str, Any]) -> str:
    """
    Generate mock ATC response (will be replaced by Claude Sonnet)
    """
    # Check communication analysis first
    comms_data = tool_results.get("comms_agent", {})
    communication_type = comms_data.get("communication_type", "GENERAL_REQUEST")
    
    if communication_type == "EMERGENCY":
        return f"{callsign}, roger emergency. Squawk 7700. Turn heading 090, descend and maintain 3000 feet. Emergency services are standing by."
    
    elif communication_type == "CLEARANCE_REQUEST":
        scheduler_data = tool_results.get("scheduler", {})
        if scheduler_data.get("clearance_issued"):
            clearance = scheduler_data.get("clearance_text", f"{callsign} cleared as filed")
            squawk = scheduler_data.get("squawk_code", "1200")
            freq = scheduler_data.get("departure_frequency", "121.9")
            return f"{callsign}, {clearance}. Squawk {squawk}. Contact departure {freq}."
        elif scheduler_data.get("taxi_clearance"):
            return scheduler_data.get("taxi_clearance")
    
    elif communication_type == "WEATHER_REQUEST":
        weather_data = tool_results.get("weather", {})
        conditions = weather_data.get("conditions_summary", "Weather information unavailable")
        wind = weather_data.get("current_conditions", {}).get("wind", "Wind calm")
        return f"{callsign}, current conditions: {conditions}. {wind}."
    
    else:
        return f"{callsign}, SkyLink Navigator. Go ahead with your request."

def _determine_next_actions(self, tool_results: Dict[str, Any]) -> List[str]:
    """
    Determine recommended next actions based on tool results
    """
    actions = []
    
    # Check for emergency
    comms_data = tool_results.get("comms_agent", {})
    if comms_data.get("communication_type") == "EMERGENCY":
        actions.extend([
            "Monitor emergency frequency",
            "Coordinate with emergency services",
            "Clear airspace as needed"
        ])
    
    # Check for clearance requirements
    if comms_data.get("requires_readback"):
        actions.append("Await pilot readback confirmation")
    
    # Check for scheduling updates
    scheduler_data = tool_results.get("scheduler", {})
    if scheduler_data.get("expected_delay"):
        actions.append(f"Monitor for departure slot in {scheduler_data.get('expected_delay')}")
    
    return actions

# Add methods to SkyLinkNavigator
SkyLinkNavigator._atc_main_agent = _atc_main_agent
SkyLinkNavigator._determine_required_tools = _determine_required_tools
SkyLinkNavigator._generate_claude_response = _generate_claude_response
SkyLinkNavigator._generate_mock_atc_response = _generate_mock_atc_response
SkyLinkNavigator._determine_next_actions = _determine_next_actions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Processing Function

# COMMAND ----------

async def process_pilot_communication(self, pilot_input: str) -> Dict[str, Any]:
    """
    Main entry point for processing pilot communications
    """
    print(f"\n🎙️ SkyLink Navigator: Processing '{pilot_input}'")
    
    initial_state = {
        "messages": [HumanMessage(content=pilot_input)],
        "pilot_callsign": None,
        "pilot_request": "",
        "tool_results": {},
        "tools_called": [],
        "atc_response": None,
        "confidence_score": 0.0,
        "next_actions": []
    }
    
    try:
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "pilot_input": pilot_input,
            "atc_response": result.get("atc_response"),
            "callsign": result.get("pilot_callsign"),
            "tools_used": result.get("tools_called", []),
            "confidence": result.get("confidence_score", 0.0),
            "next_actions": result.get("next_actions", []),
            "tool_results": result.get("tool_results", {})
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "pilot_input": pilot_input,
            "atc_response": "SkyLink Navigator technical difficulties. Please repeat request.",
            "error": str(e)
        }

# Add method to SkyLinkNavigator
SkyLinkNavigator.process_pilot_communication = process_pilot_communication

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize and Test

# COMMAND ----------

# Initialize SkyLink Navigator with tools
print("🚀 Initializing SkyLink Navigator with Tools...")
navigator = SkyLinkNavigator()
print("✅ ATC Agent with tools ready!")
print(f"🔧 Available tools: {[tool.name for tool in navigator.tools]}")

# COMMAND ----------

from IPython.display import Image, display

display(Image(navigator.graph.get_graph().draw_mermaid_png()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Scenarios

# COMMAND ----------

async def test_atc_scenarios():
    """Test various ATC scenarios with the tool-based approach"""
    
    scenarios = [
        {
            "name": "IFR Clearance Request",
            "input": "SkyLink, Delta 123 requesting IFR clearance to Seattle"
        },
        {
            "name": "Emergency Declaration", 
            "input": "Mayday Mayday, United 456, engine failure, requesting immediate assistance"
        },
        {
            "name": "Weather Request",
            "input": "American 789, requesting current weather conditions"
        },
        {
            "name": "Taxi Clearance",
            "input": "Southwest 321, ready to taxi, requesting clearance to runway"
        },
        {
            "name": "Traffic Advisory",
            "input": "Cessna N123AB, requesting traffic advisory on final approach"
        }
    ]
    
    print("🛫 Testing ATC Tool-Based System...")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\n📻 Scenario: {scenario['name']}")
        print(f"Pilot: {scenario['input']}")
        print("-" * 40)
        
        result = await navigator.process_pilot_communication(scenario['input'])
        
        print(f"📡 ATC: {result['atc_response']}")
        print(f"🔧 Tools Used: {', '.join(result['tools_used'])}")
        print(f"📊 Confidence: {result['confidence']*100:.0f}%")
        
        if result['next_actions']:
            print(f"📋 Next Actions:")
            for action in result['next_actions']:
                print(f"   • {action}")
        
        print("=" * 60)

# FIXED: Run in async context
async def run_tests():
    await test_atc_scenarios()

# Run the tests
await run_tests()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Testing

# COMMAND ----------

# FIXED: Test with custom message in async context
async def run_interactive_test():
    test_input = "SkyLink, this is Learjet N789XY, requesting vectors around weather ahead"
    
    print(f"📻 Pilot: {test_input}")
    result = await navigator.process_pilot_communication(test_input)
    print(f"📡 ATC: {result['atc_response']}")
    print(f"🔧 Tools Used: {', '.join(result['tools_used'])}")
    print(f"📋 Next Actions: {result['next_actions']}")

# Run the interactive test
await run_interactive_test()

# COMMAND ----------

# MAGIC %md  
# MAGIC ## System Summary
# MAGIC
# MAGIC ### ✅ Tool-Based ATC System Complete:
# MAGIC - **GeoTracker Tool**: Aircraft position, trajectory, conflicts
# MAGIC - **Scheduler Tool**: Clearances, slots, taxi instructions
# MAGIC - **Weather Tool**: Meteorological data and alerts  
# MAGIC - **CommsAgent Tool**: Communication analysis and intent detection
# MAGIC
# MAGIC ### 🎯 Workflow:
# MAGIC 1. **Input**: Pilot communication received
# MAGIC 2. **Analysis**: ATC agent determines required tools
# MAGIC 3. **Tool Execution**: Calls appropriate tools based on request
# MAGIC 4. **Claude Integration**: Uses tool results to generate professional response
# MAGIC 5. **Output**: Professional ATC response with next actions
# MAGIC
# MAGIC ### 🔌 Integration Points:
# MAGIC - **Real Tools**: Replace mock tools with actual implementations
# MAGIC - **Claude Sonnet**: Integrate with Databricks Model Serving for response generation
# MAGIC - **ADS-B Data**: Connect GeoTracker to real aircraft position feeds
# MAGIC - **Weather APIs**: Connect Weather tool to live meteorological data
# MAGIC
# MAGIC ### 📋 Next Steps:
# MAGIC 1. Integrate with Databricks Claude Sonnet endpoint
# MAGIC 2. Replace tool mocks with real sub-agent implementations
# MAGIC 3. Add parallel tool execution for efficiency
# MAGIC 4. Implement error handling and fallback responses

# COMMAND ----------


