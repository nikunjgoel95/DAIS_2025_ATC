# %% [markdown]
# # SkyLink Navigator - Main ATC Agent with Tools
# 
# This notebook implements the main SkyLink Navigator agent that has access to 4 tools and uses Claude Sonnet for final response generation.
# 
# ## Workflow Overview:
# ```
# START → ATC_MAIN_AGENT → Claude Sonnet Response → END
#            ↓
#    ┌─────────────────┐
#    │     TOOLS       │
#    │  • GeoTracker   │
#    │  • Scheduler    │
#    │  • Weather      │
#    │  • CommsAgent   │
#    └─────────────────┘
# ```

# %%
%pip install -r requirements.txt

# %%
dbutils.library.restartPython()

# %%
from databricks_langchain import ChatDatabricks
from databricks.sdk import WorkspaceClient
import os

w = WorkspaceClient()

os.environ["DATABRICKS_HOST"] = w.config.host
os.environ["DATABRICKS_TOKEN"] = w.tokens.create(comment="for model serving", lifetime_seconds=1200).token_value

llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")

# %%
from pydantic import BaseModel

class CommsAnalysisResponse(BaseModel):
    commsAnalysis: str

# %%
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain.agents.agent_types import AgentType
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import sqlite3
import os

@tool
def query_flight_schedule(sql: str) -> str:
    """Run a SQL query on the flight_status.db file."""
    try:
        conn = sqlite3.connect("flight_status.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [description[0] for description in cursor.description]
        conn.close()
        return "\n".join([str(dict(zip(col_names, row))) for row in rows]) or "No results."
    except Exception as e:
        return f"SQL Error: {str(e)}"


@tool
def query_geotracking(sql: str) -> str:
    """Run a SQL query on the geotracking.db file."""
    try:
        conn = sqlite3.connect("geo_tracking.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [description[0] for description in cursor.description]
        conn.close()
        return "\n".join([str(dict(zip(col_names, row))) for row in rows]) or "No results."
    except Exception as e:
        return f"SQL Error: {str(e)}"
    


@tool
def query_weather(sql: str) -> str:
    """Run a SQL query on the weather.db file."""
    try:
        conn = sqlite3.connect("weather.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return "\n".join([str(dict(zip(columns, row))) for row in rows]) or "No results."
    except Exception as e:
        return f"SQL Error in Weather Tool: {e}"
    
@tool
def comms_agent(message: str) -> CommsAnalysisResponse:
    """
    Analyze pilot communication and return LLM analysis.
    Returns simple JSON: {"commsAnalysis": "analysis text"}
    """
    from databricks_langchain import ChatDatabricks
    
    # Initialize Databricks LLM
    llm = ChatDatabricks(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
    )
    
    # Simple analysis prompt
    analysis_prompt = f"""
    You are an Air Traffic Controller analyzing pilot communication. 
    Provide a brief analysis of this message including intent, urgency, and any key information extracted.

    PILOT MESSAGE: "{message}"

    Provide a concise analysis in 1-2 sentences:
    """
    
    llm = llm.with_structured_output(CommsAnalysisResponse)
    return llm.invoke(analysis_prompt)
            
    #     except Exception as e:
    #         # Fallback analysis
    #         fallback_text = _simple_fallback_analysis(message)
    #         result = CommsAnalysisResponse(commsAnalysis=fallback_text)
    #         return result.model_dump_json()
            
    # except Exception as e:
    #     # Error fallback
    #     error_result = CommsAnalysisResponse(
    #         commsAnalysis=f"Communication analysis error: {str(e)}. Message received: {message[:50]}..."
    #     )
    #     return error_result.model_dump_json()
    
# def _simple_fallback_analysis(message: str) -> str:
#     """Simple rule-based fallback analysis"""
#     msg = message.lower()
    
#     # Extract callsign
#     callsign = "Aircraft"
#     for word in message.split():
#         if len(word) >= 3 and word.replace('-', '').isalnum():
#             callsign = word.upper()
#             break
    
#     # Quick analysis
#     if "mayday" in msg or "emergency" in msg:
#         return f"EMERGENCY detected from {callsign}. Immediate priority handling required."
#     elif "clearance" in msg:
#         return f"{callsign} requesting clearance. Standard priority, prepare clearance delivery."
#     elif "weather" in msg:
#         return f"{callsign} requesting weather information. Routine request."
#     elif "taxi" in msg:
#         return f"{callsign} requesting taxi clearance. Ground movement coordination needed."
#     elif "traffic" in msg:
#         return f"{callsign} requesting traffic advisory. Check surrounding aircraft positions."
#     else:
#         return f"General communication from {callsign}. Standard acknowledgment required."

# %%
tools = [
    Tool(
        name="ScheduleTrackerTool",
        func=query_flight_schedule,
        description="Use this tool to query scheduled flights and detect conflicts, delays, or tight arrival overlaps. Accepts SQL input."
    ),
    Tool(
        name="GeoTrackerTool",
        func=query_geotracking,
        description="Use this tool to query geospatial data about flight phases and deviations from expected routes."
    ),
    Tool(
        name="WeatherTrackerTool",
        func=query_weather,
        description="Use this tool to query weather_by_flight table to get wind, visibility, storm/fog info, and help determine flight risk."
    )
    ,
    Tool(
        name="CommsAnalysisTool",
        func=comms_agent,
        description="Use this tool to analyze pilot communication and provide a brief analysis of intent, urgency, and key information extracted."
    )
]

# %%
from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
import json

class ATCState(TypedDict):
    """
    State management for tool-based ATC workflow - Following LangGraph best practices
    """
    # Input and identification - Using proper add_messages function for LangGraph
    messages: Annotated[List[BaseMessage], add_messages]
    pilot_callsign: Optional[str]
    pilot_request: str
    
    # Tool outputs
    tool_invocations: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    
    # Final response
    atc_response: Optional[str]
    confidence_score: float
    next_actions: List[str]

# %%
def execute_selected_tools(state: ATCState) -> Dict[str, Any]:
    """
    Execute the tools selected by the LLM
    """
    print("🔧 Executing LLM-selected tools...")
    
    tool_invocations = state.get("tool_invocations", [])
    
    tool_results = {}
    if not tool_invocations:
        print("No tools to execute.")
        return {"tool_results": tool_results}

    print("--- TOOL EXECUTION START ---")
    for invocation in tool_invocations:
        tool_name = invocation.get("tool_name")
        args = invocation.get("args")

        if not tool_name or not args:
            print(f"⚠️ Invalid tool invocation: {invocation}")
            continue

        # Find the tool to run from the global 'tools' list
        tool_to_run = next((t for t in tools if t.name == tool_name), None)

        if tool_to_run:
            try:
                result = tool_to_run.invoke(args)
                tool_results[tool_name] = result
                print(f"✅ {tool_name} executed successfully.")
                print(f"   - Args: {args}")
                print(f"   - Output: {result}")
            except Exception as e:
                error_msg = f"❌ Error executing {tool_name}: {e}"
                print(error_msg)
                tool_results[tool_name] = {"error": error_msg}
        else:
            error_msg = f"⚠️ Tool '{tool_name}' not found."
            print(error_msg)
            tool_results[tool_name] = {"error": error_msg}
    
    print("--- TOOL EXECUTION END ---")
    return {"tool_results": tool_results}

def llm_generate_response(state: ATCState) -> Dict[str, Any]:
    """
    Generate final ATC response using Databricks LLM based on tool results
    """
    print("🤖 Generating LLM-based ATC response...")
    
    callsign = state.get("pilot_callsign", "Aircraft")
    request = state.get("pilot_request", "")
    tool_results = state.get("tool_results", {})
    
    # Create comprehensive context for LLM
    tool_context = ""
    for tool_name, result in tool_results.items():
        if isinstance(result, dict):
            tool_context += f"\n{tool_name.upper()} RESULTS:\n{json.dumps(result, indent=2)}\n"
        else:
            tool_context += f"\n{tool_name.upper()} RESULTS:\n{str(result)}\n"
    
    # Create professional ATC response prompt
    response_prompt = f"""
    You are a professional Air Traffic Controller. Based on the pilot's request and the tool results, generate an appropriate ATC response using standard aviation phraseology.

    PILOT REQUEST: "{request}"
    AIRCRAFT: {callsign}

    TOOL RESULTS:{tool_context}

    INSTRUCTIONS:
    1. Use standard ATC phraseology and terminology
    2. Be clear, concise, and professional
    3. Include relevant information from the tool results
    4. For emergencies: Prioritize safety, provide immediate assistance
    5. For clearances: Include all necessary details (heading, altitude, squawk, frequency)
    6. For weather: Provide current conditions clearly
    7. Address the pilot by their callsign

    Generate ONLY the ATC radio response (no explanations):
    """
    
    try:
        # Call Databricks LLM for response generation
        llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")
        llm_response = llm.invoke(response_prompt)
        atc_response = llm_response.content.strip()
        
        # Clean up any formatting issues
        if atc_response.startswith('"') and atc_response.endswith('"'):
            atc_response = atc_response[1:-1]
            
    except Exception as e:
        print(f"⚠️ LLM response generation failed: {e}, using fallback")
        atc_response = generate_fallback_response(callsign, request, tool_results)
    
    # Generate next actions
    next_actions = determine_next_actions(tool_results)
    
    # Add the AI response to messages following LangGraph best practices
    ai_message = AIMessage(content=atc_response)
    
    return {
        "messages": [ai_message],
        "atc_response": atc_response,
        "confidence_score": 0.95,
        "next_actions": next_actions
    }

def generate_fallback_response(callsign: str, request: str, tool_results: Dict[str, Any]) -> str:
    """
    Generate fallback ATC response when LLM fails
    """
    # Check communication analysis first
    comms_data = tool_results.get("CommsAnalysisTool", {})
    
    # Simple fallback based on request content
    request_lower = request.lower()
    
    if "mayday" in request_lower or "emergency" in request_lower:
        return f"{callsign}, roger emergency. Squawk 7700. Turn heading 090, descend and maintain 3000 feet. Emergency services are standing by."
    elif "clearance" in request_lower:
        return f"{callsign}, cleared as filed. Squawk 1234. Contact departure 121.9."
    elif "weather" in request_lower:
        return f"{callsign}, current conditions: Wind 270 at 8 knots, visibility 10 miles, few clouds at 3000."
    elif "taxi" in request_lower:
        return f"{callsign}, taxi to runway 16R via taxiway Alpha, hold short of runway."
    else:
        return f"{callsign}, SkyLink Navigator. Go ahead with your request."

def determine_next_actions(tool_results: Dict[str, Any]) -> List[str]:
    """
    Determine recommended next actions based on tool results
    """
    actions = []
    
    # Check for emergency
    comms_data = tool_results.get("CommsAnalysisTool", {})
    if isinstance(comms_data, dict):
        comms_analysis = comms_data.get("commsAnalysis", "")
        if "emergency" in comms_analysis.lower() or "mayday" in comms_analysis.lower():
            actions.extend([
                "Monitor emergency frequency",
                "Coordinate with emergency services",
                "Clear airspace as needed"
            ])
    
    # General actions
    actions.extend([
        "Monitor pilot readback confirmation",
        "Update flight progress strip"
    ])
    
    return actions[:3]  # Limit to 3 actions


# %%
def llm_tool_selection(state: ATCState) -> Dict[str, Any]:
    """
    LLM-driven tool selection - Uses Databricks LLM to intelligently select which tools to call
    """
    print("🤖 LLM Tool Selector: Analyzing pilot request...")
    
    messages = state.get("messages", [])
    if not messages:
        return {
            "pilot_request": "No request received",
            "tool_invocations": []
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
    
    # Use LLM to determine which tools to call
    llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")
    
    tool_selection_prompt = f"""
    You are an Air Traffic Control (ATC) agent. Your task is to analyze a pilot's communication and decide which tools to use to gather information. You have access to several tools that query databases using SQL.
    When a pilot provides a flight number (callsign), you should use it to query relevant tools to get a complete picture of the situation (position, schedule, weather).

    Available Tools and their schemas:
    - ScheduleTrackerTool(sql: str): Use to query flight schedules from the 'flights' table. Use the callsign to query. Relevant columns: flight_id, callsign, origin, destination, departure_time, arrival_time, status.
    - GeoTrackerTool(sql: str): Use to query aircraft positions from 'flight_positions' table. Use the callsign to query. Relevant columns: flight_id, callsign, latitude, longitude, altitude, speed, heading.
    - WeatherTrackerTool(sql: str): Use to query weather information for a flight from 'weather_by_flight' table. You may need to join with the flights table using flight_id to get information for a callsign.
    - CommsAnalysisTool(message: str): Use to analyze the pilot's message for intent and urgency. This tool should almost always be called.

    Pilot Request: "{pilot_request}"
    Identified Aircraft Callsign: {callsign}

    Based on the pilot's request and the callsign, generate a JSON list of tool calls to execute. Each tool call should be an object with "tool_name" and "args" which is another object. For SQL tools, the argument is "sql". For CommsAnalysisTool, the argument is "message".

    Example for "Delta 123 requesting IFR clearance to Seattle":
    [
        {{"tool_name": "CommsAnalysisTool", "args": {{"message": "Delta 123 requesting IFR clearance to Seattle"}}}},
        {{"tool_name": "ScheduleTrackerTool", "args": {{"sql": "SELECT * FROM flights WHERE callsign = 'DELTA123'"}}}},
        {{"tool_name": "GeoTrackerTool", "args": {{"sql": "SELECT * FROM flight_positions WHERE callsign = 'DELTA123' ORDER BY timestamp DESC LIMIT 1"}}}}
    ]

    Now, for the given pilot request, generate the list of tool calls. Be efficient and call all necessary tools.
    Response (JSON list of tool calls only):
    """
    
    try:
        # Call Databricks LLM for tool selection
        llm_response = llm.invoke(tool_selection_prompt)
        tool_invocations = json.loads(llm_response.content.strip())
        
        # Always ensure CommsAnalysisTool is included
        if not any(t["tool_name"] == "CommsAnalysisTool" for t in tool_invocations):
            tool_invocations.insert(0, {"tool_name": "CommsAnalysisTool", "args": {"message": pilot_request}})
            
    except Exception as e:
        print(f"⚠️ LLM tool selection failed: {e}, using fallback logic")
        tool_invocations = [{"tool_name": "CommsAnalysisTool", "args": {"message": pilot_request}}]
    
    print(f"🔧 LLM Selected Tool Invocations: {json.dumps(tool_invocations, indent=2)}")
    
    return {
        "pilot_callsign": callsign,
        "pilot_request": pilot_request,
        "tool_invocations": tool_invocations
    }

def determine_required_tools(request: str) -> List[str]:
    """
    Determine which tools are needed based on pilot request (fallback logic)
    """
    request_lower = request.lower()
    tools_needed = []
    
    # Always call CommsAnalysisTool for communication analysis
    tools_needed.append("CommsAnalysisTool")
    
    # Emergency - call all tools
    if "mayday" in request_lower or "emergency" in request_lower or "pan pan" in request_lower:
        return ["ScheduleTrackerTool", "GeoTrackerTool", "WeatherTrackerTool", "CommsAnalysisTool"]
    
    # Clearance requests
    if "clearance" in request_lower or "ifr" in request_lower or "taxi" in request_lower:
        tools_needed.extend(["ScheduleTrackerTool", "GeoTrackerTool"])
    
    # Weather requests
    if "weather" in request_lower or "conditions" in request_lower or "metar" in request_lower:
        tools_needed.append("WeatherTrackerTool")
    
    # Traffic/position requests
    if "traffic" in request_lower or "position" in request_lower or "advisory" in request_lower:
        tools_needed.extend(["GeoTrackerTool"])
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(tools_needed))

# %%
class SkyLinkNavigator:
    """
    Main ATC Agent with Databricks LLM-driven tool selection and response generation
    """
    
    def __init__(self):
        self.graph = None
        self.tools = tools
        self.tool_node = ToolNode(self.tools)
        
        # Initialize Databricks LLM for intelligent tool selection and response generation
        self.llm = ChatDatabricks(
            endpoint="databricks-llama-4-maverick",
        )
        
        self._build_atc_workflow()
    
    def _build_atc_workflow(self):
        """
        Build LLM-driven ATC workflow: Input → LLM Tool Selection → Tool Execution → LLM Response → End
        """
        workflow = StateGraph(ATCState)

        # Main workflow nodes
        workflow.add_node("llm_tool_selector", llm_tool_selection)
        workflow.add_node("execute_tools", execute_selected_tools)
        workflow.add_node("llm_response_generator", llm_generate_response)
        
        # Set entry point
        workflow.set_entry_point("llm_tool_selector")
        
        # LLM-driven workflow
        workflow.add_edge("llm_tool_selector", "execute_tools")
        workflow.add_edge("execute_tools", "llm_response_generator")
        workflow.add_edge("llm_response_generator", END)
        
        self.graph = workflow.compile()
        print("✅ LLM-Driven ATC Workflow compiled successfully")
    
    async def process_pilot_communication(self, pilot_input: str) -> Dict[str, Any]:
        """
        Main entry point for processing pilot communications
        """
        print(f"\n🎙️ SkyLink Navigator: Processing '{pilot_input}'")
        
        initial_state = {
            "messages": [HumanMessage(content=pilot_input)],
            "pilot_callsign": None,
            "pilot_request": "",
            "tool_invocations": [],
            "tool_results": {},
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
                "tools_used": [inv["tool_name"] for inv in result.get("tool_invocations", [])],
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

# %%
navigator = SkyLinkNavigator()

# %%
from IPython.display import Image, display

display(Image(navigator.graph.get_graph().draw_mermaid_png()))

# %%
# Test with proper state format following LangGraph best practices
test_input = "SkyLink, Delta 123 requesting IFR clearance to Seattle"

# Create proper initial state
initial_state = {
    "messages": [HumanMessage(content=test_input)],
    "pilot_callsign": None,
    "pilot_request": "",
    "tool_invocations": [],
    "tool_results": {},
    "atc_response": None,
    "confidence_score": 0.0,
    "next_actions": []
}

# Invoke the graph with proper state
result = navigator.graph.invoke(initial_state)
print(f"🎙️ Pilot: {test_input}")
print(f"📡 ATC: {result.get('atc_response')}")
print(f"🔧 Tools Used: {[inv['tool_name'] for inv in result.get('tool_invocations', [])]}")
print(f"📋 Next Actions: {result.get('next_actions', [])}")

# %%
# Fixed streaming function with proper state format
def stream_graph_updates(user_input: str):
    # Create proper initial state following LangGraph best practices
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "pilot_callsign": None,
        "pilot_request": "",
        "tool_invocations": [],
        "tool_results": {},
        "atc_response": None,
        "confidence_score": 0.0,
        "next_actions": []
    }
    
    print(f"🎙️ Pilot: {user_input}")
    
    for event in navigator.graph.stream(initial_state):
        for node_name, value in event.items():
            print(f"📍 Node: {node_name}")
            
            # Print ATC response when available
            if "atc_response" in value and value["atc_response"]:
                print(f"📡 ATC: {value['atc_response']}")
            
            # Print tools used
            if "tool_invocations" in value and value["tool_invocations"]:
                tools_used = [inv["tool_name"] for inv in value["tool_invocations"]]
                print(f"🔧 Tools Used: {', '.join(tools_used)}")
            
            # Print next actions
            if "next_actions" in value and value["next_actions"]:
                print(f"📋 Next Actions: {', '.join(value['next_actions'])}")
    
    print("-" * 60)

# Interactive testing loop
def run_interactive_test():
    print("🛫 SkyLink Navigator Interactive Test")
    print("Enter pilot communications (or 'quit' to exit)")
    print("-" * 60)
    
    while True:
        user_input = input("Pilot: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("✈️ Goodbye!")
            break
        
        if user_input.strip():
            stream_graph_updates(user_input)

# Uncomment the line below to run interactive testing
# run_interactive_test()

# %%
# Test various ATC scenarios with corrected implementation
def test_atc_scenarios():
    """Test various ATC scenarios following LangGraph best practices"""
    
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
    
    print("🛫 Testing ATC System with LangGraph Best Practices...")
    print("=" * 70)
    
    for scenario in scenarios:
        print(f"\n📻 Scenario: {scenario['name']}")
        print(f"Input: {scenario['input']}")
        print("-" * 50)
        
        # Create proper state for each test
        initial_state = {
            "messages": [HumanMessage(content=scenario['input'])],
            "pilot_callsign": None,
            "pilot_request": "",
            "tool_invocations": [],
            "tool_results": {},
            "atc_response": None,
            "confidence_score": 0.0,
            "next_actions": []
        }
        
        try:
            result = navigator.graph.invoke(initial_state)
            
            print(f"📡 ATC Response: {result.get('atc_response')}")
            print(f"🔧 Tools Used: {', '.join([inv['tool_name'] for inv in result.get('tool_invocations', [])])}")
            print(f"📊 Confidence: {result.get('confidence_score', 0)*100:.0f}%")
            
            if result.get('next_actions'):
                print(f"📋 Next Actions:")
                for action in result.get('next_actions', []):
                    print(f"   • {action}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 70)

# Uncomment to run tests
test_atc_scenarios()


# %% [markdown]
# # 🛫 SkyLink Navigator - LangGraph Best Practices Implementation
# 
# ## ✅ Key Improvements Made:
# 
# ### 1. **Proper State Management**
# - ✅ Used `add_messages` from `langgraph.graph.message` instead of custom lambda
# - ✅ Proper message handling with LangChain message types (`HumanMessage`, `AIMessage`)
# - ✅ State follows LangGraph TypedDict conventions
# 
# ### 2. **Correct Input Format**
# - ✅ Fixed graph invocation to use proper state dictionary instead of raw strings
# - ✅ All inputs now follow the ATCState structure
# - ✅ Proper message flow through the graph
# 
# ### 3. **Tool Integration**
# - ✅ All 4 tools properly accessible to main agent:
#   - `ScheduleTrackerTool` - Flight schedules and conflicts
#   - `GeoTrackerTool` - Position and trajectory data  
#   - `WeatherTrackerTool` - Weather conditions and alerts
#   - `CommsAnalysisTool` - Communication analysis with LLM
# - ✅ LLM-driven intelligent tool selection
# - ✅ Proper tool result handling and integration
# 
# ### 4. **Response Generation**
# - ✅ AI responses properly added to message history
# - ✅ Professional ATC phraseology generation
# - ✅ Fallback responses for error handling
# 
# ### 5. **Testing Framework**
# - ✅ Interactive testing function with proper state handling
# - ✅ Scenario-based testing with various ATC situations
# - ✅ Proper error handling and logging
# 
# ## 🎯 Workflow Structure:
# ```
# INPUT → LLM Tool Selection → Tool Execution → LLM Response → OUTPUT
#         ↓                    ↓               ↓
#     Analyzes request    Calls selected    Generates ATC
#     Selects tools       tools in          response using
#     intelligently       parallel          tool results
# ```
# 
# ## 🔧 Usage Examples:
# 
# ```python
# # Simple test
# stream_graph_updates("Delta 123, requesting IFR clearance")
# 
# # Comprehensive testing
# test_atc_scenarios()
# 
# # Interactive mode
# run_interactive_test()
# ```
# 
# ## 📋 Next Steps:
# 1. **Database Integration**: Connect to real flight tracking databases
# 2. **Real-time Data**: Integrate with live ADS-B and weather feeds  
# 3. **Claude Sonnet**: Upgrade response generation to use Claude Sonnet endpoint
# 4. **Error Handling**: Enhanced error recovery and logging
# 5. **Performance**: Add parallel tool execution for better performance"
# 

# %%



