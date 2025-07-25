from langgraph.graph import StateGraph
from state.state_schema import GraphState
from langgraph_nodes.null_check import null_check_node
from langgraph_nodes.null_critic import null_critic_node
from langgraph_nodes.null_router import null_router_node
from langgraph_nodes.duplicate_check import duplicate_check_node
from langgraph_nodes.duplicate_critic import duplicate_critic_node
from langgraph_nodes.duplicate_router import duplicate_router_node
from langgraph_nodes.type_check import type_check_node
from langgraph_nodes.type_critic import type_critic_node
from langgraph_nodes.type_router import type_router_node

builder = StateGraph(GraphState)

# Add only processing nodes, not router nodes
builder.add_node("null_check", null_check_node)
builder.add_node("null_critic", null_critic_node)

builder.add_node("duplicate_check", duplicate_check_node)
builder.add_node("duplicate_critic", duplicate_critic_node)

builder.add_node("type_check", type_check_node)
builder.add_node("type_critic", type_critic_node)

# Add placeholder nodes for missing ones
builder.add_node("feature_engineer", lambda state: state)
builder.add_node("drop_check", lambda state: state)

builder.set_entry_point("null_check")

# Connect nodes with direct edges and conditional edges
builder.add_edge("null_check", "null_critic")
builder.add_conditional_edges(
    "null_critic", 
    null_router_node,
    {
        "null_check": "null_check",
        "duplicate_check": "duplicate_check"
    }
)

builder.add_edge("duplicate_check", "duplicate_critic")
builder.add_conditional_edges(
    "duplicate_critic",
    duplicate_router_node,
    {
        "type_check": "type_check",
        "duplicate_check": "duplicate_check"
    }
)

builder.add_edge("type_check", "type_critic")
builder.add_conditional_edges(
    "type_critic",
    type_router_node,
    {
        "feature_engineer": "feature_engineer",
        "drop_check": "drop_check"
    }
)

builder.add_edge("feature_engineer", "drop_check")
builder.add_edge("drop_check", "__end__")
