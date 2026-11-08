
"""Tool-using refund agent with policy retrieval, safeguards, and checkpoints."""

from pathlib import Path
import json
import sqlite3
import sys
import time
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm

DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
AGENT_RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"
STATE_DIR = ROOT / "state"

from src import policy_assistant

_SCENARIOS = None
TOOLS_SPEC = []
TOOL_MAP = {}
# Backward-compatible internal aliases used by the existing pipeline.
_TOOLS_SPEC = TOOLS_SPEC
_TOOL_MAP = TOOL_MAP


def load_scenarios():
    """Load the seven supplied refund scenarios and initialize policy retrieval if needed."""
    global _SCENARIOS
    if policy_assistant._DOCUMENTS is None:
        policy_assistant.load_policy()
    if policy_assistant._CHUNKS is None:
        policy_assistant.create_chunks()
    if policy_assistant._VECTORIZER is None or policy_assistant._TFIDF_MATRIX is None:
        policy_assistant.fit_tfidf()
    _SCENARIOS = pd.read_csv(ROOT / "data" / "raw" / "agent_scenarios.csv")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nPolicy RAG initialized successfully.")
    print("Total policy chunks:", len(policy_assistant._CHUNKS))
    print(_SCENARIOS.to_string(index=False))
    return _SCENARIOS

def validate_completed_order(order_id):
    """Allow refund processing only for completed orders."""
    order = get_order(order_id)

    if "error" in order:
        return order

    if str(order["status"]).strip().lower() != "completed":
        return {
            "error": "Only completed orders are eligible for refunds."
        }

    return order

def get_order(order_id):
    with sqlite3.connect(DB_PATH) as con:
        order = pd.read_sql_query(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            con,
            params=(order_id,),
        )
    if order.empty:
        return {"error": "order not found"}
    order_record = order.iloc[0].to_dict()
    for key, value in order_record.items():
        if pd.isna(value):
            order_record[key] = None
    return order_record


def search_policy(query):
    return {"results": policy_assistant.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or pd.isna(delivery_minutes):
        return {"error": "Delivery time is unavailable"}
    if delivery_minutes > 45:
        refund_amount = float(order_amount)
    else:
        refund_amount = 0.0
    return {"refund_inr": round(refund_amount, 2)}


def append_ledger(tool_name, details):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        **details,
    }
    with open(LEDGER_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(ledger_entry) + "\n")
    return ledger_entry


def append_agent_run(result):
    """Save one complete agent run as a JSONL record."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_entry = {
        "scenario_id": result["scenario_id"],
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
    }

    with open(AGENT_RUNS_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(run_entry, default=str) + "\n")


def issue_refund(order_id, refund_inr, reason="Late delivery refund"):
    order = validate_completed_order(order_id)

    if "error" in order:
        return order

    if refund_inr <= 0:
        return {"error": "Refund amount must be greater than zero."}

    if refund_inr > 1000:
        return {
            "error": "Refunds above INR 1000 require human approval."
        }

    append_ledger(
        "issue_refund",
        {
            "order_id": order_id,
            "refund_inr": round(float(refund_inr), 2),
            "reason": reason,
            "status": "REFUND_ISSUED",
        },
    )

    return {"refund_inr": round(float(refund_inr), 2)}


def escalate_to_human(
    order_id,
    refund_inr,
    reason="Refund requires human approval",
):
    order = validate_completed_order(order_id)

    if "error" in order:
        return order

    if refund_inr <= 0:
        return {"error": "Refund amount must be greater than zero."}

    append_ledger(
        "escalate_to_human",
        {
            "order_id": order_id,
            "refund_inr": round(float(refund_inr), 2),
            "reason": reason,
            "status": "PENDING_HUMAN_APPROVAL",
        },
    )

    return {"refund_inr": round(float(refund_inr), 2)}

def define_tools():
    """Register the original five agent tools and their unchanged schemas."""
    global TOOLS_SPEC, TOOL_MAP, _TOOLS_SPEC, _TOOL_MAP
    TOOLS_SPEC = [
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Retrieve order details using the order ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Unique order ID",
                        }
                    },
                    "required": ["order_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_policy",
                "description": "Search the company policy documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Policy-related search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_refund",
                "description": "Calculate refund based on order amount and delivery time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_amount": {"type": "number"},
                        "delivery_minutes": {"type": "number"},
                    },
                    "required": ["order_amount", "delivery_minutes"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "issue_refund",
                "description": "Issue an approved refund of INR 1000 or less.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "refund_inr": {"type": "number"},
                    },
                    "required": ["order_id", "refund_inr"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "escalate_to_human",
                "description": "Escalate a refund request for human approval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "refund_inr": {"type": "number"},
                    },
                    "required": ["order_id", "refund_inr"],
                },
            },
        },
    ]
    TOOL_MAP = {
        "get_order": get_order,
        "search_policy": search_policy,
        "calculate_refund": calculate_refund,
        "issue_refund": issue_refund,
        "escalate_to_human": escalate_to_human,
    }
    _TOOLS_SPEC = TOOLS_SPEC
    _TOOL_MAP = TOOL_MAP
    return TOOLS_SPEC


def save_checkpoint(run_id, messages, steps, llm_calls, tokens_used):
    checkpoint = {
        "run_id": run_id,
        "scenario_id": run_id,
        "step_number": len(steps),
        "executed_tools": [step["tool"] for step in steps],
        "messages": messages,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }
    checkpoint_file = STATE_DIR / f"{run_id}.json"
    with open(checkpoint_file, "w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=4, default=str)
    return checkpoint_file


def _run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None,
):
    messages = [
        {
            "role": "system",
            "content": "You are an AGENT that processes refund requests using the available tools.",
        },
        {"role": "user", "content": user_message},
    ]
    steps = []
    llm_calls = 0
    tokens_used = 0
    start_time = time.monotonic()

    for iteration in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: Request transferred to human support.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        if tokens_used > token_limit:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: Request transferred to human support.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }
        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            final_answer = assistant_message.content
            status = final_answer.split(":")[0]
            return {
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in assistant_message.tool_calls
            ],
        })
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.name
            arguments = tool_call.arguments
            tool_function = TOOL_MAP[tool_name]
            observation = tool_function(**arguments)
            steps.append({
                "iteration": iteration + 1,
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(observation, default=str),
            })
            save_checkpoint(scenario_id, messages, steps, llm_calls, tokens_used)
            if crash_after_tools is not None and len(steps) >= crash_after_tools:
                return {
                    "scenario_id": scenario_id,
                    "status": "SIMULATED_CRASH",
                    "final_answer": "Agent interrupted. Checkpoint saved.",
                    "steps": steps,
                    "llm_calls": llm_calls,
                    "tokens_used": tokens_used,
                }

    return {
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK_MAX_ITERATIONS: Request transferred to human support.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    checkpoint_file = STATE_DIR / f"{run_id}.json"
    if not checkpoint_file.exists():
        return {
            "scenario_id": run_id,
            "status": "CHECKPOINT_NOT_FOUND",
            "final_answer": "No saved checkpoint found.",
            "steps": [],
            "llm_calls": 0,
            "tokens_used": 0,
        }
    with open(checkpoint_file, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)
    messages = checkpoint["messages"]
    steps = checkpoint["steps"]
    llm_calls = checkpoint["llm_calls"]
    tokens_used = checkpoint["tokens_used"]
    start_time = time.monotonic()

    for iteration in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            return {
                "scenario_id": run_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "TIMEOUT: Human review required.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        if tokens_used > token_limit:
            return {
                "scenario_id": run_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "TOKEN_LIMIT: Human review required.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }
        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            final_answer = assistant_message.content
            return {
                "scenario_id": run_id,
                "status": final_answer.split(":")[0],
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in assistant_message.tool_calls
            ],
        })
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.name
            arguments = tool_call.arguments
            tool_function = TOOL_MAP[tool_name]
            observation = tool_function(**arguments)
            steps.append({
                "iteration": iteration + 1,
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(observation, default=str),
            })
            save_checkpoint(run_id, messages, steps, llm_calls, tokens_used)

    return {
        "scenario_id": run_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "MAX_ITERATIONS: Human review required.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
):
    """Run one ReAct refund scenario using the assignment-required interface."""
    return _run_agent(
        user_message=user_message,
        scenario_id=scenario_id,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )





def run_safeguard_tests():
    """Run token-limit and timeout safeguard tests using the actual S1 scenario."""


    scenarios = load_scenarios()

    
    s1_rows = scenarios.loc[scenarios["scenario_id"] == "S1"]

    if s1_rows.empty:
        raise ValueError("S1 scenario not found in agent_scenarios.csv")

    s1_scenario = s1_rows.iloc[0]

    
    s1_message = s1_scenario["message"]


    token_result = _run_agent(
        s1_message,
        "S1_TOKEN_TEST",
        token_limit=200,
    )

    append_agent_run(token_result)
    print("Token Limit Test:", token_result["status"])

    timeout_result = _run_agent(
        s1_message,
        "TIMEOUT_TEST",
        time_limit_seconds=0,
    )

    append_agent_run(timeout_result)
    print("Timeout Test:", timeout_result["status"])

    return {
        "token_limit_test": token_result,
        "timeout_test": timeout_result,
    }



def manage_state():
    """Run the original S1 simulated crash, checkpoint recovery, and verification."""
    if _SCENARIOS is None:
        load_scenarios()
    s1 = _SCENARIOS[_SCENARIOS["scenario_id"] == "S1"].iloc[0]
    result_s1 = _run_agent(
        user_message=s1["message"],
        scenario_id="S1",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10,
        crash_after_tools=2,
    )
    append_agent_run(result_s1)

    print("\n--- S1 Checkpoint Test ---")
    print("Status:", result_s1["status"])
    print("Final Answer:", result_s1.get("final_answer"))
    print("Tool Steps:", len(result_s1["steps"]))

    resumed_s1 = resume_agent(
        run_id="S1", max_iterations=6, token_limit=6000, time_limit_seconds=10
    )
    append_agent_run(resumed_s1)

    print("\n--- S1 Resume Test ---")
    print("Status:", resumed_s1["status"])
    print("Final Answer:", resumed_s1.get("final_answer"))
    print("Tool Steps:", len(resumed_s1["steps"]))
    print("\n--- Checkpoint Resume Verification ---")
    for step in resumed_s1["steps"]:
        print("Iteration:", step.get("iteration"), "| Tool:", step.get("tool"))
    return resumed_s1


def evaluate_agent():
    """Evaluate the ReAct agent against all seven original scenarios."""
    if _SCENARIOS is None:
        load_scenarios()
    print("\n--- ReAct Agent Performance ---")
    results = []
    for _, scenario in _SCENARIOS.iterrows():
        result = _run_agent(
            user_message=scenario["message"],
            scenario_id=scenario["scenario_id"],
        )

        append_agent_run(result)
        results.append(result)

        print(
            scenario["scenario_id"],
            "| Status:", result["status"],
            "| Model Calls:", result["llm_calls"],
            "| Tokens:", result["tokens_used"],
        )
    return results


if __name__ == "__main__":
    load_scenarios()
    define_tools()
    run_safeguard_tests()
    manage_state()
    evaluate_agent()
