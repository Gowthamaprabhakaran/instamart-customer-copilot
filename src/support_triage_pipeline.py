"""Ticket preparation, PII redaction, mock-LLM classification, budgets, and routing."""

from pathlib import Path
import json
import random
import re
import time
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm
INPUT_FILE = ROOT / "data" / "raw" / "tickets.csv"
PREPARED_FILE = ROOT / "outputs" / "tickets_prepared.csv"
BUDGET_FILE = ROOT / "outputs" / "triage_budget_demo.csv"
FINAL_FILE = ROOT / "outputs" / "triage_results.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
TOKEN_BUDGET = 6000
FINAL_TOKEN_BUDGET = 100000
ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"

_TICKETS = None
_ZERO_SHOT_PROMPT = None
_FEW_SHOT_PROMPT = None
_FINAL_RESULTS = None
_FINAL_TOKENS = 0
_FINAL_RETRIES = 0


def prepare_tickets():
    """Load the original raw ticket dataset."""
    global _TICKETS
    _TICKETS = pd.read_csv(INPUT_FILE)
    _TICKETS["text"] = _TICKETS["text"].fillna("").astype(str)
    return _TICKETS


def _redact_pii(tickets):
    def redact_text(text):
        text, email_count = re.subn(EMAIL_PATTERN, "[EMAIL]", text)
        text, phone_count = re.subn(PHONE_PATTERN, "[PHONE]", text)
        text, order_count = re.subn(ORDER_PATTERN, "[ORDER_ID]", text)
        return text, email_count, phone_count, order_count

    results = tickets["text"].apply(redact_text)
    tickets["redacted_text"] = results.apply(lambda x: x[0])
    tickets["email_count"] = results.apply(lambda x: x[1])
    tickets["phone_count"] = results.apply(lambda x: x[2])
    tickets["order_count"] = results.apply(lambda x: x[3])

    total_emails = tickets["email_count"].sum()
    total_phones = tickets["phone_count"].sum()
    total_orders = tickets["order_count"].sum()
    print("\n--- PII Redaction Summary ---")
    print("Emails redacted:", total_emails)
    print("Phone numbers redacted:", total_phones)
    print("Order IDs redacted:", total_orders)
    print("Total substitutions:", total_emails + total_phones + total_orders)

    remaining_emails = 0
    remaining_phones = 0
    remaining_orders = 0
    for text in tickets["redacted_text"]:
        remaining_emails += len(re.findall(EMAIL_PATTERN, text))
        remaining_phones += len(re.findall(PHONE_PATTERN, text))
        remaining_orders += len(re.findall(ORDER_PATTERN, text))
    print("\n--- Redaction Verification ---")
    print("Remaining emails:", remaining_emails)
    print("Remaining phone numbers:", remaining_phones)
    print("Remaining order IDs:", remaining_orders)
    assert remaining_emails == 0
    assert remaining_phones == 0
    assert remaining_orders == 0
    return tickets


def input_guardrails(tickets=None):
    """Apply empty/long-ticket guardrails and redact sensitive patterns."""
    global _TICKETS
    if tickets is None:
        if _TICKETS is None:
            prepare_tickets()
        tickets = _TICKETS

    def check_ticket(text):
        if not text.strip():
            return "skipped_empty"
        if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            return "flagged_long"
        return "ready"

    tickets["status"] = tickets["text"].apply(check_ticket)
    tickets["estimated_tokens"] = tickets["text"].apply(mock_llm.est_tokens)
    print("\n--- Input Guardrail Summary ---")
    print(tickets["status"].value_counts())
    print(
        "\nLong ticket token estimates:",
        sorted(
            tickets.loc[
                tickets["status"] == "flagged_long", "estimated_tokens"
            ].unique()
        ),
    )
    tickets = _redact_pii(tickets)

    PREPARED_FILE.parent.mkdir(parents=True, exist_ok=True)
    tickets[[
        "ticket_id",
        "customer_id",
        "true_category",
        "redacted_text",
        "status",
        "estimated_tokens",
        "email_count",
        "phone_count",
        "order_count",
    ]].to_csv(PREPARED_FILE, index=False)
    print("\nPrepared tickets saved successfully.")
    print("Total tickets:", len(tickets))
    _TICKETS = tickets
    return tickets


def _create_prompts():
    zero_shot_prompt = """
You are a customer support ticket classifier.

CLASSIFY the customer support ticket into exactly one category:
billing, delivery, technical, fraud.

Return a JSON-only response in this format:
{"category": "...", "reason": "..."}
Do not include any additional text outside the JSON.
"""
    few_shot_prompt = """
You are a customer support ticket classifier.

CLASSIFY the customer support ticket into exactly one category:
billing, delivery, technical, fraud.

EXAMPLES:

Ticket: I was charged twice for the same order.
Response: {"category": "billing", "reason": "Duplicate charge"}

Ticket: I received an order confirmation for an order I did not place.
Response: {"category": "fraud", "reason": "Possible unauthorized order"}

Ticket: My order has not arrived yet.
Response: {"category": "delivery", "reason": "Order not received"}


Return a JSON-only response in this format:
{"category": "...", "reason": "..."}
Do not include any additional text outside the JSON.
"""
    return zero_shot_prompt, few_shot_prompt


def _retry_model_call(messages, ticket_id):
    retry_count = 0
    for retry_number in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            return response, retry_count
        except mock_llm.TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2 ** retry_number) * random.uniform(0.9, 1.15)
            print(
                f"Ticket {ticket_id} | Retry {retry_number} | Wait: {wait:.4f} seconds"
            )
            retry_count += 1
            time.sleep(wait)


def _validate_response(reply_text):
    try:
        result = json.loads(reply_text)
        if (
            isinstance(result, dict)
            and result.get("category") in ALLOWED_CATEGORIES
        ):
            return "classified_json", result["category"]
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", reply_text.lower())
        found_categories = [word for word in words if word in ALLOWED_CATEGORIES]
        if len(found_categories) == 1:
            return "classified_keyword", found_categories[0]
        return "flagged_invalid_output", None


def _run_pipeline(tickets, prompt, run_name, token_budget=None):
    mock_llm.reset()
    print(f"\n========== {run_name} RUN ==========")
    results = []
    total_tokens = 0
    total_retries = 0

    for _, row in tickets.iterrows():
        ticket_id = row["ticket_id"]
        status = row["status"]
        result = {
            "ticket_id": ticket_id,
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "status": status,
            "category": None,
            "tokens_used": 0,
        }
        if status != "ready":
            results.append(result)
            continue
        if token_budget is not None and total_tokens >= token_budget:
            result["status"] = "budget_exhausted"
            results.append(result)
            continue

        user_text = row["redacted_text"]
        assert not re.search(EMAIL_PATTERN, user_text)
        assert not re.search(PHONE_PATTERN, user_text)
        assert not re.search(ORDER_PATTERN, user_text)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ]
        response, retries = _retry_model_call(messages, ticket_id)
        total_retries += retries
        reply_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        total_tokens += tokens_used
        result_status, category = _validate_response(reply_text)
        result["status"] = result_status
        result["category"] = category
        result["tokens_used"] = tokens_used
        results.append(result)

    results_df = pd.DataFrame(results)
    print("\nTotal eligible tickets:", (tickets["status"] == "ready").sum())
    print("Recovered transient errors:", total_retries)
    print("Total tokens used:", total_tokens)
    return results_df, total_tokens, total_retries


def _compare_prompts(zero_results, few_results, zero_tokens, few_tokens):
    zero_classified = zero_results[
        zero_results["status"].isin(["classified_json", "classified_keyword"])
    ]
    few_classified = few_results[
        few_results["status"].isin(["classified_json", "classified_keyword"])
    ]
    zero_accuracy = (
        zero_classified["category"] == zero_classified["true_category"]
    ).mean() * 100
    few_accuracy = (
        few_classified["category"] == few_classified["true_category"]
    ).mean() * 100
    zero_count = len(zero_classified)
    few_count = len(few_classified)

    print("\n--- Zero-shot Results ---")
    print("Classified tickets:", zero_count)
    print(f"Accuracy: {zero_accuracy:.1f}%")
    print("Total tokens:", zero_tokens)
    print("Tokens per classified ticket:", round(zero_tokens / zero_count, 2))
    print("\nStatus counts:")
    print(zero_results["status"].value_counts())
    print("\n--- Few-shot Results ---")
    print("Classified tickets:", few_count)
    print(f"Accuracy: {few_accuracy:.1f}%")
    print("Total tokens:", few_tokens)
    print("Tokens per classified ticket:", round(few_tokens / few_count, 2))
    print("\nStatus counts:")
    print(few_results["status"].value_counts())

    confusion = pd.crosstab(
        zero_classified["true_category"], zero_classified["category"]
    ).reindex(index=ALLOWED_CATEGORIES, columns=ALLOWED_CATEGORIES, fill_value=0)
    print("\n--- Zero-shot Confusion Matrix ---")
    print(confusion)
    print("\n--- Prompt Comparison ---")
    print(f"Zero-shot accuracy: {zero_accuracy:.1f}%")
    print(f"Few-shot accuracy: {few_accuracy:.1f}%")
    print("Zero-shot total tokens:", zero_tokens)
    print("Few-shot total tokens:", few_tokens)
    print("Zero-shot tokens per classified ticket:", round(zero_tokens / zero_count, 2))
    print("Few-shot tokens per classified ticket:", round(few_tokens / few_count, 2))


def classify_tickets(tickets=None):
    """Run zero-shot and few-shot classification comparisons and final few-shot pass."""
    global _ZERO_SHOT_PROMPT, _FEW_SHOT_PROMPT
    global _FINAL_RESULTS, _FINAL_TOKENS, _FINAL_RETRIES
    if tickets is None:
        if _TICKETS is None:
            prepare_tickets()
            input_guardrails()
        tickets = _TICKETS

    _ZERO_SHOT_PROMPT, _FEW_SHOT_PROMPT = _create_prompts()
    zero_results, zero_tokens, zero_retries = _run_pipeline(
        tickets, _ZERO_SHOT_PROMPT, "ZERO-SHOT"
    )
    few_results, few_tokens, few_retries = _run_pipeline(
        tickets, _FEW_SHOT_PROMPT, "FEW-SHOT"
    )
    print("\n--- Retry Verification ---")
    print("Zero-shot recovered errors:", zero_retries)
    print("Few-shot recovered errors:", few_retries)
    assert zero_retries == 19
    assert few_retries == 19
    _compare_prompts(zero_results, few_results, zero_tokens, few_tokens)

    # The final classification pass is performed here; run_pipeline resets mock_llm.
    _FINAL_RESULTS, _FINAL_TOKENS, _FINAL_RETRIES = _run_pipeline(
        tickets,
        _FEW_SHOT_PROMPT,
        "FINAL FEW-SHOT",
        token_budget=FINAL_TOKEN_BUDGET,
    )
    return _FINAL_RESULTS


def token_budget(tickets=None):
    """Run the original zero-shot token-budget demonstration and save its output."""
    global _ZERO_SHOT_PROMPT, _FEW_SHOT_PROMPT
    if tickets is None:
        if _TICKETS is None:
            prepare_tickets()
            input_guardrails()
        tickets = _TICKETS
    if _ZERO_SHOT_PROMPT is None:
        _ZERO_SHOT_PROMPT, _FEW_SHOT_PROMPT = _create_prompts()

    budget_results, total_tokens, retries = _run_pipeline(
        tickets,
        _ZERO_SHOT_PROMPT,
        "ZERO-SHOT TOKEN BUDGET",
        token_budget=TOKEN_BUDGET,
    )
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    budget_results[[
        "ticket_id", "customer_id", "status", "category", "tokens_used"
    ]].to_csv(BUDGET_FILE, index=False)
    exhausted_count = (budget_results["status"] == "budget_exhausted").sum()
    print("\n--- Token Budget Summary ---")
    print("Token budget:", TOKEN_BUDGET)
    print("Tokens used:", total_tokens)
    print("Budget exhausted tickets:", exhausted_count)
    print("\nBudget status counts:")
    print(budget_results["status"].value_counts())
    assert exhausted_count >= 1
    return budget_results


def route_tickets(final_results=None):
    """Apply the original routing conditions and save the final triage output."""
    global _FINAL_RESULTS
    if final_results is None:
        if _FINAL_RESULTS is None:
            if _TICKETS is None:
                prepare_tickets()
                input_guardrails()
            classify_tickets()
        final_results = _FINAL_RESULTS

    segments = pd.read_csv(SEGMENTS_FILE)
    segment_data = segments[
        ["customer_id", "segment_name", "is_premium"]
    ].drop_duplicates("customer_id")
    final_results = final_results.merge(
        segment_data, on="customer_id", how="left"
    )
    final_results["segment_name"] = final_results["segment_name"].fillna("")
    final_results["is_premium"] = (
        final_results["is_premium"].fillna(False).astype(bool)
    )

    def assign_route(row):
        if row["status"] in [
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        ]:
            return "human_review", "NORMAL"
        if row["category"] == "fraud":
            return "security_team", "HIGH"
        if row["is_premium"]:
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"

    routing = final_results.apply(assign_route, axis=1, result_type="expand")
    final_results["route"] = routing[0]
    final_results["priority"] = routing[1]
    final_columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]
    final_results[final_columns].to_csv(FINAL_FILE, index=False)
    print("\n--- Final Routing Summary ---")
    print(final_results["route"].value_counts())
    print("\n--- Final Priority Summary ---")
    print(final_results["priority"].value_counts())
    print("\nFinal ticket rows:", len(final_results))
    print("Final tokens used:", _FINAL_TOKENS)
    print("Final recovered errors:", _FINAL_RETRIES)
    assert len(final_results) == 164
    assert _FINAL_RETRIES == 19
    _FINAL_RESULTS = final_results
    return final_results


if __name__ == "__main__":
    prepare_tickets()
    input_guardrails()
    classify_tickets()
    token_budget()
    route_tickets()
