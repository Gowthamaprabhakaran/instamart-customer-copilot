"""Run all six Instamart Customer Intelligence & Support Copilot components."""

from src import data_pipeline
from src import models
from src import support_triage_pipeline
from src import policy_assistant
from src import refund_agent


def main():
    print("\n========== DATA PIPELINE ==========")
    data_pipeline.audit()
    data_pipeline.clean_data()
    data_pipeline.sql_analysis()
    data_pipeline.validate()


    print("\n========== MODELS ==========")
    models.rfm_and_segments()
    models.build_lapse_dataset()
    models.train_lapse_model()
    models.create_retention_list()

    print("\n========== SUPPORT-TRIAGE PIPELINE ==========")
    tickets = support_triage_pipeline.prepare_tickets()
    tickets = support_triage_pipeline.input_guardrails(tickets)
    support_triage_pipeline.classify_tickets(tickets)
    support_triage_pipeline.token_budget(tickets)
    support_triage_pipeline.route_tickets()

    print("\n========== POLICY ASSISTANT ==========")
    policy_assistant.load_policy()
    policy_assistant.create_chunks()
    policy_assistant.fit_tfidf()
    policy_assistant.retrieve()
    policy_assistant.answers()
    policy_assistant.evaluate_policy()

    print("\n========== REFUND AGENT ==========")
    refund_agent.load_scenarios()
    refund_agent.define_tools()
    refund_agent.run_safeguard_tests()
    refund_agent.manage_state()
    refund_agent.evaluate_agent()

    print("\nInstamart Customer Intelligence & Support Copilot completed.")


if __name__ == "__main__":
    main()
