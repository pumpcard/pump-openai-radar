"""
Pre-built Radar Agent for openai-agents SDK.
Requires: pip install pump-openai-radar[agents]
"""

from __future__ import annotations


def build_radar_agent(model: str = "gpt-4o", name: str = "OpenAI Radar Agent"):
    """
    Returns an openai-agents Agent pre-loaded with all Radar tools.

    Parameters
    ----------
    model:
        The model to use for the agent (default gpt-4o).
    name:
        Display name for the agent.

    Example
    -------
        from openai_radar.agents import build_radar_agent
        from agents import Runner

        agent = build_radar_agent()
        result = Runner.run_sync(agent, "Scan my org and flag any cost anomalies")
        print(result.final_output)
    """
    try:
        from agents import Agent
    except ImportError as e:
        raise ImportError(
            "openai-agents SDK not found. Install with: pip install pump-openai-radar[agents]"
        ) from e

    from openai_radar.agents.tools import (
        export_csv,
        export_drawio,
        run_findings,
        scan_assistants,
        scan_batch_jobs,
        scan_fine_tunes,
        scan_usage,
        scan_vector_stores,
    )

    return Agent(
        name=name,
        model=model,
        instructions="""
You are an OpenAI infrastructure FinOps analyst powered by pump-openai-radar.

Your capabilities:
- Scan assistants, vector stores, fine-tunes, batch jobs, and token usage
- Detect external service relationships (AWS, GCP, Azure, databases, etc.)
- Flag cost anomalies and misconfigurations via the findings engine
- Export inventory as CSV files or a draw.io architecture diagram

When the user asks to scan, analyze, or report on their OpenAI org:
1. Run the appropriate scan tools
2. Summarize what you found — resource counts, key costs, top findings
3. Highlight HIGH and CRITICAL findings first
4. Offer to export CSV or draw.io if not already done

Be concise, data-driven, and actionable. Translate token counts into cost
estimates where possible (use rough industry pricing as a proxy if not provided).
""",
        tools=[
            scan_assistants,
            scan_vector_stores,
            scan_fine_tunes,
            scan_batch_jobs,
            scan_usage,
            run_findings,
            export_csv,
            export_drawio,
        ],
    )
