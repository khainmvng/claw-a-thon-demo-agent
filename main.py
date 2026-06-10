import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from greennode_agentbase import (
    GreenNodeAgentBaseApp,
    RequestContext,
    PingStatus,
)

load_dotenv()

app = GreenNodeAgentBaseApp()

LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
if not LLM_MODEL or not LLM_BASE_URL or not LLM_API_KEY:
    raise ValueError(
        "LLM_MODEL, LLM_BASE_URL, and LLM_API_KEY environment variables are required. "
        "Set them in your .env file or use /agentbase-llm to get a platform API key."
    )

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

SYSTEM_PROMPT = """You are a senior technical interviewer with 10+ years of experience.
Your job is to evaluate a candidate's answer to an interview question and provide constructive feedback.

Evaluate the answer based on:
1. Accuracy — is the answer technically correct?
2. Depth — does the candidate show deep understanding or just surface knowledge?
3. Clarity — is the answer well-structured and easy to follow?

Respond in the following JSON format (no markdown, pure JSON):
{{
  "score": <integer 1-10>,
  "verdict": "<Excellent|Good|Average|Below Average|Poor>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<area to improve 1>", "<area to improve 2>"],
  "feedback": "<2-3 sentence overall feedback>",
  "model_answer_hint": "<brief hint at what an ideal answer would include>"
}}"""


@app.entrypoint
def handler(payload: dict, context: RequestContext) -> dict:
    question = payload.get("question", "").strip()
    answer = payload.get("answer", "").strip()
    domain = payload.get("domain", "General").strip()
    level = payload.get("level", "Mid").strip()

    if not question or not answer:
        return {
            "status": "error",
            "message": "Both 'question' and 'answer' fields are required.",
        }

    user_message = (
        f"Domain: {domain}\n"
        f"Level: {level}\n\n"
        f"Interview Question:\n{question}\n\n"
        f"Candidate's Answer:\n{answer}"
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    import json
    try:
        evaluation = json.loads(response.content)
    except json.JSONDecodeError:
        evaluation = {"raw": response.content}

    return {
        "status": "success",
        "domain": domain,
        "level": level,
        "evaluation": evaluation,
        "timestamp": datetime.now().isoformat(),
    }


@app.ping
def health_check() -> PingStatus:
    return PingStatus.HEALTHY


if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")
