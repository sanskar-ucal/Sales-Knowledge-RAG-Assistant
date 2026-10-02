"""LLM-as-judge scoring for correctness, faithfulness and hallucination."""

import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict evaluator of a retrieval-augmented sales assistant.
Score the ASSISTANT ANSWER using the QUESTION, the REFERENCE ANSWER and the RETRIEVED CONTEXT.

- correctness (1-5): does the answer agree with the reference answer? 5 = fully correct and \
complete, 3 = partially correct, 1 = wrong or missing.
- faithfulness (1-5): is every claim supported by the retrieved context? 5 = fully supported.
- hallucination (true/false): true if the answer contains ANY claim not supported by the context.
- If the reference says the question is unanswerable, a correct abstention scores 5/5 and false.

Respond with JSON only: {{"correctness": int, "faithfulness": int, "hallucination": bool, \
"rationale": str}}""",
        ),
        (
            "human",
            "QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\n"
            "RETRIEVED CONTEXT:\n{context}\n\nASSISTANT ANSWER:\n{answer}",
        ),
    ]
)


class JudgeScore(BaseModel):
    correctness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    hallucination: bool
    rationale: str = ""


def parse_judge_output(text: str) -> JudgeScore:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    return JudgeScore.model_validate(json.loads(match.group(0)))


class LLMJudge:
    def __init__(self, llm: Runnable):
        self.chain = JUDGE_PROMPT | llm

    async def score(
        self, question: str, reference: str, context: str, answer: str
    ) -> JudgeScore | None:
        try:
            msg = await self.chain.ainvoke(
                {"question": question, "reference": reference, "context": context, "answer": answer}
            )
            return parse_judge_output(getattr(msg, "content", str(msg)))
        except Exception:
            logger.exception("Judge failed")
            return None
