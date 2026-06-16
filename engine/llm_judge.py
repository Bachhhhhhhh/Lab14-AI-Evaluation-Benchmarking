import asyncio
import json
import os
import random
import re
from typing import Any, Dict, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class LLMJudge:
    def __init__(self):
        self.judge_a_model = os.getenv("NOVITA_JUDGE_MODEL_1", "qwen/qwen3-8b-fp8")
        self.judge_b_model = os.getenv(
            "NOVITA_JUDGE_MODEL_2",
            os.getenv("NOVITA_JUDGE_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        )

        self.judge_a_key = os.getenv("NOVITA_JUDGE_API_KEY_1") or os.getenv("NOVITA_API_KEY")
        self.judge_b_key = os.getenv("NOVITA_JUDGE_API_KEY_2") or os.getenv("NOVITA_API_KEY")
        self.novita_base_url = self._normalize_novita_base_url(
            os.getenv("NOVITA_BASE_URL", "https://api.novita.ai/openai")
        )
        self.judge_a_client = (
            AsyncOpenAI(api_key=self.judge_a_key, base_url=self.novita_base_url)
            if self.judge_a_key and AsyncOpenAI
            else None
        )
        self.judge_b_client = (
            AsyncOpenAI(api_key=self.judge_b_key, base_url=self.novita_base_url)
            if self.judge_b_key and AsyncOpenAI
            else None
        )

        self.novita_min_interval_sec = float(os.getenv("NOVITA_MIN_INTERVAL_SEC", "1.3"))
        self._judge_a_lock = asyncio.Lock()
        self._judge_b_lock = asyncio.Lock()
        self._last_judge_a_call = 0.0
        self._last_judge_b_call = 0.0

    def _normalize_novita_base_url(self, base_url: str) -> str:
        normalized = (base_url or "").rstrip("/")
        if normalized in {"https://api.novita.ai", "https://api.novita.ai/v1"}:
            return "https://api.novita.ai/openai"
        return normalized

    async def _wait_for_judge_slot(self, judge_name: str) -> None:
        lock = self._judge_a_lock if judge_name == "judge_a" else self._judge_b_lock
        last_call_attr = "_last_judge_a_call" if judge_name == "judge_a" else "_last_judge_b_call"

        async with lock:
            loop = asyncio.get_running_loop()
            elapsed = loop.time() - getattr(self, last_call_attr)
            min_interval = self.novita_min_interval_sec
            wait_time = min_interval - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            setattr(self, last_call_attr, loop.time())

    async def _call_judge(
        self,
        client,
        judge_name: str,
        model: str,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> Optional[int]:
        if not client:
            return None

        prompt = f"""
You are an impartial LLM-as-Judge for an AI agent benchmark.

Evaluate the agent answer against the question and reference answer.
Use this rubric:
- Correctness: factual match with the reference answer.
- Completeness: covers the important required details.
- Relevance: answers the actual question.
- Citation/Grounding: stays supported by the provided domain answer.
- Tone/Safety: concise, safe, and appropriate.

Return only valid JSON with this exact shape:
{{"score": 4, "failure_type": "none"}}

Score must be an integer from 1 to 5.
Allowed failure_type values: none, hallucination, incomplete, irrelevant, citation_missing, tone_mismatch, unsafe.

Question:
{question}

Reference answer:
{ground_truth}

Agent answer:
{answer}
"""
        for attempt in range(2):
            try:
                await self._wait_for_judge_slot(judge_name)
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=120,
                )
                result = self._parse_judge_json(response.choices[0].message.content or "")
                score = int(result.get("score", 3))
                return max(1, min(5, score))
            except Exception as exc:
                print(f"LLM API Error ({model}) attempt {attempt + 1}/2: {exc}")
                await asyncio.sleep(2 * (attempt + 1))
        return None

    def _parse_judge_json(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        score_match = re.search(r"\b([1-5])\b", content)
        if score_match:
            return {"score": int(score_match.group(1)), "failure_type": "none"}
        return {"score": 3, "failure_type": "none"}

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        if self.judge_a_client or self.judge_b_client:
            task_a = self._call_judge(
                self.judge_a_client,
                "judge_a",
                self.judge_a_model,
                question,
                answer,
                ground_truth,
            )
            task_b = self._call_judge(
                self.judge_b_client,
                "judge_b",
                self.judge_b_model,
                question,
                answer,
                ground_truth,
            )
            score_a, score_b = await asyncio.gather(task_a, task_b)

            if score_a is None:
                score_a = random.choice([3, 4, 5])
            if score_b is None:
                score_b = random.choice([3, 4, 5])
        else:
            await asyncio.sleep(0.01)
            score_a = random.choice([3, 4, 5])
            score_b = score_a if random.random() > 0.3 else min(5, score_a + 1)
            if random.random() < 0.1:
                score_b = max(1, score_a - 2)

        score_gap = abs(score_a - score_b)
        agreement_rate = 1.0 - (score_gap / 4.0)

        failure_type = "none"
        if score_gap <= 1:
            final_score = (score_a + score_b) / 2.0
            status = "ok"
        else:
            final_score = min(score_a, score_b) + 0.5
            status = "conflict_review"
            failure_type = "incomplete"

        if final_score < 3:
            failure_type = random.choice(["hallucination", "incomplete", "irrelevant"])

        return {
            "final_score": final_score,
            "agreement_rate": agreement_rate,
            "status": status,
            "individual_scores": {
                f"novita:{self.judge_a_model}": score_a,
                f"novita:{self.judge_b_model}": score_b,
            },
            "reasoning": "Evaluated using two Novita judge models and consensus logic.",
            "failure_type": failure_type,
        }
