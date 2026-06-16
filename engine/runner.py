import asyncio
import os
import time
from typing import List, Dict

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        concurrency = int(os.getenv("BENCHMARK_CONCURRENCY", "3"))
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run_single_test(self, test_case: Dict) -> Dict:
        async with self.semaphore:
            start_time = time.perf_counter()
            
            try:
                response = await self.agent.query(test_case["question"])
                ragas_scores = await self.evaluator.score(test_case, response)
                judge_result = await self.judge.evaluate_multi_judge(
                    test_case["question"], 
                    response["answer"], 
                    test_case["expected_answer"]
                )
                
                latency = time.perf_counter() - start_time
                
                final_score = judge_result["final_score"]
                hit_rate = ragas_scores["retrieval"]["hit_rate"]
                status = "pass" if final_score >= 3 and hit_rate == 1 else "fail"
                
                return {
                    "case_id": test_case["id"],
                    "question": test_case["question"],
                    "expected_answer": test_case["expected_answer"],
                    "agent_response": response["answer"],
                    "difficulty": test_case["difficulty"],
                    "category": test_case["category"],
                    "retrieval": ragas_scores["retrieval"],
                    "judge": judge_result,
                    "latency_sec": latency,
                    "tokens_used": response["metadata"]["tokens_used"],
                    "estimated_cost_usd": response["metadata"]["estimated_cost_usd"],
                    "status": status
                }
            except Exception as e:
                print(f"Error on case {test_case['id']}: {e}")
                return {}

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        tasks = [self.run_single_test(case) for case in dataset]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]
