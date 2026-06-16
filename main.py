import asyncio
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge


def calculate_judge_reliability(results):
    pairs = []
    for result in results:
        scores = list(result["judge"].get("individual_scores", {}).values())
        if len(scores) >= 2:
            pairs.append((round(scores[0]), round(scores[1])))

    if not pairs:
        return {
            "exact_agreement_rate": 0.0,
            "within_1_agreement_rate": 0.0,
            "avg_score_gap": 0.0,
            "cohen_kappa": 0.0,
            "samples": 0,
        }

    exact_matches = sum(1 for score_a, score_b in pairs if score_a == score_b)
    within_1 = sum(1 for score_a, score_b in pairs if abs(score_a - score_b) <= 1)
    avg_gap = sum(abs(score_a - score_b) for score_a, score_b in pairs) / len(pairs)
    labels = [1, 2, 3, 4, 5]

    observed_agreement = exact_matches / len(pairs)
    expected_agreement = 0.0
    for label in labels:
        p_a = sum(1 for score_a, _ in pairs if score_a == label) / len(pairs)
        p_b = sum(1 for _, score_b in pairs if score_b == label) / len(pairs)
        expected_agreement += p_a * p_b

    if expected_agreement == 1.0:
        cohen_kappa = 1.0
    else:
        cohen_kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

    return {
        "exact_agreement_rate": observed_agreement,
        "within_1_agreement_rate": within_1 / len(pairs),
        "avg_score_gap": avg_gap,
        "cohen_kappa": cohen_kappa,
        "samples": len(pairs),
    }

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    case_limit = int(os.getenv("BENCHMARK_CASE_LIMIT", "0"))
    if case_limit > 0:
        dataset = dataset[:case_limit]
        print(f"🧪 Smoke test mode: running first {len(dataset)} cases only.")

    agent = MainAgent(version=agent_version)
    evaluator = RetrievalEvaluator()
    judge = LLMJudge()
    
    runner = BenchmarkRunner(agent, evaluator, judge)
    results = await runner.run_all(dataset)

    total = len(results)
    if total == 0:
        return None, None
        
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    hit_rate = sum(r["retrieval"]["hit_rate"] for r in results) / total
    mrr = sum(r["retrieval"]["mrr"] for r in results) / total
    agreement_rate = sum(r["judge"]["agreement_rate"] for r in results) / total
    pass_rate = sum(1 for r in results if r["status"] == "pass") / total
    avg_latency = sum(r["latency_sec"] for r in results) / total
    total_cost = sum(r["estimated_cost_usd"] for r in results)
    judge_reliability = calculate_judge_reliability(results)

    summary = {
        "metadata": {
            "version": agent_version, 
            "total": total, 
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": {
            "avg_score": avg_score,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "agreement_rate": agreement_rate,
            "pass_rate": pass_rate,
            "avg_latency_sec": avg_latency,
            "total_cost_usd": total_cost
        },
        "judge_reliability": judge_reliability
    }
    return results, summary

async def main():
    print("\n--- RUNNING V1 ---")
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base")
    
    print("\n--- RUNNING V2 ---")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta_score = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    delta_hit_rate = v2_summary["metrics"]["hit_rate"] - v1_summary["metrics"]["hit_rate"]
    
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.2f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.2f}")
    print(f"Delta Score: {'+' if delta_score >= 0 else ''}{delta_score:.2f}")

    # Build final regression report
    v2_summary["regression"] = {
        "v1_avg_score": v1_summary["metrics"]["avg_score"],
        "v2_avg_score": v2_summary["metrics"]["avg_score"],
        "delta_score": delta_score,
        "v1_hit_rate": v1_summary["metrics"]["hit_rate"],
        "v2_hit_rate": v2_summary["metrics"]["hit_rate"],
        "delta_hit_rate": delta_hit_rate
    }
    
    # Release gate
    approve = True
    if delta_score < 0: approve = False
    if delta_hit_rate < -0.02: approve = False
    if v2_summary["metrics"]["agreement_rate"] < 0.70: approve = False
    
    v2_summary["metadata"]["release_decision"] = "APPROVE" if approve else "BLOCK_RELEASE"

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if approve:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
