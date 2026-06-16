from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def score(self, case: Dict, response: Dict) -> Dict:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        
        # Determine top_k dynamically based on retrieved_ids or default to len
        top_k = len(retrieved_ids) if retrieved_ids else 3
        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        
        return {
            "retrieval": {
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
                "hit_rate": hit_rate,
                "mrr": mrr
            }
        }
