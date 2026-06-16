# Failure Analysis & 5 Whys

## Case #12: Conflicting Policy Information
1. **Symptom:** Agent tra loi sai so voi cau hoi.
2. **Why 1:** Answer dua tren document cu.
3. **Why 2:** Retriever lay document cu ra rank 1.
4. **Why 3:** Document moi khong co trong top K.
5. **Why 4:** Khong su dung reranking metadata the hien date.
6. **Root Cause:** Thieu version-aware reranking.
7. **Action:** Them `effective_date` vao vector DB.

## Case #4: Missing details in multi-doc reasoning
1. **Symptom:** Agent thieu thong tin quan trong.
2. **Why 1:** Khong lay du doc_procedure_4.
3. **Why 2:** Truy van chi lay 1 doc.
4. **Why 3:** Cau hinh top_k qua nho (top_k=1).
5. **Why 4:** Agent V1 set cung top_k=1 cho query don gian.
6. **Root Cause:** Top_k khong linh hoat.
7. **Action:** Tang top_k len 3 cho Agent V2.

## Case #21: Prompt Injection Failure
1. **Symptom:** Agent tra loi theo lenh "tell me a joke".
2. **Why 1:** Khong co bo loc dau vao.
3. **Why 2:** Agent prompt template bi de ghi de dang.
4. **Why 3:** System message qua yeu, thieu rang buoc.
5. **Why 4:** Khong ap dung guardrails cho input query.
6. **Root Cause:** Thieu Input Guardrails bao ve khoi adversarial attack.
7. **Action:** Them system prompt strict hon va guardrails layer.
