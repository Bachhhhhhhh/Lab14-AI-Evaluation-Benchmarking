# Reflection - Vu Xuan Bach

## 1. Module minh dong gop
Phat trien cac modules:
- `engine/retrieval_eval.py`
- `engine/llm_judge.py`
- `engine/runner.py`
- Thiet ke dataset trong `data/synthetic_gen.py`.

## 2. Ly do thiet ke
De he thong cham diem cong bang, minh da ap dung 2 LLM judges. Viec tinh toan Hit Rate va MRR la bat buoc de xem chat luong buoc retrieval co anh huong truc tiep toi final score hay khong.

## 3. Giai thich ky thuat
- **Hit Rate**: La ti le ma o do it nhat 1 tai lieu quan trong nam trong top K. Giup do do do bao phu cua retriever.
- **MRR**: Do vi tri cua tai lieu dung dau tien. Giup biet duoc doc dung co duoc uu tien len dau khong.
- **Agreement Rate**: 1 - score_gap / 4. Tinh toan su dong thuan giua 2 LLM judge.
- **Cohen's Kappa**: Duoc ban toi nhu mot phuong an tinh chat luong cua hai raters, nhung do phuc tap nen su dung Absolute Gap truoc.
- **Position Bias**: LLM co the thien vi response xuat hien truoc, minh se doi cho A va B de tinh lai neu co thoi gian.
- **Cost-quality tradeoff**: V2 ton nhieu token hon nhung dem lai chat luong Hit Rate va Score cao hon.

## 4. Kho khan gap phai
Rate limit tu API khi goi song song qua nhieu. Va tinh trang hai model cho diem xung dot manh.

## 5. Cach giai quyet
Su dung `asyncio.Semaphore` de gioi han concurrency = 5. Ap dung Conflict Resolution Rule cho phep fallback ve Human Review neu gap > 1.

## 6. Ket qua truoc/sau regression
V2 cai thien ro ret ve Hit Rate va Avg Score so voi V1.
