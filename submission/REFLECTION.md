# Reflection — Lab 19

**Tên:** Phạm Hải Đăng
**Cohort:** 2A202601367
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

- **Exact** (15 query): BM25 **96,7%** thắng sát hybrid 96,0%; semantic 80,0%. Query chứa đúng token xuất hiện verbatim trong doc nên thứ hạng BM25 gần tối ưu — RRF trộn thêm vài doc semantic sai làm hybrid mất 0,7 điểm.
- **Paraphrase** (15): cả 3 đều yếu — kw 33,3%, sem 28,7%, hyb 34,7%. Semantic *lẽ ra* phải thắng, nhưng `bge-small-en` là model **tiếng Anh** nên embed câu tiếng Việt diễn đạt lại rất kém. Bài học: **chọn model quan trọng hơn chọn thuật toán fusion**.
- **Mixed** (20): hybrid thắng rõ — **99,5%** vs kw 97,0%, sem 94,0%. Query có cả từ khoá exact lẫn ý diễn đạt lại, hai retriever bù khuyết nhau đúng như RRF thiết kế.
- **Trung bình 50 query:** hyb 79,0% > kw 77,8% > sem 70,2%.

**Khi nào KHÔNG dùng Hybrid:**
- **Pure BM25** cho ID, SKU, error code — token phải khớp chính xác, vector chỉ thêm nhiễu và tốn latency.
- **Pure Vector** khi query là câu hỏi tự nhiên dài, không chia token nào với doc.
- Khi **latency là ràng buộc cứng**: NB3 đo keyword P99 = 20,4 ms, còn hybrid phải embed thêm một câu query — bước embed đó chiếm gần toàn bộ P99.

---

## Điều ngạc nhiên nhất khi làm lab này

RRF trộn thứ hạng của BM25 và vector mà **không cần tune weight** cho từng
retriever — chỉ cần rank, không cần score, nên hai thang điểm hoàn toàn khác
nhau (BM25 không chuẩn hoá vs cosine 0–1) vẫn ghép được. Bất ngờ thứ hai là
latency: bước embed câu query chiếm gần như toàn bộ P99, không phải bước ANN.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
