# Reflection — Lab 19

**Tên:** Phạm Hải Đăng
**Cohort:** 2A202601367
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

- **Exact**: Keyword (BM25) thường thắng hoặc ngang bằng Hybrid, vì chứa từ khoá kỹ thuật chính xác xuất hiện trong tài liệu.
- **Paraphrase**: Semantic (Vector) có tiềm năng thắng cao nhất do bắt được ngữ nghĩa dù không trùng từ khóa. (Lưu ý với model mặc định của lite path là `bge-small` tiếng Anh, điểm semantic tiếng Việt có thể thấp, nhưng với model tốt như `bge-m3` thì semantic sẽ thắng).
- **Mixed**: Hybrid (RRF) thắng tuyệt đối vì tận dụng được cả signal từ khoá exact và ý tưởng paraphrase bổ sung cho nhau.

**Khi nào KHÔNG dùng Hybrid:**
- Dùng **pure BM25** khi query chủ yếu là tìm kiếm ID, mã số đơn hàng, hay từ khóa đặc thù cần độ chính xác tuyệt đối mà vector khó thể hiện.
- Dùng **pure Vector** khi query là dạng câu hỏi tự nhiên thuần túy, tóm tắt dài, và không hề có keyword cụ thể để match. Ngoài ra có thể không dùng Hybrid nếu hệ thống yêu cầu độ trễ (latency) siêu thấp vì Hybrid yêu cầu chạy song song 2 retriever rồi tính RRF.

---

## Điều ngạc nhiên nhất khi làm lab này

Việc sử dụng RRF trong Hybrid search giúp tự động "hòa trộn" thứ hạng từ BM25 và Vector một cách cực kì hiệu quả mà không cần phải tinh chỉnh weight cho từng mô hình.


---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
