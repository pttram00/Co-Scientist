"""Chatbot RAG tích hợp vào Co-Scientist.

Cho phép:
- Hỏi đáp tương tác về các giả thuyết/paper/báo cáo đã sinh (RAG với embedding
  local sentence-transformers).
- Comment trực tiếp từng giả thuyết: comment được LLM chấm điểm 0-10 theo 3 tiêu
  chí (correctness/novelty/feasibility) rồi lưu như một Review (review_type
  "user_comment"). Từ đó tự động chảy vào các agent ở vòng sau (Evolution đọc
  h.reviews[-2:], Reflection/ MetaReview đọc tất cả reviews) — không cần sửa
  agent nào.
- Tiếp tục luồng: chạy thêm iteration từ state.json đã có (resume).
"""
