"""
Hệ thực thi agent của HAgent.

Hệ điều phối đa agent cho học máy tự động được xây dựng trực tiếp trên
LangGraph và tích hợp với HAutoML.

Kiến trúc:
    Agent điều phối chính
        ├── Sub-agent phân tích dữ liệu
        ├── Sub-agent chọn mô hình
        ├── Sub-agent theo dõi huấn luyện
        └── Sub-agent đánh giá
"""
