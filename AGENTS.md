# Quy tắc vận hành Agent

File này áp dụng cho toàn bộ dự án. `AGENTS.md` ở thư mục con có thể bổ sung
quy tắc chuyên biệt nhưng không được làm yếu các ràng buộc tại đây.

## 1. WIP = 1

- Agent chỉ được thực hiện đúng một tính năng tại một thời điểm.
- `feature_list.json` là nguồn sự thật về task và trạng thái task.
- Tối đa một task có `status: "in_progress"`. Task đó phải trùng với
  `current_task_id`.
- Không bắt đầu task mới khi task hiện tại chưa ở trạng thái `done` hoặc
  `blocked`.
- Không gộp yêu cầu mới vào task đang chạy. Hãy ghi yêu cầu đó thành task
  `backlog` riêng.

## 2. Hoàn thành triệt để

Một task chỉ được chuyển sang `done` khi đồng thời thỏa mãn tất cả điều kiện:

1. Mọi `acceptance_criteria` đã được đáp ứng.
2. Mọi lệnh trong `test_commands` đã thực sự chạy và trả về mã thoát `0`.
3. Không còn lỗi lint, typecheck hoặc test thuộc phạm vi task.
4. Checker/reviewer không còn phát hiện chặn bàn giao.
5. Không có thay đổi ngoài whitelist và không còn file rác do task tạo ra.
6. Bằng chứng kiểm thử đã được ghi vào `verification` và
   `claude-progress.md`.

Không chạy được test vì thiếu môi trường, dependency hoặc dịch vụ nghĩa là
task đang `blocked`, không phải `done`. Kiểm tra cú pháp hoặc đọc code không
được xem là bằng chứng tương đương với test thực thi.

## 3. Whitelist bắt buộc

Trước khi ghi file, Agent phải đọc task `in_progress` trong
`feature_list.json`.

Các file được phép sửa là hợp của:

- `policy.control_files`, chỉ để cập nhật trạng thái và nhật ký; và
- `allowed_files` của task `in_progress`.

Mọi đường dẫn trong `allowed_files` phải là đường dẫn tương đối chính xác tới
một file. Không dùng đường dẫn tuyệt đối, thư mục, `..`, glob hoặc wildcard.

Nếu cần sửa một file chưa có trong whitelist:

1. Dừng sửa file đó.
2. Giải thích lý do và ảnh hưởng.
3. Xin người dùng chấp thuận mở rộng phạm vi.
4. Chỉ cập nhật whitelist sau khi được chấp thuận.

Agent không được tự thêm file vào whitelist để hợp thức hóa thay đổi đã làm.
Không được sửa, xóa, format hàng loạt hoặc refactor module ngoài tác vụ chính.

## 4. File nhạy cảm và lõi hệ thống

Các đường dẫn khớp `policy.protected_paths` bị khóa mặc định. Chỉ được sửa khi
người dùng chấp thuận rõ ràng, task dành riêng cho thay đổi đó đã được tạo và
từng file đích xuất hiện chính xác trong `allowed_files`.

Không bao giờ:

- đọc, in, sao chép hoặc commit bí mật từ `.env`, key, token hay credential;
- sửa nội dung trong `.git/`;
- dùng lệnh dọn dẹp diện rộng như `git clean -fd`, `git reset --hard` hoặc xóa
  đệ quy;
- ghi đè hay xóa thay đổi có sẵn của người dùng;
- thay dependency, lockfile, CI/CD, auth, migration hoặc lõi Agent như một
  thay đổi phụ.

## 5. Vòng đời một phiên

### Bước 1 — Initialize

1. Chạy `bash init.sh`.
2. Đọc `feature_list.json` và phần bàn giao mới nhất trong
   `claude-progress.md`.
3. Kiểm tra trạng thái working tree nếu Git khả dụng.
4. Nếu có thay đổi ngoài phạm vi, giữ nguyên và không chạm vào chúng.

### Bước 2 — Execute

1. Chọn đúng một task.
2. Đặt task thành `in_progress`, đặt `current_task_id`, rồi xác nhận whitelist.
3. Chỉ triển khai thay đổi tối thiểu cần thiết cho acceptance criteria.
4. Không refactor ngoài phạm vi, kể cả khi thấy code có thể cải thiện.

### Bước 3 — Maker–Checker

1. Maker triển khai và chạy `test_commands`.
2. Khi test lỗi, lưu lỗi, chẩn đoán nguyên nhân và sửa trong whitelist.
3. Chạy lại cho tới khi đạt; không bỏ qua, làm yếu hoặc xóa test để có màu xanh.
4. Checker review độc lập nếu môi trường hỗ trợ; nếu không, Agent tự review
   diff theo acceptance criteria và ghi rõ giới hạn này.
5. Có lỗi chặn thì quay lại bước Execute với cùng task.

### Bước 4 — Clean State & Handoff

1. Chỉ xóa artifact tạm do chính task hiện tại tạo ra, sau khi xác minh từng
   đường dẫn cụ thể.
2. Kiểm tra lại danh sách file thay đổi và chạy bộ test cuối.
3. Cập nhật `verification`, chuyển task sang `done`, đặt
   `current_task_id: null`.
4. Ghi mục bàn giao vào `claude-progress.md`: phạm vi, quyết định, file đổi,
   lệnh test, kết quả và rủi ro còn lại.

## 6. Định dạng trạng thái

Trạng thái hợp lệ: `backlog`, `in_progress`, `blocked`, `done`.

- `backlog`: chưa bắt đầu.
- `in_progress`: task duy nhất đang được thực hiện.
- `blocked`: chưa hoàn thành và đang thiếu điều kiện bên ngoài.
- `done`: đã thỏa Definition of Done và có bằng chứng test.

Kết luận bắt buộc: **Không có test pass = chưa hoàn thành.**
