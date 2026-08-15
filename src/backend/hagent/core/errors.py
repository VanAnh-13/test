"""
Hệ phân cấp exception cho toàn bộ hệ thống HAgent.

Thay thế mẫu trả lỗi dạng dict bằng cách phát sinh lớp con của HAgentError
với đầy đủ ngữ cảnh.

Phân cấp:
    HAgentError
    ├── PlanningError     — lỗi khi tạo hoặc kiểm tra kế hoạch
    ├── ExecutionError    — lỗi khi thực thi bước hoặc công cụ
    ├── WorldModelError   — lỗi khi mã hóa, dự đoán hoặc cập nhật World Model
    ├── LLMError          — lỗi thời gian chờ, tần suất hoặc phản hồi LLM
    └── ToolError         — lỗi cụ thể từ một lần gọi công cụ
"""

from __future__ import annotations

from typing import Any, ClassVar


class HAgentError(Exception):
    """Exception cơ sở cho toàn bộ hệ thống HAgent.

    Tham số:
        message: Mô tả lỗi ngắn gọn, không chứa dữ liệu nhạy cảm.
        context: Dict chứa thêm thông tin chẩn đoán (tool_name, plan_id, v.v.).
        cause: Exception gốc gây ra lỗi này để giữ chuỗi nguyên nhân.
    """

    error_code: ClassVar[str] = "HAGENT_ERROR"
    default_http_status_code: ClassVar[int] = 500

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    @property
    def http_status_code(self) -> int:
        """Mã HTTP công khai mà Bridge dùng cho loại lỗi này."""
        return self.default_http_status_code

    def _normalize_upstream_status(self, status_code: int | None) -> int:
        """Giữ lỗi phía gọi; chuẩn hóa lỗi dịch vụ phía trên thành 502/504."""
        if status_code is None:
            return self.default_http_status_code
        if 400 <= status_code < 500:
            return status_code
        if status_code == 504:
            return 504
        return self.default_http_status_code

    def __repr__(self) -> str:
        ctx = f", context={self.context!r}" if self.context else ""
        return f"{type(self).__name__}({self.message!r}{ctx})"

    def to_dict(self) -> dict[str, Any]:
        """Chuyển sang dict nội bộ để ghi log; không dùng làm phản hồi API."""
        d: dict[str, Any] = {
            "error_type": type(self).__name__,
            "message": self.message,
        }
        if self.context:
            d["context"] = self.context
        return d

    def to_public_dict(self) -> dict[str, str]:
        """Tạo payload ổn định mà không làm lộ context hoặc exception gốc."""
        return {"code": self.error_code, "message": self.message}


class PlanningError(HAgentError):
    """Lỗi trong quá trình tạo hoặc kiểm tra kế hoạch.

    Ví dụ:
        - LLM trả về kế hoạch không phân tích được.
        - Kế hoạch vi phạm ràng buộc về ngân sách hoặc công cụ.
        - Quá trình kiểm tra kế hoạch thất bại.
    """

    error_code = "PLANNING_ERROR"
    default_http_status_code = 422


class ExecutionError(HAgentError):
    """Lỗi khi thực thi một bước hoặc công cụ trong kế hoạch.

    Ví dụ:
        - Công cụ hết thời gian chờ.
        - Agent con không phản hồi.
        - Bước không thể thực thi do trạng thái không hợp lệ.
    """

    error_code = "EXECUTION_ERROR"
    default_http_status_code = 500

    def __init__(
        self,
        message: str,
        *,
        step_index: int | None = None,
        tool_name: str | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        ctx = dict(context or {})
        if step_index is not None:
            ctx["step_index"] = step_index
        if tool_name is not None:
            ctx["tool_name"] = tool_name
        super().__init__(message, context=ctx, cause=cause)
        self.step_index = step_index
        self.tool_name = tool_name


class WorldModelError(HAgentError):
    """Lỗi trong World Model khi mã hóa, dự đoán hoặc cập nhật.

    Ví dụ:
        - Bộ mã hóa không thể mã hóa quan sát.
        - Bộ dự đoán trả về NaN/Inf do bất ổn số học.
        - Kho trạng thái không thể lưu dữ liệu.
    """

    error_code = "WORLD_MODEL_ERROR"
    default_http_status_code = 503

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,  # mã hóa | dự đoán | cập nhật | độ bất ngờ
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        ctx = dict(context or {})
        if operation is not None:
            ctx["operation"] = operation
        super().__init__(message, context=ctx, cause=cause)
        self.operation = operation


class LLMError(HAgentError):
    """Lỗi khi gọi nhà cung cấp LLM.

    Ví dụ:
        - API giới hạn tần suất.
        - Hết thời gian chờ.
        - Định dạng phản hồi không hợp lệ.
        - Vượt quá cửa sổ ngữ cảnh.
    """

    error_code = "LLM_ERROR"
    default_http_status_code = 502

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        ctx = dict(context or {})
        if provider is not None:
            ctx["provider"] = provider
        if model is not None:
            ctx["model"] = model
        if status_code is not None:
            ctx["status_code"] = status_code
        super().__init__(message, context=ctx, cause=cause)
        self.provider = provider
        self.model = model
        self.status_code = status_code

    @property
    def http_status_code(self) -> int:
        """Ánh xạ mã trạng thái của nhà cung cấp LLM sang mã công khai."""
        return self._normalize_upstream_status(self.status_code)


class ToolError(HAgentError):
    """Lỗi cụ thể từ một lần gọi công cụ.

    Ví dụ:
        - API AutoML trả về 4xx/5xx.
        - Tập dữ liệu không tồn tại.
        - Tác vụ huấn luyện thất bại.
    """

    error_code = "TOOL_ERROR"
    default_http_status_code = 502

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        http_status: int | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        ctx = dict(context or {})
        if tool_name:
            ctx["tool_name"] = tool_name
        if http_status is not None:
            ctx["http_status"] = http_status
        super().__init__(message, context=ctx, cause=cause)
        self.tool_name = tool_name
        self.http_status = http_status

    @property
    def http_status_code(self) -> int:
        """Ánh xạ mã trạng thái của công cụ sang mã công khai."""
        return self._normalize_upstream_status(self.http_status)
