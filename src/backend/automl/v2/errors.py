"""Ngoại lệ chuyên biệt cho tầng lưu trữ đối tượng (MinIO/S3).

Dùng thay cho ``raise Exception(...)`` để caller có thể phân biệt loại lỗi
và không phải bắt ``Exception`` chung chung. Tất cả kế thừa ``StorageError``.
Thông điệp KHÔNG chứa credential; chi tiết gốc được giữ qua ``raise ... from e``.
"""

from __future__ import annotations


class StorageError(RuntimeError):
    """Lỗi cơ sở cho mọi thao tác lưu trữ đối tượng."""


class StorageConfigurationError(StorageError):
    """Cấu hình storage thiếu hoặc không hợp lệ (không kèm credential)."""


class BucketNotFoundError(StorageError):
    """Không tìm thấy bucket yêu cầu."""

    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        super().__init__(f"Không tìm thấy bucket: {bucket_name}")


class BucketCreationError(StorageError):
    """Không tạo được bucket (khác lỗi 'đã tồn tại')."""

    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        super().__init__(f"Không tạo được bucket: {bucket_name}")


class ObjectUploadError(StorageError):
    """Không upload được object lên storage."""

    def __init__(self, bucket_name: str, object_name: str) -> None:
        self.bucket_name = bucket_name
        self.object_name = object_name
        super().__init__(f"Lỗi upload object: s3://{bucket_name}/{object_name}")


class ObjectDownloadError(StorageError):
    """Không tải được object từ storage về đĩa."""

    def __init__(self, bucket_name: str, object_name: str) -> None:
        self.bucket_name = bucket_name
        self.object_name = object_name
        super().__init__(f"Lỗi download object: s3://{bucket_name}/{object_name}")


class ObjectCopyError(StorageError):
    """Không copy/move được object giữa các vị trí."""

    def __init__(self, source_key: str, dest_key: str) -> None:
        self.source_key = source_key
        self.dest_key = dest_key
        super().__init__(f"Lỗi copy object: {source_key} -> {dest_key}")


class ObjectRemoveError(StorageError):
    """Không xóa được object khỏi storage."""

    def __init__(self, bucket_name: str, object_name: str) -> None:
        self.bucket_name = bucket_name
        self.object_name = object_name
        super().__init__(f"Lỗi xóa object: s3://{bucket_name}/{object_name}")


class ObjectAccessError(StorageError):
    """Không truy cập/đọc được object (stat, get, tạo presigned URL)."""

    def __init__(self, bucket_name: str, object_name: str) -> None:
        self.bucket_name = bucket_name
        self.object_name = object_name
        super().__init__(f"Lỗi truy cập object: s3://{bucket_name}/{object_name}")
