# Thao tác với MinIO

"""
minio-data/
    user_id/
        job_id/
            {model_name}_{version}.pkl
"""

# Standard libraries
import io
import logging
import math
import os
import threading

# Third-party libraries
from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error
from urllib3 import PoolManager, Timeout
from urllib3.util.retry import Retry

from automl.v2.errors import (
    BucketCreationError,
    BucketNotFoundError,
    ObjectAccessError,
    ObjectCopyError,
    ObjectDownloadError,
    ObjectRemoveError,
    ObjectUploadError,
    StorageConfigurationError,
)

logger = logging.getLogger(__name__)

# Truy cập các biến môi trường (đã được entrypoint nạp qua load_dotenv)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")


def _healthcheck_timeout_seconds() -> float:
    raw = os.getenv("MINIO_HEALTHCHECK_TIMEOUT_SECONDS", "2")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError("MINIO_HEALTHCHECK_TIMEOUT_SECONDS không hợp lệ") from None
    if not math.isfinite(value) or value < 0.05 or value > 30:
        raise ValueError("MINIO_HEALTHCHECK_TIMEOUT_SECONDS không hợp lệ")
    return value


class MinIOStorage:
    def __init__(
        self,
        endpoint,
        access_key,
        secret_key,
        secure=False,
        healthcheck_timeout_seconds=2.0,
    ):
        try:
            if not all([endpoint, access_key, secret_key]):
                raise ValueError("Not found environment variables")

            self.__client = Minio(
                endpoint, access_key=access_key, secret_key=secret_key, secure=secure
            )
            health_http_client = PoolManager(
                timeout=Timeout(
                    connect=healthcheck_timeout_seconds,
                    read=healthcheck_timeout_seconds,
                ),
                retries=Retry(total=0, connect=0, read=0, redirect=0),
            )
            self.__healthcheck_client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                http_client=health_http_client,
            )
            self.__healthcheck_lock = threading.Lock()
        except Exception as exc:
            raise StorageConfigurationError(
                "MinIO configuration is unavailable"
            ) from exc

    def healthcheck(self) -> None:
        """Kiểm tra kết nối bằng thao tác đọc, không tạo bucket hoặc object."""
        if not self.__healthcheck_lock.acquire(blocking=False):
            raise RuntimeError("MinIO healthcheck đang được thực thi")
        try:
            self.__healthcheck_client.list_buckets()
        finally:
            self.__healthcheck_lock.release()

    def uploaded_object(self, bucket_name: str, object_name: str, object_bytes: bytes):
        try:
            self.__client.make_bucket(bucket_name)
        except S3Error as e:
            if e.code == "BucketAlreadyOwnedByYou" or e.code == "BucketAlreadyExists":
                pass
            else:
                raise BucketCreationError(bucket_name) from e

        with io.BytesIO(object_bytes) as data_stream:
            try:
                self.__client.put_object(
                    bucket_name,
                    object_name,
                    data=data_stream,
                    length=len(object_bytes),
                    content_type="application/octet-stream",
                )
                logger.info(
                    "Đã upload model lên MinIO: s3://%s/%s",
                    bucket_name,
                    object_name,
                )
            except Exception as e:
                raise ObjectUploadError(bucket_name, object_name) from e

    def move_model(
        self, source_bucket: str, source_model: str, dest_bucket: str, dest_model: str
    ):
        if not self.__client.bucket_exists(source_bucket):
            raise BucketNotFoundError(source_bucket)

        try:
            self.__client.make_bucket(dest_bucket)
        except S3Error as e:
            if e.code == "BucketAlreadyOwnedByYou" or e.code == "BucketAlreadyExists":
                pass
            else:
                raise BucketCreationError(dest_bucket) from e

        try:
            self.__client.copy_object(
                bucket_name=dest_bucket,
                object_name=dest_model,
                source=CopySource(source_bucket, source_model),
            )
            self.__client.remove_object(source_bucket, source_model)
        except Exception as e:
            raise ObjectCopyError(source_model, dest_model) from e

    def copy_object(
        self, source_bucket: str, source_key: str, dest_bucket: str, dest_key: str
    ):
        if not self.__client.bucket_exists(source_bucket):
            raise BucketNotFoundError(source_bucket)

        try:
            self.__client.make_bucket(dest_bucket)
        except S3Error as e:
            if e.code == "BucketAlreadyOwnedByYou" or e.code == "BucketAlreadyExists":
                pass
            else:
                raise BucketCreationError(dest_bucket) from e

        try:
            self.__client.copy_object(
                bucket_name=dest_bucket,
                object_name=dest_key,
                source=CopySource(source_bucket, source_key),
            )
        except Exception as e:
            raise ObjectCopyError(source_key, dest_key) from e

    def uploaded_dataset(self, bucket_name: str, object_name: str, parquet_buffer):
        try:
            self.__client.make_bucket(bucket_name)
        except S3Error as e:
            if e.code == "BucketAlreadyOwnedByYou" or e.code == "BucketAlreadyExists":
                pass
            else:
                raise BucketCreationError(bucket_name) from e

        try:
            self.__client.put_object(
                bucket_name,
                object_name,
                data=parquet_buffer,
                length=len(parquet_buffer.getvalue()),
                content_type="application/x-parquet",
            )
            logger.info(
                "Đã upload dataset lên MinIO: s3://%s/%s",
                bucket_name,
                object_name,
            )
        except Exception as e:
            raise ObjectUploadError(bucket_name, object_name) from e

    def check_object_exists(self, bucket_name: str, object_name: str) -> bool:
        try:
            self.__client.stat_object(bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey" or e.code == "NoSuchBucket" or "404" in str(e):
                return False
            raise ObjectAccessError(bucket_name, object_name) from e
        except Exception as e:
            raise ObjectAccessError(bucket_name, object_name) from e

    def get_object(self, bucket_name: str, object_name: str):
        data_stream = None
        try:
            data_stream = self.__client.get_object(bucket_name, object_name)
            data_bytes = data_stream.read()
            buffer = io.BytesIO(data_bytes)
            buffer.seek(0)

            return buffer
        except Exception as e:
            raise ObjectAccessError(bucket_name, object_name) from e
        finally:
            if data_stream:
                try:
                    data_stream.close()
                except Exception:
                    logger.warning(
                        "Không đóng được data stream cho s3://%s/%s",
                        bucket_name,
                        object_name,
                    )

    def remove_object(self, bucket_name: str, object_name: str):
        try:
            self.__client.remove_object(bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.debug(
                    "Object không tồn tại, bỏ qua: s3://%s/%s",
                    bucket_name,
                    object_name,
                )
                return True
            raise ObjectRemoveError(bucket_name, object_name) from e
        except Exception as e:
            raise ObjectRemoveError(bucket_name, object_name) from e

    def get_url(self, bucket_name: str, object_name: str):
        try:
            url = self.__client.presigned_get_object(bucket_name, object_name)
            # Không log URL đã ký (chứa credential tạm thời); chỉ log định danh object.
            logger.debug(
                "Đã tạo presigned URL cho s3://%s/%s",
                bucket_name,
                object_name,
            )
            return url
        except Exception as e:
            raise ObjectAccessError(bucket_name, object_name) from e

    def download_model(self, bucket_name: str, object_name: str, local_temp_path: str):
        try:
            os.makedirs(os.path.dirname(local_temp_path), exist_ok=True)

            self.__client.fget_object(bucket_name, object_name, local_temp_path)
            logger.info("Đã tải model về: %s", local_temp_path)
            return local_temp_path
        except Exception as e:
            raise ObjectDownloadError(bucket_name, object_name) from e

    def list_objects(self, bucket_name: str) -> list[str]:
        """Liệt kê tên các object trong bucket (đệ quy)."""
        return [
            obj.object_name
            for obj in self.__client.list_objects(bucket_name, recursive=True)
        ]


minIOStorage = MinIOStorage(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    healthcheck_timeout_seconds=_healthcheck_timeout_seconds(),
)
