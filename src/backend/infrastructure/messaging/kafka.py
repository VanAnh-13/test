"""Producer và consumer Kafka cho hàng đợi huấn luyện AutoML."""

import asyncio
import io
import json
import logging
import os

import joblib
import numpy as np
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import TopicPartition
from pymongo.asynchronous.database import AsyncDatabase

from automl.v2.master import get_config_hash, setup_job_tasks
from automl.v2.minio import minIOStorage
from database.get_dataset import MongoDataLoader

logger = logging.getLogger(__name__)

_DEFAULT_KAFKA_SERVER = "localhost:9092"
_DEFAULT_KAFKA_TOPIC = "example-topic"
_PRODUCER_START_ATTEMPTS = 15
_PRODUCER_RETRY_DELAY_SECONDS = 2
_MAX_CONCURRENT_HANDLERS = 1


# Producer dùng chung trong vòng đời ứng dụng.
producer_instance: AIOKafkaProducer | None = None


async def _ensure_topic(bootstrap: str, topic: str) -> None:
    """Tạo topic huấn luyện khi cụm Kafka mới chưa có topic."""
    try:
        from aiokafka.admin import AIOKafkaAdminClient, NewTopic

        admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
        await admin.start()
        try:
            existing = await admin.list_topics()
            if topic not in existing:
                await admin.create_topics(
                    [NewTopic(name=topic, num_partitions=1, replication_factor=1)]
                )
                logger.info("Đã tạo Kafka topic", extra={"topic": topic})
            else:
                logger.info("Kafka topic đã tồn tại", extra={"topic": topic})
        finally:
            await admin.close()
    # Đây là bước tối ưu tùy chọn; broker vẫn có thể tự tạo topic khi gửi lần đầu.
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Không thể bảo đảm Kafka topic trước khi gửi",
            extra={"topic": topic, "error_type": type(exc).__name__},
        )


async def start_producer() -> None:
    """Khởi tạo producer dùng chung với retry hữu hạn."""
    global producer_instance
    bootstrap = os.getenv("KAFKA_SERVER", _DEFAULT_KAFKA_SERVER)
    topic = os.getenv("KAFKA_TOPIC", _DEFAULT_KAFKA_TOPIC)

    # Kafka thường cần một khoảng ngắn để sẵn sàng sau khi container khởi động.
    last_error: Exception | None = None
    for attempt in range(1, _PRODUCER_START_ATTEMPTS + 1):
        try:
            await _ensure_topic(bootstrap, topic)
            producer_instance = AIOKafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            )
            await producer_instance.start()
            logger.info(
                "Kafka producer đã khởi động",
                extra={"bootstrap": bootstrap, "topic": topic},
            )
            return
        # Retry là boundary chủ động, cần giữ loại lỗi gốc nhưng không log credential.
        except Exception as exc:  # noqa: BLE001
            producer_instance = None
            last_error = exc
            logger.warning(
                "Khởi động Kafka producer chưa thành công",
                extra={
                    "attempt": attempt,
                    "max_attempts": _PRODUCER_START_ATTEMPTS,
                    "error_type": type(exc).__name__,
                },
            )
            await asyncio.sleep(_PRODUCER_RETRY_DELAY_SECONDS)
    raise RuntimeError("Không thể khởi động Kafka producer") from last_error


async def stop_producer() -> None:
    """Dừng và xóa tham chiếu producer dùng chung."""
    global producer_instance
    producer = producer_instance
    producer_instance = None
    if producer is not None:
        await producer.stop()
        logger.info("Kafka producer đã dừng")


def get_producer() -> AIOKafkaProducer:
    """Trả producer đã khởi động hoặc fail closed khi chưa sẵn sàng."""
    if producer_instance is None:
        raise RuntimeError("Kafka producer chưa được khởi động trong lifespan")
    return producer_instance


async def handle_training_job(
    job_id: str,
    id_data: str,
    id_user: str,
    config: dict,
    db: AsyncDatabase,
) -> None:
    """Chuẩn bị artifact dữ liệu rồi đăng ký job huấn luyện vào master."""
    dataset = MongoDataLoader(db)
    try:
        # Tiền xử lý một lần tại master để các worker không lặp lại công việc tốn tài nguyên.
        cache_bucket = "cache"
        models_bucket = "models"
        problem_type = config.get("problem_type") or "classification"
        list_feature = config.get("list_feature", [])
        target = config.get("target", "")

        config_hash = get_config_hash(id_data, list_feature, target, problem_type)
        data_cache_path = f"{id_data}/{config_hash}.npz"
        preprocessor_cache_path = f"{id_data}/{config_hash}_preprocessor.joblib"
        le_target_cache_path = f"{id_data}/{config_hash}_le_target.joblib"

        cache_exists = await asyncio.to_thread(
            minIOStorage.check_object_exists, cache_bucket, data_cache_path
        )

        if not cache_exists:
            (
                X_processed,
                y_processed,
                preprocessor,
                le_target,
            ) = await dataset.get_processed_data(
                id_data, list_feature, target, problem_type
            )

            with (
                io.BytesIO() as f_data,
                io.BytesIO() as f_pre,
                io.BytesIO() as f_target,
            ):
                # Chạy cả 3 tác vụ nén/dump song song trên các thread
                await asyncio.gather(
                    asyncio.to_thread(
                        np.savez_compressed, f_data, X=X_processed, y=y_processed
                    ),
                    asyncio.to_thread(joblib.dump, preprocessor, f_pre),
                    asyncio.to_thread(joblib.dump, le_target, f_target),
                )

                f_data.seek(0)
                f_pre.seek(0)
                f_target.seek(0)

                # Tải 3 file lên song song
                await asyncio.gather(
                    asyncio.to_thread(
                        minIOStorage.uploaded_object,
                        cache_bucket,
                        data_cache_path,
                        f_data.read(),
                    ),
                    asyncio.to_thread(
                        minIOStorage.uploaded_object,
                        cache_bucket,
                        preprocessor_cache_path,
                        f_pre.read(),
                    ),
                    asyncio.to_thread(
                        minIOStorage.uploaded_object,
                        cache_bucket,
                        le_target_cache_path,
                        f_target.read(),
                    ),
                )

        preprocessor_job_path = f"{id_user}/{job_id}/preprocessor.joblib"
        target_job_path = f"{id_user}/{job_id}/target.joblib"

        await asyncio.gather(
            asyncio.to_thread(
                minIOStorage.copy_object,
                source_bucket=cache_bucket,
                source_key=preprocessor_cache_path,
                dest_bucket=models_bucket,
                dest_key=preprocessor_job_path,
            ),
            asyncio.to_thread(
                minIOStorage.copy_object,
                source_bucket=cache_bucket,
                source_key=le_target_cache_path,
                dest_bucket=models_bucket,
                dest_key=target_job_path,
            ),
        )

        # Đăng ký task vào hàng đợi
        await setup_job_tasks(job_id, id_data, id_user, config, config_hash, db)
        logger.info("Đã gửi job huấn luyện tới master", extra={"job_id": job_id})
    except Exception as exc:
        logger.error(
            "Không thể chuẩn bị job huấn luyện",
            extra={"job_id": job_id, "error_type": type(exc).__name__},
        )
        raise


async def kafka_consumer_process(db: AsyncDatabase) -> None:
    """Chạy consumer Kafka và commit từng message sau khi xử lý."""
    consumer = None

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_HANDLERS)

    async def process_message_safely(message) -> None:
        """Xử lý một message và luôn tiến offset theo policy hiện tại."""
        job_id = message.key.decode("utf-8")
        topic_partition = TopicPartition(message.topic, message.partition)

        try:
            async with semaphore:
                id_data = message.value.get("id_data")
                id_user = message.value.get("id_user")
                config = message.value.get("config")

                await handle_training_job(job_id, id_data, id_user, config, db)

            await consumer.commit({topic_partition: message.offset + 1})
            logger.info("Đã commit Kafka offset", extra={"job_id": job_id})
        # Consumer hiện chủ động bỏ qua job lỗi sau khi ghi log để không kẹt partition.
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Xử lý Kafka message thất bại",
                extra={"job_id": job_id, "error_type": type(exc).__name__},
            )
            await consumer.commit({topic_partition: message.offset + 1})

    try:
        consumer = AIOKafkaConsumer(
            os.getenv("KAFKA_TOPIC", _DEFAULT_KAFKA_TOPIC),
            bootstrap_servers=os.getenv("KAFKA_SERVER", _DEFAULT_KAFKA_SERVER),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            group_id="train-consumer-group",
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        )

        await consumer.start()
        logger.info("Kafka consumer đang chạy")

        # Mỗi task vẫn đi qua semaphore nên số handler đồng thời không vượt cấu hình.
        async for message in consumer:
            asyncio.create_task(process_message_safely(message))
    except asyncio.CancelledError:
        logger.info("Kafka consumer đã nhận yêu cầu hủy")
        raise
    # Lifespan giám sát task consumer; boundary này giữ tiến trình không làm sập app.
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Kafka consumer gặp lỗi",
            extra={"error_type": type(exc).__name__},
        )
    finally:
        if consumer is not None:
            await consumer.stop()
        logger.info("Kafka consumer đã đóng")
