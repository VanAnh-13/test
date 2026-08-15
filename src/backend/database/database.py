import os

from pymongo import AsyncMongoClient

_DEFAULT_DATABASE_NAME = "AutoML"


def _application_database_name() -> str:
    configured_name = os.getenv("MONGODB_DB_NAME")
    if configured_name is None:
        return _DEFAULT_DATABASE_NAME
    if not configured_name.strip():
        raise ValueError("MONGODB_DB_NAME không được rỗng")
    return configured_name


async def connection():
    database_name = _application_database_name()
    client = AsyncMongoClient(os.getenv("MONGODB_CONNECT", "localhost:27017"))
    return client[database_name], client


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    dbname = connection()
    # Kiểm tra kết nối đã được thiết lập thành công hay không
    if dbname is not None:
        print("Kết nối đến cơ sở dữ liệu MongoDB thành công.")
    else:
        print("Kết nối đến cơ sở dữ liệu MongoDB không thành công.")
