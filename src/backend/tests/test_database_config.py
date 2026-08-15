from __future__ import annotations

from typing import ClassVar

import pytest

from database import database as database_module


class _FakeDatabase:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeMongoClient:
    instances: ClassVar[list[_FakeMongoClient]] = []

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.selected_names: list[str] = []
        type(self).instances.append(self)

    def __getitem__(self, name: str) -> _FakeDatabase:
        self.selected_names.append(name)
        return _FakeDatabase(name)


@pytest.fixture(autouse=True)
def _reset_fake_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeMongoClient.instances.clear()
    monkeypatch.setattr(database_module, "AsyncMongoClient", _FakeMongoClient)


@pytest.mark.asyncio
async def test_connection_selects_configured_application_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONGODB_CONNECT", "mongodb://mongo.internal:27017/")
    monkeypatch.setenv("MONGODB_DB_NAME", "hagent_application")

    database, client = await database_module.connection()

    assert database.name == "hagent_application"
    assert client is _FakeMongoClient.instances[0]
    assert client.uri == "mongodb://mongo.internal:27017/"
    assert client.selected_names == ["hagent_application"]


@pytest.mark.asyncio
async def test_connection_keeps_development_database_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONGODB_DB_NAME", raising=False)

    database, client = await database_module.connection()

    assert database.name == "AutoML"
    assert client is _FakeMongoClient.instances[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_name", ["", " ", "\t\r\n"])
async def test_connection_rejects_blank_database_name_before_creating_client(
    monkeypatch: pytest.MonkeyPatch,
    configured_name: str,
) -> None:
    monkeypatch.setenv("MONGODB_DB_NAME", configured_name)

    with pytest.raises(ValueError) as error:
        await database_module.connection()

    assert str(error.value) == "MONGODB_DB_NAME không được rỗng"
    assert _FakeMongoClient.instances == []
