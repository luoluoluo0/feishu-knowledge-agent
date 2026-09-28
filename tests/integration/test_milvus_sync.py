from __future__ import annotations

import os
from uuid import uuid4

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MILVUS_INTEGRATION") != "1",
    reason="需要显式启动 Milvus 集成测试",
)


def test_milvus_insert_query_delete_roundtrip():
    from pymilvus import DataType, MilvusClient

    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    collection = f"sync_ci_{uuid4().hex[:10]}"
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=32)
    schema.add_field("item_id", DataType.VARCHAR, max_length=16)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)
    indexes = client.prepare_index_params()
    indexes.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    try:
        client.create_collection(
            collection_name=collection, schema=schema, index_params=indexes
        )
        client.insert(
            collection_name=collection,
            data=[{"id": "chunk-1", "item_id": "1234567890abcdef", "vector": [1, 0, 0, 0]}],
        )
        client.flush(collection)
        rows = client.query(
            collection_name=collection,
            filter='item_id == "1234567890abcdef"',
            output_fields=["id", "item_id"],
        )
        assert rows == [{"id": "chunk-1", "item_id": "1234567890abcdef"}]
        client.delete(collection_name=collection, filter='id == "chunk-1"')
        client.flush(collection)
        assert client.query(
            collection_name=collection,
            filter='item_id == "1234567890abcdef"',
            output_fields=["id"],
        ) == []
    finally:
        if client.has_collection(collection):
            client.drop_collection(collection)
