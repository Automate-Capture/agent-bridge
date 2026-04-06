"""
Unit and functional tests for EventStreamAdapter.

Tests verify:
1. Single chunk ingest/egress
2. 100-chunk sequence order preservation
3. Round-trip lossless conversion
4. Edge cases: out-of-order input, field mismatches, malformed JSON
"""

import json
import pytest
from openclaw_gateway.adapters.event_stream import EventStreamAdapter
from openclaw_gateway.canonical_message import CanonicalMessage


@pytest.fixture
def adapter():
    """Create a fresh EventStreamAdapter instance for testing."""
    return EventStreamAdapter()


@pytest.fixture
def adapter_custom_id():
    """Create EventStreamAdapter with custom agent_id."""
    return EventStreamAdapter(agent_id="custom_stream_agent")


class TestEventStreamAdapterBasics:
    """Test basic adapter functionality."""

    @pytest.mark.asyncio
    async def test_single_chunk_ingest(self, adapter):
        """Test ingesting a single event stream chunk."""
        raw = json.dumps({
            "event": "chunk",
            "chunk_index": 0,
            "total_chunks": 1,
            "data": "hello world"
        }).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert isinstance(canonical, CanonicalMessage)
        assert canonical.intent == "stream_result"
        assert canonical.payload["chunk_index"] == 0
        assert canonical.payload["total_chunks"] == 1
        assert canonical.payload["data"] == "hello world"

    @pytest.mark.asyncio
    async def test_single_chunk_egress(self, adapter):
        """Test egressing a single canonical message to event stream format."""
        canonical = CanonicalMessage(
            message_id="550e8400-e29b-41d4-a716-446655440000",
            source_agent_id="event_stream_agent",
            conversation_id="test_conv",
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": 0,
                "total_chunks": 1,
                "data": "test data"
            }
        )

        result = await adapter.egress(canonical)

        assert isinstance(result, bytes)
        decoded = json.loads(result.decode("utf-8"))
        assert decoded["event"] == "chunk"
        assert decoded["chunk_index"] == 0
        assert decoded["total_chunks"] == 1
        assert decoded["data"] == "test data"

    @pytest.mark.asyncio
    async def test_agent_id_property(self, adapter):
        """Test that agent_id property returns consistent identifier."""
        assert adapter.agent_id == "event_stream_agent"
        assert adapter.protocol_name == "event_stream"

    @pytest.mark.asyncio
    async def test_agent_id_custom(self, adapter_custom_id):
        """Test custom agent_id."""
        assert adapter_custom_id.agent_id == "custom_stream_agent"


class TestOrderPreservation:
    """Test chunk ordering preservation."""

    @pytest.mark.asyncio
    async def test_100_chunk_ingest_order(self, adapter):
        """Test that 100 chunks preserve order via chunk_index sequence."""
        # Ingest 100 chunks in sequence
        for i in range(100):
            raw = json.dumps({
                "event": "chunk",
                "chunk_index": i,
                "total_chunks": 100,
                "data": f"chunk_{i}"
            }).encode("utf-8")

            canonical = await adapter.ingest(raw)

            # Verify chunk_index preserved
            assert canonical.payload["chunk_index"] == i
            assert canonical.payload["total_chunks"] == 100
            assert canonical.payload["data"] == f"chunk_{i}"

    @pytest.mark.asyncio
    async def test_10_chunk_order_sequence(self, adapter):
        """Test 10-chunk sequence for scaling verification."""
        chunk_indices = []
        for i in range(10):
            raw = json.dumps({
                "event": "chunk",
                "chunk_index": i,
                "total_chunks": 10,
                "data": f"data_{i}"
            }).encode("utf-8")

            canonical = await adapter.ingest(raw)
            chunk_indices.append(canonical.payload["chunk_index"])

        # Verify order matches expected sequence
        assert chunk_indices == list(range(10))

    @pytest.mark.asyncio
    async def test_1000_chunk_order_sequence(self, adapter):
        """Test 1000-chunk sequence for scaling verification."""
        chunk_indices = []
        for i in range(1000):
            raw = json.dumps({
                "event": "chunk",
                "chunk_index": i,
                "total_chunks": 1000,
                "data": f"data_{i}"
            }).encode("utf-8")

            canonical = await adapter.ingest(raw)
            chunk_indices.append(canonical.payload["chunk_index"])

        # Verify order matches expected sequence
        assert chunk_indices == list(range(1000))
        assert len(chunk_indices) == 1000


class TestRoundTrip:
    """Test round-trip conversion preservation."""

    @pytest.mark.asyncio
    async def test_round_trip_single_chunk(self, adapter):
        """Test round-trip: JSON → ingest → egress → JSON preserves all fields."""
        original_raw = json.dumps({
            "event": "chunk",
            "chunk_index": 5,
            "total_chunks": 10,
            "data": "test_payload"
        }).encode("utf-8")

        # Ingest
        canonical = await adapter.ingest(original_raw)

        # Egress
        egressed = await adapter.egress(canonical)

        # Verify structure preserved
        result_dict = json.loads(egressed.decode("utf-8"))
        assert result_dict["chunk_index"] == 5
        assert result_dict["total_chunks"] == 10
        assert result_dict["data"] == "test_payload"
        assert result_dict["event"] == "chunk"

    @pytest.mark.asyncio
    async def test_round_trip_100_chunks(self, adapter):
        """Test round-trip with 100 chunks → canonical → chunks preserves all chunk_indices."""
        # Create 100 chunks in forward order
        ingested_chunks = []
        for i in range(100):
            raw = json.dumps({
                "event": "chunk",
                "chunk_index": i,
                "total_chunks": 100,
                "data": f"chunk_data_{i}"
            }).encode("utf-8")

            canonical = await adapter.ingest(raw)
            ingested_chunks.append(canonical)

        # Egress all back to event stream format
        egressed_chunks = []
        for canonical in ingested_chunks:
            result = await adapter.egress(canonical)
            result_dict = json.loads(result.decode("utf-8"))
            egressed_chunks.append(result_dict)

        # Verify chunk_indices match [0..99]
        chunk_indices = [chunk["chunk_index"] for chunk in egressed_chunks]
        assert chunk_indices == list(range(100))

        # Verify all data preserved
        for i, chunk in enumerate(egressed_chunks):
            assert chunk["data"] == f"chunk_data_{i}"
            assert chunk["total_chunks"] == 100
            assert chunk["event"] == "chunk"

    @pytest.mark.asyncio
    async def test_round_trip_100_chunks_full_sequence(self, adapter):
        """Test full round-trip preserves exact sequence."""
        original_chunks = [
            {
                "event": "chunk",
                "chunk_index": i,
                "total_chunks": 100,
                "data": f"data_{i:03d}"
            }
            for i in range(100)
        ]

        # Ingest all chunks
        canonical_messages = []
        for chunk in original_chunks:
            raw = json.dumps(chunk).encode("utf-8")
            canonical = await adapter.ingest(raw)
            canonical_messages.append(canonical)

        # Egress all chunks
        egressed_chunks = []
        for canonical in canonical_messages:
            result = await adapter.egress(canonical)
            egressed_chunks.append(json.loads(result.decode("utf-8")))

        # Extract and verify chunk_index sequence
        chunk_indices = [chunk["chunk_index"] for chunk in egressed_chunks]
        assert chunk_indices == list(range(100)), "chunk_index sequence mismatch"


class TestEdgeCases:
    """Test edge case behavior."""

    @pytest.mark.asyncio
    async def test_out_of_order_input_preserves_index(self, adapter):
        """Test that out-of-order chunk input still preserves chunk_index correctly."""
        # Send chunk 5 before chunk 3
        raw_5 = json.dumps({
            "event": "chunk",
            "chunk_index": 5,
            "total_chunks": 10,
            "data": "chunk_5"
        }).encode("utf-8")

        raw_3 = json.dumps({
            "event": "chunk",
            "chunk_index": 3,
            "total_chunks": 10,
            "data": "chunk_3"
        }).encode("utf-8")

        canonical_5 = await adapter.ingest(raw_5)
        canonical_3 = await adapter.ingest(raw_3)

        # Verify indices are preserved as-is (adapter doesn't reorder)
        assert canonical_5.payload["chunk_index"] == 5
        assert canonical_3.payload["chunk_index"] == 3

    @pytest.mark.asyncio
    async def test_total_chunks_mismatch_preserved(self, adapter):
        """Test that total_chunks mismatch is preserved (not validated by adapter)."""
        # Claim total_chunks=100 but only send one chunk
        raw = json.dumps({
            "event": "chunk",
            "chunk_index": 0,
            "total_chunks": 100,
            "data": "single_chunk"
        }).encode("utf-8")

        canonical = await adapter.ingest(raw)

        # Adapter should preserve the claimed total_chunks
        assert canonical.payload["total_chunks"] == 100
        assert canonical.payload["chunk_index"] == 0

    @pytest.mark.asyncio
    async def test_missing_chunk_index_defaults_to_zero(self, adapter):
        """Test that missing chunk_index defaults to 0."""
        raw = json.dumps({
            "event": "chunk",
            "total_chunks": 1,
            "data": "test"
        }).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.payload["chunk_index"] == 0
        assert canonical.payload["total_chunks"] == 1

    @pytest.mark.asyncio
    async def test_missing_total_chunks_defaults_to_one(self, adapter):
        """Test that missing total_chunks defaults to 1."""
        raw = json.dumps({
            "event": "chunk",
            "chunk_index": 0,
            "data": "test"
        }).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.payload["chunk_index"] == 0
        assert canonical.payload["total_chunks"] == 1

    @pytest.mark.asyncio
    async def test_missing_data_defaults_to_empty_string(self, adapter):
        """Test that missing data defaults to empty string."""
        raw = json.dumps({
            "event": "chunk",
            "chunk_index": 0,
            "total_chunks": 1
        }).encode("utf-8")

        canonical = await adapter.ingest(raw)

        assert canonical.payload["data"] == ""

    @pytest.mark.asyncio
    async def test_malformed_json_raises_error(self, adapter):
        """Test that malformed JSON raises ValueError."""
        raw = b"{invalid json}"

        with pytest.raises(ValueError, match="Failed to parse event stream JSON"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_non_utf8_bytes_raises_error(self, adapter):
        """Test that non-UTF-8 bytes raise ValueError."""
        raw = b"\xff\xfe invalid utf-8"

        with pytest.raises(ValueError, match="Failed to parse event stream JSON"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_chunk_index_non_integer_raises_error(self, adapter):
        """Test that non-integer chunk_index raises ValueError."""
        raw = json.dumps({
            "event": "chunk",
            "chunk_index": "not_an_int",
            "total_chunks": 1,
            "data": "test"
        }).encode("utf-8")

        with pytest.raises(ValueError, match="chunk_index must be an integer"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_total_chunks_non_integer_raises_error(self, adapter):
        """Test that non-integer total_chunks raises ValueError."""
        raw = json.dumps({
            "event": "chunk",
            "chunk_index": 0,
            "total_chunks": "not_an_int",
            "data": "test"
        }).encode("utf-8")

        with pytest.raises(ValueError, match="total_chunks must be an integer"):
            await adapter.ingest(raw)

    @pytest.mark.asyncio
    async def test_egress_missing_chunk_index_raises_error(self, adapter):
        """Test that egress without chunk_index in payload raises ValueError."""
        # CanonicalMessage validation will prevent creation without required fields,
        # so we test by modifying the payload after creation
        canonical = CanonicalMessage(
            message_id="550e8400-e29b-41d4-a716-446655440000",
            source_agent_id="test_agent",
            conversation_id="test_conv",
            message_type="response",
            intent="stream_result",
            payload={
                "chunk_index": 0,
                "total_chunks": 1,
                "data": "test"
            }
        )

        # Remove chunk_index from payload
        del canonical.payload["chunk_index"]

        with pytest.raises(ValueError, match="Missing critical fields"):
            await adapter.egress(canonical)

    @pytest.mark.asyncio
    async def test_egress_wrong_intent_raises_error(self, adapter):
        """Test that egress with non-stream_result intent raises ValueError."""
        canonical = CanonicalMessage(
            message_id="550e8400-e29b-41d4-a716-446655440000",
            source_agent_id="test_agent",
            conversation_id="test_conv",
            message_type="request",
            intent="analyze",  # Wrong intent
            payload={
                "document_id": "doc_123",
                "format": "pdf",
                "size": 1000
            }
        )

        with pytest.raises(ValueError, match="intent='stream_result'"):
            await adapter.egress(canonical)
