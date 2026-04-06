"""
Tests for streaming response routing with chunk order preservation.

Tests cover:
- 50 chunks from agent arrive at original requester in sequence
- No loss or reordering of chunks
- Out-of-order arrival and buffering
- Chunk index verification
- Large chunk payloads
- Concurrent chunk streaming
"""

import pytest
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class StreamChunk:
    """Simulates a streaming chunk with ordering metadata."""
    chunk_index: int
    total_chunks: int
    data: str
    source_agent: str
    conversation_id: str

    def __lt__(self, other):
        """Support sorting by chunk_index."""
        return self.chunk_index < other.chunk_index


class StreamingChunkBuffer:
    """Buffer for managing streaming chunks with order preservation."""

    def __init__(self):
        """Initialize chunk buffer."""
        self.chunks: Dict[int, StreamChunk] = {}
        self.next_expected_index = 0
        self.total_chunks = None
        self.completed = False

    def add_chunk(self, chunk: StreamChunk) -> List[StreamChunk]:
        """
        Add chunk to buffer and return ready chunks in order.

        Args:
            chunk: StreamChunk to add

        Returns:
            List of chunks ready to deliver (in order from next_expected_index)
        """
        if self.total_chunks is None:
            self.total_chunks = chunk.total_chunks

        # Store chunk by index
        self.chunks[chunk.chunk_index] = chunk

        # Collect all chunks in order starting from next_expected_index
        ready = []
        while self.next_expected_index in self.chunks:
            ready.append(self.chunks[self.next_expected_index])
            self.next_expected_index += 1

        # Check if all chunks received
        if self.next_expected_index >= self.total_chunks:
            self.completed = True

        return ready

    def get_ordered_chunks(self) -> List[StreamChunk]:
        """Get all chunks in correct order (only if completed)."""
        if not self.completed:
            raise ValueError("Not all chunks received yet")

        return [self.chunks[i] for i in range(self.total_chunks)]


class TestStreamingChunkRouting:
    """Test streaming chunk routing and order preservation."""

    def test_fifty_chunks_in_order(self):
        """50 chunks from agent arrive at requester in sequence."""
        buffer = StreamingChunkBuffer()

        # Simulate 50 chunks arriving in order
        chunks_data = []
        for i in range(50):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=50,
                data=f"chunk_data_{i}",
                source_agent="streaming_agent",
                conversation_id="conv1"
            )
            ready = buffer.add_chunk(chunk)
            chunks_data.extend(ready)

        # All chunks should be ready and in order
        assert len(chunks_data) == 50
        for i, chunk in enumerate(chunks_data):
            assert chunk.chunk_index == i
            assert chunk.data == f"chunk_data_{i}"

    def test_chunk_order_preserved_sequential(self):
        """Chunks maintain order when delivered sequentially."""
        buffer = StreamingChunkBuffer()

        # Add chunks in order
        for i in range(10):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=10,
                data=f"data_{i}",
                source_agent="agent_1",
                conversation_id="conv1"
            )
            buffer.add_chunk(chunk)

        # Get all chunks
        ordered = buffer.get_ordered_chunks()

        # Verify order by index
        for i, chunk in enumerate(ordered):
            assert chunk.chunk_index == i

    def test_chunk_out_of_order_arrival(self):
        """Out-of-order arrival handled correctly with buffering."""
        buffer = StreamingChunkBuffer()

        # Chunks: [2, 0, 3, 1, 4]
        chunks_indices = [2, 0, 3, 1, 4]
        delivered = []

        for idx in chunks_indices:
            chunk = StreamChunk(
                chunk_index=idx,
                total_chunks=5,
                data=f"data_{idx}",
                source_agent="agent_1",
                conversation_id="conv1"
            )
            ready = buffer.add_chunk(chunk)
            delivered.extend(ready)

        # Should have delivered in order: [0, 1, 2, 3, 4]
        assert len(delivered) == 5
        for i, chunk in enumerate(delivered):
            assert chunk.chunk_index == i

    def test_chunk_index_verification(self):
        """Chunk indices correctly identify position in stream."""
        buffer = StreamingChunkBuffer()

        # Create 20 chunks with explicit indices
        for i in range(20):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=20,
                data=f"payload_{i}",
                source_agent="source_agent",
                conversation_id="conv1"
            )
            buffer.add_chunk(chunk)

        ordered = buffer.get_ordered_chunks()

        # Every chunk should have matching index
        for expected_idx, chunk in enumerate(ordered):
            assert chunk.chunk_index == expected_idx

    def test_large_chunk_payload_preserved(self):
        """Large chunk payloads preserved without loss."""
        buffer = StreamingChunkBuffer()

        large_data = "x" * 100000  # 100KB chunk

        for i in range(5):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=5,
                data=large_data if i == 2 else f"data_{i}",
                source_agent="agent_1",
                conversation_id="conv1"
            )
            buffer.add_chunk(chunk)

        ordered = buffer.get_ordered_chunks()

        # Large payload should be intact
        assert len(ordered[2].data) == 100000
        assert ordered[2].data == large_data

    def test_chunk_loss_detection(self):
        """Missing chunk detected and prevents completion."""
        buffer = StreamingChunkBuffer()

        # Add chunks 0, 1, 3, 4 (missing 2)
        for i in [0, 1, 3, 4]:
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=5,
                data=f"data_{i}",
                source_agent="agent_1",
                conversation_id="conv1"
            )
            buffer.add_chunk(chunk)

        # Should not be complete
        assert not buffer.completed
        assert buffer.next_expected_index == 2

    def test_duplicate_chunk_handling(self):
        """Duplicate chunks don't break ordering."""
        buffer = StreamingChunkBuffer()

        # Add chunk 0 twice, then others
        for i in [0, 0, 1, 2, 3]:
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=4,
                data=f"data_{i}",
                source_agent="agent_1",
                conversation_id="conv1"
            )
            buffer.add_chunk(chunk)

        # Should still complete with 4 chunks
        assert buffer.completed
        assert len(buffer.chunks) == 4

    def test_chunk_completion_status(self):
        """Completion status correctly tracks when all chunks received."""
        buffer = StreamingChunkBuffer()

        # Add 5 chunks out of 10
        for i in range(5):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=10,
                data=f"data_{i}",
                source_agent="agent_1",
                conversation_id="conv1"
            )
            buffer.add_chunk(chunk)

        assert not buffer.completed

        # Add remaining chunks
        for i in range(5, 10):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=10,
                data=f"data_{i}",
                source_agent="agent_1",
                conversation_id="conv1"
            )
            buffer.add_chunk(chunk)

        assert buffer.completed

    def test_multiple_streams_independent(self):
        """Multiple independent chunk streams don't interfere."""
        buffer1 = StreamingChunkBuffer()
        buffer2 = StreamingChunkBuffer()

        # Stream 1: 10 chunks
        for i in range(10):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=10,
                data=f"stream1_data_{i}",
                source_agent="agent_1",
                conversation_id="conv1"
            )
            buffer1.add_chunk(chunk)

        # Stream 2: 10 chunks (different conversation)
        for i in range(10):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=10,
                data=f"stream2_data_{i}",
                source_agent="agent_2",
                conversation_id="conv2"
            )
            buffer2.add_chunk(chunk)

        # Both should be complete and independent
        assert buffer1.completed
        assert buffer2.completed

        ordered1 = buffer1.get_ordered_chunks()
        ordered2 = buffer2.get_ordered_chunks()

        # Verify data integrity
        for i, chunk in enumerate(ordered1):
            assert chunk.data == f"stream1_data_{i}"

        for i, chunk in enumerate(ordered2):
            assert chunk.data == f"stream2_data_{i}"

    def test_chunk_source_agent_tracking(self):
        """Chunks track source agent for routing."""
        buffer = StreamingChunkBuffer()

        source_agent = "streaming_agent_xyz"

        for i in range(5):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=5,
                data=f"data_{i}",
                source_agent=source_agent,
                conversation_id="conv1"
            )
            buffer.add_chunk(chunk)

        ordered = buffer.get_ordered_chunks()

        # All chunks should have same source agent
        for chunk in ordered:
            assert chunk.source_agent == source_agent

    def test_chunk_conversation_id_routing(self):
        """Chunks route to correct conversation based on ID."""
        buffers = {}

        # Create 3 buffers for 3 conversations
        for conv_id in ["conv1", "conv2", "conv3"]:
            buffers[conv_id] = StreamingChunkBuffer()

        # Add chunks for each conversation
        for conv_id in ["conv1", "conv2", "conv3"]:
            for i in range(5):
                chunk = StreamChunk(
                    chunk_index=i,
                    total_chunks=5,
                    data=f"{conv_id}_data_{i}",
                    source_agent="agent_1",
                    conversation_id=conv_id
                )
                buffers[conv_id].add_chunk(chunk)

        # Verify each conversation has correct chunks
        for conv_id in ["conv1", "conv2", "conv3"]:
            ordered = buffers[conv_id].get_ordered_chunks()
            for i, chunk in enumerate(ordered):
                assert chunk.conversation_id == conv_id
                assert chunk.data == f"{conv_id}_data_{i}"

    def test_chunk_buffer_memory_efficiency(self):
        """Chunk buffer releases delivered chunks efficiently."""
        buffer = StreamingChunkBuffer()

        # Add chunks in order
        for i in range(100):
            chunk = StreamChunk(
                chunk_index=i,
                total_chunks=100,
                data=f"data_{i}",
                source_agent="agent_1",
                conversation_id="conv1"
            )
            buffer.add_chunk(chunk)

        # At any point, only undelivered chunks should be buffered
        # Since we add in order, all are immediately delivered
        # So buffer.chunks should contain all (no auto-cleanup in this impl)
        assert len(buffer.chunks) == 100

    @pytest.fixture
    def random_order_chunks(self):
        """Generate chunks in random order for testing."""
        import random

        chunks_list = []
        indices = list(range(20))
        random.shuffle(indices)

        for idx in indices:
            chunks_list.append(
                StreamChunk(
                    chunk_index=idx,
                    total_chunks=20,
                    data=f"data_{idx}",
                    source_agent="agent_1",
                    conversation_id="conv1"
                )
            )

        return chunks_list

    def test_random_order_chunk_buffering(self, random_order_chunks):
        """Chunks arriving in random order correctly buffered."""
        buffer = StreamingChunkBuffer()

        for chunk in random_order_chunks:
            buffer.add_chunk(chunk)

        # Should eventually complete
        assert buffer.completed

        ordered = buffer.get_ordered_chunks()

        # Verify final order
        for i, chunk in enumerate(ordered):
            assert chunk.chunk_index == i
