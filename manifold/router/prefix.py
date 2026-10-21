"""Approximate prefix index, built from the router's own routing decisions (llm-d style)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

BlockHash = int


def block_hashes(tokens: Sequence[int], block_tokens: int) -> list[BlockHash]:
    """Chained hashes over full blocks only, so a block's hash commits to its whole prefix."""
    hashes: list[BlockHash] = []
    parent: BlockHash = 0
    for start in range(0, len(tokens) - block_tokens + 1, block_tokens):
        parent = hash((parent, tuple(tokens[start : start + block_tokens])))
        hashes.append(parent)
    return hashes


class PrefixIndex:
    """Per-replica LRU of block hashes the router believes each replica holds."""

    def __init__(self, n_replicas: int, capacity_blocks: int):
        self.n_replicas = n_replicas
        self.capacity_blocks = capacity_blocks
        self._lru: list[OrderedDict[BlockHash, None]] = [OrderedDict() for _ in range(n_replicas)]

    def matched(self, replica_index: int, blocks: Sequence[BlockHash]) -> int:
        lru = self._lru[replica_index]
        n = 0
        for h in blocks:
            if h not in lru:
                break
            n += 1
        return n

    def match(self, blocks: Sequence[BlockHash]) -> dict[int, int]:
        return {i: self.matched(i, blocks) for i in range(self.n_replicas)}

    def record(self, replica_index: int, blocks: Sequence[BlockHash]) -> None:
        lru = self._lru[replica_index]
        for h in blocks:
            lru[h] = None
            lru.move_to_end(h)
        while len(lru) > self.capacity_blocks:
            lru.popitem(last=False)

    def forget(self, replica_index: int) -> None:
        self._lru[replica_index].clear()

    def size(self, replica_index: int) -> int:
        return len(self._lru[replica_index])
