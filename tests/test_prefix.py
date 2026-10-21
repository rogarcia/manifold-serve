from manifold.router.prefix import PrefixIndex, block_hashes


def test_full_blocks_only():
    assert len(block_hashes(list(range(10)), 4)) == 2
    assert block_hashes([1, 2, 3], 4) == []


def test_shared_prefix_shares_leading_hashes():
    a = block_hashes([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    b = block_hashes([1, 2, 3, 4, 5, 6, 0, 0, 0], 3)
    assert a[:2] == b[:2]
    assert a[2] != b[2]


def test_hash_is_chained_on_the_whole_prefix():
    a = block_hashes([1, 1, 7, 7], 2)
    b = block_hashes([2, 2, 7, 7], 2)
    assert a[1] != b[1], "same block tokens after a different prefix must hash differently"


def test_match_counts_leading_blocks_per_replica():
    index = PrefixIndex(3, 100)
    index.record(0, [1, 2, 3])
    index.record(1, [1, 3])
    assert index.match([1, 2, 3, 4]) == {0: 3, 1: 1, 2: 0}


def test_lru_evicts_least_recently_recorded():
    index = PrefixIndex(1, 3)
    index.record(0, [1, 2, 3])
    index.record(0, [1])
    index.record(0, [9])
    assert index.matched(0, [1]) == 1
    assert index.matched(0, [2]) == 0
    assert index.size(0) == 3


def test_forget_drops_only_that_replica():
    index = PrefixIndex(2, 100)
    index.record(0, [1, 2])
    index.record(1, [1, 2])
    index.forget(0)
    assert index.match([1, 2]) == {0: 0, 1: 2}
