import pandas as pd

from md_emg.data import PHONEME_WORDS


def test_phoneme_word_sets_partition_1_to_40():
    sets = [set(PHONEME_WORDS[k]) for k in ["t", "theta", "d", "eth"]]
    assert all(len(s) == 10 for s in sets)
    assert set.union(*sets) == set(range(1, 41))
    for i, a in enumerate(sets):
        for b in sets[i+1:]:
            assert a.isdisjoint(b)

def test_expected_original_word_mapping():
    assert PHONEME_WORDS["t"] == [1,3,5,7,9,12,14,16,18,20]
    assert PHONEME_WORDS["theta"] == [2,4,6,8,10,11,13,15,17,19]
    assert PHONEME_WORDS["d"] == [21,23,25,27,29,32,34,36,38,40]
    assert PHONEME_WORDS["eth"] == [22,24,26,28,30,31,33,35,37,39]
