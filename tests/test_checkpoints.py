from nds import checkpoints


def test_there_are_154_pythia_checkpoints():
    steps = checkpoints.all_pythia_steps()

    assert len(steps) == 154
    assert steps[:12] == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    assert steps[-1] == 143000


def test_week_one_is_a_sorted_subset():
    week_one = checkpoints.week_one_steps()

    assert len(week_one) == 25
    assert week_one == sorted(week_one)
    assert set(week_one) <= set(checkpoints.all_pythia_steps())


def test_tokens_seen_at_the_end_is_about_300_billion():
    assert abs(checkpoints.tokens_seen(143000) - 3.0e11) < 0.01e11
