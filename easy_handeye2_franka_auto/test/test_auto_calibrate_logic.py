from easy_handeye2_franka_auto.handeye_auto_calibrate import should_save, sample_added


def test_should_save_requires_min_samples():
    assert should_save(5, 5) is True
    assert should_save(4, 5) is False


def test_should_save_never_below_three():
    assert should_save(2, 1) is False
    assert should_save(3, 1) is True


def test_sample_added_detects_list_growth():
    assert sample_added(0, 1) is True
    assert sample_added(3, 3) is False
