from easy_handeye2_franka_auto.handeye_auto_calibrate import _parse_args, should_save, sample_added

_REQUIRED_ARGV = [
    '--robot-config', '/tmp/robot.yaml',
    '--name', 'my_eob_calib',
    '--robot-base-frame', 'fr3_link0',
    '--robot-effector-frame', 'handeye_ee',
]


def test_motion_mode_defaults_to_dls():
    cli = _parse_args(_REQUIRED_ARGV)
    assert cli.motion_mode == 'dls'


def test_motion_mode_accepts_cartesian():
    cli = _parse_args(_REQUIRED_ARGV + ['--motion-mode', 'cartesian'])
    assert cli.motion_mode == 'cartesian'


def test_diversity_guard_defaults():
    cli = _parse_args(_REQUIRED_ARGV)
    assert cli.min_rotation_diff_deg == 2.0
    assert cli.min_translation_diff_m == 0.005


def test_should_save_requires_min_samples():
    assert should_save(5, 5) is True
    assert should_save(4, 5) is False


def test_should_save_never_below_three():
    assert should_save(2, 1) is False
    assert should_save(3, 1) is True


def test_sample_added_detects_list_growth():
    assert sample_added(0, 1) is True
    assert sample_added(3, 3) is False
