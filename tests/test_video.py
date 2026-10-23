from truthlens.data.video import uniform_sample_indices


def test_two_samples_cover_full_long_video():
    indices = uniform_sample_indices(duration_sec=100, fps=30, target_frames=2)

    assert indices == [0, 2999]


def test_multiple_samples_span_full_video():
    indices = uniform_sample_indices(duration_sec=10, fps=10, target_frames=5)

    assert indices == [0, 25, 50, 74, 99]


def test_one_requested_sample_preserves_first_frame_behavior():
    assert uniform_sample_indices(duration_sec=10, fps=10, target_frames=1) == [0]


def test_zero_or_negative_sample_count_returns_empty_list():
    assert uniform_sample_indices(duration_sec=10, fps=10, target_frames=0) == []
    assert uniform_sample_indices(duration_sec=10, fps=10, target_frames=-1) == []


def test_one_frame_video_returns_its_only_index():
    assert uniform_sample_indices(duration_sec=0.1, fps=10, target_frames=4) == [0]


def test_request_larger_than_frame_count_returns_each_frame_once():
    assert uniform_sample_indices(duration_sec=0.3, fps=10, target_frames=5) == [0, 1, 2]


def test_indices_are_ordered_unique_and_in_range_when_possible():
    total_frames = 100
    indices = uniform_sample_indices(duration_sec=10, fps=10, target_frames=7)

    assert indices == sorted(set(indices))
    assert len(indices) == 7
    assert all(0 <= index < total_frames for index in indices)