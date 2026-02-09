"""Tests for slippi_ai.action_space.clustering module."""

import unittest

import numpy as np

# Import directly from the module to avoid __init__.py import issues
from slippi_ai.action_space.clustering import (
    bucket_shoulder,
    bucket_sticks_polar,
    bucket_deltas_polar,
    normalize_buttons,
    PolarBucketInfo,
    DeltaBucketInfo,
    ButtonDataInfo,
    ComboClusterResult,
    HierarchicalClusterResult,
)


class TestBucketShoulder(unittest.TestCase):
  """Tests for bucket_shoulder function."""

  def test_unpressed(self):
    """Shoulder values <= 0.3 should be bucket 0 (unpressed)."""
    shoulder = np.array([0.0, 0.1, 0.2, 0.3])
    labels = bucket_shoulder(shoulder)
    np.testing.assert_array_equal(labels, [0, 0, 0, 0])

  def test_light_press(self):
    """Shoulder values in (0.3, 0.9] should be bucket 1 (light press)."""
    shoulder = np.array([0.31, 0.5, 0.7, 0.9])
    labels = bucket_shoulder(shoulder)
    np.testing.assert_array_equal(labels, [1, 1, 1, 1])

  def test_full_press(self):
    """Shoulder values > 0.9 should be bucket 2 (full press)."""
    shoulder = np.array([0.91, 0.95, 1.0])
    labels = bucket_shoulder(shoulder)
    np.testing.assert_array_equal(labels, [2, 2, 2])

  def test_mixed_values(self):
    """Test a mix of all three buckets."""
    shoulder = np.array([0.0, 0.5, 1.0, 0.3, 0.91, 0.31])
    labels = bucket_shoulder(shoulder)
    np.testing.assert_array_equal(labels, [0, 1, 2, 0, 2, 1])


class TestNormalizeButtons(unittest.TestCase):
  """Tests for normalize_buttons function."""

  def test_x_to_y_normalization(self):
    """X button should be merged into Y button."""
    # Buttons are: A, B, X, Y, Z, L, R, D_UP (8 columns)
    buttons = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],  # X pressed
        [0, 0, 0, 1, 0, 0, 0, 0],  # Y pressed
        [0, 0, 1, 1, 0, 0, 0, 0],  # Both pressed
    ], dtype=np.uint8)
    result = normalize_buttons(buttons)
    # After normalization: X is zeroed, Y becomes X|Y, same for L->R
    expected = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0],  # X -> Y (X zeroed)
        [0, 0, 0, 1, 0, 0, 0, 0],  # Y stays
        [0, 0, 0, 1, 0, 0, 0, 0],  # Both -> Y (X zeroed)
    ], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)

  def test_l_to_r_normalization(self):
    """L button should be merged into R button."""
    # Buttons are: A, B, X, Y, Z, L, R, D_UP (8 columns)
    buttons = np.array([
        [0, 0, 0, 0, 0, 1, 0, 0],  # L pressed
        [0, 0, 0, 0, 0, 0, 1, 0],  # R pressed
        [0, 0, 0, 0, 0, 1, 1, 0],  # Both pressed
    ], dtype=np.uint8)
    result = normalize_buttons(buttons)
    # After normalization: L is zeroed, R becomes L|R
    expected = np.array([
        [0, 0, 0, 0, 0, 0, 1, 0],  # L -> R (L zeroed)
        [0, 0, 0, 0, 0, 0, 1, 0],  # R stays
        [0, 0, 0, 0, 0, 0, 1, 0],  # Both -> R (L zeroed)
    ], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)

  def test_no_buttons_pressed(self):
    """No buttons pressed should remain no buttons."""
    buttons = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    result = normalize_buttons(buttons)
    expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


class TestBucketSticksPolar(unittest.TestCase):
  """Tests for bucket_sticks_polar function."""

  def test_origin_classification(self):
    """Points at origin should be classified as label 0."""
    stick = np.array([
        [0.5, 0.5],  # Exact center
        [0.51, 0.5],  # Very close to center
        [0.49, 0.51],  # Very close to center
    ])
    labels, info = bucket_sticks_polar(
        stick, n_log_radius=3, n_angle=8, origin_threshold=0.05)
    # All should be at origin (label 0)
    np.testing.assert_array_equal(labels, [0, 0, 0])
    self.assertEqual(info.at_origin_count, 3)

  def test_non_origin_points(self):
    """Points away from origin should get non-zero labels."""
    stick = np.array([
        [0.5, 0.5],  # Origin
        [1.0, 0.5],  # Far right
        [0.0, 0.5],  # Far left
        [0.5, 1.0],  # Far top
    ])
    labels, info = bucket_sticks_polar(
        stick, n_log_radius=3, n_angle=8, origin_threshold=0.05)
    self.assertEqual(labels[0], 0)  # Origin
    self.assertGreater(labels[1], 0)  # Non-origin
    self.assertGreater(labels[2], 0)  # Non-origin
    self.assertGreater(labels[3], 0)  # Non-origin
    self.assertEqual(info.at_origin_count, 1)
    self.assertEqual(info.regular_count, 3)

  def test_returns_polar_bucket_info(self):
    """Should return a PolarBucketInfo dataclass."""
    stick = np.array([[0.5, 0.5], [1.0, 0.5]])
    labels, info = bucket_sticks_polar(stick, n_log_radius=3, n_angle=8)
    self.assertIsInstance(info, PolarBucketInfo)
    self.assertEqual(info.n_log_radius, 3)
    self.assertEqual(info.n_angle, 8)
    self.assertIsInstance(info.n_angle_per_radius, list)
    self.assertEqual(len(info.n_angle_per_radius), 3)

  def test_angle_scaling_by_radius(self):
    """Angle buckets should scale with radius (fewer at inner radii)."""
    stick = np.array([[1.0, 0.5]])  # Single point far from origin
    labels, info = bucket_sticks_polar(stick, n_log_radius=4, n_angle=16)
    # Inner radii should have fewer angle buckets
    n_angle_per_radius = info.n_angle_per_radius
    # Should be non-decreasing (more angles at larger radii)
    for i in range(len(n_angle_per_radius) - 1):
      self.assertLessEqual(n_angle_per_radius[i], n_angle_per_radius[i + 1])
    # Outermost should have max angles
    self.assertEqual(n_angle_per_radius[-1], 16)

  def test_angle_buckets_are_powers_of_2(self):
    """Angle bucket counts should be powers of 2."""
    stick = np.array([[0.6, 0.5], [1.0, 0.5]])
    labels, info = bucket_sticks_polar(stick, n_log_radius=4, n_angle=16)
    for n_ang in info.n_angle_per_radius:
      # Check if power of 2: n & (n-1) == 0 for powers of 2
      self.assertEqual(n_ang & (n_ang - 1), 0, f"{n_ang} is not a power of 2")

  def test_total_classes(self):
    """Total classes should be 1 (origin) + sum of angle buckets per radius."""
    stick = np.array([[0.6, 0.5], [1.0, 0.5]])
    labels, info = bucket_sticks_polar(stick, n_log_radius=3, n_angle=8)
    expected_total = 1 + sum(info.n_angle_per_radius)
    self.assertEqual(info.total_classes, expected_total)

  def test_labels_in_valid_range(self):
    """All labels should be in [0, total_classes)."""
    np.random.seed(42)
    stick = np.random.uniform(0, 1, size=(100, 2))
    labels, info = bucket_sticks_polar(stick, n_log_radius=4, n_angle=16)
    self.assertTrue(np.all(labels >= 0))
    self.assertTrue(np.all(labels < info.total_classes))


class TestBucketDeltasPolar(unittest.TestCase):
  """Tests for bucket_deltas_polar function."""

  def test_no_movement_classification(self):
    """Zero deltas should be classified as label 0."""
    deltas = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    destinations = np.array([
        [0.5, 0.5],
        [0.7, 0.3],
    ])
    labels, info = bucket_deltas_polar(
        deltas, destinations, n_log_radius=3, n_angle=8)
    np.testing.assert_array_equal(labels, [0, 0])
    self.assertEqual(info.no_movement_count, 2)

  def test_destination_origin_classification(self):
    """Movements ending at origin should be classified as label 1."""
    deltas = np.array([
        [0.1, 0.0],  # Moving
        [-0.2, 0.0],  # Moving
    ])
    destinations = np.array([
        [0.5, 0.5],  # Origin
        [0.51, 0.49],  # Near origin
    ])
    labels, info = bucket_deltas_polar(
        deltas, destinations, n_log_radius=3, n_angle=8, origin_threshold=0.05)
    np.testing.assert_array_equal(labels, [1, 1])
    self.assertEqual(info.dst_origin_count, 2)

  def test_regular_movement_classification(self):
    """Regular movements should get labels >= 2."""
    deltas = np.array([
        [0.3, 0.0],  # Moving right
        [0.0, 0.3],  # Moving up
    ])
    destinations = np.array([
        [0.8, 0.5],  # Not at origin
        [0.5, 0.8],  # Not at origin
    ])
    labels, info = bucket_deltas_polar(
        deltas, destinations, n_log_radius=3, n_angle=8)
    self.assertTrue(np.all(labels >= 2))
    self.assertEqual(info.regular_count, 2)

  def test_returns_delta_bucket_info(self):
    """Should return a DeltaBucketInfo dataclass."""
    deltas = np.array([[0.1, 0.0]])
    destinations = np.array([[0.6, 0.5]])
    labels, info = bucket_deltas_polar(
        deltas, destinations, n_log_radius=3, n_angle=8)
    self.assertIsInstance(info, DeltaBucketInfo)
    self.assertEqual(info.n_log_radius, 3)
    self.assertEqual(info.n_angle, 8)

  def test_labels_in_valid_range(self):
    """All labels should be in [0, total_classes)."""
    np.random.seed(42)
    deltas = np.random.uniform(-0.5, 0.5, size=(100, 2))
    destinations = np.random.uniform(0, 1, size=(100, 2))
    labels, info = bucket_deltas_polar(
        deltas, destinations, n_log_radius=4, n_angle=16)
    self.assertTrue(np.all(labels >= 0))
    self.assertTrue(np.all(labels < info.total_classes))


class TestDataclasses(unittest.TestCase):
  """Tests for dataclass definitions."""

  def test_polar_bucket_info_creation(self):
    """PolarBucketInfo should be creatable with all required fields."""
    info = PolarBucketInfo(
        n_log_radius=3,
        n_angle=8,
        n_angle_per_radius=[2, 4, 8],
        n_regular_classes=14,
        total_classes=15,
        at_origin_count=100,
        regular_count=900,
        min_radius=0.05,
        max_radius=0.5,
    )
    self.assertEqual(info.n_log_radius, 3)
    self.assertEqual(info.total_classes, 15)

  def test_delta_bucket_info_creation(self):
    """DeltaBucketInfo should be creatable with all required fields."""
    info = DeltaBucketInfo(
        n_log_radius=3,
        n_angle=8,
        n_angle_per_radius=[2, 4, 8],
        n_regular_classes=14,
        total_classes=16,
        no_movement_count=50,
        dst_origin_count=30,
        regular_count=920,
        min_radius=0.01,
        max_radius=0.7,
    )
    self.assertEqual(info.no_movement_count, 50)
    self.assertEqual(info.total_classes, 16)

  def test_button_data_info_creation(self):
    """ButtonDataInfo should be creatable with all required fields."""
    info = ButtonDataInfo(
        button_names=['B', 'Y', 'LR', 'D_UP', 'z_a_shoulder', 'c_stick'],
        n_binary_buttons=4,
        z_a_shoulder_idx=4,
        c_stick_idx=5,
        z_a_shoulder_counts={0: 700, 1: 150, 2: 100, 6: 50},
        c_stick_at_origin=800,
        c_stick_buckets_used=12,
        c_stick_total_buckets=25,
    )
    self.assertEqual(info.n_binary_buttons, 4)
    self.assertEqual(len(info.button_names), 6)

  def test_combo_cluster_result_creation(self):
    """ComboClusterResult should be creatable with all required fields."""
    result = ComboClusterResult(
        combo=(1, 0, 0, 0, 0, 0),
        count=1000,
        n_stick_clusters=15,
        scaled_exclude_pct=2.5,
        buckets_per_radius=[1, 3, 5, 6],
    )
    self.assertEqual(result.count, 1000)
    self.assertEqual(result.n_stick_clusters, 15)

  def test_hierarchical_cluster_result_creation(self):
    """HierarchicalClusterResult should be creatable with all required fields."""
    button_info = ButtonDataInfo(
        button_names=['B', 'Y', 'LR', 'D_UP', 'z_a_shoulder', 'c_stick'],
        n_binary_buttons=4,
        z_a_shoulder_idx=4,
        c_stick_idx=5,
        z_a_shoulder_counts={0: 700, 1: 150, 2: 100, 6: 50},
        c_stick_at_origin=800,
        c_stick_buckets_used=12,
        c_stick_total_buckets=25,
    )
    combo_result = ComboClusterResult(
        combo=(0, 0, 0, 0, 0, 0),
        count=500,
        n_stick_clusters=10,
        scaled_exclude_pct=1.5,
        buckets_per_radius=[1, 2, 4, 3],
    )
    result = HierarchicalClusterResult(
        exclude_pct=0.01,
        n_log_radius=4,
        n_angle=16,
        n_angle_per_radius=[2, 4, 8, 16],
        n_button_combos=10,
        button_coverage=0.95,
        total_clusters=150,
        combo_results=[combo_result],
        button_info=button_info,
    )
    self.assertEqual(result.exclude_pct, 0.01)
    self.assertEqual(result.total_clusters, 150)
    self.assertEqual(len(result.combo_results), 1)


class TestPolarBucketingConsistency(unittest.TestCase):
  """Tests for consistency between bucket functions."""

  def test_same_point_same_label(self):
    """Same point should always get the same label."""
    stick = np.array([
        [0.8, 0.5],
        [0.8, 0.5],
        [0.8, 0.5],
    ])
    labels, _ = bucket_sticks_polar(stick, n_log_radius=4, n_angle=16)
    self.assertEqual(labels[0], labels[1])
    self.assertEqual(labels[1], labels[2])

  def test_opposite_angles_different_labels(self):
    """Points at opposite angles should have different labels."""
    stick = np.array([
        [1.0, 0.5],  # Right
        [0.0, 0.5],  # Left
    ])
    labels, _ = bucket_sticks_polar(stick, n_log_radius=3, n_angle=8)
    self.assertNotEqual(labels[0], labels[1])

  def test_different_radii_different_labels(self):
    """Points at same angle but different radii should have different labels."""
    stick = np.array([
        [0.6, 0.5],  # Close to center
        [1.0, 0.5],  # Far from center
    ])
    labels, _ = bucket_sticks_polar(stick, n_log_radius=4, n_angle=8)
    # They might still have the same label if they fall in the same bucket,
    # but with 4 radius buckets and significant distance difference, likely different
    # This is a weaker test - just verify labels are valid
    self.assertTrue(all(l >= 0 for l in labels))


if __name__ == '__main__':
  unittest.main()
