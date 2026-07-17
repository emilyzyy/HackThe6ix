import numpy as np
import pytest

from coverage import CoverageGrid
from tests.synthetic import look_at_pose
from transforms import intrinsics_to_K


BOUNDS = [[-0.3, 0.3], [-0.3, 0.3]]


def _straight_down_setup(height=0.4):
    """Small depth camera looking straight down at the z=0 table."""
    K = intrinsics_to_K(20.0, 20.0, 8.0, 6.0)  # 16x12, ~43 deg FOV
    T = look_at_pose([0.0, 0.0, height], [0.0, 0.0, 0.0])
    depth = np.full((12, 16), np.nan, dtype=np.float32)
    # Straight-down camera: z-depth to the plane is the height everywhere.
    depth[:] = height
    return depth, K, T


def test_mark_frame_marks_only_footprint_cells():
    grid = CoverageGrid(BOUNDS, cell_m=0.01)
    depth, K, T = _straight_down_setup()
    W2T = np.eye(4)  # world == table (plane z=0)
    grid.mark_frame(depth, K, T, W2T)
    assert grid.seen_count.max() == 1
    # Footprint half-extent: (8 px / 20 fx) * 0.4 m = 0.16 m in x, 0.12 in y.
    ys, xs = np.nonzero(grid.seen_count)
    cx = (xs + 0.5) * 0.01 + BOUNDS[0][0]
    cy = (ys + 0.5) * 0.01 + BOUNDS[1][0]
    assert np.abs(cx).max() < 0.17 and np.abs(cy).max() < 0.13
    assert grid.seen_count.sum() > 0


def test_mark_frame_counts_once_per_frame_per_cell():
    grid = CoverageGrid(BOUNDS, cell_m=0.05)  # coarse: many points per cell
    depth, K, T = _straight_down_setup()
    grid.mark_frame(depth, K, T, np.eye(4))
    assert grid.seen_count.max() == 1
    grid.mark_frame(depth, K, T, np.eye(4))
    assert grid.seen_count.max() == 2


def test_mark_frame_ignores_out_of_band_points():
    grid = CoverageGrid(BOUNDS, cell_m=0.01)
    depth, K, T = _straight_down_setup()
    depth -= 0.1  # surface floats 10 cm above the table plane
    grid.mark_frame(depth, K, T, np.eye(4), z_band=0.02)
    assert grid.seen_count.sum() == 0


def test_opposing_views_set_distinct_angle_bins():
    grid = CoverageGrid(BOUNDS, cell_m=0.02)
    K = intrinsics_to_K(20.0, 20.0, 8.0, 6.0)
    for xcam in (0.5, -0.5):  # two low, opposing viewpoints
        T = look_at_pose([xcam, 0.0, 0.25], [0.0, 0.0, 0.0])
        # Exact per-pixel z-depth to the plane for this pose.
        from tests.synthetic import _ray_plane_zdepth
        depth, _ = _ray_plane_zdepth(K, T, (12, 16))
        grid.mark_frame(depth.astype(np.float32), K, T, np.eye(4))
    both = grid.seen_count >= 2
    assert both.any()
    bins = grid.angle_bins[both]
    assert all(bin(int(b)).count("1") >= 2 for b in bins)


def test_coverage_fraction_on_handmade_grid():
    grid = CoverageGrid([[0.0, 0.1], [0.0, 0.1]], cell_m=0.01)  # 10x10
    grid.seen_count[2:8, 2:8] = 5  # a solid 6x6 block
    # Hull of the block is the block itself: fully covered.
    assert grid.hull_coverage_fraction(min_seen=2) == pytest.approx(1.0)
    grid.seen_count[4:6, 4:6] = 0  # punch a hole inside the hull
    frac = grid.hull_coverage_fraction(min_seen=2)
    assert 0.8 < frac < 1.0


def test_coverage_fraction_empty_grid():
    grid = CoverageGrid(BOUNDS, cell_m=0.01)
    assert grid.hull_coverage_fraction() == 0.0


def test_save_load_round_trip(tmp_path):
    grid = CoverageGrid(BOUNDS, cell_m=0.01)
    depth, K, T = _straight_down_setup()
    grid.mark_frame(depth, K, T, np.eye(4))
    grid.save(tmp_path / "cov.npz")
    g2 = CoverageGrid.load(tmp_path / "cov.npz")
    np.testing.assert_array_equal(g2.seen_count, grid.seen_count)
    np.testing.assert_array_equal(g2.angle_bins, grid.angle_bins)
    assert g2.cell_m == grid.cell_m
    assert g2.bounds == grid.bounds


def test_hull_area_m2():
    grid = CoverageGrid([[0.0, 0.1], [0.0, 0.1]], cell_m=0.01)
    grid.seen_count[2:8, 2:8] = 5  # 6x6 cells of 1 cm2 each
    area = grid.hull_area_m2(min_seen=2)
    assert area == pytest.approx(36e-4, rel=0.35)  # hull rasterization slack
    assert CoverageGrid(BOUNDS, cell_m=0.01).hull_area_m2() == 0.0
