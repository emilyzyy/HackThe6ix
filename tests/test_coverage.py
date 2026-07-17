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


def _grid_with(cells_value_5):
    grid = CoverageGrid([[0.0, 0.2], [0.0, 0.2]], cell_m=0.01)  # 20x20
    for ys, xs in cells_value_5:
        grid.seen_count[ys, xs] = 5
    return grid


def test_workspace_keeps_only_dominant_component():
    grid = _grid_with([(np.s_[2:12], np.s_[2:12]),   # 10x10 main blob
                       (np.s_[16:18], np.s_[16:18])])  # far 2x2 junk patch
    ws = grid.workspace_mask(min_seen=2)
    assert ws[2:12, 2:12].all()
    assert not ws[16:18, 16:18].any()


def test_workspace_excludes_connected_low_density_background():
    grid = CoverageGrid([[0.0, 0.3], [0.0, 0.3]], cell_m=0.01)
    # The central workspace is revisited throughout the scan.  A thin path
    # connects it to a far background patch that was only seen a few times.
    grid.seen_count[8:22, 8:22] = 20
    grid.seen_count[14:16, 22:27] = 2
    grid.seen_count[12:18, 27:30] = 2

    ws = grid.workspace_mask(min_seen=2)

    assert ws[8:22, 8:22].all()
    assert not ws[12:18, 27:30].any()


def test_workspace_concave_region_excludes_open_mouth():
    # U shape: hull-based coverage is stuck below 1, workspace-based is 1.0.
    grid = _grid_with([(np.s_[2:18], np.s_[2:6]),
                       (np.s_[2:18], np.s_[14:18]),
                       (np.s_[14:18], np.s_[2:18])])
    assert grid.hull_coverage_fraction(min_seen=2) < 0.9
    assert grid.workspace_coverage_fraction(min_seen=2) == pytest.approx(1.0)


def test_workspace_counts_enclosed_holes():
    # Ring: the enclosed hole must count as unswept workspace.
    grid = _grid_with([(np.s_[2:18], np.s_[2:18])])
    grid.seen_count[8:12, 8:12] = 0  # enclosed 4x4 hole
    ws = grid.workspace_mask(min_seen=2)
    assert ws[8:12, 8:12].all()  # hole is part of the workspace
    frac = grid.workspace_coverage_fraction(min_seen=2)
    assert 0.9 < frac < 1.0
    grid.seen_count[8:12, 8:12] = 5  # sweep the hole
    assert grid.workspace_coverage_fraction(min_seen=2) == pytest.approx(1.0)


def test_workspace_bounds_xy():
    grid = _grid_with([(np.s_[2:12], np.s_[4:14])])
    (x0, x1), (y0, y1) = grid.workspace_bounds_xy(min_seen=2)
    assert x0 == pytest.approx(0.04, abs=0.011)
    assert x1 == pytest.approx(0.14, abs=0.011)
    assert y0 == pytest.approx(0.02, abs=0.011)
    assert y1 == pytest.approx(0.12, abs=0.011)


def test_workspace_empty_grid():
    grid = CoverageGrid([[0.0, 0.2], [0.0, 0.2]], cell_m=0.01)
    assert not grid.workspace_mask().any()
    assert grid.workspace_coverage_fraction() == 0.0
    assert grid.workspace_bounds_xy() is None


def test_admissible_workspace_keeps_outer_repeated_plane_not_only_dense_core():
    grid = CoverageGrid([[0.0, 0.1], [0.0, 0.1]], cell_m=0.01)
    grid.seen_count[1:9, 1:9] = 2
    grid.seen_count[3:7, 3:7] = 10

    admissible = grid.admissible_workspace_mask()
    dense = grid.workspace_mask()

    assert admissible[1:9, 1:9].all()
    assert admissible.sum() >= 64
    assert dense.sum() < admissible.sum()


def test_admissible_contour_is_reported_in_table_metres():
    grid = CoverageGrid([[-0.05, 0.05], [-0.05, 0.05]], cell_m=0.01)
    grid.seen_count[2:8, 3:9] = 3

    contours = grid.admissible_workspace_contours_xy(simplify_m=0.0)

    assert contours
    assert all(
        -0.05 <= value <= 0.05
        for contour in contours
        for point in contour
        for value in point
    )


def test_admissible_workspace_is_empty_without_repeated_plane_cells():
    grid = CoverageGrid([[0.0, 0.2], [0.0, 0.2]], cell_m=0.01)

    assert not grid.admissible_workspace_mask().any()
    assert grid.admissible_workspace_contours_xy() == []
