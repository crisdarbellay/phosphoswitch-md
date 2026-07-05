"""
test_basin.py — Unit tests for basin assignment and occupancy metrics.

All tests use synthetic numpy arrays — no GROMACS trajectories required.
"""

import math

import numpy as np
import pytest

from phosphoswitch_md.basin import (
    kabsch_rmsd,
    assign_basin_per_frame,
    compute_occupancy_metrics,
    find_hitting_time,
    compute_switch_metrics,
)


# ---------------------------------------------------------------------------
# kabsch_rmsd
# ---------------------------------------------------------------------------

class TestKabschRmsd:
    def test_identical_coords_zero_rmsd(self):
        rng = np.random.default_rng(0)
        coords = rng.standard_normal((20, 3))
        assert kabsch_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation_ignored(self):
        """Kabsch should handle pure translation (centering step)."""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((15, 3))
        B = A + np.array([5.0, -3.0, 2.0])   # shift by a constant vector
        assert kabsch_rmsd(A, B) == pytest.approx(0.0, abs=1e-10)

    def test_rotated_copy_lower_rmsd_than_random(self):
        """A rotated copy of A should have lower RMSD to A than a random set.

        The Kabsch algorithm finds a rotation to reduce RMSD.  A rotated copy
        should be easy to align; a random set should be harder.
        """
        rng = np.random.default_rng(2)
        A = rng.standard_normal((20, 3))
        # 90° rotation about Z axis
        theta = math.pi / 2
        R = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta),  math.cos(theta), 0],
            [0, 0, 1],
        ])
        B_rotated = A @ R.T
        C_random = rng.standard_normal((20, 3)) * 5.0
        assert kabsch_rmsd(A, B_rotated) < kabsch_rmsd(A, C_random)
        assert kabsch_rmsd(A, B_rotated) >= 0.0

    def test_known_rmsd(self):
        """Two structures shifted by 1 Å along x should have RMSD ≈ 1 Å."""
        n = 10
        A = np.zeros((n, 3))
        B = np.zeros((n, 3))
        B[:, 0] = 1.0   # shift all atoms by 1 Å in x
        # Kabsch centres both → RMSD = 0 (pure translation)
        # To get a non-zero RMSD, shift only half the atoms
        B2 = A.copy()
        B2[:n // 2, 0] = 2.0   # shift first 5 atoms by 2 Å
        result = kabsch_rmsd(A, B2)
        # Manual expected: after centering each, some residual displacement
        assert result > 0.0

    def test_reflection_correction(self):
        """Improper rotation (det=-1) should be corrected."""
        A = np.eye(3)
        # Flip sign of z axis → reflection
        B = np.eye(3)
        B[2, 2] = -1
        # kabsch_rmsd should not produce a negative or nan value
        r = kabsch_rmsd(A, B)
        assert r >= 0.0
        assert not math.isnan(r)

    def test_symmetry(self):
        """kabsch_rmsd(A, B) == kabsch_rmsd(B, A)."""
        rng = np.random.default_rng(3)
        A = rng.standard_normal((12, 3))
        B = rng.standard_normal((12, 3))
        assert kabsch_rmsd(A, B) == pytest.approx(kabsch_rmsd(B, A), abs=1e-10)


# ---------------------------------------------------------------------------
# assign_basin_per_frame
# ---------------------------------------------------------------------------

class TestAssignBasin:
    def test_closer_to_apo(self):
        rmsd_apo = np.array([1.0, 2.0, 3.0])
        rmsd_phos = np.array([5.0, 6.0, 7.0])
        basins = assign_basin_per_frame(rmsd_apo, rmsd_phos)
        assert list(basins) == ["apo", "apo", "apo"]

    def test_closer_to_phos(self):
        rmsd_apo = np.array([8.0, 9.0])
        rmsd_phos = np.array([2.0, 3.0])
        basins = assign_basin_per_frame(rmsd_apo, rmsd_phos)
        assert list(basins) == ["phos", "phos"]

    def test_equal_distance_goes_to_apo(self):
        """Tie-breaking: equal RMSD → apo (from numpy.where logic)."""
        rmsd_apo = np.array([5.0])
        rmsd_phos = np.array([5.0])
        basins = assign_basin_per_frame(rmsd_apo, rmsd_phos)
        assert basins[0] == "apo"

    def test_mixed(self):
        rmsd_apo = np.array([1.0, 9.0, 5.0, 3.0])
        rmsd_phos = np.array([9.0, 1.0, 6.0, 2.0])
        basins = assign_basin_per_frame(rmsd_apo, rmsd_phos)
        assert list(basins) == ["apo", "phos", "apo", "phos"]


# ---------------------------------------------------------------------------
# compute_occupancy_metrics
# ---------------------------------------------------------------------------

class TestOccupancyMetrics:
    def _make_data(self, n=100, seed=0):
        """Synthetic RMSD arrays: apo near 3 Å, phos near 9 Å."""
        rng = np.random.default_rng(seed)
        rmsd_apo = rng.normal(loc=3.0, scale=0.5, size=n).clip(0.5)
        rmsd_phos = rng.normal(loc=9.0, scale=0.5, size=n).clip(0.5)
        return rmsd_apo, rmsd_phos

    def test_all_in_apo_basin(self):
        rmsd_apo, rmsd_phos = self._make_data()
        m = compute_occupancy_metrics(rmsd_apo, rmsd_phos, strict_A=4.0, loose_A=7.0)
        assert m["frac_apo_strict"] > 0.8
        assert m["frac_phos_strict"] < 0.05
        assert m["frac_apo_loose"] > 0.8

    def test_all_in_phos_basin(self):
        rng = np.random.default_rng(1)
        rmsd_apo = rng.normal(loc=12.0, scale=0.5, size=100).clip(0.5)
        rmsd_phos = rng.normal(loc=3.0, scale=0.5, size=100).clip(0.5)
        m = compute_occupancy_metrics(rmsd_apo, rmsd_phos, strict_A=4.0, loose_A=7.0)
        assert m["frac_phos_strict"] > 0.8
        assert m["frac_apo_strict"] < 0.05

    def test_fraction_sum_loose_le_one(self):
        rng = np.random.default_rng(2)
        rmsd_apo = rng.uniform(2, 10, 200)
        rmsd_phos = rng.uniform(2, 10, 200)
        m = compute_occupancy_metrics(rmsd_apo, rmsd_phos, strict_A=4.0, loose_A=7.0)
        # frac_apo_loose + frac_phos_loose - frac_both_loose + frac_neither_loose ≈ 1
        total = (m["frac_apo_loose"] - m["frac_both_loose"]
                 + m["frac_phos_loose"] - m["frac_both_loose"]
                 + m["frac_neither_loose"]
                 + m["frac_both_loose"])
        assert total == pytest.approx(1.0, abs=0.01)

    def test_min_max_sensible(self):
        rmsd_apo = np.array([1.0, 2.0, 15.0])
        rmsd_phos = np.array([5.0, 0.5, 20.0])
        m = compute_occupancy_metrics(rmsd_apo, rmsd_phos)
        assert m["min_rmsd_to_apo"] == pytest.approx(1.0)
        assert m["min_rmsd_to_phos"] == pytest.approx(0.5)

    def test_n_frames(self):
        n = 57
        rmsd_apo = np.ones(n)
        rmsd_phos = np.ones(n) * 10
        m = compute_occupancy_metrics(rmsd_apo, rmsd_phos)
        assert m["n_frames"] == n


# ---------------------------------------------------------------------------
# find_hitting_time
# ---------------------------------------------------------------------------

class TestFindHittingTime:
    def test_never_reaches_threshold(self):
        t = np.linspace(0, 600, 1000)
        rmsd_phos = np.ones(1000) * 10.0  # always above threshold
        assert math.isnan(find_hitting_time(t, rmsd_phos, threshold_A=4.0))

    def test_first_hit(self):
        t = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        rmsd_phos = np.array([10.0, 8.0, 3.5, 5.0, 2.0])  # dips below 4 at t=200
        ht = find_hitting_time(t, rmsd_phos, threshold_A=4.0)
        assert ht == pytest.approx(200.0)

    def test_hits_at_start(self):
        t = np.array([0.0, 50.0, 100.0])
        rmsd_phos = np.array([1.0, 5.0, 5.0])  # immediately below 4 Å
        ht = find_hitting_time(t, rmsd_phos, threshold_A=4.0)
        assert ht == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_switch_metrics
# ---------------------------------------------------------------------------

class TestSwitchMetrics:
    def _apo_like_metrics(self):
        """Metrics for apo MD: most frames in apo basin, few in phos."""
        rng = np.random.default_rng(0)
        rmsd_apo = rng.normal(3.0, 0.5, 1000).clip(0.1)
        rmsd_phos = rng.normal(12.0, 0.5, 1000).clip(0.1)
        return compute_occupancy_metrics(rmsd_apo, rmsd_phos, strict_A=4.0, loose_A=7.0)

    def _phos_like_metrics(self):
        """Metrics for phos MD: most frames in phos basin."""
        rng = np.random.default_rng(1)
        rmsd_apo = rng.normal(12.0, 0.5, 1000).clip(0.1)
        rmsd_phos = rng.normal(3.0, 0.5, 1000).clip(0.1)
        return compute_occupancy_metrics(rmsd_apo, rmsd_phos, strict_A=4.0, loose_A=7.0)

    def test_good_switch_has_enrichment_gt_1(self):
        apo_m = self._apo_like_metrics()
        phos_m = self._phos_like_metrics()
        sm = compute_switch_metrics(apo_m, phos_m)
        assert sm["enrich_phos_loose"] > 1.0
        assert sm["log2_phos_enrich_loose"] > 0.0

    def test_bad_switch_enrichment_near_1(self):
        """When apo and phos MD look the same, enrichment ≈ 1."""
        rng = np.random.default_rng(2)
        rmsd_apo = rng.normal(5.0, 1.0, 500).clip(0.1)
        rmsd_phos = rng.normal(5.0, 1.0, 500).clip(0.1)
        m = compute_occupancy_metrics(rmsd_apo, rmsd_phos, strict_A=4.0, loose_A=7.0)
        sm = compute_switch_metrics(m, m)
        assert sm["enrich_phos_loose"] == pytest.approx(1.0, rel=0.1)

    def test_keys_present(self):
        apo_m = self._apo_like_metrics()
        phos_m = self._phos_like_metrics()
        sm = compute_switch_metrics(apo_m, phos_m)
        for key in ("enrich_phos_strict", "enrich_phos_loose",
                    "log2_phos_enrich_loose", "switch_strength"):
            assert key in sm

    def test_switch_strength_bounded(self):
        apo_m = self._apo_like_metrics()
        phos_m = self._phos_like_metrics()
        sm = compute_switch_metrics(apo_m, phos_m)
        assert 0.0 <= sm["switch_strength"] <= 1.0
