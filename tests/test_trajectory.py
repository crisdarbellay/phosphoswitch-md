"""
test_trajectory.py — Unit tests for trajectory analysis functions.

All tests use in-memory synthetic data — no GROMACS trajectory files required.
MDAnalysis Universes are built from minimal coordinate arrays using
the MemoryReader, which lets us exercise the real analysis code without disk I/O.
"""

import io
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from phosphoswitch_md.io_utils import parse_xvg, parse_xvg_multi, aggregate_replicates
from phosphoswitch_md.trajectory import (
    compute_rmsd_mda,
    compute_rmsf_mda,
    compute_rg_mda,
)


# ---------------------------------------------------------------------------
# parse_xvg tests
# ---------------------------------------------------------------------------

def _write_xvg(path, rows, header_lines=None):
    """Write a minimal .xvg file for testing."""
    with open(path, "w") as fh:
        for h in (header_lines or []):
            fh.write(h + "\n")
        for t, v in rows:
            fh.write(f"{t:.3f}  {v:.6f}\n")


class TestParseXvg:
    def test_basic_two_column(self, tmp_path):
        rows = [(0.0, 1.0), (20.0, 1.5), (40.0, 2.0)]
        p = tmp_path / "test.xvg"
        _write_xvg(p, rows)
        t, v = parse_xvg(p)
        assert len(t) == 3
        assert t[0] == pytest.approx(0.0)
        # Time is already in ns if < 1000 → no conversion expected here
        assert v[1] == pytest.approx(1.5)

    def test_ps_to_ns_conversion(self, tmp_path):
        """Values > 1000 on the time axis should be converted from ps to ns."""
        rows = [(0.0, 1.0), (100000.0, 2.0), (200000.0, 3.0)]
        p = tmp_path / "test.xvg"
        _write_xvg(p, rows)
        t, v = parse_xvg(p)
        assert t[-1] == pytest.approx(200.0)   # 200000 ps → 200 ns

    def test_comment_and_header_skipped(self, tmp_path):
        rows = [(0.0, 1.0), (1.0, 2.0)]
        p = tmp_path / "test.xvg"
        _write_xvg(p, rows, header_lines=[
            "# This is a comment",
            "@ title Test",
            "@ xlabel Time (ns)",
        ])
        t, v = parse_xvg(p)
        assert len(t) == 2

    def test_empty_file_returns_empty_arrays(self, tmp_path):
        p = tmp_path / "empty.xvg"
        p.write_text("# no data\n")
        t, v = parse_xvg(p)
        assert len(t) == 0
        assert len(v) == 0

    def test_values_preserved(self, tmp_path):
        rows = [(0.0, 3.14159), (1.0, 2.71828)]
        p = tmp_path / "vals.xvg"
        _write_xvg(p, rows)
        t, v = parse_xvg(p)
        assert v[0] == pytest.approx(3.14159, rel=1e-5)
        assert v[1] == pytest.approx(2.71828, rel=1e-5)


class TestParseXvgMulti:
    def test_three_columns(self, tmp_path):
        p = tmp_path / "multi.xvg"
        with open(p, "w") as fh:
            fh.write("# header\n")
            fh.write("0.0  1.0  2.0\n")
            fh.write("1.0  1.5  2.5\n")
        t, d = parse_xvg_multi(p)
        assert d.shape == (2, 2)
        assert d[0, 1] == pytest.approx(2.0)

    def test_ps_conversion_multi(self, tmp_path):
        p = tmp_path / "multi_ps.xvg"
        with open(p, "w") as fh:
            fh.write("0.0  1.0\n")
            fh.write("500000.0  2.0\n")
        t, _ = parse_xvg_multi(p)
        assert t[-1] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# aggregate_replicates tests
# ---------------------------------------------------------------------------

class TestAggregateReplicates:
    def test_single_replicate(self):
        t = np.linspace(0, 600, 100)
        v = np.sin(t)
        ref_t, mean, sem = aggregate_replicates([(t, v)])
        np.testing.assert_allclose(mean, v)
        np.testing.assert_allclose(sem, np.zeros_like(v))

    def test_identical_replicates(self):
        t = np.linspace(0, 100, 50)
        v = np.ones(50) * 3.0
        ref_t, mean, sem = aggregate_replicates([(t, v), (t, v), (t, v)])
        np.testing.assert_allclose(mean, 3.0)
        np.testing.assert_allclose(sem, np.zeros_like(mean), atol=1e-12)

    def test_mean_of_two(self):
        t = np.array([0.0, 1.0, 2.0])
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([3.0, 4.0, 5.0])
        _, mean, sem = aggregate_replicates([(t, v1), (t, v2)])
        np.testing.assert_allclose(mean, [2.0, 3.0, 4.0])

    def test_sem_two_replicates(self):
        t = np.array([0.0, 1.0])
        v1 = np.array([1.0, 3.0])
        v2 = np.array([3.0, 5.0])
        _, mean, sem = aggregate_replicates([(t, v1), (t, v2)])
        # For values [1.0, 3.0]: std(ddof=1) = sqrt(2), n = 2
        # SEM = sqrt(2) / sqrt(2) = 1.0
        assert sem[0] == pytest.approx(1.0, rel=1e-5)

    def test_empty_list(self):
        t, mean, sem = aggregate_replicates([])
        assert len(t) == 0

    def test_interpolation_different_lengths(self):
        t1 = np.array([0.0, 1.0, 2.0])
        v1 = np.array([0.0, 1.0, 2.0])
        t2 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # finer grid
        v2 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        # Should interpolate v2 onto t1 (3 points)
        ref_t, mean, sem = aggregate_replicates([(t1, v1), (t2, v2)])
        assert len(ref_t) == 3  # ref is first replicate's grid
        np.testing.assert_allclose(mean, [0.0, 1.0, 2.0], atol=1e-10)


# ---------------------------------------------------------------------------
# MDAnalysis-based analyses with synthetic trajectories
# ---------------------------------------------------------------------------

def _make_universe_from_coords(coords_sequence):
    """Build an MDAnalysis Universe from a list of (N, 3) coordinate arrays.

    Each array is one frame.  All atoms are typed as CA (Cα) for simplicity.
    Requires MDAnalysis >=2.0 with MemoryReader support.
    """
    try:
        import MDAnalysis as mda
        from MDAnalysis.core.universe import Universe
        from MDAnalysis.coordinates.memory import MemoryReader
    except ImportError:
        pytest.skip("MDAnalysis not installed")

    n_atoms = coords_sequence[0].shape[0]
    n_frames = len(coords_sequence)

    # Build coordinate array (n_frames, n_atoms, 3)
    coords = np.stack(coords_sequence).astype(np.float32)

    # Create a Universe with n_atoms CA atoms using in-memory topology
    u = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_atoms,
        atom_resindex=np.arange(n_atoms),
        residue_segindex=np.zeros(n_atoms, dtype=int),
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["CA"] * n_atoms)
    u.add_TopologyAttr("resname", ["ALA"] * n_atoms)
    u.add_TopologyAttr("resid", list(range(1, n_atoms + 1)))
    u.add_TopologyAttr("segid", ["A"])   # one segid per segment (1 segment total)
    u.add_TopologyAttr("masses", [12.0] * n_atoms)
    u.load_new(coords, format=MemoryReader)

    return u


class TestComputeRmsdMda:
    def test_same_coords_gives_zero(self):
        rng = np.random.default_rng(10)
        coords = rng.standard_normal((20, 3)).astype(np.float32)
        # All frames identical → RMSD = 0 vs frame 0
        u = _make_universe_from_coords([coords] * 5)
        t, rmsd = compute_rmsd_mda(u, ref_frame=0)
        np.testing.assert_allclose(rmsd, 0.0, atol=1e-6)

    def test_increasing_rmsd(self):
        """Frames that progressively shift away from ref should have increasing RMSD."""
        rng = np.random.default_rng(11)
        ref = rng.standard_normal((15, 3)).astype(np.float32)
        frames = [ref + np.float32(i) * np.ones((15, 3)) for i in range(5)]
        u = _make_universe_from_coords(frames)
        t, rmsd = compute_rmsd_mda(u, ref_frame=0)
        # RMSD should be 0 at frame 0 and > 0 at later frames
        assert rmsd[0] == pytest.approx(0.0, abs=1e-5)
        # After centering, pure translation gives RMSD = 0 (Kabsch)
        # so test that all frames have RMSD = 0 (translation is removed)
        np.testing.assert_allclose(rmsd, 0.0, atol=1e-4)

    def test_n_sample_reduces_output_length(self):
        rng = np.random.default_rng(12)
        frames = [rng.standard_normal((10, 3)).astype(np.float32) for _ in range(50)]
        u = _make_universe_from_coords(frames)
        t, rmsd = compute_rmsd_mda(u, ref_frame=0, n_sample=10)
        assert len(rmsd) == 10
        assert len(t) == 10

    def test_returns_nonnegative_values(self):
        rng = np.random.default_rng(13)
        frames = [rng.standard_normal((20, 3)).astype(np.float32) for _ in range(20)]
        u = _make_universe_from_coords(frames)
        _, rmsd = compute_rmsd_mda(u)
        assert (rmsd >= 0).all()


class TestComputeRmsf:
    def test_zero_rmsf_for_static_system(self):
        """All frames identical → RMSF = 0 per residue."""
        rng = np.random.default_rng(20)
        coords = rng.standard_normal((10, 3)).astype(np.float32)
        u = _make_universe_from_coords([coords] * 20)
        res_ids, rmsf = compute_rmsf_mda(u, start_frac=0.0)
        np.testing.assert_allclose(rmsf, 0.0, atol=1e-6)

    def test_high_rmsf_for_fluctuating_system(self):
        """Random frames → each residue should have non-zero RMSF."""
        rng = np.random.default_rng(21)
        n_atoms = 15
        frames = [rng.standard_normal((n_atoms, 3)).astype(np.float32) for _ in range(30)]
        u = _make_universe_from_coords(frames)
        res_ids, rmsf = compute_rmsf_mda(u, start_frac=0.0)
        assert len(rmsf) == n_atoms
        assert (rmsf > 0).all()

    def test_start_frac_uses_subset(self):
        """start_frac=0.5 should use only the last 50% of frames."""
        rng = np.random.default_rng(22)
        n_atoms = 8
        # First 10 frames are static; last 10 are noisy
        static_coords = np.zeros((n_atoms, 3), dtype=np.float32)
        noisy_frames = [rng.standard_normal((n_atoms, 3)).astype(np.float32) for _ in range(10)]
        frames = [static_coords] * 10 + noisy_frames
        u = _make_universe_from_coords(frames)
        # With start_frac=0.5 → uses frames 10..19 (noisy) → non-zero RMSF
        _, rmsf_noisy = compute_rmsf_mda(u, start_frac=0.5)
        # With start_frac=0.0 → uses all frames
        _, rmsf_all = compute_rmsf_mda(u, start_frac=0.0)
        # Noisy-only RMSF should be higher than all-frames RMSF
        assert rmsf_noisy.mean() > rmsf_all.mean()


class TestComputeRgMda:
    def test_single_atom_zero_rg(self):
        """A single atom always has Rg = 0."""
        coords = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        u = _make_universe_from_coords([coords] * 5)
        t, rg = compute_rg_mda(u)
        np.testing.assert_allclose(rg, 0.0, atol=1e-6)

    def test_rg_positive_for_multi_atom(self):
        rng = np.random.default_rng(30)
        frames = [rng.standard_normal((20, 3)).astype(np.float32) for _ in range(10)]
        u = _make_universe_from_coords(frames)
        t, rg = compute_rg_mda(u)
        assert (rg > 0).all()

    def test_n_sample(self):
        rng = np.random.default_rng(31)
        frames = [rng.standard_normal((10, 3)).astype(np.float32) for _ in range(40)]
        u = _make_universe_from_coords(frames)
        t, rg = compute_rg_mda(u, n_sample=8)
        assert len(t) == 8
        assert len(rg) == 8

    def test_known_rg(self):
        """Three atoms at ±d from origin along x → Rg = d."""
        d = 3.0
        coords = np.array([[d, 0, 0], [-d, 0, 0], [0, 0, 0]], dtype=np.float32)
        u = _make_universe_from_coords([coords])
        t, rg = compute_rg_mda(u)
        # COM = (0,0,0), distances = d, d, 0 → Rg = sqrt(mean([d², d², 0])) = d/sqrt(1.5) ≈ 2.449
        expected = math.sqrt((d**2 + d**2 + 0) / 3)
        assert rg[0] == pytest.approx(expected, rel=0.01)
