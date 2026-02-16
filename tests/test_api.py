"""
Tests for the high-level minkowski_functionals() API.
Uses a box mesh (a=2, b=3, c=4) built as numpy arrays (no file I/O).
"""

import math
import numpy as np
import pytest

from karambola_py.api import minkowski_functionals


def _box_mesh(a, b, c):
    """Build a triangulated axis-aligned box centered at the origin.

    Returns (verts, faces) numpy arrays. The box spans
    [-a/2, a/2] x [-b/2, b/2] x [-c/2, c/2].
    Each face of the box is split into 2 triangles (12 triangles total).
    Normals point outward.
    """
    ha, hb, hc = a / 2.0, b / 2.0, c / 2.0
    verts = np.array([
        [-ha, -hb, -hc],  # 0
        [ ha, -hb, -hc],  # 1
        [ ha,  hb, -hc],  # 2
        [-ha,  hb, -hc],  # 3
        [-ha, -hb,  hc],  # 4
        [ ha, -hb,  hc],  # 5
        [ ha,  hb,  hc],  # 6
        [-ha,  hb,  hc],  # 7
    ], dtype=np.float64)

    # Each face: 2 triangles, winding for outward normals
    faces = np.array([
        # -z face (normal -z): 0,3,2 and 0,2,1
        [0, 3, 2], [0, 2, 1],
        # +z face (normal +z): 4,5,6 and 4,6,7
        [4, 5, 6], [4, 6, 7],
        # -y face (normal -y): 0,1,5 and 0,5,4
        [0, 1, 5], [0, 5, 4],
        # +y face (normal +y): 2,3,7 and 2,7,6
        [2, 3, 7], [2, 7, 6],
        # -x face (normal -x): 0,4,7 and 0,7,3
        [0, 4, 7], [0, 7, 3],
        # +x face (normal +x): 1,2,6 and 1,6,5
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)

    return verts, faces


class TestStandardBox:
    """Test standard functionals on centered box a=2, b=3, c=4."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a, self.b, self.c = 2.0, 3.0, 4.0
        self.verts, self.faces = _box_mesh(self.a, self.b, self.c)
        self.result = minkowski_functionals(self.verts, self.faces)

    def test_w000_volume(self):
        expected = self.a * self.b * self.c  # 24
        assert self.result['w000'] == pytest.approx(expected, rel=1e-4)

    def test_w100_area(self):
        expected = 2.0 / 3.0 * (self.a * self.b + self.a * self.c + self.b * self.c)
        assert self.result['w100'] == pytest.approx(expected, rel=1e-4)

    def test_w200_mean_curvature(self):
        expected = math.pi / 3.0 * (self.a + self.b + self.c)
        assert self.result['w200'] == pytest.approx(expected, rel=1e-4)

    def test_w300_euler(self):
        expected = 4.0 * math.pi / 3.0
        assert self.result['w300'] == pytest.approx(expected, rel=1e-4)

    def test_vectors_zero(self):
        for name in ['w010', 'w110', 'w210', 'w310']:
            np.testing.assert_allclose(self.result[name], [0, 0, 0], atol=1e-3,
                                       err_msg=f"{name} should be zero for centered box")

    def test_w020_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        expected = sorted([a**3 * b * c / 12, b**3 * a * c / 12, c**3 * a * b / 12])
        actual = sorted(self.result['w020_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w120_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        expected = sorted([
            1.0/6 * (1.0/3 * a**3 * (b + c) + a**2 * b * c),
            1.0/6 * (1.0/3 * b**3 * (a + c) + b**2 * a * c),
            1.0/6 * (1.0/3 * c**3 * (a + b) + c**2 * a * b),
        ])
        actual = sorted(self.result['w120_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w220_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = math.pi
        expected = sorted([
            pi / 36 * (a**3 + 3 * a**2 * (b + c)),
            pi / 36 * (b**3 + 3 * b**2 * (a + c)),
            pi / 36 * (c**3 + 3 * c**2 * (a + b)),
        ])
        actual = sorted(self.result['w220_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w320_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = math.pi
        expected = sorted([pi / 3 * a**2, pi / 3 * b**2, pi / 3 * c**2])
        actual = sorted(self.result['w320_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w102_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        expected = sorted([2.0/3 * a * b, 2.0/3 * a * c, 2.0/3 * b * c])
        actual = sorted(self.result['w102_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_w202_eigenvalues(self):
        a, b, c = self.a, self.b, self.c
        pi = math.pi
        expected = sorted([pi / 6 * (a + b), pi / 6 * (a + c), pi / 6 * (b + c)])
        actual = sorted(self.result['w202_eigvals'])
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_eigvecs_shape(self):
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            assert self.result[f'{name}_eigvecs'].shape == (3, 3)

    def test_standard_keys(self):
        expected_keys = {
            'w000', 'w100', 'w200', 'w300',
            'w010', 'w110', 'w210', 'w310',
            'w020', 'w120', 'w220', 'w320', 'w102', 'w202',
        }
        # Plus eigvals/eigvecs for each rank-2
        for name in ['w020', 'w120', 'w220', 'w320', 'w102', 'w202']:
            expected_keys.add(f'{name}_eigvals')
            expected_keys.add(f'{name}_eigvecs')
        assert set(self.result.keys()) == expected_keys


class TestCenterOptions:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a, self.b, self.c = 2.0, 3.0, 4.0
        self.verts, self.faces = _box_mesh(self.a, self.b, self.c)

    def test_centroid_runs(self):
        result = minkowski_functionals(self.verts, self.faces, center='centroid')
        assert 'w000' in result

    def test_explicit_center_scalars(self):
        shift = np.array([1.0, 2.0, 3.0])
        shifted_verts = self.verts + shift
        result = minkowski_functionals(shifted_verts, self.faces, center=shift)
        # Scalars should match centered box
        expected_vol = self.a * self.b * self.c
        assert result['w000'] == pytest.approx(expected_vol, rel=1e-4)
        expected_area = 2.0 / 3.0 * (self.a * self.b + self.a * self.c + self.b * self.c)
        assert result['w100'] == pytest.approx(expected_area, rel=1e-4)

    def test_explicit_center_vectors_zero(self):
        shift = np.array([1.0, 2.0, 3.0])
        shifted_verts = self.verts + shift
        result = minkowski_functionals(shifted_verts, self.faces, center=shift)
        # Vectors should be zero for re-centered box
        for name in ['w010', 'w110', 'w210', 'w310']:
            np.testing.assert_allclose(result[name], [0, 0, 0], atol=1e-3)


class TestComputeOptions:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verts, self.faces = _box_mesh(2.0, 3.0, 4.0)

    def test_compute_all_has_extras(self):
        result = minkowski_functionals(self.verts, self.faces, compute='all')
        assert 'w103' in result
        assert 'w104' in result
        assert 'msm_ql' in result
        assert 'msm_wl' in result
        assert result['w103'].shape == (3, 3, 3)
        assert result['w104'].shape == (6, 6)
        assert result['msm_ql'].shape[0] > 0
        assert result['msm_wl'].shape[0] > 0

    def test_compute_subset(self):
        result = minkowski_functionals(self.verts, self.faces, compute=['w000', 'w100'])
        assert set(result.keys()) == {'w000', 'w100'}

    def test_compute_single_tensor(self):
        result = minkowski_functionals(self.verts, self.faces, compute=['w102'])
        assert 'w102' in result
        assert 'w102_eigvals' in result
        assert 'w102_eigvecs' in result


class TestMultiLabel:

    def test_labels_returns_per_label_dict(self):
        verts, faces = _box_mesh(2.0, 3.0, 4.0)
        # Split faces into two groups
        labels = np.array([0]*6 + [1]*6, dtype=np.int64)
        result = minkowski_functionals(verts, faces, labels=labels)
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert 'w000' in result[0]
        assert 'w000' in result[1]
        # Total volume should equal box volume
        total_vol = result[0]['w000'] + result[1]['w000']
        assert total_vol == pytest.approx(2.0 * 3.0 * 4.0, rel=1e-4)
