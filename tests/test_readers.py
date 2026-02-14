"""
Tests for .poly and .off file readers.
"""

import os
import pytest
import numpy as np

from karambola_py.io_poly import parse_poly_file
from karambola_py.io_off import parse_off_file, is_off_file
from karambola_py.triangulation import LABEL_UNASSIGNED
from karambola_py.surface import check_surface
from karambola_py.results import CalcOptions
from karambola_py.minkowski import calculate_w000, calculate_w100

TEST_INPUTS = os.path.join(os.path.dirname(__file__), "..", "test_suite", "inputs")


class TestPolyReader:
    """Tests for .poly file parsing."""

    def test_box_poly(self):
        """Read box.poly and verify vertex/triangle counts."""
        filepath = os.path.join(TEST_INPUTS, "box_a=2_b=3_c=4.poly")
        surface = parse_poly_file(filepath)
        assert surface.n_vertices() == 8
        assert surface.n_triangles() == 12

    def test_box_poly_volume(self):
        """Read box.poly and verify volume computation."""
        filepath = os.path.join(TEST_INPUTS, "box_a=2_b=3_c=4.poly")
        surface = parse_poly_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        assert w000[LABEL_UNASSIGNED].result == pytest.approx(24.0, rel=1e-4)

    def test_empty_poly(self):
        """Empty poly should raise an error during surface check."""
        filepath = os.path.join(TEST_INPUTS, "empty.poly")
        surface = parse_poly_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        co = CalcOptions()
        with pytest.raises(RuntimeError, match="no polygons"):
            check_surface(co, surface)


class TestOffReader:
    """Tests for .off file parsing."""

    def test_cuboid_off(self):
        """Read cuboid.off and verify vertex/triangle counts."""
        filepath = os.path.join(TEST_INPUTS, "cuboid.off")
        surface = parse_off_file(filepath)
        assert surface.n_vertices() == 8
        # 6 faces * 2 triangles each = 12 triangles (quads are triangulated)
        assert surface.n_triangles() == 12

    def test_cuboid_off_volume(self):
        """Read cuboid.off and verify volume."""
        filepath = os.path.join(TEST_INPUTS, "cuboid.off")
        surface = parse_off_file(filepath)
        surface.create_vertex_polygon_lookup_table()
        surface.create_polygon_polygon_lookup_table()
        w000 = calculate_w000(surface)
        # Cuboid volume should be positive
        assert w000[LABEL_UNASSIGNED].result > 0

    def test_cuboid_with_labels(self):
        """Read cuboid with labels from alpha channel."""
        filepath = os.path.join(TEST_INPUTS, "cuboid-labels.off")
        if not os.path.exists(filepath):
            pytest.skip("cuboid-labels.off not found")
        surface = parse_off_file(filepath, with_labels=True)
        assert surface.n_vertices() >= 8

    def test_cuboid_with_comments(self):
        """Read cuboid with comments in .off file."""
        filepath = os.path.join(TEST_INPUTS, "cuboid-labelsWithComments.off")
        if not os.path.exists(filepath):
            pytest.skip("cuboid-labelsWithComments.off not found")
        surface = parse_off_file(filepath)
        assert surface.n_vertices() >= 8

    def test_is_off_file(self):
        assert is_off_file("test.off")
        assert is_off_file("test.OFF")
        assert not is_off_file("test.poly")
        assert not is_off_file("test.txt")
