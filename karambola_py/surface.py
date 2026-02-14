"""
Surface validation and statistics.
"""

import math
import numpy as np
from .triangulation import NEIGHBOUR_UNASSIGNED
from .results import SurfaceStatistics


def check_surface(calc_options, surface):
    """Check surface properties (closure status, statistics, validation).

    Parameters
    ----------
    calc_options : CalcOptions
        Will be updated with label closure status.
    surface : Triangulation
        The surface to check.

    Returns
    -------
    SurfaceStatistics
    """
    stats = SurfaceStatistics()

    if surface.n_triangles() == 0:
        raise RuntimeError("There are no polygons in your .poly file")

    # Initialize all labels as closed (status 0)
    for i in range(surface.n_triangles()):
        calc_options.create_label(surface.label_of_triangle(i), 0)

    # Check for shared edges (different labels on neighbours) -> status 1
    for i in range(surface.n_triangles()):
        for j in range(3):
            nb = surface.ith_neighbour_of_triangle(i, j)
            if nb != NEIGHBOUR_UNASSIGNED:
                if surface.label_of_triangle(i) != surface.label_of_triangle(nb):
                    calc_options.create_label(surface.label_of_triangle(i), 1)

    # Check for open edges (no neighbour) -> status 2
    for i in range(surface.n_triangles()):
        for j in range(3):
            if surface.ith_neighbour_of_triangle(i, j) == NEIGHBOUR_UNASSIGNED:
                calc_options.create_label(surface.label_of_triangle(i), 2)

    # Edge length statistics
    shortest_edge = math.inf
    longest_edge = 0.0
    for i in range(surface.n_triangles()):
        for j in range(3):
            v1 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(i, j))
            v2 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(i, (j + 1) % 3))
            e = np.linalg.norm(v1 - v2)
            if e < shortest_edge:
                shortest_edge = e
            if e > longest_edge:
                longest_edge = e
    stats.shortest_edge = shortest_edge
    stats.longest_edge = longest_edge

    # Area statistics
    smallest_area = math.inf
    largest_area = 0.0
    for i in range(surface.n_triangles()):
        area = surface.area_of_triangle(i)
        if area < smallest_area:
            smallest_area = area
        if area > largest_area:
            largest_area = area
    stats.smallest_area = smallest_area
    stats.largest_area = largest_area

    if largest_area > 0 and smallest_area / largest_area < 1e-12:
        print(f"\nWARNING: there is something wrong with the area of your facets!")
        print(f"smallest area: {smallest_area}")
        print(f"largest area:  {largest_area}\n")

    # Check for multiple objects at one vertex and normal consistency
    for i in range(surface.n_vertices()):
        tris = surface.get_triangles_of_vertex(i)
        if len(tris) == 0:
            continue

        sum_of_triangles = 1
        start_triangle = tris[0]
        old_triangle = NEIGHBOUR_UNASSIGNED
        triangle = tris[0]
        new_triangle = tris[0]

        neigh_un = False
        normal_wrong = False

        for _ in range(len(tris)):
            triangle = new_triangle
            vertex_id = 0
            for k in range(3):
                if surface.ith_vertex_of_triangle(triangle, k) == i:
                    vertex_id = k

            if surface.ith_neighbour_of_triangle(triangle, vertex_id) == NEIGHBOUR_UNASSIGNED:
                neigh_un = True
                break

            new_triangle = surface.ith_neighbour_of_triangle(triangle, vertex_id)

            if old_triangle == new_triangle:
                normal_wrong = True
                break

            if new_triangle == start_triangle:
                break

            sum_of_triangles += 1
            old_triangle = triangle

        if normal_wrong:
            raise RuntimeError(
                f"shortest edge = {stats.shortest_edge}\n"
                f"longest edge  = {stats.longest_edge}\n"
                f"smallest area = {stats.smallest_area}\n"
                f"largest area  = {stats.largest_area}\n"
                f"your polyfile is damaged\n"
                f"your normals are messed up!!\n"
                f"error occurs at vertex: {surface.get_original_number_of_vertex_in_file(i)}"
            )

        if len(tris) != sum_of_triangles and not neigh_un:
            raise RuntimeError(
                f"your polyfile is damaged\n"
                f"there are more than one objects at vertex "
                f"{surface.get_original_number_of_vertex_in_file(i)}: "
                f"{surface.get_pos_of_vertex(i)}"
            )

    return stats
