"""
All Minkowski functional calculations.

Each function returns a dict mapping label -> MinkValResult.
"""

import numpy as np
from collections import defaultdict
from .triangulation import LABEL_UNASSIGNED, NEIGHBOUR_UNASSIGNED
from .results import MinkValResult
from .tensor import SymmetricMatrix3, Rank3Tensor, SymmetricRank4Tensor, fourth_tensorial_power


def _default_scalar():
    return MinkValResult(result=0.0)


def _default_vector():
    return MinkValResult(result=np.zeros(3, dtype=np.float64))


def _default_matrix():
    return MinkValResult(result=SymmetricMatrix3())


def _default_rank3():
    return MinkValResult(result=Rank3Tensor())


def _default_rank4():
    return MinkValResult(result=SymmetricRank4Tensor())


def _get_or_create(d, label, factory):
    if label not in d:
        d[label] = factory()
    return d[label]


def get_ref_vec(label, w_scalar, w_vector):
    """Compute reference vector (centroid) = w_vector / w_scalar for a label."""
    return w_vector[label].result / w_scalar[label].result


# ============================================================================
# Scalar functionals
# ============================================================================

def calculate_w000(surface):
    """Volume via divergence theorem: (1/3) * sum(com . normal * area)."""
    results = {}
    for i in range(surface.n_triangles()):
        label = surface.label_of_triangle(i)
        r = _get_or_create(results, label, _default_scalar)
        com = surface.com_of_triangle(i)
        n = surface.normal_vector_of_triangle(i)
        area = surface.area_of_triangle(i)
        r.result += np.dot(com, n) * area * (1.0 / 3.0)
    return results


def calculate_w100(surface):
    """Surface area: (1/3) * sum(area)."""
    results = {}
    for i in range(surface.n_triangles()):
        label = surface.label_of_triangle(i)
        r = _get_or_create(results, label, _default_scalar)
        r.result += surface.area_of_triangle(i) * (1.0 / 3.0)
    return results


def calculate_w200(surface):
    """Mean curvature integral via dihedral angles."""
    results = {}
    for i in range(surface.n_triangles()):
        label = surface.label_of_triangle(i)
        _get_or_create(results, label, _default_scalar)
        for j in range(3):
            nb = surface.ith_neighbour_of_triangle(i, j)
            if nb == NEIGHBOUR_UNASSIGNED:
                continue
            a = surface.normal_vector_of_triangle(nb)
            b = surface.normal_vector_of_triangle(i)
            c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            if c >= 1.0:
                c = 1.0
            alpha = np.arccos(c)

            # Determine convexity
            convex = (surface.com_of_triangle(i) + b
                      - (surface.com_of_triangle(nb) + a))
            concave = (surface.com_of_triangle(i) - b
                       - (surface.com_of_triangle(nb) - a))
            if np.linalg.norm(convex) < np.linalg.norm(concave):
                alpha = -alpha

            results[label].result += alpha * surface.get_edge_length(i, j) / (2.0 * 6.0)
    return results


def calculate_w300(surface):
    """Gaussian curvature (Euler characteristic) via angle deficit."""
    results = {}
    for i in range(surface.n_vertices()):
        angle_sum = surface.sum_of_angles_of_ith_vertex(i)
        for tri_idx in surface.get_triangles_of_vertex(i):
            for k in range(3):
                if surface.ith_vertex_of_triangle(tri_idx, k) == i:
                    label = surface.label_of_triangle(tri_idx)
                    _get_or_create(results, label, _default_scalar)
                    angle = surface.get_angle_of_ith_edge(tri_idx, k)
                    w3_part = (2.0 * np.pi * (angle / angle_sum)) - angle
                    results[label].result += w3_part / 3.0
    return results


# ============================================================================
# Vector functionals
# ============================================================================

def calculate_w010(surface):
    """Volume integral of position (first moment)."""
    results = {}
    for j in range(surface.n_triangles()):
        label = surface.label_of_triangle(j)
        r = _get_or_create(results, label, _default_vector)
        c1 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(j, 0))
        c2 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(j, 1))
        c3 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(j, 2))
        v = c2 - c1
        w = c3 - c1
        for i in range(3):
            ip1 = (i + 1) % 3
            part1 = (2 * v[i] * v[ip1]
                     + v[ip1] * w[i]
                     + 4 * c1[ip1] * (v[i] + w[i])
                     + v[i] * w[ip1]
                     + 2 * w[i] * w[ip1]
                     + 4 * c1[i] * (3 * c1[ip1] + v[ip1] + w[ip1]))
            vf = np.cross(v, w)[ip1]
            r.result[i] += vf * part1 / 24.0
    return results


def calculate_w110(surface):
    """Surface-weighted position: (1/3) * sum(com * area)."""
    results = {}
    for j in range(surface.n_triangles()):
        label = surface.label_of_triangle(j)
        r = _get_or_create(results, label, _default_vector)
        com = surface.com_of_triangle(j)
        area = surface.area_of_triangle(j)
        r.result += com * area / 3.0
    return results


def calculate_w210(surface):
    """Curvature-weighted position via dihedral angles."""
    results = {}
    for j in range(surface.n_triangles()):
        label = surface.label_of_triangle(j)
        _get_or_create(results, label, _default_vector)
        for k in range(3):
            nb = surface.ith_neighbour_of_triangle(j, k)
            if nb == NEIGHBOUR_UNASSIGNED:
                continue
            a = surface.normal_vector_of_triangle(nb)
            b = surface.normal_vector_of_triangle(j)
            c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            if c >= 1.0:
                c = 1.0
            alpha = np.arccos(c)

            convex = (surface.com_of_triangle(j) + b
                      - (surface.com_of_triangle(nb) + a))
            concave = (surface.com_of_triangle(j) - b
                       - (surface.com_of_triangle(nb) - a))
            if np.linalg.norm(convex) < np.linalg.norm(concave):
                alpha = -alpha

            e = surface.get_edge_length(j, k)
            e_c1 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(j, k))
            e_c2 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(j, (k + 1) % 3))
            for i in range(3):
                w210_part = alpha * e * (e_c1[i] + e_c2[i])
                results[label].result[i] += w210_part / (12.0 * 2.0)
    return results


def calculate_w310(surface):
    """Gaussian curvature-weighted position via angle deficit."""
    results = {}
    for j in range(surface.n_vertices()):
        angle_sum = surface.sum_of_angles_of_ith_vertex(j)
        for tri_idx in surface.get_triangles_of_vertex(j):
            for k in range(3):
                if surface.ith_vertex_of_triangle(tri_idx, k) == j:
                    label = surface.label_of_triangle(tri_idx)
                    r = _get_or_create(results, label, _default_vector)
                    angle = surface.get_angle_of_ith_edge(tri_idx, k)
                    pos = surface.get_pos_of_vertex(j)
                    w310_part = ((2.0 * np.pi * (angle / angle_sum)) - angle)
                    r.result += w310_part * pos / 3.0
    return results


# ============================================================================
# Matrix functionals
# ============================================================================

def calculate_w020(surface, w000=None, w010=None):
    """Second moment (inertia) tensor."""
    results = {}
    for k in range(surface.n_triangles()):
        label = surface.label_of_triangle(k)
        r = _get_or_create(results, label, _default_matrix)

        ref_vec = np.zeros(3)
        if w000 and len(w000) > 0:
            ref_vec = get_ref_vec(label, w000, w010)

        c1 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(k, 0)) - ref_vec
        c2 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(k, 1)) - ref_vec
        c3 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(k, 2)) - ref_vec
        n = surface.normal_vector_of_triangle(k)
        area = surface.area_of_triangle(k)

        # Ixx
        Ixx = (3 * c2[0]**2 * c2[2] + 2 * c2[0] * c2[2] * c3[0]
               + c2[2] * c3[0]**2
               + c1[2] * (c2[0]**2 + c2[0] * c3[0] + c3[0]**2)
               + c2[0]**2 * c3[2] + 2 * c2[0] * c3[0] * c3[2]
               + 3 * c3[0]**2 * c3[2]
               + c1[0]**2 * (3 * c1[2] + c2[2] + c3[2])
               + c1[0] * (2 * c1[2] * (c2[0] + c3[0])
                          + c2[0] * (2 * c2[2] + c3[2])
                          + c3[0] * (c2[2] + 2 * c3[2]))) / 60.0
        r.result[0, 0] += Ixx * 2 * area * n[2]

        # Iyy
        Iyy = (3 * c2[1]**2 * c2[2] + 2 * c2[1] * c2[2] * c3[1]
               + c2[2] * c3[1]**2
               + c1[2] * (c2[1]**2 + c2[1] * c3[1] + c3[1]**2)
               + c2[1]**2 * c3[2] + 2 * c2[1] * c3[1] * c3[2]
               + 3 * c3[1]**2 * c3[2]
               + c1[1]**2 * (3 * c1[2] + c2[2] + c3[2])
               + c1[1] * (2 * c1[2] * (c2[1] + c3[1])
                          + c2[1] * (2 * c2[2] + c3[2])
                          + c3[1] * (c2[2] + 2 * c3[2]))) / 60.0
        r.result[1, 1] += Iyy * 2 * area * n[2]

        # Izz
        Izz = (3 * c2[1] * c2[2]**2 + c2[2]**2 * c3[1]
               + c1[2]**2 * (c2[1] + c3[1])
               + 2 * c2[1] * c2[2] * c3[2] + 2 * c2[2] * c3[1] * c3[2]
               + c2[1] * c3[2]**2 + 3 * c3[1] * c3[2]**2
               + c1[1] * (3 * c1[2]**2 + c2[2]**2 + c2[2] * c3[2]
                          + c3[2]**2 + 2 * c1[2] * (c2[2] + c3[2]))
               + c1[2] * (c2[1] * (2 * c2[2] + c3[2])
                          + c3[1] * (c2[2] + 2 * c3[2]))) / 60.0
        r.result[2, 2] += Izz * 2 * area * n[1]

        # Ixy
        Ixy = (2 * c1[2] * c2[0] * c2[1] + 6 * c2[0] * c2[1] * c2[2]
               + c1[2] * c2[1] * c3[0] + 2 * c2[1] * c2[2] * c3[0]
               + c1[2] * c2[0] * c3[1] + 2 * c2[0] * c2[2] * c3[1]
               + 2 * c1[2] * c3[0] * c3[1] + 2 * c2[2] * c3[0] * c3[1]
               + 2 * c2[0] * c2[1] * c3[2] + 2 * c2[1] * c3[0] * c3[2]
               + 2 * c2[0] * c3[1] * c3[2] + 6 * c3[0] * c3[1] * c3[2]
               + c1[0] * (2 * c2[1] * c2[2] + c2[2] * c3[1]
                          + 2 * c1[2] * (c2[1] + c3[1])
                          + c2[1] * c3[2] + 2 * c3[1] * c3[2]
                          + 2 * c1[1] * (3 * c1[2] + c2[2] + c3[2]))
               + c1[1] * (2 * c1[2] * (c2[0] + c3[0])
                          + c2[0] * (2 * c2[2] + c3[2])
                          + c3[0] * (c2[2] + 2 * c3[2]))) / 120.0

        r.result[0, 1] += Ixy * 2 * area * n[2]
        r.result[0, 2] += Ixy * 2 * area * n[1]
        r.result[1, 2] += Ixy * 2 * area * n[0]

    return results


def calculate_w120(surface, w100=None, w110=None):
    """Surface-weighted tensor product."""
    results = {}
    for k in range(surface.n_triangles()):
        label = surface.label_of_triangle(k)
        r = _get_or_create(results, label, _default_matrix)
        area = surface.area_of_triangle(k)

        ref_vec = np.zeros(3)
        if w100 and len(w100) > 0:
            ref_vec = get_ref_vec(label, w100, w110)

        c1 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(k, 0)) - ref_vec
        c2 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(k, 1)) - ref_vec
        c3 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(k, 2)) - ref_vec

        for i in range(3):
            for j in range(i + 1):
                part_1 = c1[i] * c1[j] + c2[i] * c2[j] + c3[i] * c3[j]
                part_2 = c1[i] * c2[j] + c2[i] * c3[j] + c3[i] * c1[j]
                part_3 = c1[j] * c2[i] + c2[j] * c3[i] + c3[j] * c1[i]
                r.result[i, j] += (1.0 / 18.0) * (part_1 + part_2 / 2.0 + part_3 / 2.0) * area
    return results


def calculate_w220(surface, w200=None, w210=None):
    """Curvature-weighted tensor product via dihedral angles."""
    results = {}
    for l in range(surface.n_triangles()):
        label = surface.label_of_triangle(l)
        _get_or_create(results, label, _default_matrix)
        for k in range(3):
            nb = surface.ith_neighbour_of_triangle(l, k)
            if nb == NEIGHBOUR_UNASSIGNED:
                continue
            a = surface.normal_vector_of_triangle(nb)
            b = surface.normal_vector_of_triangle(l)
            c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            if c >= 1.0:
                c = 1.0
            alpha = np.arccos(c)

            convex = (surface.com_of_triangle(l) + b
                      - (surface.com_of_triangle(nb) + a))
            concave = (surface.com_of_triangle(l) - b
                       - (surface.com_of_triangle(nb) - a))
            if np.linalg.norm(convex) < np.linalg.norm(concave):
                alpha = -alpha

            e = surface.get_edge_length(l, k)

            ref_vec = np.zeros(3)
            if w200 and len(w200) > 0:
                ref_vec = get_ref_vec(label, w200, w210)

            e_c1 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(l, k)) - ref_vec
            e_c2 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(l, (k + 1) % 3)) - ref_vec

            for i in range(3):
                for j in range(i + 1):
                    w220_part = alpha * e * (
                        e_c1[i] * e_c1[j]
                        + 0.5 * (e_c1[i] * e_c2[j] + e_c1[j] * e_c2[i])
                        + e_c2[i] * e_c2[j]
                    )
                    results[label].result[i, j] += w220_part / (18.0 * 2.0)
    return results


def calculate_w320(surface, w300=None, w310=None):
    """Gaussian curvature-weighted tensor product via angle deficit."""
    results = {}
    for m in range(surface.n_vertices()):
        angle_sum = surface.sum_of_angles_of_ith_vertex(m)
        for tri_idx in surface.get_triangles_of_vertex(m):
            for k in range(3):
                if surface.ith_vertex_of_triangle(tri_idx, k) == m:
                    label = surface.label_of_triangle(tri_idx)
                    _get_or_create(results, label, _default_matrix)
                    angle = surface.get_angle_of_ith_edge(tri_idx, k)
                    angle_part = (2.0 * np.pi * (angle / angle_sum)) - angle

                    ref_vec = np.zeros(3)
                    if w300 and len(w300) > 0:
                        ref_vec = get_ref_vec(label, w300, w310)

                    c = surface.get_pos_of_vertex(m) - ref_vec
                    for i in range(3):
                        for j in range(i + 1):
                            results[label].result[i, j] += angle_part * c[i] * c[j] / 3.0
    return results


def calculate_w102(surface):
    """Normal tensor: (1/3) * sum(area * n_i * n_j)."""
    results = {}
    for k in range(surface.n_triangles()):
        label = surface.label_of_triangle(k)
        r = _get_or_create(results, label, _default_matrix)
        area = surface.area_of_triangle(k)
        n = surface.normal_vector_of_triangle(k)
        for i in range(3):
            for j in range(i + 1):
                r.result[i, j] += (1.0 / 3.0) * area * n[i] * n[j]
    return results


def calculate_w202(surface):
    """Curvature-weighted normal tensor."""
    results = {}
    for l in range(surface.n_triangles()):
        label = surface.label_of_triangle(l)
        _get_or_create(results, label, _default_matrix)
        for k in range(3):
            nb = surface.ith_neighbour_of_triangle(l, k)
            if nb == NEIGHBOUR_UNASSIGNED:
                continue
            n2 = surface.normal_vector_of_triangle(nb)
            n1 = surface.normal_vector_of_triangle(l)
            c = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
            if c >= 1.0:
                c = 1.0
            alpha = np.arccos(c)

            convex = (surface.com_of_triangle(l) + n1
                      - (surface.com_of_triangle(nb) + n2))
            concave = (surface.com_of_triangle(l) - n1
                       - (surface.com_of_triangle(nb) - n2))
            if np.linalg.norm(convex) < np.linalg.norm(concave):
                alpha = -alpha

            e_norm = surface.get_edge_length(l, k)
            e_c1 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(l, k))
            e_c2 = surface.get_pos_of_vertex(surface.ith_vertex_of_triangle(l, (k + 1) % 3))

            e_vec = (e_c2 - e_c1) / e_norm
            n_sum = n1 + n2
            n_a = n_sum / np.linalg.norm(n_sum)
            n_i = np.cross(e_vec, n_a)

            sin_alpha = np.sin(alpha)
            for i in range(3):
                for j in range(i + 1):
                    local_value = e_norm * (
                        (alpha + sin_alpha) * n_a[i] * n_a[j]
                        + (alpha - sin_alpha) * n_i[i] * n_i[j]
                    )
                    results[label].result[i, j] += (1.0 / 12.0) * (1.0 / 2.0) * local_value
    return results


# ============================================================================
# Higher-rank tensor functionals
# ============================================================================

def calculate_w103(surface):
    """Rank-3 normal tensor: (1/3) * sum(area * n_i * n_j * n_k)."""
    results = {}
    for l in range(surface.n_triangles()):
        label = surface.label_of_triangle(l)
        r = _get_or_create(results, label, _default_rank3)
        area = surface.area_of_triangle(l)
        n = surface.normal_vector_of_triangle(l)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r.result[i, j, k] += (1.0 / 3.0) * area * n[i] * n[j] * n[k]
    return results


def calculate_w104(surface):
    """Rank-4 normal tensor via fourth tensorial power."""
    results = {}
    for k in range(surface.n_triangles()):
        label = surface.label_of_triangle(k)
        r = _get_or_create(results, label, _default_rank4)
        area = surface.area_of_triangle(k)
        n = surface.normal_vector_of_triangle(k)
        t = fourth_tensorial_power(n)
        r.result.addmul(area / 3.0, t)
    return results
