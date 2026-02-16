"""
Triangulation data structure for representing 3D triangulated surfaces.
"""

import numpy as np

LABEL_UNASSIGNED = -300
NEIGHBOUR_UNASSIGNED = -200


class Triangulation:
    """Stores a triangulated surface mesh with vertices, triangles, and labels."""

    @classmethod
    def from_arrays(cls, verts, faces, labels=None):
        """Construct a Triangulation from numpy arrays.

        Parameters
        ----------
        verts : (V, 3) array_like
            Vertex positions.
        faces : (F, 3) array_like
            Triangle vertex indices.
        labels : (F,) array_like or None
            Per-face labels. If None, all faces get label 0.

        Returns
        -------
        Triangulation
        """
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        tri = cls()
        for i in range(len(verts)):
            tri.append_vertex(verts[i, 0], verts[i, 1], verts[i, 2], number_in_file=i)
        for i in range(len(faces)):
            lbl = int(labels[i]) if labels is not None else 0
            tri.append_triangle(int(faces[i, 0]), int(faces[i, 1]), int(faces[i, 2]), label=lbl)
        tri.create_vertex_polygon_lookup_table()
        tri.create_polygon_polygon_lookup_table()
        return tri

    def __init__(self):
        self._vertices = []          # list of (x, y, z)
        self._vertex_numbers = []    # original numbering from file
        self._triangles = []         # list of (v0, v1, v2)
        self._labels = []            # label per triangle
        self._vertex_triangles = []  # list of lists: triangles per vertex
        self._neighbour_triangles = []  # Mx3: neighbour across each edge

    def append_vertex(self, x, y, z, number_in_file=0):
        idx = len(self._vertices)
        self._vertices.append(np.array([x, y, z], dtype=np.float64))
        self._vertex_numbers.append(number_in_file)
        self._vertex_triangles.append([])
        return idx

    def change_pos_of_vertex(self, i, x, y, z):
        self._vertices[i] = np.array([x, y, z], dtype=np.float64)

    def append_triangle(self, v0, v1, v2, label=LABEL_UNASSIGNED):
        self._triangles.append((v0, v1, v2))
        self._labels.append(label)

    def n_triangles(self):
        return len(self._triangles)

    def n_vertices(self):
        return len(self._vertices)

    def ith_vertex_of_triangle(self, a, i):
        return self._triangles[a][i]

    def label_of_triangle(self, i):
        return self._labels[i]

    def get_pos_of_vertex(self, a):
        return self._vertices[a]

    def get_original_number_of_vertex_in_file(self, a):
        return self._vertex_numbers[a]

    def get_triangles_of_vertex(self, a):
        return self._vertex_triangles[a]

    def ith_neighbour_of_triangle(self, a, i):
        return self._neighbour_triangles[a][i]

    def area_of_triangle(self, i):
        v0 = self._vertices[self._triangles[i][0]]
        v1 = self._vertices[self._triangles[i][1]]
        v2 = self._vertices[self._triangles[i][2]]
        a = v1 - v0
        b = v2 - v0
        return np.linalg.norm(np.cross(a, b)) / 2.0

    def normal_vector_of_triangle(self, i):
        v0 = self._vertices[self._triangles[i][0]]
        v1 = self._vertices[self._triangles[i][1]]
        v2 = self._vertices[self._triangles[i][2]]
        a = v1 - v0
        b = v2 - v0
        c = np.cross(a, b)
        n = np.linalg.norm(c)
        return c / n

    def com_of_triangle(self, i):
        v0 = self._vertices[self._triangles[i][0]]
        v1 = self._vertices[self._triangles[i][1]]
        v2 = self._vertices[self._triangles[i][2]]
        return (v0 + v1 + v2) / 3.0

    def get_edge_length(self, a, i):
        v0 = self._vertices[self._triangles[a][i]]
        v1 = self._vertices[self._triangles[a][(i + 1) % 3]]
        return np.linalg.norm(v0 - v1)

    def get_angle_of_ith_edge(self, a, i):
        vi = self._vertices[self._triangles[a][i]]
        vj = self._vertices[self._triangles[a][(i + 1) % 3]]
        vk = self._vertices[self._triangles[a][(i + 2) % 3]]
        aa = vi - vj
        bb = vi - vk
        c = np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb))
        c = min(c, 1.0)
        return np.arccos(c)

    def sum_of_angles_of_ith_vertex(self, i):
        total = 0.0
        for tri_idx in self._vertex_triangles[i]:
            for k in range(3):
                if self._triangles[tri_idx][k] == i:
                    total += self.get_angle_of_ith_edge(tri_idx, k)
        return total

    def create_vertex_polygon_lookup_table(self):
        for i in range(len(self._vertices)):
            self._vertex_triangles[i] = []
        for i in range(len(self._triangles)):
            for j in range(3):
                self._vertex_triangles[self._triangles[i][j]].append(i)

    def create_polygon_polygon_lookup_table(self):
        self._neighbour_triangles = []
        for i in range(len(self._triangles)):
            neighbours = [NEIGHBOUR_UNASSIGNED, NEIGHBOUR_UNASSIGNED, NEIGHBOUR_UNASSIGNED]
            for j in range(3):
                vj = self._triangles[i][j]
                vj_next = self._triangles[i][(j + 1) % 3]
                for k_tri in self._vertex_triangles[vj]:
                    if k_tri == i:
                        continue
                    for l in range(3):
                        if self._triangles[k_tri][l] == vj_next:
                            neighbours[j] = k_tri
            self._neighbour_triangles.append(neighbours)
