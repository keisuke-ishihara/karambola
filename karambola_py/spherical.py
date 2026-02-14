"""
Spherical Minkowski functionals using spherical harmonics.
"""

import numpy as np
try:
    from scipy.special import sph_harm_y
    def _sph_harm(m, l, phi, theta):
        """Wrapper matching old sph_harm(m, l, phi, theta) convention."""
        return sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm as _sph_harm
from .results import MinkValResult

MAX_L = 8


class SphMinkData:
    """Container for ql and wl arrays."""
    def __init__(self):
        self.ql = [0.0] * 12
        self.wl = [0.0] * 12


def _default_sphmink():
    return MinkValResult(result=SphMinkData())


class SphericalMinkowskis:
    """Accumulates spherical harmonic coefficients for Minkowski functionals."""

    def __init__(self):
        # Store d_lm coefficients: for each l, store m = 0..l as complex
        self._d = {}  # (l, m) -> complex
        for l in range(MAX_L + 1):
            for m in range(l + 1):
                self._d[(l, m)] = 0.0 + 0.0j
        self.total_area = 0.0

    def add_facet(self, f):
        """Add a facet contribution (f = normal * area)."""
        area = np.linalg.norm(f)
        f_normalized = f / area
        cos_th = f_normalized[2]
        phi = np.arctan2(f_normalized[1], f_normalized[0])
        self.total_area += area

        for l in range(MAX_L + 1):
            l_prefactor = np.sqrt(4.0 * np.pi / (2 * l + 1))
            for m in range(l + 1):
                # Use scipy's normalized associated Legendre functions
                # sph_harm(m, l, phi, theta) where theta = polar angle
                # But we need the real spherical harmonic Plm * exp(i*m*phi)
                # gsl_sf_legendre_sphPlm returns the normalized Plm
                # scipy.special.sph_harm(m, l, phi, theta) returns Y_l^m
                # We need: Plm(cos_th) * exp(i*m*phi) * l_prefactor * area
                # scipy's lpmv is unnormalized; use sph_harm directly

                # sph_harm convention: Y_l^m(theta, phi) where theta is polar
                theta = np.arccos(np.clip(cos_th, -1, 1))
                ylm = _sph_harm(m, l, phi, theta)

                # The C++ code uses:
                #   leg = gsl_sf_legendre_sphPlm(l, m, cos_th) = sqrt((2l+1)/4pi * (l-m)!/(l+m)!) * Plm(cos_th)
                #   ylm = leg * exp(i*m*phi)
                # scipy sph_harm already computes:
                #   Y_l^m = sqrt((2l+1)/4pi * (l-m)!/(l+m)!)) * Plm(cos_th) * exp(i*m*phi)
                # So ylm from scipy is already the full Y_l^m.
                # The C++ multiplies by l_prefactor = sqrt(4pi/(2l+1)), so:
                #   contribution = area * l_prefactor * gsl_sf_legendre_sphPlm * exp(i*m*phi)
                #                = area * l_prefactor * Y_l^m / exp(i*m*phi) * exp(i*m*phi) ... no
                # Actually: Y_l^m = normalization * Plm * exp(i*m*phi)
                # And gsl_sf_legendre_sphPlm = normalization * Plm  (same normalization)
                # So gsl_sf_legendre_sphPlm * exp(i*m*phi) = Y_l^m
                # Therefore: contribution = area * l_prefactor * Y_l^m

                self._d[(l, m)] += area * l_prefactor * ylm

    def ql(self, l):
        """Compute q_l rotation invariant."""
        r = abs(self._d[(l, 0)])**2
        for m in range(1, l + 1):
            r += 2 * abs(self._d[(l, m)])**2
        return np.sqrt(r) / self.total_area

    def wl(self, l):
        """Compute w_l rotation invariant using Wigner 3j symbols."""
        try:
            from sympy.physics.wigner import wigner_3j
        except ImportError:
            return 0.0

        v = 0.0 + 0.0j

        for ma in range(-l, l + 1):
            for mb in range(-l, l + 1):
                mc = -(ma + mb)
                if abs(mc) > l:
                    continue

                # Wigner 3j symbol
                w3j = float(wigner_3j(l, l, l, ma, mb, mc))
                if w3j == 0.0:
                    continue

                # Get coefficients, handling negative m
                a = self._get_coeff(l, ma)
                b = self._get_coeff(l, mb)
                c = self._get_coeff(l, mc)

                v += w3j * a * b * c

        if abs(v.imag) > 1e-4:
            import sys
            print(f"large spurious imaginary component (l = {l}: "
                  f"{v.real}, {v.imag}i)", file=sys.stderr)

        absv = abs(v)
        if absv == 0:
            return 0.0
        sign = 1.0 if v.real >= 0 else -1.0
        return absv**(1.0 / 3.0) / self.total_area * sign

    def _get_coeff(self, l, m):
        """Get C_lm coefficient, handling negative m via conjugate relation."""
        if m >= 0:
            return self._d[(l, m)]
        else:
            val = np.conj(self._d[(l, -m)])
            if (-m) % 2:
                val = -val
            return val


def calculate_sphmink(surface):
    """Calculate spherical Minkowski functionals for each label."""
    results = {}
    data = {}

    for k in range(surface.n_triangles()):
        label = surface.label_of_triangle(k)
        if label not in data:
            data[label] = SphericalMinkowskis()
        area = surface.area_of_triangle(k)
        n = surface.normal_vector_of_triangle(k)
        data[label].add_facet(n * area)

    for label, sm in data.items():
        r = _default_sphmink()
        for l in range(MAX_L + 1):
            r.result.ql[l] = sm.ql(l)
            r.result.wl[l] = sm.wl(l)
        results[label] = r

    return results
