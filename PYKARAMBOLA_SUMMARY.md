# pykarambola Development Summary

A complete Python port of the C++ Karambola package, documenting all steps taken to create the Python implementation.

## Development Timeline

### 1. Initial Python Port (commit 4964706)
Created a complete Python reimplementation of the C++ Karambola library, porting all core functionality:
- **Minkowski functionals**: All 16 quantities (w000-w320, w102, w202, w103, w104)
- **Spherical Minkowski functionals**: Full computation pipeline
- **Tensor calculations**: Eigensystem decomposition for 3D analysis
- **File I/O**: Parsers for `.poly` and `.off` file formats; output writers matching C++ format
- **CLI interface**: Command-line tool replicating C++ behavior
- **Tests**: 36 pytest tests validating against known analytical box formulas
- **Dependencies**: Replaced GSL with NumPy/SciPy

### 2. Spherical Harmonics Fix (commit 365b752)
- Replaced SymPy's Wigner 3j symbols with a pure-Python Racah formula
- Removed SymPy dependency (scipy >= 1.14 removed `sph_harm`; adapted to use `sph_harm_y` with swapped argument order)

### 3. API Refinement (commits 0207a6c, 7b7ac6a)
- Added high-level `minkowski_functionals()` API for NumPy arrays
- Added volume-level `minkowski_functionals_from_label_image()` API for batch processing

### 4. File Format Expansion (commit 5a0d058)
- Added OBJ and GLB file parsers for broader model format support

### 5. Project Infrastructure (commits a28f50c, b4d57c2)
- Renamed package from `karambola_py` to `pykarambola`
- Added `pyproject.toml` for modern Python packaging
- Created demo notebook showcasing usage

### 6. Performance Optimization (commit 987b885)
- Vectorized core operations with NumPy for **5-10x speedup**

### 7. Parallel Processing (commit 4b1876d)
- Added demo notebooks for parallel processing using ProcessPoolExecutor and joblib

### 8. Cython Acceleration (commit 3ea6e6d)
- Vectorized `check_surface()` loops in surface.py
- Added optional Cython extension (`_accel.pyx`) for:
  - Neighbour table computation
  - Vertex-triangle CSR lookup
  - Pure-Python fallback when not compiled

### 9. Dependencies & Gitignore (commits b2d48bb, 2a907b1, 97552e3)
- Added proper `.gitignore` entries for build artifacts and caches

### 10. Recent Merges (commits 2edabba, 2416bd4, 80581de)
- Merged updates from `cython` branch
- Integrated upstream `morphometry:master` changes
- Kept in sync with main `master` branch

## Outcome

A fully functional, performant Python replacement for the C++ library with:
- Modern Python packaging and distribution
- Comprehensive pytest test suite
- Optional performance accelerations via Cython
- Vectorized NumPy operations (5-10x speedup)
- Support for multiple file formats (.poly, .off, .obj, .glb)
- Parallel processing capabilities
- Full feature parity with C++ implementation
