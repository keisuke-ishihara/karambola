# pykarambola: Python Port of Karambola

## High-Level Summary

**pykarambola** is a complete Python reimplementation of the C++ Karambola library for computing Minkowski functionals and tensor quantities on 3D meshes.

### Core Implementation
- Full Python port of all 16 Minkowski functionals, spherical Minkowski functionals, and tensor calculations
- Replaced C++ GSL dependency with NumPy/SciPy
- Comprehensive pytest test suite with 36+ tests validating against analytical formulas
- CLI interface matching C++ behavior
- File I/O for `.poly`, `.off`, `.obj`, and `.glb` formats

### Mathematical Refinements
- Replaced SymPy Wigner 3j symbols with pure-Python Racah formula
- Removed SymPy dependency; adapted to scipy >= 1.14 spherical harmonic API changes

### API Design
- High-level `minkowski_functionals()` API for direct NumPy array processing
- Batch processing API `minkowski_functionals_from_label_image()` for volume-level computation
- Clean, user-friendly interfaces abstracting algorithmic complexity

### Performance Optimization
- **5-10x speedup** through NumPy vectorization of core loops
- Optional Cython extension (`_accel.pyx`) for hot-path operations (neighbor table, CSR lookup)
- Pure-Python fallback ensuring portability

### Project Infrastructure
- Modern Python packaging with `pyproject.toml`
- Proper build configuration with optional Cython compilation
- Demo notebooks for basic usage and parallel processing
- Documentation and examples
