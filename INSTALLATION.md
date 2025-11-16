# MeshAnalyzer Installation Guide

## Quick Start

### Installation

From the Scripts directory, install MeshAnalyzer in your conda environment:

```bash
cd /Users/philippkaintoch/Documents/Projects/02_Bleb3D/Pipeline/Scripts
conda activate bleb3d
pip install -e .
```

The `-e` flag installs in **editable mode**, meaning:
- Changes to the source code are immediately available (no reinstall needed)
- The package is linked to your development directory
- Perfect for active development

### Verify Installation

```bash
python -c "from MeshAnalyzer import MeshAnalyzer; print('✓ MeshAnalyzer installed successfully!')"
```

---

## Usage

### Basic Import

Now you can import MeshAnalyzer from **anywhere** in your system:

```python
from MeshAnalyzer import MeshAnalyzer

# Create analyzer instance
analyzer = MeshAnalyzer(
    surface_path="path/to/surface_1_1.mat",
    curvature_path="path/to/meanCurvature_1_1.mat",
    pixel_size_xy=0.1661,  # micrometers
    pixel_size_z=0.5       # micrometers
)

# Load data
analyzer.load_data()

# Calculate statistics
results = analyzer.calculate_statistics()
print(results.summary())
```

### Advanced Usage with Dataclasses

```python
from MeshAnalyzer import MeshAnalyzer, AnalysisResults

analyzer = MeshAnalyzer(surface_path, curvature_path)
analyzer.load_data()
results = analyzer.calculate_statistics()

# Type-safe access to results
print(f"Volume: {results.mesh_stats.volume_um3:.2f} μm³")
print(f"Surface area: {results.mesh_stats.surface_area_um2:.2f} μm²")
print(f"Mean curvature: {results.curvature_stats.mean:.4f}")

# Check mesh quality
warnings = results.quality_metrics.get_warnings()
if warnings:
    print("Mesh quality warnings:")
    for warning in warnings:
        print(f"  - {warning}")
```

### Import Available Functions

```python
from MeshAnalyzer import (
    MeshAnalyzer,
    AnalysisResults,
    MeshStatistics,
    CurvatureStatistics,
    QualityMetrics,
    plot_curvature_distribution,
    basic_spatial_plot,
    load_surface_data,
    load_curvature_data,
    convert_pixels_to_um
)
```

---

## Development Workflow

### Running Tests from Example Directory

The import issue is now **solved**. You can run test scripts from anywhere:

```bash
# From Example directory
cd /Users/.../Scripts/Example
python test.py

# Or with absolute path
python /Users/.../Scripts/Example/test.py

# Or as a module
cd /Users/.../Scripts
python -m Example.test
```

### Making Changes

Since the package is installed in editable mode:

1. Edit source files in `MeshAnalyzer/`
2. Changes are immediately available
3. **No need to reinstall**
4. Just restart your Python session or kernel

```python
# In a Jupyter notebook, reload modules after changes:
import importlib
import MeshAnalyzer
importlib.reload(MeshAnalyzer)
```

---

## Package Structure

After installation, your package structure is:

```
Scripts/
├── setup.py              # Package configuration
├── pyproject.toml        # Modern Python packaging config
├── MANIFEST.in           # Include/exclude files for distribution
├── MeshAnalyzer/         # Source package (linked via -e)
│   ├── __init__.py       # Exports main classes
│   ├── analyzer.py       # MeshAnalyzer class
│   ├── datatypes.py      # Type-safe dataclasses
│   ├── io.py             # MATLAB file I/O
│   ├── utils.py          # Utility functions
│   └── visualization.py  # Plotting functions
├── Example/              # Can now import MeshAnalyzer!
│   └── test.py
├── Notebooks/            # Can now import MeshAnalyzer!
└── Experimental/         # Can now import MeshAnalyzer!
```

---

## Uninstalling

If needed, uninstall with:

```bash
pip uninstall MeshAnalyzer
```

---

## Troubleshooting

### Import Error After Installation

If you still get `ModuleNotFoundError`:

1. **Check correct environment:**
   ```bash
   conda activate bleb3d
   which python  # Should show bleb3d environment
   ```

2. **Verify installation:**
   ```bash
   pip list | grep MeshAnalyzer
   # Should show: meshanalyzer 1.0.0 /path/to/Scripts
   ```

3. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   # Should include site-packages from bleb3d environment
   ```

### Changes Not Reflected

If changes to source files don't appear:

1. **Restart Python/Jupyter kernel**
2. **Verify editable install:**
   ```bash
   pip show MeshAnalyzer
   # Location should point to your Scripts directory
   ```

3. **Force reload in Jupyter:**
   ```python
   %load_ext autoreload
   %autoreload 2
   ```

---

## Advanced: Distribution

### Create Distribution Package

To share with others:

```bash
# Install build tools
pip install build

# Create distribution
python -m build

# This creates:
# dist/meshanalyzer-1.0.0.tar.gz
# dist/meshanalyzer-1.0.0-py3-none-any.whl
```

### Install from Wheel

Others can install from the wheel file:

```bash
pip install meshanalyzer-1.0.0-py3-none-any.whl
```

### Upload to PyPI (Optional)

```bash
# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*
```

Then anyone can install with:
```bash
pip install MeshAnalyzer
```

---

## Dependencies

The package automatically installs these dependencies:

- `numpy>=1.20.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.4.0` - Plotting
- `vedo>=2023.4.0` - 3D mesh operations
- `mat73>=0.59` - MATLAB v7.3 file reading
- `h5py>=3.0.0` - HDF5 file support

### Optional Development Dependencies

Install with:
```bash
pip install -e ".[dev]"
```

Includes:
- pytest - Testing framework
- black - Code formatter
- flake8 - Linter
- jupyter - Notebook support

---

## Summary

**Before:** Import only worked when running from specific directories

**After:** Import works from anywhere because MeshAnalyzer is installed in your Python environment

**Key Command:**
```bash
pip install -e .
```

This solved the import issue permanently while maintaining development flexibility!
