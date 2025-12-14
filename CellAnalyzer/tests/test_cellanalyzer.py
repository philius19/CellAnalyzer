"""
Minimal tests for CellAnalyzer v2.0 API.
"""
import pytest
import numpy as np
from pathlib import Path

# Test data path
TEST_DATA_DIR = Path('/Users/philippkaintoch/Documents/Projects/02_Codebase/Z_Marching_Cubes/TestOutput')


class TestImports:
    """Test package imports."""

    def test_import_load_cell(self):
        from CellAnalyzer import load_cell
        assert callable(load_cell)

    def test_import_cellanalyzer(self):
        from CellAnalyzer import CellAnalyzer
        assert CellAnalyzer is not None

    def test_import_datatypes(self):
        from CellAnalyzer import MeshFrame, CellData, AnalysisResults
        assert MeshFrame is not None
        assert CellData is not None
        assert AnalysisResults is not None

    def test_import_visualization(self):
        from CellAnalyzer import basic_spatial_plot, publication_style
        assert callable(basic_spatial_plot)


@pytest.mark.skipif(not TEST_DATA_DIR.exists(), reason="Test data not available")
class TestCellAnalyzerLoading:
    """Test CellAnalyzer loading functionality."""

    def test_load_cell(self):
        from CellAnalyzer import load_cell
        cell = load_cell(TEST_DATA_DIR, channel=1)
        assert cell.n_frames > 0

    def test_frame_access(self):
        from CellAnalyzer import load_cell
        cell = load_cell(TEST_DATA_DIR, channel=1)

        # Access first available frame
        first_idx = cell.time_indices[0]
        frame = cell[first_idx]

        assert frame.n_vertices > 0
        assert frame.n_faces > 0
        assert len(frame.curvature) == frame.n_faces

    def test_iteration(self):
        from CellAnalyzer import load_cell
        cell = load_cell(TEST_DATA_DIR, channel=1)

        count = 0
        for time_idx, frame in cell:
            assert isinstance(time_idx, int)
            assert frame.n_vertices > 0
            count += 1

        assert count == cell.n_frames

    def test_contains(self):
        from CellAnalyzer import load_cell
        cell = load_cell(TEST_DATA_DIR, channel=1)

        first_idx = cell.time_indices[0]
        assert first_idx in cell
        assert -999 not in cell


@pytest.mark.skipif(not TEST_DATA_DIR.exists(), reason="Test data not available")
class TestStatistics:
    """Test statistics calculation."""

    def test_calculate_statistics(self):
        from CellAnalyzer import load_cell
        cell = load_cell(TEST_DATA_DIR, channel=1)

        first_idx = cell.time_indices[0]
        stats = cell.calculate_statistics(first_idx)

        # Mesh statistics
        assert stats.mesh_stats.n_vertices > 0
        assert stats.mesh_stats.n_faces > 0
        assert stats.mesh_stats.volume_um3 > 0
        assert stats.mesh_stats.surface_area_um2 > 0

        # Curvature statistics
        assert np.isfinite(stats.curvature_stats.mean)
        assert stats.curvature_stats.std >= 0

    def test_calculate_all_statistics(self):
        from CellAnalyzer import load_cell
        cell = load_cell(TEST_DATA_DIR, channel=1)

        all_stats = cell.calculate_all_statistics()

        assert len(all_stats) == cell.n_frames
        for t in cell.time_indices:
            assert t in all_stats


@pytest.mark.skipif(not TEST_DATA_DIR.exists(), reason="Test data not available")
class TestMetadata:
    """Test metadata loading."""

    def test_pixel_sizes(self):
        from CellAnalyzer import load_cell
        cell = load_cell(TEST_DATA_DIR, channel=1)

        assert cell.pixel_size_xy_um > 0
        assert cell.pixel_size_z_um > 0

    def test_metadata_access(self):
        from CellAnalyzer import load_cell
        cell = load_cell(TEST_DATA_DIR, channel=1)

        metadata = cell.metadata
        assert metadata.pixel_size_xy_nm > 0
        assert metadata.pixel_size_z_nm > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
