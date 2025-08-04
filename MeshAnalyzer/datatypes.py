"""
Data structures for mesh analysis results.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass(frozen=True)
class MeshStatistics:
    """Statistics for mesh geometry."""
    n_vertices: int
    n_faces: int
    n_edges: int
    volume_pixels3: float
    volume_um3: float
    surface_area_pixels2: float
    surface_area_um2: float
    is_watertight: bool
    euler_number: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for backwards compatibility."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class CurvatureStatistics:
    """Statistics for curvature distribution."""
    mean: float
    std: float
    sem: float
    median: float
    min: float
    max: float
    percentiles: Dict[int, float] = field(default_factory=dict)
    
    @classmethod
    def from_array(cls, curvature: np.ndarray) -> 'CurvatureStatistics':
        """Create from curvature array."""
        percentile_values = [1, 5, 25, 50, 75, 95, 99]
        return cls(
            mean=float(np.mean(curvature)),
            std=float(np.std(curvature)),
            sem=float(np.std(curvature) / np.sqrt(len(curvature))),
            median=float(np.median(curvature)),
            min=float(np.min(curvature)),
            max=float(np.max(curvature)),
            percentiles={p: float(np.percentile(curvature, p)) for p in percentile_values}
        )


@dataclass(frozen=True)
class QualityMetrics:
    """Mesh quality metrics."""
    mean_edge_length: float
    std_edge_length: float
    min_edge_length: float
    max_edge_length: float
    mean_face_area: float
    std_face_area: float
    aspect_ratio_mean: float
    aspect_ratio_std: float
    
    def get_warnings(self) -> list[str]:
        """Check for quality issues."""
        warnings = []
        if self.aspect_ratio_mean > 3.0:
            warnings.append(f"High aspect ratio: {self.aspect_ratio_mean:.2f}")
        if self.std_edge_length / self.mean_edge_length > 0.5:
            warnings.append("High edge length variation")
        return warnings


@dataclass
class AnalysisResults:
    """Container for all analysis results."""
    mesh_stats: Optional[MeshStatistics] = None
    curvature_stats: Optional[CurvatureStatistics] = None
    quality_metrics: Optional[QualityMetrics] = None
    
    def is_complete(self) -> bool:
        """Check if all analyses have been run."""
        return all([self.mesh_stats, self.curvature_stats, self.quality_metrics])
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.mesh_stats:
            return "No analysis performed yet."
        
        lines = [
            "=== Analysis Summary ===",
            f"Vertices: {self.mesh_stats.n_vertices:,}",
            f"Faces: {self.mesh_stats.n_faces:,}",
            f"Volume: {self.mesh_stats.volume_um3:.2f} μm³",
            f"Surface Area: {self.mesh_stats.surface_area_um2:.2f} μm²",
        ]
        
        if self.curvature_stats:
            lines.extend([
                f"\nCurvature: {self.curvature_stats.mean:.4f} ± {self.curvature_stats.std:.4f}",
                f"Range: [{self.curvature_stats.min:.4f}, {self.curvature_stats.max:.4f}]"
            ])
        
        if self.quality_metrics:
            warnings = self.quality_metrics.get_warnings()
            if warnings:
                lines.append("\nWarnings:")
                lines.extend(f"  - {w}" for w in warnings)
        
        return "\n".join(lines)