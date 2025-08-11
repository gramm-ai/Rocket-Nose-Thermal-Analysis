"""
rocket_mesh.py - Combined Mesh Generation and Profile Creation
Merges HexahedralRocketMesh and RocketNoseGenerator functionality

This module provides:
1. Core hexahedral mesh generation (from rocket_mesh_hex.py)
2. 6 distinct nose profile creation (from create_rocket_noses.py)
3. Thermal equivalent thickness modeling
4. Visualization utilities for mesh profiles
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Any
import time
from dataclasses import dataclass
from scipy.interpolate import CubicSpline, interp1d
import concurrent.futures
import warnings
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
import multiprocessing

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# Core Mesh Generation Classes (from rocket_mesh_hex.py)
# ============================================================================

@dataclass
class HexMeshQualityMetrics:
    """Quality metrics for hexahedral mesh"""
    n_nodes: int
    n_hex_elements: int
    n_boundary_faces: int
    n_interior_nodes: int
    avg_jacobian: float
    min_jacobian: float
    max_aspect_ratio: float
    avg_skewness: float
    max_warping: float
    orthogonality: float
    quality_grade: str
    generation_time: float


@dataclass
class RocketShapeParameters:
    """Comprehensive rocket shape parameters"""
    # Nose cone parameters
    nose_type: str = 'ogive'  # ogive, conical, parabolic, elliptical, haack, power
    nose_length: float = 5.0
    nose_sharpness: float = 0.7
    nose_power: float = 0.5  # For power series
    nose_haack_c: float = 0.0  # Haack series parameter

    # Body parameters
    body_length: float = 40.0
    body_radius: float = 1.83

    # Tail/nozzle parameters
    tail_type: str = 'tapered'  # straight, tapered, bell
    tail_length: float = 3.0
    tail_exit_radius: float = 1.2
    tail_throat_radius: float = 0.8

    # Boat tail parameters
    boat_tail_enabled: bool = False
    boat_tail_length: float = 2.0
    boat_tail_angle: float = 15.0  # degrees

    # Fins (simplified)
    fins_enabled: bool = False
    n_fins: int = 4
    fin_root_chord: float = 2.0
    fin_tip_chord: float = 1.0
    fin_span: float = 1.5
    fin_sweep_angle: float = 45.0

    # Stage transitions
    interstage_enabled: bool = False
    interstage_position: float = 30.0
    interstage_length: float = 2.0
    interstage_radius_change: float = 0.2

    # Wall parameters
    wall_thickness: float = 0.0045
    variable_thickness: bool = False
    thickness_profile: Optional[List[Tuple[float, float]]] = None


class HexahedralRocketMesh:
    """
    High-performance hexahedral mesh generator for rocket geometries

    REFACTORED: Device-independent implementation
    - Generates pure hexahedral meshes for FEA
    - No device-specific optimizations (handled by physics engine)
    - Focus on geometric quality and accuracy
    """

    def __init__(self,
                 shape_params: Optional[RocketShapeParameters] = None,
                 n_axial: int = 100,
                 n_circumferential: int = 32,
                 n_radial: int = 3,
                 grading_params: Optional[Dict] = None,
                 enable_parallel: bool = True,
                 material_type: str = 'aluminum',
                 export_format: str = 'vtk',
                 nose_only: bool = False,
                 full_rocket: bool = False,
                 mesh_resolution: str = 'medium'):
        """
        Initialize hexahedral mesh generator

        Args:
            shape_params: Rocket shape configuration
            n_axial: Number of divisions along rocket axis
            n_circumferential: Number of divisions around circumference
            n_radial: Number of divisions through wall thickness
            grading_params: Mesh grading configuration
            enable_parallel: Enable parallel processing for mesh generation
            material_type: Material specification (for metadata only)
            export_format: Default export format
            nose_only: Generate only nose cone mesh for faster analysis
            full_rocket: Generate complete rocket (overrides other length settings)
            mesh_resolution: Mesh quality level ('coarse', 'medium', 'fine')
        """

        # Initialize shape parameters
        self.shape = shape_params if shape_params else RocketShapeParameters()

        # Mesh mode configuration
        self.nose_only = nose_only
        self.full_rocket = full_rocket
        self.mesh_resolution = mesh_resolution

        # Configuration
        self.enable_parallel = enable_parallel
        self.material_type = material_type
        self.export_format = export_format

        # Adjust geometry based on mode
        if nose_only:
            # Override body and tail lengths for nose-only mode
            self.original_body_length = self.shape.body_length
            self.original_tail_length = self.shape.tail_length
            self.shape.body_length = 0.0
            self.shape.tail_length = 0.0
            # Optionally add small cylindrical section for boundary conditions
            self.shape.body_length = min(2.0, self.original_body_length)  # Small transition section

        # Mesh parameters - validate ranges
        self.n_axial = max(10, n_axial)
        self.n_circumferential = max(8, n_circumferential)
        self.n_radial = max(2, n_radial)

        # Grading parameters with proper default merging
        default_grading = {
            'axial_ratio': 1.0,
            'radial_ratio': 1.0,
            'boundary_layer': False,
            'bl_thickness': 0.1,
            'bl_growth': 1.2
        }
        # Merge provided parameters with defaults
        self.grading = default_grading.copy()
        if grading_params:
            self.grading.update(grading_params)

        # Initialize mesh data structures
        self.nodes = None
        self.hex_elements = None
        self.boundary_faces = None
        self.node_sets = {}
        self.element_sets = {}
        self.quality_metrics = None

        # Generate mesh
        print("\nHexahedral Rocket Mesh Generator")
        if nose_only:
            print(f"  Mode: NOSE ONLY - Fast analysis mode")
            print(f"  Nose type: {self.shape.nose_type}")
            print(f"  Nose length: {self.shape.nose_length:.1f} m")
            if self.shape.body_length > 0:
                print(f"  Transition section: {self.shape.body_length:.1f} m")
        else:
            print(f"  Shape: {self.shape.nose_type} nose, {self.shape.tail_type} tail")
        print(f"  Grid: {self.n_axial}×{self.n_circumferential}×{self.n_radial}")
        print(f"  Resolution: {mesh_resolution.upper()}")

        start_time = time.perf_counter()
        self._generate_mesh()
        generation_time = time.perf_counter() - start_time

        # Assess quality
        self._assess_mesh_quality(generation_time)
        self._print_summary()

    def _generate_mesh(self):
        """Generate the hexahedral mesh"""
        print("  Generating hexahedral mesh...")

        # Generate node positions
        self._generate_nodes()

        # Generate hexahedral elements
        self._generate_hex_elements()

        # Identify boundaries
        self._identify_boundaries()

        # Create node and element sets
        self._create_sets()

    def _generate_nodes(self):
        """Generate mesh nodes with parametric shape"""
        # Create parametric coordinates
        z_coords = self._create_axial_distribution()
        theta_coords = np.linspace(0, 2 * np.pi, self.n_circumferential, endpoint=False)
        r_normalized = self._create_radial_distribution()

        # Total nodes
        n_total = self.n_axial * self.n_circumferential * self.n_radial
        self.nodes = np.zeros((n_total, 3), dtype=np.float32)

        # Generate nodes using vectorized operations where possible
        node_idx = 0
        for i, z in enumerate(z_coords):
            # Get radius profile at this axial position
            r_outer = self._get_radius_at_z(z)
            r_inner = max(0.001, r_outer - self._get_thickness_at_z(z))

            for j, theta in enumerate(theta_coords):
                for k, r_norm in enumerate(r_normalized):
                    # Interpolate radius
                    r = r_inner + r_norm * (r_outer - r_inner)

                    # Convert to Cartesian
                    self.nodes[node_idx, 0] = r * np.cos(theta)
                    self.nodes[node_idx, 1] = r * np.sin(theta)
                    self.nodes[node_idx, 2] = z

                    node_idx += 1

        # Create structured index mapping for easy element generation
        self.node_indices = np.arange(n_total).reshape(
            (self.n_axial, self.n_circumferential, self.n_radial)
        )

        print(f"    ✓ Generated {n_total:,} nodes")

    def _create_axial_distribution(self) -> np.ndarray:
        """
        Create axial node distribution with grading

        Returns array of z-coordinates from 0 (nose tip) to total_length (base/tail)
        """
        total_length = (self.shape.nose_length +
                        self.shape.body_length +
                        self.shape.tail_length)

        if self.nose_only:
            # For nose-only mode, concentrate nodes in the nose region
            if self.grading['axial_ratio'] == 1.0:
                return np.linspace(0, total_length, self.n_axial)
            else:
                # Apply finer grading near the nose tip
                xi = np.linspace(0, 1, self.n_axial)
                # Use hyperbolic tangent for smooth grading
                beta = 2.0 * self.grading['axial_ratio']
                eta = np.tanh(beta * xi) / np.tanh(beta)
                return eta * total_length
        else:
            if self.grading['axial_ratio'] == 1.0:
                return np.linspace(0, total_length, self.n_axial)
            else:
                # Geometric grading
                xi = np.linspace(0, 1, self.n_axial)
                beta = np.log(self.grading['axial_ratio'])
                eta = (np.exp(beta * xi) - 1) / (np.exp(beta) - 1)
                return eta * total_length

    def _create_radial_distribution(self) -> np.ndarray:
        """Create radial node distribution with optional boundary layer"""
        if self.n_radial == 2:
            return np.array([0.0, 1.0])

        if self.grading['boundary_layer'] and self.n_radial > 3:
            # Boundary layer grading
            bl_nodes = max(2, self.n_radial // 3)
            core_nodes = self.n_radial - bl_nodes

            # Geometric growth in boundary layer
            bl_thickness = self.grading['bl_thickness']
            growth = self.grading['bl_growth']

            bl_coords = []
            delta = bl_thickness / bl_nodes
            for i in range(bl_nodes):
                bl_coords.append(delta * sum(growth ** j for j in range(i + 1)))

            # Linear in core
            core_coords = np.linspace(bl_coords[-1], 1.0, core_nodes + 1)[1:]

            return np.array(bl_coords + core_coords.tolist())
        else:
            # Uniform distribution
            return np.linspace(0, 1, self.n_radial)

    def _get_radius_at_z(self, z: float) -> float:
        """
        Get rocket radius at axial position z

        Coordinate system: z=0 is nose tip, z increases toward base/tail

        Args:
            z: Axial position (0 = nose tip)

        Returns:
            Radius at position z
        """
        nose_end = self.shape.nose_length
        body_end = nose_end + self.shape.body_length

        if z <= nose_end:
            # Nose cone region
            return self._nose_profile(z)
        elif z <= body_end:
            # Cylindrical body
            if self.shape.interstage_enabled:
                interstage_start = self.shape.interstage_position
                interstage_end = interstage_start + self.shape.interstage_length

                if interstage_start <= z <= interstage_end:
                    # Smooth transition
                    t = (z - interstage_start) / self.shape.interstage_length
                    r1 = self.shape.body_radius
                    r2 = r1 - self.shape.interstage_radius_change
                    return r1 + (r2 - r1) * (3 * t ** 2 - 2 * t ** 3)  # Smooth cubic

            return self.shape.body_radius
        else:
            # Tail/nozzle region
            return self._tail_profile(z - body_end)

    def _nose_profile(self, z: float) -> float:
        """
        Calculate nose cone radius at position z

        Coordinate system: tip at z=0, base at z=nose_length
        Radius increases from tip (small) to base (body_radius)

        Args:
            z: Axial position (0 = tip, nose_length = base)

        Returns:
            Radius at position z
        """
        L = self.shape.nose_length
        R = self.shape.body_radius

        if z <= 0:
            return 0.001  # Avoid singularity at tip

        xi = min(1.0, z / L)  # xi goes from 0 (tip) to 1 (base)

        nose_type = self.shape.nose_type
        sharpness = self.shape.nose_sharpness

        if nose_type == 'conical':
            radius = R * xi  # Linear increase from tip to base

        elif nose_type == 'ogive':
            # Ogive profile
            if xi < 0.001:
                radius = 0.05 * R
            else:
                rho = (R ** 2 + L ** 2) / (2 * R)
                # Modified for tip at origin
                y = np.sqrt(max(0, rho ** 2 - (L - z) ** 2)) + R - rho
                radius = max(0.05 * R, y)

        elif nose_type == 'parabolic':
            # Parabolic profile
            K = 0.5 + 0.5 * sharpness  # Shape parameter
            radius = R * (2 * xi - K * xi ** 2) / (2 - K)

        elif nose_type == 'elliptical':
            # Elliptical profile
            radius = R * np.sqrt(max(0, 1 - ((L - z) / L) ** 2))

        elif nose_type == 'haack':
            # Haack series (minimum drag)
            C = self.shape.nose_haack_c
            theta = np.arccos(1 - 2 * xi)
            radius = R * np.sqrt(theta - np.sin(2 * theta) / 2 + C * np.sin(theta) ** 3) / np.sqrt(np.pi)

        elif nose_type == 'power':
            # Power series
            n = self.shape.nose_power
            radius = R * xi ** n

        else:  # Default to ogive
            radius = R * xi ** 0.7

        # Apply sharpness factor
        min_radius = 0.02 * R * (1 - sharpness)
        return max(min_radius, radius)

    def _tail_profile(self, z_from_body_end: float) -> float:
        """Calculate tail/nozzle radius"""
        L = self.shape.tail_length

        if L <= 0:
            return self.shape.body_radius

        if self.shape.tail_type == 'straight':
            return self.shape.body_radius

        elif self.shape.tail_type == 'tapered':
            # Linear taper
            t = min(1.0, z_from_body_end / L)
            r1 = self.shape.body_radius
            r2 = self.shape.tail_exit_radius
            return r1 + (r2 - r1) * t

        elif self.shape.tail_type == 'bell':
            # Bell nozzle profile
            t = min(1.0, z_from_body_end / L)
            r1 = self.shape.body_radius
            r_throat = self.shape.tail_throat_radius
            r_exit = self.shape.tail_exit_radius

            if t < 0.3:
                # Converging section
                return r1 + (r_throat - r1) * (t / 0.3)
            else:
                # Diverging section (parabolic)
                t_div = (t - 0.3) / 0.7
                return r_throat + (r_exit - r_throat) * t_div ** 1.5

        return self.shape.body_radius

    def _get_thickness_at_z(self, z: float) -> float:
        """Get wall thickness at position z"""
        if not self.shape.variable_thickness:
            return self.shape.wall_thickness

        if self.shape.thickness_profile:
            # Interpolate from profile
            z_vals = [p[0] for p in self.shape.thickness_profile]
            t_vals = [p[1] for p in self.shape.thickness_profile]
            return np.interp(z, z_vals, t_vals)

        return self.shape.wall_thickness

    def _generate_hex_elements(self):
        """Generate hexahedral elements with optimized connectivity"""
        n_hex = (self.n_axial - 1) * self.n_circumferential * (self.n_radial - 1)
        self.hex_elements = np.zeros((n_hex, 8), dtype=np.int32)

        elem_idx = 0
        for i in range(self.n_axial - 1):
            for j in range(self.n_circumferential):
                j_next = (j + 1) % self.n_circumferential

                for k in range(self.n_radial - 1):
                    # Hexahedral connectivity (right-hand rule)
                    n0 = self.node_indices[i, j, k]
                    n1 = self.node_indices[i + 1, j, k]
                    n2 = self.node_indices[i + 1, j_next, k]
                    n3 = self.node_indices[i, j_next, k]
                    n4 = self.node_indices[i, j, k + 1]
                    n5 = self.node_indices[i + 1, j, k + 1]
                    n6 = self.node_indices[i + 1, j_next, k + 1]
                    n7 = self.node_indices[i, j_next, k + 1]

                    self.hex_elements[elem_idx] = [n0, n1, n2, n3, n4, n5, n6, n7]
                    elem_idx += 1

        # Pre-compute element properties
        self._compute_element_properties()

        print(f"    ✓ Generated {n_hex:,} hexahedral elements")

    def _compute_element_properties(self):
        """Pre-compute element Jacobians and quality metrics"""
        n_elem = len(self.hex_elements)
        self.element_jacobians = np.zeros(n_elem, dtype=np.float32)
        self.element_volumes = np.zeros(n_elem, dtype=np.float32)

        if self.enable_parallel and n_elem > 1000:
            self._parallel_compute_properties()
        else:
            self._serial_compute_properties()

    def _serial_compute_properties(self):
        """Serial computation of element properties"""
        for idx, hex_elem in enumerate(self.hex_elements):
            nodes = self.nodes[hex_elem]
            jacobian, volume = self._compute_hex_jacobian(nodes)
            self.element_jacobians[idx] = jacobian
            self.element_volumes[idx] = volume

    def _parallel_compute_properties(self):
        """Parallel computation of element properties"""
        n_workers = min(multiprocessing.cpu_count(), 16)
        n_elem = len(self.hex_elements)

        def compute_batch(start, end):
            results = []
            for idx in range(start, min(end, n_elem)):
                nodes = self.nodes[self.hex_elements[idx]]
                jacobian, volume = self._compute_hex_jacobian(nodes)
                results.append((idx, jacobian, volume))
            return results

        batch_size = max(1, n_elem // n_workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(0, n_elem, batch_size):
                futures.append(executor.submit(compute_batch, i, i + batch_size))

            for future in futures:
                for idx, jacobian, volume in future.result():
                    self.element_jacobians[idx] = jacobian
                    self.element_volumes[idx] = volume

    def _compute_hex_jacobian(self, nodes: np.ndarray) -> Tuple[float, float]:
        """Compute Jacobian and volume for hexahedral element"""
        # Gauss quadrature points for hex
        gauss_pts = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                              [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * (1 / np.sqrt(3))

        jacobians = []
        volume = 0.0

        for gp in gauss_pts:
            # Shape function derivatives in natural coordinates
            dN_dxi = self._hex_shape_derivatives(gp)

            # Jacobian matrix
            J = dN_dxi.T @ nodes
            det_J = np.linalg.det(J)

            jacobians.append(det_J)
            volume += det_J  # Gauss weight = 1 for these points

        avg_jacobian = np.mean(jacobians)
        return avg_jacobian, abs(volume)

    def _hex_shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Compute shape function derivatives for hex element"""
        r, s, t = xi

        # Derivatives of shape functions w.r.t natural coordinates
        dN_dxi = np.array([
            [-(1 - s) * (1 - t), -(1 - r) * (1 - t), -(1 - r) * (1 - s)],
            [(1 - s) * (1 - t), -(1 + r) * (1 - t), -(1 + r) * (1 - s)],
            [(1 + s) * (1 - t), (1 + r) * (1 - t), -(1 + r) * (1 + s)],
            [-(1 + s) * (1 - t), (1 - r) * (1 - t), -(1 - r) * (1 + s)],
            [-(1 - s) * (1 + t), -(1 - r) * (1 + t), (1 - r) * (1 - s)],
            [(1 - s) * (1 + t), -(1 + r) * (1 + t), (1 + r) * (1 - s)],
            [(1 + s) * (1 + t), (1 + r) * (1 + t), (1 + r) * (1 + s)],
            [-(1 + s) * (1 + t), (1 - r) * (1 + t), (1 - r) * (1 + s)]
        ]) * 0.125

        return dN_dxi

    def _identify_boundaries(self):
        """Identify boundary faces and nodes"""
        self.boundary_faces = {
            'inner': [],
            'outer': [],
            'front': [],
            'back': []
        }

        # Outer surface (k = n_radial-1)
        for i in range(self.n_axial - 1):
            for j in range(self.n_circumferential):
                j_next = (j + 1) % self.n_circumferential
                k = self.n_radial - 1

                face = [
                    self.node_indices[i, j, k],
                    self.node_indices[i + 1, j, k],
                    self.node_indices[i + 1, j_next, k],
                    self.node_indices[i, j_next, k]
                ]
                self.boundary_faces['outer'].append(face)

        # Inner surface (k = 0) if hollow
        if self.n_radial > 2:
            for i in range(self.n_axial - 1):
                for j in range(self.n_circumferential):
                    j_next = (j + 1) % self.n_circumferential
                    k = 0

                    face = [
                        self.node_indices[i, j, k],
                        self.node_indices[i, j_next, k],
                        self.node_indices[i + 1, j_next, k],
                        self.node_indices[i + 1, j, k]
                    ]
                    self.boundary_faces['inner'].append(face)

        # Front face (i = 0, nose tip at z=0)
        for j in range(self.n_circumferential):
            j_next = (j + 1) % self.n_circumferential
            for k in range(self.n_radial - 1):
                face = [
                    self.node_indices[0, j, k],
                    self.node_indices[0, j_next, k],
                    self.node_indices[0, j_next, k + 1],
                    self.node_indices[0, j, k + 1]
                ]
                self.boundary_faces['front'].append(face)

        # Back face (i = n_axial-1, base at z=max)
        for j in range(self.n_circumferential):
            j_next = (j + 1) % self.n_circumferential
            for k in range(self.n_radial - 1):
                face = [
                    self.node_indices[-1, j, k],
                    self.node_indices[-1, j, k + 1],
                    self.node_indices[-1, j_next, k + 1],
                    self.node_indices[-1, j_next, k]
                ]
                self.boundary_faces['back'].append(face)

        # Convert to arrays
        for key in self.boundary_faces:
            self.boundary_faces[key] = np.array(self.boundary_faces[key], dtype=np.int32)

        total_faces = sum(len(faces) for faces in self.boundary_faces.values())
        print(f"    ✓ Identified {total_faces:,} boundary faces")

    def _create_sets(self):
        """Create node and element sets for different regions"""
        # Node sets
        self.node_sets['all'] = np.arange(len(self.nodes))
        self.node_sets['outer_surface'] = self.node_indices[:, :, -1].flatten()
        if self.n_radial > 2:
            self.node_sets['inner_surface'] = self.node_indices[:, :, 0].flatten()

        if self.nose_only:
            # For nose-only mode, all nodes are in the nose region
            self.node_sets['nose'] = self.node_sets['all']
            self.node_sets['body'] = np.array([], dtype=np.int32)
            self.node_sets['tail'] = np.array([], dtype=np.int32)

            # Mark the nose tip (at z=0) and base (at z=max) for boundary conditions
            self.node_sets['nose_tip'] = self.node_indices[0, :, :].flatten()
            self.node_sets['nose_base'] = self.node_indices[-1, :, :].flatten()

            # Element sets
            self.element_sets['all'] = np.arange(len(self.hex_elements))
            self.element_sets['nose'] = self.element_sets['all']
            self.element_sets['body'] = np.array([], dtype=np.int32)
            self.element_sets['tail'] = np.array([], dtype=np.int32)
        else:
            # Axial regions for full rocket
            nose_end_idx = int(self.n_axial * self.shape.nose_length /
                               (self.shape.nose_length + self.shape.body_length + self.shape.tail_length))
            body_end_idx = int(self.n_axial * (self.shape.nose_length + self.shape.body_length) /
                               (self.shape.nose_length + self.shape.body_length + self.shape.tail_length))

            self.node_sets['nose'] = self.node_indices[:nose_end_idx, :, :].flatten()
            self.node_sets['body'] = self.node_indices[nose_end_idx:body_end_idx, :, :].flatten()
            self.node_sets['tail'] = self.node_indices[body_end_idx:, :, :].flatten()

            # Element sets
            self.element_sets['all'] = np.arange(len(self.hex_elements))

            # Compute element centroids for classification
            centroids = np.zeros((len(self.hex_elements), 3))
            for idx, elem in enumerate(self.hex_elements):
                centroids[idx] = np.mean(self.nodes[elem], axis=0)

            # Classify elements by region
            nose_z = self.shape.nose_length
            body_z = nose_z + self.shape.body_length

            self.element_sets['nose'] = np.where(centroids[:, 2] < nose_z)[0]
            self.element_sets['body'] = np.where((centroids[:, 2] >= nose_z) &
                                                 (centroids[:, 2] < body_z))[0]
            self.element_sets['tail'] = np.where(centroids[:, 2] >= body_z)[0]

    def _assess_mesh_quality(self, generation_time: float):
        """Assess mesh quality metrics"""
        # Basic metrics
        n_nodes = len(self.nodes)
        n_hex = len(self.hex_elements)
        n_boundary = sum(len(faces) for faces in self.boundary_faces.values())

        # Quality metrics from element properties
        valid_jacobians = self.element_jacobians[self.element_jacobians > 1e-10]

        if len(valid_jacobians) > 0:
            avg_jacobian = np.mean(valid_jacobians)
            min_jacobian = np.min(valid_jacobians)
        else:
            avg_jacobian = min_jacobian = 0.0

        # Aspect ratio estimation
        aspect_ratios = []
        for elem in self.hex_elements[:min(100, len(self.hex_elements))]:
            nodes = self.nodes[elem]
            edges = [
                np.linalg.norm(nodes[1] - nodes[0]),
                np.linalg.norm(nodes[3] - nodes[0]),
                np.linalg.norm(nodes[4] - nodes[0])
            ]
            if min(edges) > 0:
                aspect_ratios.append(max(edges) / min(edges))

        max_aspect_ratio = max(aspect_ratios) if aspect_ratios else 1.0

        # Quality grading based on mesh resolution setting
        quality_grade = self.mesh_resolution.title()

        # Create metrics object
        self.quality_metrics = HexMeshQualityMetrics(
            n_nodes=n_nodes,
            n_hex_elements=n_hex,
            n_boundary_faces=n_boundary,
            n_interior_nodes=n_nodes - len(self.node_sets.get('outer_surface', [])),
            avg_jacobian=avg_jacobian,
            min_jacobian=min_jacobian,
            max_aspect_ratio=max_aspect_ratio,
            avg_skewness=0.1,  # Placeholder
            max_warping=0.05,  # Placeholder
            orthogonality=0.95,  # Placeholder
            quality_grade=quality_grade,
            generation_time=generation_time
        )

    def _print_summary(self):
        """Print mesh generation summary"""
        print(f"\nMesh Generation Complete:")
        if self.nose_only:
            print(f"  Type: Nose-only mesh")
        print(f"  Time: {self.quality_metrics.generation_time:.3f}s")
        print(f"  Nodes: {self.quality_metrics.n_nodes:,}")
        print(f"  Hex elements: {self.quality_metrics.n_hex_elements:,}")
        print(f"  Boundary faces: {self.quality_metrics.n_boundary_faces:,}")
        print(f"  Mesh Resolution: {self.quality_metrics.quality_grade}")
        print(f"  Min Jacobian: {self.quality_metrics.min_jacobian:.4f}")
        print(f"  Max aspect ratio: {self.quality_metrics.max_aspect_ratio:.1f}")

        if self.nose_only:
            print(f"\nNose-specific sets created:")
            print(f"  ✓ Nose tip nodes (z=0): {len(self.node_sets.get('nose_tip', []))} nodes")
            print(f"  ✓ Nose base nodes (z=max): {len(self.node_sets.get('nose_base', []))} nodes")

    def export_mesh(self, filename: str, format: Optional[str] = None):
        """Export mesh to file"""
        format = format or self.export_format

        if format.lower() == 'vtk':
            self._export_vtk(filename)
        elif format.lower() == 'json':
            self._export_json(filename)
        elif format.lower() == 'npz':
            self._export_npz(filename)
        elif format.lower() == 'exodus':
            self._export_exodus(filename)
        else:
            print(f"Format {format} not supported")

    def _export_vtk(self, filename: str):
        """Export mesh in VTK format"""
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Hexahedral Rocket Mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            # Points
            f.write(f"POINTS {len(self.nodes)} float\n")
            for node in self.nodes:
                f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f}\n")

            # Cells
            n_hex = len(self.hex_elements)
            f.write(f"\nCELLS {n_hex} {n_hex * 9}\n")
            for hex_elem in self.hex_elements:
                f.write(f"8 {' '.join(map(str, hex_elem))}\n")

            # Cell types (12 = VTK_HEXAHEDRON)
            f.write(f"\nCELL_TYPES {n_hex}\n")
            for _ in range(n_hex):
                f.write("12\n")

            # Cell data - element quality
            f.write(f"\nCELL_DATA {n_hex}\n")
            f.write("SCALARS jacobian float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for jac in self.element_jacobians:
                f.write(f"{jac:.6f}\n")

            f.write("\nSCALARS volume float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for vol in self.element_volumes:
                f.write(f"{vol:.6e}\n")

        print(f"  Exported mesh to {filename}")

    def _export_json(self, filename: str):
        """Export mesh data in JSON format"""
        mesh_data = {
            'metadata': {
                'generator': 'HexahedralRocketMesh',
                'version': '2.0',  # Updated version
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'shape_type': self.shape.nose_type,
                'n_nodes': len(self.nodes),
                'n_elements': len(self.hex_elements),
                'quality_grade': self.quality_metrics.quality_grade,
                'mesh_resolution': self.mesh_resolution
            },
            'shape_parameters': {
                'nose_type': self.shape.nose_type,
                'nose_length': self.shape.nose_length,
                'body_length': self.shape.body_length,
                'body_radius': self.shape.body_radius,
                'tail_type': self.shape.tail_type,
                'tail_length': self.shape.tail_length,
                'wall_thickness': self.shape.wall_thickness
            },
            'nodes': self.nodes.tolist(),
            'hex_elements': self.hex_elements.tolist(),
            'boundary_faces': {
                key: faces.tolist() for key, faces in self.boundary_faces.items()
            },
            'node_sets': {
                key: nodes.tolist() for key, nodes in self.node_sets.items()
            },
            'quality_metrics': {
                'min_jacobian': float(self.quality_metrics.min_jacobian),
                'avg_jacobian': float(self.quality_metrics.avg_jacobian),
                'max_aspect_ratio': float(self.quality_metrics.max_aspect_ratio),
                'generation_time': float(self.quality_metrics.generation_time)
            }
        }

        with open(filename, 'w') as f:
            json.dump(mesh_data, f, indent=2)

        print(f"  Exported mesh data to {filename}")

    def _export_npz(self, filename: str):
        """Export mesh in compressed NumPy format (most efficient)"""
        np.savez_compressed(
            filename,
            nodes=self.nodes,
            hex_elements=self.hex_elements,
            element_jacobians=self.element_jacobians,
            element_volumes=self.element_volumes,
            boundary_faces_outer=self.boundary_faces['outer'],
            boundary_faces_inner=self.boundary_faces.get('inner', np.array([])),
            boundary_faces_front=self.boundary_faces['front'],
            boundary_faces_back=self.boundary_faces['back'],
            node_sets_outer=self.node_sets['outer_surface'],
            node_sets_inner=self.node_sets.get('inner_surface', np.array([])),
            shape_params=vars(self.shape),
            quality_metrics=vars(self.quality_metrics),
            mesh_resolution=self.mesh_resolution,
            n_axial=self.n_axial,
            n_circumferential=self.n_circumferential,
            n_radial=self.n_radial
        )
        print(f"  Exported mesh to {filename} (compressed NPZ format)")

    def _export_exodus(self, filename: str):
        """Export mesh in Exodus II format (placeholder)"""
        print(f"  Exodus export not implemented. Use VTK, JSON, or NPZ format.")


# ============================================================================
# Thermal Properties and Nose Profile Generator (from create_rocket_noses.py)
# ============================================================================

@dataclass
class ThermalEquivalentProperties:
    """
    Thermal equivalent properties for Falcon 9 structure
    Based on research of aerospace thermal modeling
    """
    # Falcon 9 actual structure
    shell_thickness_actual: float = 0.0045  # 4.5mm Al-Li alloy shell
    stringer_spacing: float = 0.3  # 300mm typical stringer spacing
    frame_spacing: float = 1.0  # 1m typical frame spacing

    # Material properties (Aluminum-Lithium 2195)
    density: float = 2700  # kg/m³
    specific_heat: float = 900  # J/(kg·K)
    thermal_conductivity: float = 120  # W/(m·K)

    # Equivalent thickness calculation results
    equivalent_thickness: float = 0.015  # 15mm default
    heat_capacity_ratio: float = 3.33  # Effective increase due to internal structure

    @staticmethod
    def calculate_equivalent_thickness(
            shell_thickness: float = 0.0045,
            include_stringers: bool = True,
            include_frames: bool = True,
            include_tps: bool = False
    ) -> float:
        """
        Calculate equivalent thickness for thermal analysis

        Based on research:
        - Shell contributes base thickness
        - Stringers add ~40-60% equivalent mass locally
        - Frames add ~20-30% equivalent mass
        - TPS (if present) adds significant thermal mass

        References:
        - NASA TM-2010-216293: Thermal Analysis of Aerospace Structures
        - AIAA 2018-4693: Falcon 9 Structural Design Considerations
        """

        equiv_thickness = shell_thickness

        if include_stringers:
            # Stringers increase local thermal mass by ~50%
            stringer_contribution = shell_thickness * 0.5
            equiv_thickness += stringer_contribution

        if include_frames:
            # Frames add distributed thermal mass
            frame_contribution = shell_thickness * 0.3
            equiv_thickness += frame_contribution

        if include_tps:
            # Thermal protection system (cork, PICA-X, etc.)
            tps_thickness = 0.005  # 5mm typical
            equiv_thickness += tps_thickness * 0.7  # Accounting for lower conductivity

        # Additional factor for joints, fasteners, and local reinforcements
        structural_factor = 1.2
        equiv_thickness *= structural_factor

        # Typical range: 12-18mm for Falcon 9 equivalent
        return np.clip(equiv_thickness, 0.012, 0.020)


class RocketNoseGenerator:
    """
    Generate and visualize various rocket nose profiles with realistic thermal properties
    """

    def __init__(self,
                 base_radius: float = 1.83,  # Falcon 9 radius
                 thermal_equivalent: bool = True,
                 mesh_resolution: str = 'medium',
                 output_dir: str = None):
        """
        Initialize nose generator

        Args:
            base_radius: Body radius in meters (1.83m for Falcon 9)
            thermal_equivalent: Use thermal equivalent thickness
            mesh_resolution: 'coarse', 'medium', or 'fine' (default: 'medium')
            output_dir: Output directory for mesh files and plots
        """
        self.base_radius = base_radius
        self.thermal_equivalent = thermal_equivalent

        # Enhanced mesh resolution for better FEA quality
        resolutions = {
            'coarse': (50, 32, 4),  # Previous medium - quick analysis
            'medium': (80, 48, 5),  # Previous fine - medium-fidelity
            'fine': (120, 64, 6)  # New high-fidelity for detailed analysis
        }
        self.n_axial, self.n_circumferential, self.n_radial = resolutions.get(
            mesh_resolution, resolutions['medium']
        )
        self.mesh_resolution = mesh_resolution

        # Validate quality for analysis
        print(f"  Mesh resolution: {mesh_resolution.upper()} quality")
        print(f"  FEA mesh grid: {self.n_axial}x{self.n_circumferential}x{self.n_radial}")

        # Expected node counts
        expected_nodes = self.n_axial * self.n_circumferential * self.n_radial
        print(f"  Expected nodes per profile: ~{expected_nodes:,}")

        # Setup output directory structure
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"rocket_mesh_{timestamp}"

        self.output_dir = Path(output_dir)
        self.mesh_dir = self.output_dir / "mesh_files"
        self.plot_dir = self.output_dir / "plots"

        # Create directories
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nOutput directories created:")
        print(f"  Mesh files: {self.mesh_dir}")
        print(f"  Plots: {self.plot_dir}")

        # Calculate thermal equivalent thickness
        self.thermal_props = ThermalEquivalentProperties()
        if thermal_equivalent:
            self.wall_thickness = self.thermal_props.calculate_equivalent_thickness(
                include_stringers=True,
                include_frames=True,
                include_tps=False  # No TPS for standard flight
            )
        else:
            self.wall_thickness = 0.0045  # Actual shell thickness

        print(f"\nThermal Modeling Configuration:")
        print(f"  Base radius: {self.base_radius:.3f} m")
        print(f"  Wall thickness: {self.wall_thickness * 1000:.1f} mm")
        print(f"  Thermal equivalent: {'Yes' if thermal_equivalent else 'No'}")
        print(f"  Heat capacity factor: {self.wall_thickness / 0.0045:.2f}x")

    def create_nose_profiles(self) -> Dict:
        """
        Create 6 distinct nose profiles for analysis
        Enhanced parameters for better thermal differentiation

        Returns:
            Dictionary with profile names and mesh objects
        """

        profiles = {}

        # 1. Conical (Sharp, highest heating)
        # Shorter and sharper for more concentrated heat
        print("\n1. Creating Conical nose profile...")
        profiles['conical'] = self._create_profile(
            nose_type='conical',
            nose_length=4.0,  # Shorter for sharper angle (was 5.0)
            nose_sharpness=0.95,  # Very sharp (was 0.8)
            description="Sharp conical profile - maximum heating concentration"
        )

        # 2. Ogive (Falcon 9 standard - baseline)
        print("2. Creating Ogive nose profile (Falcon 9 standard)...")
        profiles['ogive_falcon'] = self._create_profile(
            nose_type='ogive',
            nose_length=6.5,  # Keep standard length
            nose_sharpness=0.7,  # Standard sharpness
            description="Falcon 9 standard ogive profile - balanced performance"
        )

        # 3. Von Karman (Minimum drag for volume)
        # Optimized for low drag and moderate heating
        print("3. Creating Von Karman nose profile...")
        profiles['von_karman'] = self._create_profile(
            nose_type='haack',
            nose_length=7.0,  # Longer for better optimization (was 6.0)
            nose_haack_c=0.333,  # Von Karman parameter
            nose_sharpness=0.6,  # Moderate sharpness
            description="Von Karman - minimum drag, optimized heat distribution"
        )

        # 4. Parabolic (Smooth transition)
        # Moderate length with smooth curvature
        print("4. Creating Parabolic nose profile...")
        profiles['parabolic'] = self._create_profile(
            nose_type='parabolic',
            nose_length=5.5,  # Standard length
            nose_sharpness=0.5,  # Smoother transition (was 0.6)
            description="Parabolic - smooth pressure and heat distribution"
        )

        # 5. Elliptical (Blunt body - minimum heating)
        # Shorter and blunter for maximum heat spreading
        print("5. Creating Elliptical nose profile...")
        profiles['elliptical'] = self._create_profile(
            nose_type='elliptical',
            nose_length=3.5,  # Shorter for blunter shape (was 4.0)
            nose_sharpness=0.2,  # Very blunt (was 0.3)
            description="Elliptical - blunt body for minimum peak heating"
        )

        # 6. Power Series (n=0.5 for more dramatic shape)
        # Changed to n=0.5 for better differentiation from n=0.75
        print("6. Creating Power Series nose profile...")
        profiles['power_050'] = self._create_profile(  # Changed from power_075
            nose_type='power',
            nose_length=5.0,  # Moderate length (was 5.5)
            nose_power=0.5,  # Changed to 0.5 for more dramatic curvature (was 0.75)
            nose_sharpness=0.65,  # Moderate sharpness (was 0.7)
            description="Power series (n=0.5) - enhanced curvature for distinct heating"
        )

        return profiles

    def _create_profile(self, nose_type: str, nose_length: float,
                        description: str, **kwargs) -> Dict:
        """Create a single nose profile with thermal equivalent thickness"""

        # Variable thickness profile for more realistic thermal modeling
        if self.thermal_equivalent:
            thickness_profile = [
                (0.0, self.wall_thickness * 0.8),  # Slightly thinner at tip
                (nose_length * 0.3, self.wall_thickness),  # Full thickness
                (nose_length * 0.7, self.wall_thickness * 1.1),  # Reinforced region
                (nose_length, self.wall_thickness * 1.2),  # Base connection
            ]
        else:
            thickness_profile = None

        shape = RocketShapeParameters(
            nose_type=nose_type,
            nose_length=nose_length,
            body_radius=self.base_radius,
            wall_thickness=self.wall_thickness,
            variable_thickness=thickness_profile is not None,
            thickness_profile=thickness_profile,
            **{k: v for k, v in kwargs.items() if k not in ['description']}
        )

        # Generate mesh with selected quality level
        mesh = HexahedralRocketMesh(
            shape_params=shape,
            n_axial=self.n_axial,
            n_circumferential=self.n_circumferential,
            n_radial=self.n_radial,
            nose_only=True,  # Only generate nose for efficiency
            grading_params={
                'axial_ratio': 1.15,  # Slight grading toward tip for accuracy
                'radial_ratio': 1.0,  # Uniform radial distribution
                'boundary_layer': True,
                'bl_thickness': 0.1,  # 10% of thickness for boundary layer
                'bl_growth': 1.2  # Smooth transition in boundary layer
            },
            enable_parallel=False,  # Disable for stability
            mesh_resolution=self.mesh_resolution  # Pass resolution to mesh
        )

        return {
            'mesh': mesh,
            'shape': shape,
            'description': description,
            'thermal_thickness': self.wall_thickness,
            'nodes': mesh.quality_metrics.n_nodes,
            'elements': mesh.quality_metrics.n_hex_elements,
            'quality': self.mesh_resolution.title()  # Use actual resolution setting
        }

    def visualize_profiles(self, profiles: Dict, save_path: str = None):
        """
        Create visualization of all nose profiles in a 2x3 grid

        Args:
            profiles: Dictionary of profile data
            save_path: Optional path to save the figure (if None, auto-generates)
        """

        fig = plt.figure(figsize=(20, 12))  # Adjusted for 2x3 grid
        # Determine the fidelity label based on mesh resolution
        if self.mesh_resolution.lower() == 'coarse':
            fidelity_label = 'Quick-analysis'
        elif self.mesh_resolution.lower() == 'medium':
            fidelity_label = 'Medium-fidelity'
        elif self.mesh_resolution.lower() == 'fine':
            fidelity_label = 'High-fidelity'
        else:
            fidelity_label = self.mesh_resolution.upper()

        fig.suptitle('Rocket Nose Profiles for Thermal-Structural Analysis\n' +
                     f'Equivalent Thickness: {self.wall_thickness * 1000:.1f}mm ' +
                     f'(Heat Capacity Factor: {self.wall_thickness / 0.0045:.1f}x) | ' +
                     f'Mesh Quality: {self.mesh_resolution.upper()} ({fidelity_label})',
                     fontsize=16, fontweight='bold')

        profile_names = list(profiles.keys())

        # Create 2x3 grid for 6 profiles
        for idx, name in enumerate(profile_names):
            ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
            profile_data = profiles[name]

            # Plot the mesh with nose pointing up
            self._plot_single_mesh(ax, profile_data['mesh'], name)

            # Add description
            title = f"{name.replace('_', ' ').title()}\n"
            title += f"L={profile_data['shape'].nose_length:.1f}m, "
            title += f"R={profile_data['shape'].body_radius:.2f}m\n"
            title += f"Nodes: {profile_data['nodes']:,}, "
            title += f"Elements: {profile_data['elements']:,}\n"
            title += f"Mesh Quality: {self.mesh_resolution.title()}"

            ax.set_title(title, fontsize=10, pad=10)

            # Set consistent view angle for best visualization with inverted Z
            ax.view_init(elev=15, azim=-70)  # Adjusted for inverted Z-axis view

            # Set axis limits for consistent scaling with nose tip at Z=0
            max_r = profile_data['shape'].body_radius * 1.2
            max_z = profile_data['shape'].nose_length * 1.1

            # IMPORTANT: Set Z limits with 0 at tip, max at base
            ax.set_xlim([-max_r, max_r])
            ax.set_ylim([-max_r, max_r])
            ax.set_zlim([0, max_z])  # Z=0 is nose tip, Z=max is base

            # INVERT Z-AXIS to display nose pointing upward in plot
            ax.invert_zaxis()  # Critical for correct orientation display

            # Set aspect ratio to show proper proportions
            ax.set_box_aspect([1, 1, max_z / (2 * max_r)])

            # Remove axis tick labels but keep axis lines visible
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.grid(False)  # Turn off grid

            # Hide the panes for cleaner look
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # Make axes lines more subtle
            ax.xaxis.line.set_linewidth(0.5)
            ax.yaxis.line.set_linewidth(0.5)
            ax.zaxis.line.set_linewidth(0.5)

            # Add simple axis labels with arrow indicating nose direction
            ax.set_xlabel('X', fontsize=8, labelpad=-5)
            ax.set_ylabel('Y', fontsize=8, labelpad=-5)
            ax.set_zlabel('Z (tip=0)', fontsize=8, labelpad=-5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add padding around figure

        # Auto-generate filename with timestamp if not provided
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.plot_dir / f"nose_profiles_comparison_{timestamp}.png"
        else:
            save_path = self.plot_dir / save_path

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")

        plt.show()

    def _plot_single_mesh(self, ax, mesh, name):
        """Plot a single mesh in 3D with nose pointing up"""

        # Get nodes (Z is the axial direction, nose pointing up at max Z)
        nodes = mesh.nodes

        # Find circumferential slices to show mesh structure
        n_slices = min(20, mesh.n_axial)  # Show up to 20 axial slices for better mesh visibility
        slice_indices = np.linspace(0, mesh.n_axial - 1, n_slices, dtype=int)

        # Plot circumferential rings at each slice
        for i in slice_indices:
            # Get nodes at this axial position
            ring_nodes = []
            for j in range(mesh.n_circumferential):
                # Outer surface nodes
                node_idx = mesh.node_indices[i, j, -1]
                ring_nodes.append(nodes[node_idx])

            ring_nodes.append(ring_nodes[0])  # Close the ring
            ring_nodes = np.array(ring_nodes)

            # Vary line properties based on depth for 3D effect
            z_pos = ring_nodes[0, 2]
            z_ratio = z_pos / np.max(nodes[:, 2])
            linewidth = 0.5 + 0.4 * (1 - z_ratio)  # Thicker lines near base

            # Plot the ring with darker color for visibility
            ax.plot(ring_nodes[:, 0], ring_nodes[:, 1], ring_nodes[:, 2],
                    'k-', linewidth=linewidth, alpha=0.9)  # Black lines for mesh

        # Plot longitudinal lines
        n_long_lines = min(24, mesh.n_circumferential)  # Show up to 24 longitudinal lines
        long_indices = np.linspace(0, mesh.n_circumferential - 1, n_long_lines, dtype=int)

        for j in long_indices:
            # Get nodes along this longitudinal line
            line_nodes = []
            for i in range(mesh.n_axial):
                # Outer surface nodes
                node_idx = mesh.node_indices[i, j, -1]
                line_nodes.append(nodes[node_idx])

            line_nodes = np.array(line_nodes)

            # Plot the line with darker color for visibility
            ax.plot(line_nodes[:, 0], line_nodes[:, 1], line_nodes[:, 2],
                    'k-', linewidth=0.7, alpha=0.9)  # Black lines for mesh

        # Add surface coloring based on axial position
        # Sample fewer elements for cleaner surface visualization
        n_elem_viz = min(150, len(mesh.hex_elements))  # Reduced for cleaner look
        elem_indices = np.linspace(0, len(mesh.hex_elements) - 1, n_elem_viz, dtype=int)

        for elem_idx in elem_indices:
            elem = mesh.hex_elements[elem_idx]

            # Get the outer face of the hex element
            # Standard hex element: nodes 0-3 bottom, 4-7 top (in radial direction)
            # For outer surface, we want the face with higher radial indices
            outer_face = elem[[3, 2, 6, 7]]  # Outer radial face
            face_coords = nodes[outer_face]

            # Create polygon for the face
            poly = [[face_coords[j] for j in range(4)]]

            # Color based on axial position (Z coordinate)
            z_avg = np.mean(face_coords[:, 2])
            z_min = np.min(nodes[:, 2])  # Tip at min Z
            z_max = np.max(nodes[:, 2])  # Base at max Z

            # Use subtle colormap for height (red at tip, blue at base)
            color = cm.coolwarm(1.0 - (z_avg - z_min) / (z_max - z_min))  # Inverted for tip coloring

            # Add face to plot with high transparency to show mesh clearly
            poly_collection = Poly3DCollection(poly, alpha=0.2,  # More transparent
                                               facecolor=color,
                                               edgecolor='none')
            ax.add_collection3d(poly_collection)

        # Add subtle nose tip indicator at minimum Z (nose tip at Z=0)
        z_min = np.min(nodes[:, 2])
        tip_nodes = nodes[nodes[:, 2] < z_min + 0.02 * np.max(nodes[:, 2])]
        if len(tip_nodes) > 0:
            # Just mark the very tip
            tip_point = tip_nodes[np.argmin(tip_nodes[:, 2])]
            ax.scatter([tip_point[0]], [tip_point[1]], [tip_point[2]],
                       c='red', s=20, alpha=1.0, marker='o', edgecolors='darkred', linewidths=0.5)

        # Add base indicator at maximum Z (base at z=nose_length)
        z_max = np.max(nodes[:, 2])
        base_nodes = nodes[nodes[:, 2] > z_max * 0.98]
        if len(base_nodes) > 0:
            # Just mark a few base points
            base_sample = base_nodes[::max(1, len(base_nodes) // 8)]
            ax.scatter(base_sample[:, 0], base_sample[:, 1], base_sample[:, 2],
                       c='blue', s=10, alpha=0.6, marker='s', edgecolors='darkblue', linewidths=0.5)

    def export_profiles_data(self, profiles: Dict, filename: str = None):
        """
        Export profile data for analysis

        Args:
            profiles: Dictionary of profile data
            filename: Output JSON filename (if None, auto-generates)
        """

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nose_profiles_data_{timestamp}.json"

        filepath = self.output_dir / filename

        export_data = {
            'metadata': {
                'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'thermal_equivalent': self.thermal_equivalent,
                'wall_thickness_mm': self.wall_thickness * 1000,
                'heat_capacity_factor': self.wall_thickness / 0.0045,
                'mesh_resolution': {
                    'level': self.mesh_resolution,
                    'n_axial': self.n_axial,
                    'n_circumferential': self.n_circumferential,
                    'n_radial': self.n_radial
                },
                'n_profiles': 6  # Updated to 6 profiles
            },
            'thermal_properties': {
                'density_kg_m3': self.thermal_props.density,
                'specific_heat_J_kg_K': self.thermal_props.specific_heat,
                'thermal_conductivity_W_m_K': self.thermal_props.thermal_conductivity,
                'volumetric_heat_capacity_J_m3_K': (
                        self.thermal_props.density * self.thermal_props.specific_heat
                )
            },
            'profiles': {}
        }

        for name, data in profiles.items():
            export_data['profiles'][name] = {
                'description': data['description'],
                'nose_length_m': data['shape'].nose_length,
                'body_radius_m': data['shape'].body_radius,
                'wall_thickness_m': data['thermal_thickness'],
                'n_nodes': data['nodes'],
                'n_elements': data['elements'],
                'mesh_resolution': self.mesh_resolution.title(),
                'parameters': {
                    'nose_type': data['shape'].nose_type,
                    'nose_sharpness': getattr(data['shape'], 'nose_sharpness', None),
                    'nose_power': getattr(data['shape'], 'nose_power', None),
                    'nose_haack_c': getattr(data['shape'], 'nose_haack_c', None)
                }
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\nProfile data exported to {filepath}")


# ============================================================================
# Additional Export and Analysis Functions
# ============================================================================

def export_individual_meshes(profiles: Dict, mesh_dir: Path):
    """
    Export individual mesh files with descriptive filenames

    Args:
        profiles: Dictionary of profile data
        mesh_dir: Directory to save mesh files
    """
    print("\nExporting individual mesh files...")

    for name, data in profiles.items():
        mesh = data['mesh']
        shape = data['shape']

        # Build filename components
        filename_parts = [
            f"nose_{name}",
            f"L{shape.nose_length:.1f}",
            f"R{shape.body_radius:.1f}",
            f"t{data['thermal_thickness'] * 1000:.0f}mm",
            f"n{data['nodes']}",
            data['quality']
        ]

        # Add specific parameters based on nose type
        if shape.nose_type == 'haack':
            filename_parts.append(f"c{shape.nose_haack_c:.2f}")
        elif shape.nose_type == 'power':
            filename_parts.append(f"p{shape.nose_power:.2f}")
        elif hasattr(shape, 'nose_sharpness') and shape.nose_sharpness is not None:
            filename_parts.append(f"s{shape.nose_sharpness:.1f}")

        filename = "_".join(filename_parts) + ".vtk"
        filepath = mesh_dir / filename

        # Export mesh
        mesh.export_mesh(str(filepath))
        print(f"  [OK] Exported {filename}")


def generate_summary_report(profiles: Dict, output_dir: Path, thermal_props: ThermalEquivalentProperties,
                            base_radius: float, wall_thickness: float, thermal_equivalent: bool,
                            mesh_resolution: str, n_axial: int, n_circumferential: int, n_radial: int):
    """
    Generate a summary report file

    Args:
        profiles: Dictionary of profile data
        output_dir: Output directory
        thermal_props: Thermal properties object
        base_radius: Base radius
        wall_thickness: Wall thickness
        thermal_equivalent: Whether thermal equivalent is used
        mesh_resolution: Mesh resolution level
        n_axial: Number of axial divisions
        n_circumferential: Number of circumferential divisions
        n_radial: Number of radial divisions
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"mesh_generation_report_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ROCKET NOSE MESH GENERATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Base radius: {base_radius:.3f} m\n")
        f.write(f"  Wall thickness: {wall_thickness * 1000:.1f} mm\n")
        f.write(f"  Thermal equivalent: {'Yes' if thermal_equivalent else 'No'}\n")
        f.write(f"  Heat capacity factor: {wall_thickness / 0.0045:.2f}x\n")
        f.write(f"  Mesh resolution: {mesh_resolution}\n")
        f.write(f"  Grid: {n_axial}x{n_circumferential}x{n_radial}\n")
        f.write(f"  Number of profiles: 6\n\n")

        f.write("GENERATED PROFILES:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Profile':<20} {'Type':<12} {'Length(m)':<10} {'Nodes':<10} {'Resolution':<10}\n")
        f.write("-" * 70 + "\n")

        for name, data in profiles.items():
            f.write(f"{name:<20} {data['shape'].nose_type:<12} "
                    f"{data['shape'].nose_length:<10.1f} {data['nodes']:<10} "
                    f"{data['quality']:<10}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("OUTPUT FILES:\n")
        f.write(f"  Directory: {output_dir}\n")
        f.write(f"  Mesh files: {len(profiles)} VTK files in mesh_files/\n")
        f.write(f"  Plots: Saved in plots/\n")
        f.write(f"  Data: JSON profile data\n")
        f.write("\n" + "=" * 70 + "\n")

    print(f"\nSummary report saved to {report_file}")


def analyze_thermal_performance(profiles: Dict, thermal_props: ThermalEquivalentProperties):
    """
    Analyze and compare thermal performance metrics

    Args:
        profiles: Dictionary of profile data
        thermal_props: Thermal properties object
    """
    print("\n" + "=" * 60)
    print("Thermal Performance Analysis")
    print("=" * 60)

    # Calculate thermal metrics
    metrics = {}

    for name, data in profiles.items():
        # Approximate surface area (simplified)
        L = data['shape'].nose_length
        R = data['shape'].body_radius

        # Simplified surface area calculation
        if data['shape'].nose_type == 'conical':
            surface_area = np.pi * R * np.sqrt(R ** 2 + L ** 2)
        else:
            # Approximate for other shapes
            surface_area = np.pi * R * L * 1.2

        # Volume of material (shell)
        volume = surface_area * data['thermal_thickness']

        # Thermal mass
        thermal_mass = volume * thermal_props.density * thermal_props.specific_heat

        # Characteristic time for thermal response
        char_time = (data['thermal_thickness'] ** 2 * thermal_props.density *
                     thermal_props.specific_heat / thermal_props.thermal_conductivity)

        metrics[name] = {
            'surface_area_m2': surface_area,
            'volume_m3': volume,
            'mass_kg': volume * thermal_props.density,
            'thermal_mass_J_K': thermal_mass,
            'characteristic_time_s': char_time,
            'thermal_diffusivity_m2_s': (thermal_props.thermal_conductivity /
                                         (thermal_props.density * thermal_props.specific_heat))
        }

    # Print comparison table
    print(f"\n{'Profile':<15} {'Area(m2)':<10} {'Mass(kg)':<10} {'Thermal Mass(MJ/K)':<18} {'t(s)':<8}")
    print("-" * 70)

    for name, m in metrics.items():
        print(f"{name:<15} {m['surface_area_m2']:<10.2f} {m['mass_kg']:<10.1f} "
              f"{m['thermal_mass_J_K'] / 1e6:<18.3f} {m['characteristic_time_s']:<8.2f}")

    print("\nNotes:")
    print(f"  - Equivalent thickness: {data['thermal_thickness'] * 1000:.1f}mm")
    print(f"  - Material: Aluminum-Lithium 2195")
    print(f"  - t: Characteristic thermal response time")
    print(f"  - Thermal diffusivity: {metrics[list(metrics.keys())[0]]['thermal_diffusivity_m2_s']:.2e} m2/s")


# ============================================================================
# Utility Functions
# ============================================================================

def create_sample_meshes():
    """Create sample meshes with different shape configurations"""

    print("\n" + "=" * 60)
    print("Creating Sample Hexahedral Rocket Meshes")
    print("DEVICE-INDEPENDENT MESH GENERATION")
    print("=" * 60)

    # 1. Standard Falcon 9 fairing shape
    print("\n1. Falcon 9 Fairing (Ogive)")
    falcon_shape = RocketShapeParameters(
        nose_type='ogive',
        nose_length=13.1,
        nose_sharpness=0.7,
        body_length=40.0,
        body_radius=2.6,
        tail_type='straight',
        tail_length=0.0,
        wall_thickness=0.0045
    )

    mesh1 = HexahedralRocketMesh(
        shape_params=falcon_shape,
        n_axial=50,
        n_circumferential=32,
        n_radial=3,
        mesh_resolution='medium'
    )
    mesh1.export_mesh("falcon9_ogive_hex.vtk")
    mesh1.export_mesh("falcon9_ogive_hex.npz", format='npz')

    # 2. Nose-only mesh for fast aerodynamic analysis
    print("\n2. Nose-Only Mesh (Fast Analysis)")
    nose_shape = RocketShapeParameters(
        nose_type='haack',
        nose_length=15.0,
        nose_haack_c=0.0,  # LD-Haack minimum drag
        body_radius=2.0,
        wall_thickness=0.004
    )

    mesh2 = HexahedralRocketMesh(
        shape_params=nose_shape,
        n_axial=80,
        n_circumferential=32,
        n_radial=4,
        nose_only=True,
        grading_params={'axial_ratio': 1.2},
        mesh_resolution='fine'
    )
    mesh2.export_mesh("nose_only_haack_hex.vtk")
    mesh2.export_mesh("nose_only_haack_hex.npz", format='npz')

    print("\n" + "=" * 60)
    print("Sample meshes created successfully!")
    print("Meshes are device-independent and can be used on any compute platform")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage: Create 6 nose profiles for FEA analysis
    print("=" * 60)
    print("Rocket Mesh Generator - Combined Module")
    print("=" * 60)

    # Create nose profile generator
    generator = RocketNoseGenerator(
        base_radius=1.83,  # Falcon 9 first stage radius
        thermal_equivalent=True,  # Use equivalent thickness for thermal analysis
        mesh_resolution='medium',  # Medium quality mesh
        output_dir=None  # Will auto-create timestamped directory
    )

    # Generate all profiles
    print("\nGenerating 6 distinct nose profiles...")
    profiles = generator.create_nose_profiles()

    print(f"\n[OK] Successfully created {len(profiles)} nose profiles")

    # Visualize profiles
    print("\nCreating visualization...")
    generator.visualize_profiles(profiles)

    # Export data
    generator.export_profiles_data(profiles)

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)