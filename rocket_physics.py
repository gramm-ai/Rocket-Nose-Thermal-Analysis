"""
rocket_physics.py - GPU-Optimized Physics Engine for Rocket Thermal Analysis
High-performance FEA engine for heat transfer simulation using RTX 3090

Key Features:
1. Aerodynamic heating from air friction
2. Heat conduction through hexahedral elements
3. Radiation and convection heat losses
4. GPU-accelerated using PyTorch and CUDA
5. Support for parallel instances
6. Real-time monitoring interface

Optimized for RTX 3090 (24GB VRAM, 10496 CUDA cores)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import time
from dataclasses import dataclass
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class SimulationConfig:
    """Configuration for physics simulation"""
    # Time stepping
    time_step: float = 0.01  # seconds
    adaptive_timestep: bool = True
    min_timestep: float = 1e-4
    max_timestep: float = 0.1

    # Physics parameters
    enable_aerodynamic_heating: bool = True
    enable_conduction: bool = True
    enable_radiation: bool = True
    enable_convection: bool = True

    # GPU settings
    use_mixed_precision: bool = True  # Use tensor cores on RTX 3090
    batch_size: int = 1024  # Elements per batch

    # Monitoring
    monitor_interval: int = 10  # Steps between monitoring updates
    enable_profiling: bool = False


@dataclass
class MaterialProperties:
    """Thermal material properties"""
    # Aluminum-Lithium 2195 (Falcon 9 material)
    thermal_conductivity: float = 120.0  # W/(m·K)
    density: float = 2700.0  # kg/m³
    specific_heat: float = 900.0  # J/(kg·K)
    emissivity: float = 0.8  # Surface emissivity
    melting_point: float = 893.15  # K

    # Temperature-dependent properties (optional)
    use_temperature_dependent: bool = False

    def get_conductivity(self, temperature: torch.Tensor) -> torch.Tensor:
        """Get temperature-dependent thermal conductivity"""
        if not self.use_temperature_dependent:
            return torch.full_like(temperature, self.thermal_conductivity)

        # Simplified temperature dependence for aluminum
        # k(T) = k0 * (1 - α * (T - T0))
        T0 = 293.15  # Reference temperature
        alpha = 0.0002  # Temperature coefficient
        k = self.thermal_conductivity * (1 - alpha * (temperature - T0))
        return torch.clamp(k, min=50.0, max=200.0)


class RocketPhysicsEngine:
    """
    GPU-Optimized Physics Engine for Rocket Thermal FEA

    Implements finite element analysis for heat transfer in rocket structures
    using hexahedral elements and GPU acceleration.
    """

    # Physical constants
    STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
    GAS_CONSTANT = 287.052874  # J/(kg·K)
    GAMMA = 1.4  # Heat capacity ratio for air

    def __init__(self,
                 mesh,
                 config: Optional[SimulationConfig] = None,
                 material: Optional[MaterialProperties] = None,
                 device_id: int = 0,
                 instance_id: int = 0):
        """
        Initialize physics engine

        Args:
            mesh: HexahedralRocketMesh object
            config: Simulation configuration
            material: Material properties
            device_id: GPU device ID (for multi-GPU systems)
            instance_id: Instance ID for parallel simulations
        """
        self.mesh = mesh
        self.config = config or SimulationConfig()
        self.material = material or MaterialProperties()
        self.instance_id = instance_id

        # Setup GPU device
        self.device = self._setup_device(device_id)

        # Initialize mesh data on GPU
        self._initialize_gpu_data()

        # Build FEA matrices
        self._build_fem_system()

        # Initialize simulation state
        self._initialize_state()

        # Setup flight profile
        self._setup_flight_profile()

        # Performance tracking
        self.step_count = 0
        self.total_time = 0.0
        self.computation_times = {}

        print(f"\n[Instance {instance_id}] Physics Engine Initialized")
        print(f"  Device: {self.device}")
        print(f"  Nodes: {self.n_nodes:,}")
        print(f"  Elements: {self.n_elements:,}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")

    def _setup_device(self, device_id: int) -> torch.device:
        """Setup and configure GPU device"""
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            return torch.device('cpu')

        # Check device availability
        if device_id >= torch.cuda.device_count():
            device_id = 0

        device = torch.device(f'cuda:{device_id}')

        # Get device properties
        props = torch.cuda.get_device_properties(device_id)
        print(f"\n[Instance {self.instance_id}] GPU Configuration:")
        print(f"  Device: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  CUDA cores: {props.multi_processor_count * 128}")  # Approx for RTX 3090

        # RTX 3090 optimizations
        if 'RTX 3090' in props.name or '3090' in props.name:
            # Enable TF32 for Ampere architecture
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  ✓ RTX 3090 detected: TF32 enabled for faster computation")

        # Set memory fraction to allow multiple instances
        torch.cuda.set_per_process_memory_fraction(0.8, device_id)

        return device

    def _initialize_gpu_data(self):
        """Transfer mesh data to GPU"""
        # Node coordinates
        self.nodes = torch.tensor(self.mesh.nodes, dtype=torch.float32, device=self.device)
        self.n_nodes = len(self.nodes)

        # Element connectivity
        self.elements = torch.tensor(self.mesh.hex_elements, dtype=torch.long, device=self.device)
        self.n_elements = len(self.elements)

        # Boundary information
        if hasattr(self.mesh, 'node_sets') and 'outer_surface' in self.mesh.node_sets:
            outer_surface = self.mesh.node_sets['outer_surface']
            if len(outer_surface) > 0:
                self.boundary_nodes = torch.tensor(
                    outer_surface,
                    dtype=torch.long, device=self.device
                )
            else:
                self.boundary_nodes = torch.tensor([], dtype=torch.long, device=self.device)

            # Identify nose region (first 30% from tip at z=0)
            z_coords = self.nodes[:, 2]
            z_min = torch.min(z_coords)
            z_max = torch.max(z_coords)
            nose_threshold = z_min + 0.3 * (z_max - z_min)

            nose_mask = z_coords < nose_threshold
            self.nose_nodes = torch.where(nose_mask)[0]
        else:
            # Fallback: estimate boundary nodes based on radius
            print("  Warning: No boundary information in mesh, estimating...")
            radii = torch.sqrt(self.nodes[:, 0] ** 2 + self.nodes[:, 1] ** 2)
            threshold = torch.quantile(radii, 0.8)
            boundary_mask = radii > threshold
            self.boundary_nodes = torch.where(boundary_mask)[0]

            # Nose region
            z_coords = self.nodes[:, 2]
            z_min = torch.min(z_coords)
            z_max = torch.max(z_coords)
            nose_threshold = z_min + 0.3 * (z_max - z_min)
            nose_mask = z_coords < nose_threshold
            self.nose_nodes = torch.where(nose_mask)[0]

        print(f"  Boundary nodes: {len(self.boundary_nodes):,}")
        print(f"  Nose region nodes: {len(self.nose_nodes):,}")

    def _build_fem_system(self):
        """Build FEM system matrices on GPU"""
        print("\n  Building FEM system matrices...")

        # Allocate mass and stiffness matrices (sparse format)
        self._build_mass_matrix()
        self._build_stiffness_matrix()

        # Precompute element volumes and shape functions
        self._precompute_element_data()

        print("  ✓ FEM system ready")

    def _build_mass_matrix(self):
        """Build lumped mass matrix"""
        self.mass_lumped = torch.zeros(self.n_nodes, dtype=torch.float32, device=self.device)

        # Process elements in batches for efficiency
        batch_size = min(self.config.batch_size, self.n_elements)

        print(f"    Building mass matrix: {self.n_elements} elements, batch size {batch_size}")

        for i in range(0, self.n_elements, batch_size):
            batch_elements = self.elements[i:i + batch_size]

            # Get element nodes
            element_nodes = self.nodes[batch_elements]  # [batch, 8, 3]

            # Compute element volumes (simplified)
            volumes = self._compute_hex_volumes_batch(element_nodes)

            # Distribute mass to nodes (lumped mass)
            # Each node gets 1/8 of the element volume
            node_mass = volumes / 8.0  # [batch] tensor

            # Accumulate to global mass matrix
            for j in range(8):
                node_indices = batch_elements[:, j]
                self.mass_lumped.index_add_(0, node_indices, node_mass)

        # Apply material density
        self.mass_lumped *= self.material.density

        # Ensure positive mass
        self.mass_lumped = torch.clamp(self.mass_lumped, min=1e-10)

        print(f"    Mass matrix complete: min={self.mass_lumped.min():.3e}, max={self.mass_lumped.max():.3e}")

    def _build_stiffness_matrix(self):
        """Build thermal stiffness matrix (conductivity matrix)"""
        # For large meshes, we use a matrix-free approach
        # Store element stiffness matrices instead of global matrix

        print(f"    Building stiffness matrix: {self.n_elements} elements")

        self.element_stiffness = []
        batch_size = min(self.config.batch_size, self.n_elements)

        for i in range(0, self.n_elements, batch_size):
            batch_elements = self.elements[i:i + batch_size]
            element_nodes = self.nodes[batch_elements]

            # Compute element stiffness matrices
            K_batch = self._compute_element_stiffness_batch(element_nodes)

            if K_batch is not None:
                self.element_stiffness.append(K_batch)

        # Concatenate all batches
        if len(self.element_stiffness) > 0:
            self.element_stiffness = torch.cat(self.element_stiffness, dim=0)
            print(f"    Element stiffness shape: {self.element_stiffness.shape}")
        else:
            # Fallback: create identity-like stiffness for stability
            print("    Warning: Could not compute stiffness matrices, using fallback")
            self.element_stiffness = torch.eye(8, device=self.device).unsqueeze(0).repeat(self.n_elements, 1, 1)

    def _compute_hex_volumes_batch(self, element_nodes: torch.Tensor) -> torch.Tensor:
        """
        Compute volumes for batch of hexahedral elements

        Args:
            element_nodes: [batch, 8, 3] tensor of node coordinates

        Returns:
            [batch] tensor of volumes
        """
        # Simplified volume calculation using decomposition
        # Hex can be decomposed into 6 tetrahedra

        # Get corner nodes
        v0 = element_nodes[:, 0]  # [batch, 3]
        v1 = element_nodes[:, 1]
        v2 = element_nodes[:, 2]
        v3 = element_nodes[:, 3]
        v4 = element_nodes[:, 4]
        v5 = element_nodes[:, 5]
        v6 = element_nodes[:, 6]
        v7 = element_nodes[:, 7]

        # Compute volume using divergence theorem
        # Approximate as sum of tetrahedral volumes
        center = element_nodes.mean(dim=1)  # [batch, 3]

        volumes = torch.zeros(element_nodes.shape[0], device=self.device)

        # Face contributions (6 faces)
        faces = [
            [v0, v1, v2, v3],  # Bottom
            [v4, v7, v6, v5],  # Top
            [v0, v4, v5, v1],  # Front
            [v2, v6, v7, v3],  # Back
            [v0, v3, v7, v4],  # Left
            [v1, v5, v6, v2],  # Right
        ]

        for face in faces:
            # Compute face area and normal
            v_a, v_b, v_c, v_d = face

            # Diagonal split into triangles
            area1 = 0.5 * torch.norm(torch.cross(v_b - v_a, v_c - v_a), dim=1)
            area2 = 0.5 * torch.norm(torch.cross(v_c - v_a, v_d - v_a), dim=1)

            # Add pyramid volume (face to center)
            height = torch.norm(center - (v_a + v_b + v_c + v_d) / 4, dim=1)
            volumes += (area1 + area2) * height / 3

        return torch.abs(volumes) / 2  # Correction factor

    def _compute_element_stiffness_batch(self, element_nodes: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute stiffness matrices for batch of elements

        Args:
            element_nodes: [batch, 8, 3] tensor

        Returns:
            [batch, 8, 8] tensor of element stiffness matrices, or None if failed
        """
        batch_size = element_nodes.shape[0]
        K = torch.zeros(batch_size, 8, 8, device=self.device)

        # Gauss quadrature points for hexahedron (2x2x2)
        gauss_points = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], device=self.device, dtype=torch.float32) / np.sqrt(3)

        gauss_weights = torch.ones(8, device=self.device)

        valid_elements_found = False

        for gp_idx, (gp, w) in enumerate(zip(gauss_points, gauss_weights)):
            # Compute shape function derivatives
            dN_dxi = self._hex_shape_derivatives(gp)  # [8, 3]

            # Compute Jacobian for each element
            J = torch.matmul(dN_dxi.T.unsqueeze(0), element_nodes)  # [batch, 3, 3]

            # Compute determinant and check for validity
            det_J = torch.det(J)

            # Skip invalid elements (singular Jacobian)
            valid_mask = torch.abs(det_J) > 1e-10
            if not torch.any(valid_mask):
                continue

            valid_elements_found = True

            # Compute inverse only for valid elements
            J_valid = J[valid_mask]
            det_J_valid = det_J[valid_mask]

            # Safe inverse computation
            try:
                J_inv = torch.inverse(J_valid)
            except:
                # If inverse fails, skip this Gauss point
                continue

            # Transform derivatives to physical coordinates
            dN_dx = torch.matmul(dN_dxi.unsqueeze(0), J_inv.transpose(-2, -1))  # [valid_batch, 8, 3]

            # Compute element stiffness contribution
            # K += w * det_J * (dN_dx @ dN_dx.T) * k_thermal
            k_thermal = self.material.thermal_conductivity

            K_contribution = torch.matmul(dN_dx, dN_dx.transpose(-2, -1))  # [valid_batch, 8, 8]

            # Update only valid elements
            K[valid_mask] += w * torch.abs(det_J_valid).unsqueeze(-1).unsqueeze(-1) * K_contribution * k_thermal

        if not valid_elements_found:
            # Return None to indicate failure
            return None

        return K

    def _hex_shape_derivatives(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Compute shape function derivatives for hexahedral element

        Args:
            xi: [3] tensor of natural coordinates (r, s, t)

        Returns:
            [8, 3] tensor of shape function derivatives
        """
        # Extract coordinates as scalars
        if xi.dim() > 0:
            r = xi[0].item()
            s = xi[1].item()
            t = xi[2].item()
        else:
            # Single scalar case (shouldn't happen but handle it)
            r = s = t = xi.item()

        # Compute derivatives of shape functions w.r.t natural coordinates
        dN_dxi_values = [
            [-(1 - s) * (1 - t), -(1 - r) * (1 - t), -(1 - r) * (1 - s)],
            [(1 - s) * (1 - t), -(1 + r) * (1 - t), -(1 + r) * (1 - s)],
            [(1 + s) * (1 - t), (1 + r) * (1 - t), -(1 + r) * (1 + s)],
            [-(1 + s) * (1 - t), (1 - r) * (1 - t), -(1 - r) * (1 + s)],
            [-(1 - s) * (1 + t), -(1 - r) * (1 + t), (1 - r) * (1 - s)],
            [(1 - s) * (1 + t), -(1 + r) * (1 + t), (1 + r) * (1 - s)],
            [(1 + s) * (1 + t), (1 + r) * (1 + t), (1 + r) * (1 + s)],
            [-(1 + s) * (1 + t), (1 - r) * (1 + t), (1 - r) * (1 + s)]
        ]

        dN_dxi = torch.tensor(dN_dxi_values, device=self.device, dtype=torch.float32) * 0.125

        return dN_dxi

    def _precompute_element_data(self):
        """Precompute element data for efficiency"""
        # Element centroids for aerodynamic heating
        self.element_centroids = torch.zeros(self.n_elements, 3, device=self.device)

        for i in range(0, self.n_elements, self.config.batch_size):
            batch_elements = self.elements[i:i + self.config.batch_size]
            element_nodes = self.nodes[batch_elements]
            self.element_centroids[i:i + len(batch_elements)] = element_nodes.mean(dim=1)

        # Surface areas for boundary elements
        if len(self.boundary_nodes) > 0:
            self._compute_surface_areas()

    def _compute_surface_areas(self):
        """Compute surface areas for boundary nodes"""
        if len(self.boundary_nodes) == 0:
            self.surface_areas = torch.tensor([], device=self.device)
            return

        # Simplified: assign uniform area based on mesh density
        total_surface = 2 * np.pi * 1.83 * 6.5  # Approximate for nose cone
        self.surface_areas = torch.full(
            (len(self.boundary_nodes),),
            total_surface / max(len(self.boundary_nodes), 1),
            device=self.device
        )

    def _initialize_state(self):
        """Initialize simulation state variables"""
        # Temperature field
        initial_temp = 288.15  # K (15°C)
        self.temperature = torch.full((self.n_nodes,), initial_temp,
                                      dtype=torch.float32, device=self.device)

        # Time tracking
        self.time = 0.0
        self.dt = self.config.time_step

        # Flight state
        self.velocity = 0.0
        self.altitude = 0.0
        self.mach_number = 0.0

        # Atmospheric properties
        self.air_temp = 288.15
        self.air_pressure = 101325.0
        self.air_density = 1.225

        # Heat flux arrays
        self.heat_flux_aero = torch.zeros(self.n_nodes, device=self.device)
        self.heat_flux_radiation = torch.zeros(self.n_nodes, device=self.device)
        self.heat_flux_convection = torch.zeros(self.n_nodes, device=self.device)

        # Monitoring data
        self.monitor_data = {
            'time': [],
            'max_temp': [],
            'avg_temp': [],
            'nose_max_temp': [],
            'heat_flux_total': []
        }

    def _setup_flight_profile(self):
        """Setup flight profile interpolators"""
        # Simplified Falcon 9 ascent profile
        profile_time = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
        profile_velocity = np.array([0, 280, 620, 850, 1100, 1500, 1800, 1950, 2000])
        profile_altitude = np.array([0, 3200, 12800, 26400, 43200, 63200, 86400, 112800, 142400])

        self.velocity_interp = interp1d(profile_time, profile_velocity,
                                        kind='linear', fill_value='extrapolate')
        self.altitude_interp = interp1d(profile_time, profile_altitude,
                                        kind='linear', fill_value='extrapolate')

    def update_flight_state(self):
        """Update flight conditions based on time"""
        # Get velocity and altitude from profile
        self.velocity = float(self.velocity_interp(self.time))
        self.altitude = float(self.altitude_interp(self.time))

        # Update atmospheric properties
        self._update_atmosphere()

        # Calculate Mach number
        sound_speed = np.sqrt(self.GAMMA * self.GAS_CONSTANT * self.air_temp)
        self.mach_number = self.velocity / sound_speed if sound_speed > 0 else 0.0

    def _update_atmosphere(self):
        """Update atmospheric properties based on altitude"""
        h = self.altitude

        if h <= 11000:  # Troposphere
            self.air_temp = 288.15 - 0.0065 * h
            self.air_pressure = 101325.0 * (self.air_temp / 288.15) ** 5.256
        elif h <= 20000:  # Lower stratosphere
            self.air_temp = 216.65
            self.air_pressure = 22632.0 * np.exp(-0.0001577 * (h - 11000))
        else:  # Upper atmosphere
            self.air_temp = 216.65
            self.air_pressure = 101325.0 * np.exp(-h / 7000)

        # Air density from ideal gas law
        self.air_density = self.air_pressure / (self.GAS_CONSTANT * self.air_temp)

    @torch.no_grad()
    def compute_aerodynamic_heating(self):
        """Compute aerodynamic heating from air friction"""
        if not self.config.enable_aerodynamic_heating or self.velocity < 10.0:
            self.heat_flux_aero.zero_()
            return

        # Timer for profiling
        if self.config.enable_profiling:
            torch.cuda.synchronize()
            start = time.perf_counter()

        # Recovery temperature (stagnation temperature)
        recovery_factor = 0.9  # Turbulent boundary layer
        T_recovery = self.air_temp * (1 + recovery_factor * (self.GAMMA - 1) / 2 * self.mach_number ** 2)

        # Heat transfer coefficient (simplified correlation)
        Re = self.air_density * self.velocity * 1.0 / 1.8e-5  # Reynolds number
        h_conv = 10.0 * (Re ** 0.5) * (self.air_density ** 0.5)
        h_conv = np.clip(h_conv, 100, 10000)

        # Apply heating to boundary nodes
        if len(self.boundary_nodes) > 0:
            boundary_temps = self.temperature[self.boundary_nodes]

            # Heat flux: q = h * (T_recovery - T_wall)
            q_aero = h_conv * (T_recovery - boundary_temps)

            # Enhanced heating for nose region (stagnation point effects)
            if len(self.nose_nodes) > 0:
                nose_mask = torch.isin(self.boundary_nodes, self.nose_nodes)
                q_aero[nose_mask] *= 2.0  # Stagnation heating factor

            # Apply to heat flux array
            self.heat_flux_aero.zero_()
            self.heat_flux_aero[self.boundary_nodes] = torch.clamp(q_aero, min=0.0)

        if self.config.enable_profiling:
            torch.cuda.synchronize()
            self.computation_times['aero_heating'] = time.perf_counter() - start

    @torch.no_grad()
    def compute_heat_conduction(self):
        """Compute heat conduction through structure using FEM"""
        if not self.config.enable_conduction:
            return torch.zeros_like(self.temperature)

        if self.config.enable_profiling:
            torch.cuda.synchronize()
            start = time.perf_counter()

        # Initialize heat flow
        heat_flow = torch.zeros_like(self.temperature)

        # Apply element stiffness matrices (matrix-free approach)
        for i in range(0, self.n_elements, self.config.batch_size):
            batch_elements = self.elements[i:i + self.config.batch_size]
            batch_stiffness = self.element_stiffness[i:i + len(batch_elements)]

            # Get element temperatures
            element_temps = self.temperature[batch_elements]  # [batch, 8]

            # Compute heat flow: q = K * T
            element_flow = torch.matmul(batch_stiffness, element_temps.unsqueeze(-1)).squeeze(-1)

            # Assemble to global heat flow
            for j in range(8):
                node_indices = batch_elements[:, j]
                heat_flow.index_add_(0, node_indices, element_flow[:, j])

        # Convert to temperature rate: dT/dt = -q / (m * cp)
        thermal_capacity = self.mass_lumped * self.material.specific_heat
        conduction_rate = -heat_flow / thermal_capacity

        if self.config.enable_profiling:
            torch.cuda.synchronize()
            self.computation_times['conduction'] = time.perf_counter() - start

        return conduction_rate

    @torch.no_grad()
    def compute_radiation_cooling(self):
        """Compute radiation heat loss"""
        if not self.config.enable_radiation:
            self.heat_flux_radiation.zero_()
            return

        if self.config.enable_profiling:
            torch.cuda.synchronize()
            start = time.perf_counter()

        # Sky temperature (simplified)
        T_sky = 220.0 if self.altitude > 10000 else self.air_temp - 20.0

        # Apply to boundary nodes only
        if len(self.boundary_nodes) > 0:
            boundary_temps = self.temperature[self.boundary_nodes]

            # Stefan-Boltzmann radiation: q = ε * σ * (T^4 - T_sky^4)
            q_rad = self.material.emissivity * self.STEFAN_BOLTZMANN * (
                    torch.pow(boundary_temps, 4) - T_sky ** 4
            )

            # Apply to heat flux array (negative for cooling)
            self.heat_flux_radiation.zero_()
            self.heat_flux_radiation[self.boundary_nodes] = -q_rad

        if self.config.enable_profiling:
            torch.cuda.synchronize()
            self.computation_times['radiation'] = time.perf_counter() - start

    @torch.no_grad()
    def compute_convection_cooling(self):
        """Compute convective heat transfer"""
        if not self.config.enable_convection or self.velocity < 1.0:
            self.heat_flux_convection.zero_()
            return

        if self.config.enable_profiling:
            torch.cuda.synchronize()
            start = time.perf_counter()

        # Convective heat transfer coefficient
        Re = self.air_density * self.velocity * 1.0 / 1.8e-5
        Nu = 0.037 * (Re ** 0.8)  # Turbulent flow over flat plate
        h_conv = Nu * 0.025 / 1.0  # Simplified
        h_conv = np.clip(h_conv, 10, 1000)

        # Apply to boundary nodes
        if len(self.boundary_nodes) > 0:
            boundary_temps = self.temperature[self.boundary_nodes]

            # Heat flux: q = h * (T_wall - T_air)
            q_conv = h_conv * (boundary_temps - self.air_temp)

            # Apply to heat flux array (negative for cooling)
            self.heat_flux_convection.zero_()
            self.heat_flux_convection[self.boundary_nodes] = -q_conv

        if self.config.enable_profiling:
            torch.cuda.synchronize()
            self.computation_times['convection'] = time.perf_counter() - start

    @torch.no_grad()
    def step(self) -> Dict[str, Any]:
        """
        Perform one simulation time step

        Returns:
            Dictionary with current state information
        """
        # Update flight conditions
        self.update_flight_state()

        # Compute heat transfer components
        self.compute_aerodynamic_heating()
        self.compute_radiation_cooling()
        self.compute_convection_cooling()

        # Combine boundary heat fluxes
        total_boundary_flux = (self.heat_flux_aero +
                               self.heat_flux_radiation +
                               self.heat_flux_convection)

        # Convert flux to temperature rate for boundary nodes
        boundary_rate = torch.zeros_like(self.temperature)
        if len(self.boundary_nodes) > 0:
            # q = h * A * (T2 - T1) => dT/dt = q * A / (m * cp)
            masses = self.mass_lumped[self.boundary_nodes]
            areas = self.surface_areas if hasattr(self, 'surface_areas') else torch.ones_like(masses)
            cp = self.material.specific_heat

            boundary_rate[self.boundary_nodes] = (
                    total_boundary_flux[self.boundary_nodes] * areas / (masses * cp)
            )

        # Compute heat conduction
        conduction_rate = self.compute_heat_conduction()

        # Total temperature rate
        total_rate = boundary_rate + conduction_rate

        # Adaptive time stepping
        if self.config.adaptive_timestep:
            max_rate = torch.max(torch.abs(total_rate)).item()
            if max_rate > 0:
                # Target maximum temperature change per step
                target_change = 1.0  # K
                new_dt = min(target_change / max_rate, self.config.max_timestep)
                self.dt = max(new_dt, self.config.min_timestep)

        # Update temperature
        self.temperature += self.dt * total_rate

        # Apply physical bounds
        self.temperature = torch.clamp(self.temperature,
                                       min=150.0,
                                       max=self.material.melting_point - 10.0)

        # Update time and step count
        self.time += self.dt
        self.step_count += 1

        # Monitoring
        if self.step_count % self.config.monitor_interval == 0:
            self._update_monitor_data()

        # Return current state
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        temps_celsius = self.temperature - 273.15

        # Compute statistics
        max_temp = torch.max(temps_celsius).item()
        min_temp = torch.min(temps_celsius).item()
        avg_temp = torch.mean(temps_celsius).item()

        # Nose region statistics
        if len(self.nose_nodes) > 0:
            nose_temps = temps_celsius[self.nose_nodes]
            nose_max = torch.max(nose_temps).item()
            nose_avg = torch.mean(nose_temps).item()
        else:
            nose_max = max_temp
            nose_avg = avg_temp

        # Total heat flux
        total_flux = torch.sum(torch.abs(
            self.heat_flux_aero + self.heat_flux_radiation + self.heat_flux_convection
        )).item()

        return {
            'time': self.time,
            'step': self.step_count,
            'dt': self.dt,
            'velocity': self.velocity,
            'altitude': self.altitude,
            'mach_number': self.mach_number,
            'air_temperature': self.air_temp - 273.15,
            'max_temperature': max_temp,
            'min_temperature': min_temp,
            'avg_temperature': avg_temp,
            'nose_max_temperature': nose_max,
            'nose_avg_temperature': nose_avg,
            'total_heat_flux': total_flux,
            'instance_id': self.instance_id
        }

    def _update_monitor_data(self):
        """Update monitoring data for visualization"""
        state = self.get_state()

        self.monitor_data['time'].append(state['time'])
        self.monitor_data['max_temp'].append(state['max_temperature'])
        self.monitor_data['avg_temp'].append(state['avg_temperature'])
        self.monitor_data['nose_max_temp'].append(state['nose_max_temperature'])
        self.monitor_data['heat_flux_total'].append(state['total_heat_flux'])

        # Keep only last 1000 points for memory efficiency
        max_points = 1000
        for key in self.monitor_data:
            if len(self.monitor_data[key]) > max_points:
                self.monitor_data[key] = self.monitor_data[key][-max_points:]

    def get_temperature_field(self) -> torch.Tensor:
        """Get full temperature field (for visualization)"""
        return self.temperature.cpu()

    def get_monitor_data(self) -> Dict[str, List[float]]:
        """Get monitoring data for visualization"""
        return self.monitor_data

    def save_state(self, filename: str):
        """Save current simulation state to file"""
        state = {
            'time': self.time,
            'step_count': self.step_count,
            'temperature': self.temperature.cpu().numpy(),
            'monitor_data': self.monitor_data,
            'flight_state': {
                'velocity': self.velocity,
                'altitude': self.altitude,
                'mach_number': self.mach_number
            }
        }
        torch.save(state, filename)
        print(f"[Instance {self.instance_id}] State saved to {filename}")

    def load_state(self, filename: str):
        """Load simulation state from file"""
        state = torch.load(filename, map_location=self.device)

        self.time = state['time']
        self.step_count = state['step_count']
        self.temperature = torch.tensor(state['temperature'], device=self.device)
        self.monitor_data = state['monitor_data']

        if 'flight_state' in state:
            self.velocity = state['flight_state']['velocity']
            self.altitude = state['flight_state']['altitude']
            self.mach_number = state['flight_state']['mach_number']

        print(f"[Instance {self.instance_id}] State loaded from {filename}")

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        stats = {
            'total_steps': self.step_count,
            'simulation_time': self.time,
            'average_dt': self.time / max(self.step_count, 1),
            'gpu_memory_used': torch.cuda.memory_allocated(self.device) / 1e9 if self.device.type == 'cuda' else 0
        }

        # Add computation times if profiling enabled
        if self.config.enable_profiling and self.computation_times:
            stats.update({
                f'time_{key}': value
                for key, value in self.computation_times.items()
            })

        return stats

    def cleanup(self):
        """Clean up GPU resources"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"[Instance {self.instance_id}] GPU memory released")


def create_test_instance():
    """Create a test instance of the physics engine"""
    from rocket_mesh_hex import HexahedralRocketMesh, RocketShapeParameters

    print("\n" + "=" * 60)
    print("Testing Rocket Physics Engine")
    print("=" * 60)

    # Create a simple nose mesh
    shape = RocketShapeParameters(
        nose_type='ogive',
        nose_length=6.5,
        body_radius=1.83,
        wall_thickness=0.015
    )

    mesh = HexahedralRocketMesh(
        shape_params=shape,
        n_axial=30,
        n_circumferential=24,
        n_radial=3,
        nose_only=True,
        mesh_resolution='coarse'
    )

    # Create physics engine
    config = SimulationConfig(
        time_step=0.01,
        adaptive_timestep=True,
        enable_profiling=True
    )

    engine = RocketPhysicsEngine(
        mesh=mesh,
        config=config,
        instance_id=0
    )

    # Run a few steps
    print("\nRunning test simulation...")
    for i in range(100):
        state = engine.step()

        if i % 20 == 0:
            print(f"  Step {i:3d}: t={state['time']:6.2f}s, "
                  f"v={state['velocity']:4.0f} m/s, "
                  f"T_max={state['nose_max_temperature']:6.1f}°C")

    # Get performance stats
    stats = engine.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Simulation time: {stats['simulation_time']:.2f}s")
    print(f"  Average dt: {stats['average_dt']:.4f}s")
    print(f"  GPU memory: {stats['gpu_memory_used']:.2f} GB")

    # Cleanup
    engine.cleanup()

    return engine


if __name__ == "__main__":
    # Test the physics engine
    engine = create_test_instance()

    print("\n" + "=" * 60)
    print("Physics Engine Test Complete")
    print("=" * 60)
    print("\nThe engine is ready to be used by rocket_simulation.py")
    print("Multiple instances can run in parallel on RTX 3090")
    print("Monitor data available for rocket_visualization.py")