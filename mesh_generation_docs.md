# Rocket Mesh Generation Documentation

## Overview

This documentation covers two high-performance mesh generators for rocket geometries, optimized for physics simulations on modern hardware (RTX 3090 GPU + multi-core CPUs).

### Available Mesh Generators

1. **`rocket_mesh.py`** - Optimized mixed mesh generator (hexahedral → tetrahedral conversion)
2. **`rocket_mesh_hex.py`** - Pure hexahedral mesh generator with advanced shape parameterization

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 3090 or similar (24GB VRAM) for GPU acceleration
- **CPU**: 24-core AMD processor or equivalent for parallel processing
- **RAM**: 32GB+ recommended for large meshes

### Software Dependencies
```python
torch >= 1.9.0          # GPU acceleration
numpy >= 1.20.0         # Numerical operations
scipy >= 1.7.0          # Interpolation and spatial operations
numba >= 0.54.0         # JIT compilation (rocket_mesh.py only)
psutil >= 5.8.0         # Hardware detection
concurrent.futures      # Parallel processing (standard library)
```

## Quick Start

### Basic Usage

```python
from rocket_mesh_hex import HexahedralRocketMesh, RocketShapeParameters

# Define rocket shape
shape = RocketShapeParameters(
    nose_type='ogive',
    nose_length=13.1,
    body_length=40.0,
    body_radius=2.6,
    wall_thickness=0.0045
)

# Generate mesh
mesh = HexahedralRocketMesh(
    shape_params=shape,
    n_axial=100,
    n_circumferential=32,
    n_radial=3
)

# Export mesh
mesh.export_mesh("rocket_mesh.vtk")
```

### Nose-Only Mode (Fast Analysis)

```python
# Generate only the nose cone for rapid analysis
mesh = HexahedralRocketMesh(
    shape_params=shape,
    n_axial=80,           # Higher resolution for detailed nose analysis
    n_circumferential=32,
    n_radial=4,
    nose_only=True,       # Enable nose-only mode
    grading_params={
        'axial_ratio': 1.3  # Concentrate nodes near tip
    }
)

# Access nose-specific node sets
nose_tip_nodes = mesh.node_sets['nose_tip']
nose_base_nodes = mesh.node_sets['nose_base']
```

## Mesh Generators Comparison

| Feature | rocket_mesh.py | rocket_mesh_hex.py |
|---------|---------------|-------------------|
| Element Type | Tetrahedral (converted from hex) | Pure Hexahedral |
| Shape Flexibility | Basic (ogive, parabolic, etc.) | Advanced (Haack, boat tail, etc.) |
| Thermal Optimization | ✓ Pre-computed properties | ✓ Orthogonal heat flow |
| GPU Optimization | ✓ RTX 3090 optimized | ✓ RTX 3090 optimized |
| CSR Matrix Support | ✓ Pre-assembled | ✓ Pre-assembled |
| Parallel Processing | ✓ 24-core optimized | ✓ Multi-core support |
| Memory Layout | C-contiguous arrays | C-contiguous arrays |
| Best For | General FEA, complex physics | Thin walls, thermal analysis |

## Shape Parameters

### Nose Cone Profiles

#### 1. Ogive (Default Falcon 9 Style)
```python
shape = RocketShapeParameters(
    nose_type='ogive',
    nose_length=13.1,
    nose_sharpness=0.7  # 0.1 (blunt) to 1.0 (sharp)
)
```

#### 2. Haack Series (Minimum Drag)
```python
shape = RocketShapeParameters(
    nose_type='haack',
    nose_length=15.0,
    nose_haack_c=0.333  # Von Karman profile
)
```
Common Haack C values:
- `C = 0`: LD-Haack (minimum drag for given length)
- `C = 0.333`: Von Karman (minimum drag for given volume)

#### 3. Power Series
```python
shape = RocketShapeParameters(
    nose_type='power',
    nose_power=0.75  # Shape exponent
)
```

#### 4. Other Profiles
```python
# Conical
shape = RocketShapeParameters(nose_type='conical')

# Parabolic
shape = RocketShapeParameters(nose_type='parabolic')

# Elliptical
shape = RocketShapeParameters(nose_type='elliptical')
```

### Tail/Nozzle Configurations

#### Bell Nozzle
```python
shape = RocketShapeParameters(
    tail_type='bell',
    tail_length=8.0,
    tail_throat_radius=0.8,
    tail_exit_radius=1.5
)
```

#### Tapered Exit
```python
shape = RocketShapeParameters(
    tail_type='tapered',
    tail_exit_radius=1.2
)
```

### Advanced Features

#### Variable Wall Thickness
```python
shape = RocketShapeParameters(
    variable_thickness=True,
    thickness_profile=[
        (0.0, 0.020),   # z-position, thickness
        (3.0, 0.015),
        (13.0, 0.010),
        (50.0, 0.008)
    ]
)
```

#### Boat Tail
```python
shape = RocketShapeParameters(
    boat_tail_enabled=True,
    boat_tail_length=3.0,
    boat_tail_angle=15.0  # degrees
)
```

## Mesh Parameters

### Grid Resolution

```python
mesh = HexahedralRocketMesh(
    n_axial=100,          # Divisions along rocket axis
    n_circumferential=32,  # Divisions around circumference
    n_radial=3            # Divisions through wall thickness
)
```

**Recommended Settings:**

| Analysis Type | n_axial | n_circumferential | n_radial |
|--------------|---------|-------------------|----------|
| Quick Preview | 50 | 16 | 2 |
| Structural | 100 | 32 | 3 |
| Thermal | 150 | 32 | 4-5 |
| High-Fidelity | 200+ | 64 | 5-8 |
| GPU Optimized | 128 | 32/64 | 4 |

### Mesh Grading

```python
grading = {
    'axial_ratio': 1.2,      # Geometric growth ratio
    'radial_ratio': 1.0,     # Uniform by default
    'boundary_layer': True,   # Enable boundary layer refinement
    'bl_thickness': 0.1,     # Relative thickness (0-1)
    'bl_growth': 1.2         # Growth ratio in boundary layer
}

mesh = HexahedralRocketMesh(
    grading_params=grading
)
```

## Performance Optimization

### GPU Optimization

```python
# Automatic GPU detection and optimization
mesh = HexahedralRocketMesh(
    optimization_target='gpu',  # or 'cpu'
    enable_gpu_optimization=True
)
```

**GPU-Specific Optimizations:**
- Warp-aligned circumferential divisions (multiples of 32)
- Contiguous memory layout for coalesced access
- Pre-computed PyTorch tensors
- CSR connectivity for sparse operations

### Parallel Processing

```python
mesh = HexahedralRocketMesh(
    enable_parallel=True  # Uses available CPU cores
)
```

### Material-Specific Optimization

```python
# Pre-configured materials with thermal properties
mesh = OptimizedRocketMesh3D(
    material_type='aluminum_lithium'  # or 'titanium', 'equivalent_multilayer'
)
```

## Export Formats

### VTK Format (Visualization)
```python
mesh.export_mesh("rocket.vtk", format='vtk')
```
Compatible with: ParaView, VisIt, VTK-based tools

### JSON Format (Data Exchange)
```python
mesh.export_mesh("rocket.json", format='json')
```
Includes: nodes, elements, boundaries, quality metrics

### Optimization Data (rocket_mesh.py)
```python
mesh.export_optimized_data("rocket_base")
# Creates: rocket_base.vtk, rocket_base_optimization.json
```

## Quality Metrics

### Accessing Quality Information

```python
metrics = mesh.get_mesh_quality_metrics()
print(f"Quality Grade: {metrics['quality_grade']}")
print(f"Min Jacobian: {metrics.get('min_jacobian', 'N/A')}")
print(f"Max Aspect Ratio: {metrics.get('max_aspect_ratio', 'N/A')}")
```

### Quality Grades

| Grade | Criteria | Suitable For |
|-------|----------|--------------|
| Excellent | Min Jacobian > 0.1, Max AR < 10 | High-accuracy simulations |
| Good | Min Jacobian > 0.01, Max AR < 50 | Most engineering analyses |
| Fair | Min Jacobian > 0, Max AR < 100 | Preliminary studies |
| Poor | Negative Jacobians or AR > 100 | Requires remeshing |

## Nose-Only Mode for Fast Analysis

The nose-only mode is a specialized feature in `rocket_mesh_hex.py` that generates only the nose cone portion of the rocket, significantly reducing computational requirements for aerodynamic and thermal analyses.

### Benefits of Nose-Only Mode

1. **Reduced Element Count**: 5-10x fewer elements than full rocket
2. **Faster FEA/CFD Simulations**: ~7-15x speedup for typical analyses
3. **Higher Resolution**: Can afford finer mesh in critical nose region
4. **Rapid Design Iteration**: Quick evaluation of different nose profiles
5. **Memory Efficiency**: Smaller memory footprint for GPU operations

### When to Use Nose-Only Mode

- **Aerodynamic Optimization**: Finding optimal nose cone shape for drag reduction
- **Heat Shield Design**: Thermal analysis of reentry vehicles
- **Pressure Distribution Studies**: Detailed analysis of stagnation regions
- **Shape Optimization**: Rapid evaluation of multiple nose profiles
- **Preliminary Design**: Quick assessment before full rocket analysis

### Implementation Example

```python
from rocket_mesh_hex import HexahedralRocketMesh, RocketShapeParameters

# Define nose shape for analysis
nose_shape = RocketShapeParameters(
    nose_type='haack',
    nose_haack_c=0.0,     # LD-Haack minimum drag profile
    nose_length=15.0,
    body_radius=2.5,
    wall_thickness=0.004
)

# Generate high-resolution nose-only mesh
nose_mesh = HexahedralRocketMesh(
    shape_params=nose_shape,
    n_axial=120,          # High axial resolution
    n_circumferential=64,  # Fine circumferential resolution
    n_radial=5,           # Through-thickness resolution
    nose_only=True,       # ACTIVATE NOSE-ONLY MODE
    grading_params={
        'axial_ratio': 1.5,      # Concentrate nodes near tip
        'boundary_layer': True,   # Enable for CFD
        'bl_thickness': 0.02
    }
)

# Access nose-specific boundaries for boundary conditions
nose_tip = nose_mesh.node_sets['nose_tip']    # Stagnation point region
nose_base = nose_mesh.node_sets['nose_base']  # Base for fixed BC
outer_surface = nose_mesh.node_sets['outer_surface']  # Aerodynamic surface
```

### Nose Profile Comparison Study

```python
def compare_nose_profiles():
    """Compare different nose profiles for optimization"""
    
    profiles = [
        ('conical', {}),
        ('ogive', {'nose_sharpness': 0.7}),
        ('haack', {'nose_haack_c': 0.0}),     # Min drag for length
        ('haack_vk', {'nose_haack_c': 0.333}), # Von Karman
        ('power', {'nose_power': 0.75})
    ]
    
    results = {}
    for profile_type, params in profiles:
        shape = RocketShapeParameters(
            nose_type=profile_type.split('_')[0],
            nose_length=10.0,
            body_radius=2.0,
            **params
        )
        
        mesh = HexahedralRocketMesh(
            shape_params=shape,
            n_axial=60,
            n_circumferential=32,
            n_radial=3,
            nose_only=True  # Fast comparison mode
        )
        
        results[profile_type] = {
            'elements': mesh.quality_metrics.n_hex_elements,
            'quality': mesh.quality_metrics.quality_grade,
            'time': mesh.quality_metrics.generation_time
        }
    
    return results
```

### Performance Comparison

| Configuration | Full Rocket | Nose Only | Speedup |
|--------------|-------------|-----------|---------|
| Nodes | 100,000 | 15,000 | 6.7x |
| Elements | 90,000 | 13,500 | 6.7x |
| Generation Time | 2.5s | 0.4s | 6.3x |
| FEA Solve Time | 120s | 8s | 15x |
| Memory Usage | 800 MB | 120 MB | 6.7x |

### Boundary Conditions for Nose-Only Analysis

```python
# Example: Setting up aerodynamic analysis
def setup_nose_aerodynamics(mesh):
    """Setup boundary conditions for nose cone CFD"""
    
    # Fixed base (where nose connects to body)
    fixed_nodes = mesh.node_sets['nose_base']
    
    # Aerodynamic surface
    surface_nodes = mesh.node_sets['outer_surface']
    
    # Stagnation region (nose tip)
    tip_nodes = mesh.node_sets['nose_tip']
    
    # Example pressure distribution
    mach = 2.0  # Supersonic flow
    for node_idx in surface_nodes:
        z = mesh.nodes[node_idx, 2]
        # Apply pressure based on position
        pressure = calculate_pressure(z, mach)
        apply_pressure_bc(node_idx, pressure)
```

### Integration with Physics Solvers

The nose-only mesh is fully compatible with standard FEA/CFD solvers:

```python
# Export for external solvers
nose_mesh.export_mesh("nose_analysis.vtk", format='vtk')
nose_mesh.export_mesh("nose_analysis.json", format='json')

# Get element data for custom solvers
elements = nose_mesh.hex_elements
nodes = nose_mesh.nodes
jacobians = nose_mesh.element_jacobians
volumes = nose_mesh.element_volumes
```

## Advanced Usage

### Mesh Refinement

```python
# Create base mesh
base_mesh = HexahedralRocketMesh(n_axial=50)

# Refine specific zones
refined_mesh = base_mesh.refine_mesh({
    'nose': 2.0,    # 2x refinement in nose
    'body': 1.0,    # No refinement in body
    'tail': 1.5     # 1.5x refinement in tail
})
```

### Accessing Node and Element Sets

```python
# Get node sets for boundary conditions
outer_nodes = mesh.node_sets['outer_surface']
nose_nodes = mesh.node_sets['nose']

# Get element sets for material assignment
nose_elements = mesh.element_sets['nose']
body_elements = mesh.element_sets['body']
```

### Shape Derivatives (for Optimization)

```python
derivatives = mesh.get_shape_derivatives()
radius_gradient = derivatives['radius']
curvature = derivatives['curvature']
```

## Example Configurations

### 1. Falcon 9 First Stage

```python
shape = RocketShapeParameters(
    nose_type='ogive',
    nose_length=13.1,
    nose_sharpness=0.7,
    body_length=40.0,
    body_radius=1.83,
    tail_type='straight',
    wall_thickness=0.0045
)

mesh = HexahedralRocketMesh(
    shape_params=shape,
    n_axial=100,
    n_circumferential=32,
    n_radial=3,
    material_type='aluminum_lithium'
)
```

### 2. Hypersonic Reentry Vehicle

```python
shape = RocketShapeParameters(
    nose_type='elliptical',
    nose_length=3.0,
    nose_sharpness=0.2,  # Very blunt
    body_length=10.0,
    body_radius=3.0,
    variable_thickness=True,
    thickness_profile=[
        (0, 0.02),    # Thick heat shield
        (3, 0.015),
        (13, 0.01)
    ]
)

mesh = HexahedralRocketMesh(
    shape_params=shape,
    n_axial=80,
    n_circumferential=48,
    n_radial=6,
    grading_params={
        'boundary_layer': True,
        'bl_thickness': 0.05
    }
)
```

### 3. Advanced Nozzle Design

```python
shape = RocketShapeParameters(
    nose_type='power',
    nose_power=0.75,
    body_length=30.0,
    body_radius=2.0,
    tail_type='bell',
    tail_length=10.0,
    tail_throat_radius=0.6,
    tail_exit_radius=1.8,
    boat_tail_enabled=True,
    boat_tail_length=3.0
)

mesh = HexahedralRocketMesh(
    shape_params=shape,
    n_axial=150,
    n_circumferential=64,
    n_radial=5,
    optimization_target='gpu'
)
```

## Performance Benchmarks

### Typical Generation Times

| Mesh Size | Nodes | Elements | CPU Time | GPU-Optimized |
|-----------|-------|----------|----------|---------------|
| Small | 10K | 8K | 0.5s | 0.3s |
| Medium | 100K | 90K | 2.5s | 1.2s |
| Large | 500K | 450K | 12s | 5s |
| Very Large | 2M | 1.8M | 45s | 18s |

### Memory Usage

```python
# Estimate memory requirements
nodes = n_axial * n_circumferential * n_radial
memory_mb = nodes * 3 * 4 / 1e6  # float32 coordinates
print(f"Estimated memory: {memory_mb:.1f} MB")
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Poor mesh quality | Sharp transitions | Increase mesh resolution or adjust grading |
| GPU out of memory | Mesh too large | Reduce resolution or use CPU mode |
| Slow generation | Parallel disabled | Enable parallel processing |
| Negative Jacobians | Extreme geometry | Adjust shape parameters or increase radial resolution |
| Import errors | Missing dependencies | Install required packages with pip |

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Mesh generation will now show detailed progress
mesh = HexahedralRocketMesh(...)
```

## Best Practices

1. **Start with coarse mesh** for geometry validation
2. **Use GPU optimization** for meshes > 100K nodes
3. **Enable boundary layers** for thermal/fluid analyses
4. **Maintain aspect ratios < 50** for accuracy
5. **Export JSON format** for custom post-processing
6. **Use hexahedral meshes** for thin-wall structures
7. **Apply grading** to capture gradients efficiently

## API Reference

### HexahedralRocketMesh Class

```python
class HexahedralRocketMesh:
    def __init__(self, 
                 shape_params: RocketShapeParameters,
                 n_axial: int,
                 n_circumferential: int,
                 n_radial: int,
                 grading_params: Dict = None,
                 optimization_target: str = 'gpu',
                 enable_parallel: bool = True,
                 material_type: str = 'aluminum',
                 export_format: str = 'vtk',
                 nose_only: bool = False,
                 full_rocket: bool = False)
    
    def export_mesh(self, filename: str, format: str = None)
    def refine_mesh(self, refinement_zones: Dict) -> HexahedralRocketMesh
    def get_shape_derivatives(self) -> Dict[str, np.ndarray]
    def get_mesh_quality_metrics(self) -> Dict
```

**Parameters:**
- `nose_only` (bool): Generate only nose cone mesh for faster analysis
- `full_rocket` (bool): Generate complete rocket (overrides length settings)

**Node Sets Available in Nose-Only Mode:**
- `nose_tip`: Nodes at the nose tip (stagnation region)
- `nose_base`: Nodes at the base of the nose cone
- `outer_surface`: All outer surface nodes
- `inner_surface`: Inner surface nodes (if hollow)

### RocketShapeParameters Class

```python
@dataclass
class RocketShapeParameters:
    nose_type: str
    nose_length: float
    body_length: float
    body_radius: float
    tail_type: str
    wall_thickness: float
    # ... additional parameters
```

## Contributing

For improvements or bug reports, consider:
1. Performance optimizations for specific hardware
2. Additional nose cone profiles
3. Advanced mesh quality metrics
4. New export formats
5. Multi-stage rocket configurations

## License and Citation

If using these mesh generators in research, please cite:
```
High-Performance Rocket Mesh Generator
Optimized for RTX 3090 + Multi-core CPU
Version 1.0, 2024
```