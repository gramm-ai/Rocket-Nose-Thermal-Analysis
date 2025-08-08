# Rocket Nose Cone Thermal Analysis System

## GPU-Accelerated Finite Element Analysis for Aerospace Thermal Protection

![Project Status](https://img.shields.io/badge/Status-Production-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Physics Simulation](#physics-simulation)
- [Implementation Details](#implementation-details)
- [Performance Optimization](#performance-optimization)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Analysis](#results--analysis)
- [System Requirements](#system-requirements)
- [Contributing](#contributing)

---

## 🚀 Overview

A high-performance thermal analysis system for rocket nose cone geometries, implementing GPU-accelerated finite element analysis (FEA) to simulate aerodynamic heating during atmospheric flight. The system evaluates six different nose cone profiles in parallel, providing comparative thermal performance metrics critical for aerospace design decisions.

### Key Features

- **Parallel Processing**: 6 simultaneous simulations optimized for multi-core CPUs
- **GPU Acceleration**: CUDA-enabled FEA using PyTorch with RTX 3090 optimization
- **Real-time Visualization**: 3D heat distribution with live updates
- **Hexahedral Mesh Generation**: Pure hex elements for superior thin-wall analysis
- **Physics-Based Modeling**: Comprehensive heat transfer including conduction, convection, and radiation

### Applications

- Thermal protection system (TPS) design
- Nose cone geometry optimization
- Material selection for heat shields
- Reentry vehicle analysis
- Launch vehicle fairing design

---

## 🏗️ Technical Architecture

### System Components

```
┌──────────────────────────────────────────────────────────┐
│                    Simulation Manager                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │           Mesh Generation Pipeline                 │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │  Conical   │  │   Ogive    │  │ Von Karman │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │ Parabolic  │  │ Elliptical │  │ Power 0.75 │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  └────────────────────────────────────────────────────┘  │
│                              ↓                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │          GPU-Accelerated Physics Engine            │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │   Aerodynamic    │   Heat      │  Radiation  │  │  │
│  │  │     Heating      │ Conduction  │   Cooling   │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
│                              ↓                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │           3D Visualization & Analysis              │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Module Structure

| Module | Description | Primary Functions |
|--------|-------------|-------------------|
| `rocket_simulation.py` | Main orchestrator | Process management, result aggregation |
| `rocket_mesh_hex.py` | Mesh generation | Hexahedral element creation, node mapping |
| `rocket_physics.py` | FEA solver | Heat transfer equations, GPU kernels |
| `rocket_visualization.py` | 3D rendering | Real-time heat distribution display |
| `create_rocket_noses.py` | Geometry definition | Profile generation, parameterization |

---

## 🔬 Physics Simulation

### Heat Transfer Mechanisms

#### 1. Aerodynamic Heating (Stagnation Point)
```
q_aero = h_conv × (T_recovery - T_wall)

where:
  T_recovery = T_∞ × (1 + r × (γ-1)/2 × M²)
  h_conv = 10 × Re^0.5 × ρ^0.5
```

**[INSERT DIAGRAM: Stagnation point heating illustration showing airflow, shock wave, and temperature distribution]**

#### 2. Heat Conduction (Fourier's Law)
```
∂T/∂t = α × ∇²T

where:
  α = k/(ρ×cp) = thermal diffusivity
  k = thermal conductivity (120 W/m·K for Al-Li 2195)
```

#### 3. Radiation Cooling (Stefan-Boltzmann)
```
q_rad = ε × σ × (T⁴ - T_sky⁴)

where:
  ε = 0.8 (surface emissivity)
  σ = 5.67×10⁻⁸ W/m²·K⁴
```

### Finite Element Formulation

#### Hexahedral Element Advantages
- **Reduced Locking**: Superior performance for thin-wall structures
- **Orthogonal Heat Flow**: Better accuracy for thermal gradients
- **Computational Efficiency**: 8 nodes vs 10 for tetrahedra


<img width="980" height="691" alt="image" src="https://github.com/user-attachments/assets/e5071305-07b5-4bcd-ac04-f7e7e434ddd1" />


#### Shape Functions (Trilinear)
```python
N_i(ξ,η,ζ) = 1/8 × (1 + ξ_i×ξ) × (1 + η_i×η) × (1 + ζ_i×ζ)
```

### Flight Profile Simulation

| Phase | Time (s) | Altitude (km) | Velocity (m/s) | Mach | Max Temp (°C) |
|-------|----------|---------------|----------------|------|---------------|
| Launch | 0 | 0 | 0 | 0.0 | 15 |
| Max-Q | 60 | 26.4 | 850 | 2.5 | 180 |
| Stage Sep | 120 | 86.4 | 1800 | 6.0 | 260 |
| MECO | 160 | 142.4 | 2000 | 7.5 | 280 |

---

## 💻 Implementation Details

### GPU Optimization Strategies

#### Memory Management
```python
# Optimized tensor allocation
self.temperature = torch.zeros(n_nodes, dtype=torch.float32, device='cuda')
self.heat_flux = torch.zeros(n_nodes, dtype=torch.float32, device='cuda')

# Batch processing for large meshes
batch_size = min(1024, n_elements)
for i in range(0, n_elements, batch_size):
    process_batch(elements[i:i+batch_size])
```

#### CUDA Kernel Efficiency
- **Warp Alignment**: Circumferential divisions = 32 (warp size)
- **Memory Coalescing**: Sequential node access patterns
- **Tensor Cores**: Mixed precision (FP16/FP32) on RTX 3090

### Parallel Processing Architecture

```python
# 6 parallel simulations with resource management
n_processes = min(
    n_simulations,           # 6 profiles
    max_gpu_simulations,     # GPU memory limit
    optimal_parallel         # CPU core limit
)
```

### Mesh Generation Pipeline

#### Profile Parameters
| Profile | Nose Length (m) | Shape Factor | Drag Coefficient |
|---------|----------------|--------------|------------------|
| Conical | 5.0 | Linear | 0.50 |
| Ogive | 6.5 | Haack C=0 | 0.42 |
| Von Karman | 6.0 | Haack C=1/3 | 0.38 |
| Parabolic | 5.5 | K=0.5 | 0.45 |
| Elliptical | 4.0 | 3:4 ratio | 0.48 |
| Power 0.75 | 5.5 | n=0.75 | 0.44 |

#### Mesh Statistics
```
Medium Resolution:
- Nodes: ~15,000 per profile
- Elements: ~12,000 hexahedra
- DOF: ~45,000 (3 per node)
- Memory: ~3.5 GB GPU per simulation
```

---

## ⚡ Performance Optimization

### Computational Benchmarks

| Configuration | Time/Simulation | Speedup | Efficiency |
|--------------|-----------------|---------|------------|
| CPU Serial | 180s | 1.0x | 100% |
| CPU Parallel (6 cores) | 35s | 5.1x | 85% |
| GPU Single | 45s | 4.0x | - |
| GPU + CPU Parallel | 12s | 15.0x | 83% |

---

## 📦 Installation

### Prerequisites
```bash
# System requirements
- Python 3.12+
- CUDA 11.8+ (for GPU acceleration)
- 32GB RAM recommended
- RTX 3090 or equivalent (optional)
```

### Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/rocket-thermal-analysis.git
cd rocket-thermal-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
torch>=2.0.0+cu118
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
psutil>=5.9.0
```

---

## 🎮 Usage

### Basic Simulation
```bash
# Run with default settings (6 parallel, GPU-enabled)
python rocket_simulation.py

# Custom configuration
python rocket_simulation.py \
    --simulation-time 120 \
    --mesh-resolution fine \
    --output-dir results/
```

### Command-Line Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--simulation-time` | 60.0 | Total simulation time (seconds) |
| `--mesh-resolution` | medium | Mesh quality: coarse/medium/fine |
| `--sequential` | False | Disable parallel processing |
| `--no-viz` | False | Disable 3D visualization |
| `--output-dir` | auto | Results directory path |

### Programmatic Interface
```python
from rocket_simulation import SimulationManager

# Initialize manager
manager = SimulationManager(
    mesh_resolution='fine',
    simulation_time=160.0,
    parallel_mode=True,
    visualization_mode=True
)

# Run simulations
manager.run()

# Access results
results = manager.results
for profile, data in results.items():
    print(f"{profile}: Max temp = {data['max_temperature']:.1f}°C")
```

---

## 📊 Results & Analysis

### Thermal Performance Ranking

**[INSERT DIAGRAM: 3D heat distribution visualization showing all 6 profiles]**

| Rank | Profile | Max Temperature | Final Temperature | Thermal Efficiency |
|------|---------|----------------|-------------------|-------------------|
| 1 | Elliptical | 259.3°C | 227.7°C | Best |
| 2 | Ogive (F9) | 259.4°C | 222.7°C | Excellent |
| 3 | Parabolic | 259.6°C | 226.1°C | Good |
| 4 | Von Karman | 260.0°C | 223.9°C | Good |
| 5 | Power 0.75 | 261.0°C | 227.5°C | Fair |
| 6 | Conical | 261.3°C | 229.9°C | Baseline |

### Heat Distribution Patterns

**[INSERT DIAGRAM: Temperature contour plots for each profile at t=60s]**

#### Key Observations
1. **Stagnation Point Heating**: Maximum at nose tip (0,0,0)
2. **Circumferential Variation**: 15% temperature difference windward/leeward
3. **Axial Gradient**: Exponential decay from tip to base
4. **Profile Impact**: Blunter shapes distribute heat more effectively

### Output Files
```
simulation_results/
├── meshes/
│   ├── mesh_6profiles.json
│   └── *.vtk (mesh files)
├── results/
│   ├── *_result.npz (simulation data)
│   └── *_result.json (summaries)
├── plots/
│   ├── heat_distribution_3d_*.png
│   └── visualization_report.json
└── logs/
    └── simulation_6profiles_*.log
```

---

## 💻 System Requirements

### Minimum Configuration
- **CPU**: 6-core processor (Intel i5-9600K or AMD Ryzen 5 3600)
- **RAM**: 16GB DDR4
- **GPU**: GTX 1660 (6GB VRAM) - optional
- **Storage**: 10GB free space

### Recommended Configuration
- **CPU**: 24-core processor (Intel i9-13900K or AMD Threadripper)
- **RAM**: 32GB DDR5
- **GPU**: RTX 3090 (24GB VRAM)
- **Storage**: 50GB SSD

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📚 References

1. Anderson, J.D. (2006). *Hypersonic and High-Temperature Gas Dynamics*
2. Bathe, K.J. (2014). *Finite Element Procedures*
3. NASA Technical Report: *Nose Cone Design Optimization* (NASA-TM-2018)
4. SpaceX Falcon 9 User's Guide (2021)

---

## 🙏 Acknowledgments

- NASA Ames Research Center for aerodynamic heating correlations
- PyTorch team for GPU acceleration framework
- SpaceX for Falcon 9 reference geometry
- Open-source FEA community

---

**Contact**: [jisoo@gramm.ai](mailto:jisoo@gramm.ai)  
**Project Link**: [https://github.com/yourusername/rocket-thermal-analysis](https://github.com/yourusername/rocket-thermal-analysis)
