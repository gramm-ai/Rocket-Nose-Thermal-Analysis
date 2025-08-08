# Rocket Thermal Simulation Suite

High-performance simulation framework for analyzing thermal behavior of rocket nose geometries under atmospheric ascent conditions. Designed for RTX 3090-class GPUs, with support for parallel simulations, real-time monitoring, and high-fidelity visualization.

---

## Key Features

- GPU-accelerated thermal FEA engine (conduction, convection, radiation, aerodynamic heating)
- High-quality hexahedral mesh generation for rocket structures
- Parallelized simulation for 6 distinct nose profiles
- 3D visualization of heat propagation and temperature distribution
- Falcon 9-inspired thermal-equivalent modeling

---

## Project Structure

| File | Description |
|------|-------------|
| `rocket_physics.py` | Core GPU-accelerated FEA engine |
| `rocket_mesh_hex.py` | Hexahedral mesh generator for rocket shapes |
| `create_rocket_noses.py` | Script to generate 6 rocket nose designs |
| `rocket_simulation.py` | Simulation manager (parallel execution) |
| `rocket_visualization.py` | 3D visualization during and after simulation |

---

## Rocket Nose Profiles

The following six geometries are included for comparative analysis:

1. Conical
2. Ogive (Falcon 9)
3. Von Karman
4. Parabolic
5. Elliptical
6. Power Series (n=0.75)

These are generated using thermal-equivalent wall thickness modeling.

**Nose Profile Meshes**

<img width="980" height="691" alt="image" src="https://github.com/user-attachments/assets/bcf83fa1-0d3c-4ff8-8d6d-544bba31e334" />


---

## Real-Time 3D Heat Distribution

The simulation features a dashboard showing:

- 2x3 grid layout of temperature profiles
- Color maps indicating nose-to-base thermal gradients
- Time, altitude, velocity, and Mach overlays

**3D Thermal Visualization Dashboard**

<img width="1558" height="1237" alt="image" src="https://github.com/user-attachments/assets/ab8d59d7-9ebb-481e-a529-7d025132e2d6" />


---

## Workflow Overview

1. **Mesh Generation**: Uses `create_rocket_noses.py` to generate parameterized, thermal-equivalent meshes.
2. **Physics Engine**: Loads mesh into `RocketPhysicsEngine`, configures simulation on GPU.
3. **Simulation Execution**: Launches 6 processes in parallel, each simulating one nose design.
4. **Visualization**: Produces interactive or saved 3D heat maps using `rocket_visualization.py`.

---

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- NumPy, SciPy, Matplotlib
- Optional: TkAgg or Qt5Agg for interactive visualization

---

## Hardware Recommendations

- GPU: NVIDIA RTX 3090 (24 GB)
- CPU: 24-core for parallel simulation
- RAM: 32 GB+

---

## Output Directory Structure

```
simulation_6profiles_YYYYMMDD/
├── meshes/
│   └── <nose mesh files>
├── results/
│   ├── conical.json
│   ├── ogive_falcon.json
│   └── ...
├── plots/
│   └── final_3d_visualization.png
└── logs/
    └── simulation.log
```

---

## License

This project is provided for research and educational use. Contact the authors for commercial licensing.
