# Simple Finite Element Simulation (1D Bar)

A small, recruiter-friendly FEM project that simulates deformation of a 1D bar under axial load using a basic finite element formulation.

## What It Does

- Simulates deformation of a 1D bar under load
- Uses the relationship `F = kx` (with sign based on direction)
- Computes nodal displacement and element stress

## Formula Used

`F = kx` and `k = EA / L`

Where:
- `F` = force
- `k` = stiffness
- `x` = displacement
- `E` = Young's modulus
- `A` = cross-sectional area
- `L` = bar length

## Features

- Input material stiffness (`E`)
- Apply external force (`F`)
- Compute displacement and stress
- Multiple nodal loads
- Per-element material properties (E and A)
- Optional displacement/stress plots

## Tech

- Python
- NumPy

## Quick Start

```bash
python main.py --length 1.0 --area 0.01 --young 200e9 --force 1000 --elements 4
```

## More Examples

```bash
# Multiple nodal loads (disable default end load with --force 0)
python main.py --elements 4 --force 0 --node-forces "1:500,4:1000"
```

```bash
# Variable material properties per element
python main.py --elements 4 --element-youngs "200e9,150e9,200e9,200e9" --element-areas "0.01,0.01,0.008,0.01"
```

```bash
# Show plots and save a copy
python main.py --plot --plot-file results.png
```

## Example Output

```
Nodes: 5 | Elements: 4
Node 0: u = 0.000000e+00 m
Node 1: u = 1.000000e-06 m
Node 2: u = 2.000000e-06 m
Node 3: u = 3.000000e-06 m
Node 4: u = 4.000000e-06 m
Max displacement: 4.000000e-06 m
Element stress (Pa): [1.0e+08 1.0e+08 1.0e+08 1.0e+08]
```

## Notes

- Boundary condition: node 0 is fixed (`u=0`)
- Default load applied at the last node (additional nodal loads supported)
- The solver assembles a global stiffness matrix and solves `K u = F`

## Tests

```bash
pytest -q
```
