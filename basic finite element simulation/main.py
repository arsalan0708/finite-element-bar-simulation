import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BarModel:
    length: float
    elements: int
    element_areas: np.ndarray
    element_youngs: np.ndarray
    nodal_forces: np.ndarray

    @property
    def nodes(self) -> int:
        return self.elements + 1

    @property
    def element_length(self) -> float:
        return self.length / self.elements


def build_uniform_model(
    *,
    length: float,
    area: float,
    young: float,
    force: float,
    elements: int,
    extra_nodal_forces: Optional[np.ndarray] = None,
) -> BarModel:
    element_areas = np.full(elements, area, dtype=float)
    element_youngs = np.full(elements, young, dtype=float)
    nodal_forces = np.zeros(elements + 1, dtype=float)
    nodal_forces[-1] = force
    if extra_nodal_forces is not None:
        if len(extra_nodal_forces) != elements + 1:
            raise ValueError("extra_nodal_forces length must match number of nodes")
        nodal_forces += extra_nodal_forces
    return BarModel(
        length=length,
        elements=elements,
        element_areas=element_areas,
        element_youngs=element_youngs,
        nodal_forces=nodal_forces,
    )


def assemble_global_stiffness(model: BarModel) -> np.ndarray:
    size = model.nodes
    k_global = np.zeros((size, size), dtype=float)

    for e in range(model.elements):
        k_local = (
            model.element_youngs[e]
            * model.element_areas[e]
            / model.element_length
        ) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        n1 = e
        n2 = e + 1
        k_global[n1 : n2 + 1, n1 : n2 + 1] += k_local

    return k_global


def solve_displacement(model: BarModel) -> np.ndarray:
    k_global = assemble_global_stiffness(model)
    force_vec = model.nodal_forces

    fixed_dofs = [0]
    free_dofs = [i for i in range(model.nodes) if i not in fixed_dofs]

    k_ff = k_global[np.ix_(free_dofs, free_dofs)]
    f_f = force_vec[free_dofs]

    u_f = np.linalg.solve(k_ff, f_f)

    u = np.zeros(model.nodes, dtype=float)
    u[free_dofs] = u_f
    return u


def compute_element_stress(model: BarModel, displacement: np.ndarray) -> np.ndarray:
    stress = np.zeros(model.elements, dtype=float)
    for e in range(model.elements):
        n1 = e
        n2 = e + 1
        strain = (displacement[n2] - displacement[n1]) / model.element_length
        stress[e] = model.element_youngs[e] * strain
    return stress


def _parse_csv_floats(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return []
    return [float(part) for part in parts]


def _parse_element_list(value: str, expected: int, label: str) -> np.ndarray:
    values = _parse_csv_floats(value)
    if len(values) != expected:
        raise ValueError(f"{label} must have {expected} entries")
    return np.array(values, dtype=float)


def _parse_node_forces(value: str, nodes: int) -> np.ndarray:
    forces = np.zeros(nodes, dtype=float)
    parts = [part.strip() for part in value.split(",") if part.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError("node_forces must use the format index:force")
        index_str, force_str = part.split(":", 1)
        index = int(index_str.strip())
        force = float(force_str.strip())
        if index < 0 or index >= nodes:
            raise ValueError(f"node index {index} out of range")
        forces[index] += force
    return forces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1D bar FEM simulation.")
    parser.add_argument("--length", type=float, default=1.0, help="Bar length (m).")
    parser.add_argument("--area", type=float, default=0.01, help="Cross-sectional area (m^2).")
    parser.add_argument("--young", type=float, default=200e9, help="Young's modulus (Pa).")
    parser.add_argument("--force", type=float, default=1000.0, help="Force at last node (N).")
    parser.add_argument("--elements", type=int, default=4, help="Number of elements.")
    parser.add_argument(
        "--element-areas",
        type=str,
        default="",
        help="Comma-separated area values for each element (m^2).",
    )
    parser.add_argument(
        "--element-youngs",
        type=str,
        default="",
        help="Comma-separated Young's modulus values for each element (Pa).",
    )
    parser.add_argument(
        "--node-forces",
        type=str,
        default="",
        help="Comma-separated nodal forces like 'index:force,index:force' (N).",
    )
    parser.add_argument("--plot", action="store_true", help="Show displacement/stress plots.")
    parser.add_argument(
        "--plot-file",
        type=str,
        default="",
        help="Save plot image to a file (png, jpg, etc.).",
    )
    args = parser.parse_args()

    if args.length <= 0:
        raise ValueError("length must be positive")
    if args.elements < 1:
        raise ValueError("elements must be >= 1")
    if not args.element_areas and args.area <= 0:
        raise ValueError("area must be positive")
    if not args.element_youngs and args.young <= 0:
        raise ValueError("young must be positive")

    return args


def build_model(args: argparse.Namespace) -> BarModel:
    elements = args.elements
    if args.element_areas:
        element_areas = _parse_element_list(args.element_areas, elements, "element_areas")
    else:
        element_areas = np.full(elements, args.area, dtype=float)

    if args.element_youngs:
        element_youngs = _parse_element_list(args.element_youngs, elements, "element_youngs")
    else:
        element_youngs = np.full(elements, args.young, dtype=float)

    if np.any(element_areas <= 0):
        raise ValueError("all element areas must be positive")
    if np.any(element_youngs <= 0):
        raise ValueError("all element Young's modulus values must be positive")

    nodes = elements + 1
    nodal_forces = np.zeros(nodes, dtype=float)
    nodal_forces[-1] += args.force

    if args.node_forces:
        nodal_forces += _parse_node_forces(args.node_forces, nodes)

    return BarModel(
        length=args.length,
        elements=elements,
        element_areas=element_areas,
        element_youngs=element_youngs,
        nodal_forces=nodal_forces,
    )


def plot_results(
    model: BarModel,
    displacement: np.ndarray,
    stress: np.ndarray,
    *,
    show: bool,
    plot_file: str,
) -> None:
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, model.length, model.nodes)
    element_centers = 0.5 * (x[:-1] + x[1:])

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
    axes[0].plot(x, displacement, marker="o", linewidth=2)
    axes[0].set_title("Displacement")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("u (m)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(element_centers, stress, marker="s", linewidth=2)
    axes[1].set_title("Element Stress")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("stress (Pa)")
    axes[1].grid(True, alpha=0.3)

    if plot_file:
        fig.savefig(plot_file, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model = build_model(args)
    displacement = solve_displacement(model)
    stress = compute_element_stress(model, displacement)

    print(f"Nodes: {model.nodes} | Elements: {model.elements}")
    for i, u in enumerate(displacement):
        print(f"Node {i}: u = {u:.6e} m")
    print(f"Max displacement: {np.max(np.abs(displacement)):.6e} m")
    print("Element stress (Pa):", np.array2string(stress, precision=3, suppress_small=False))
    if args.plot or args.plot_file:
        plot_results(
            model,
            displacement,
            stress,
            show=args.plot,
            plot_file=args.plot_file,
        )


if __name__ == "__main__":
    main()
