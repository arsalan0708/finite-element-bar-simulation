import numpy as np

from main import (
    BarModel,
    assemble_global_stiffness,
    build_uniform_model,
    compute_element_stress,
    solve_displacement,
)


def test_uniform_bar_end_load_matches_analytic() -> None:
    model = build_uniform_model(
        length=2.0,
        area=0.5,
        young=100.0,
        force=10.0,
        elements=4,
    )
    displacement = solve_displacement(model)
    x = np.linspace(0.0, model.length, model.nodes)
    expected = model.nodal_forces[-1] * x / (model.element_youngs[0] * model.element_areas[0])
    assert np.allclose(displacement, expected)


def test_uniform_stress_constant() -> None:
    model = build_uniform_model(
        length=2.0,
        area=0.5,
        young=100.0,
        force=10.0,
        elements=4,
    )
    displacement = solve_displacement(model)
    stress = compute_element_stress(model, displacement)
    expected = model.nodal_forces[-1] / model.element_areas[0]
    assert np.allclose(stress, expected)


def test_variable_element_stiffness_assembly() -> None:
    length = 2.0
    elements = 2
    element_length = length / elements
    element_areas = np.array([1.0, 2.0])
    element_youngs = np.array([3.0, 4.0])
    nodal_forces = np.zeros(elements + 1)

    model = BarModel(
        length=length,
        elements=elements,
        element_areas=element_areas,
        element_youngs=element_youngs,
        nodal_forces=nodal_forces,
    )

    k1 = element_youngs[0] * element_areas[0] / element_length
    k2 = element_youngs[1] * element_areas[1] / element_length
    expected = np.array(
        [
            [k1, -k1, 0.0],
            [-k1, k1 + k2, -k2],
            [0.0, -k2, k2],
        ]
    )

    k_global = assemble_global_stiffness(model)
    assert np.allclose(k_global, expected)


def test_multiple_nodal_forces_solution() -> None:
    model = build_uniform_model(
        length=2.0,
        area=1.0,
        young=1.0,
        force=0.0,
        elements=2,
        extra_nodal_forces=np.array([0.0, 1.0, 1.0]),
    )
    displacement = solve_displacement(model)
    expected = np.array([0.0, 2.0, 3.0])
    assert np.allclose(displacement, expected)
