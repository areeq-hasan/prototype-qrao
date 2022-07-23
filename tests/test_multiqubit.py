from itertools import product

import numpy as np

from qrao.utils import get_random_maxcut_qp

from qrao.multiqubit import encode, dvar_values_to_state

from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp


def test_11p():
    problem = get_random_maxcut_qp(degree=1, num_nodes=6, seed=1)

    operator, partitions, offset = encode(
        problem, max_qubits_per_partition=1, max_dvars_per_partition=1
    )

    for dvar_values in product([0, 1], repeat=6):
        dvar_values = list(dvar_values)
        state = dvar_values_to_state(dvar_values, partitions)
        encoding_eval = np.real(state.expectation_value(operator)) + offset
        objective_eval = (
            problem.objective.evaluate(dvar_values) * problem.objective.sense.value
        )
        assert np.isclose(encoding_eval, objective_eval)


def test_21p():
    problem = get_random_maxcut_qp(degree=2, num_nodes=6, seed=1)

    operator, partitions, offset = encode(
        problem, max_qubits_per_partition=1, max_dvars_per_partition=2
    )

    for dvar_values in product([0, 1], repeat=6):
        dvar_values = list(dvar_values)
        state = dvar_values_to_state(dvar_values, partitions)
        encoding_eval = np.real(state.expectation_value(operator)) + offset
        objective_eval = (
            problem.objective.evaluate(dvar_values) * problem.objective.sense.value
        )
        assert np.isclose(encoding_eval, objective_eval)


def test_31p():
    problem = get_random_maxcut_qp(degree=1, num_nodes=6, seed=1)

    operator, partitions, offset = encode(
        problem, max_qubits_per_partition=1, max_dvars_per_partition=3
    )

    for dvar_values in product([0, 1], repeat=6):
        dvar_values = list(dvar_values)
        state = dvar_values_to_state(dvar_values, partitions)
        encoding_eval = np.real(state.expectation_value(operator)) + offset
        objective_eval = (
            problem.objective.evaluate(dvar_values) * problem.objective.sense.value
        )
        assert np.isclose(encoding_eval, objective_eval)


def test_adaptive_11p_21p():
    problem = get_random_maxcut_qp(degree=2, num_nodes=5, seed=1)

    operator, partitions, offset = encode(
        problem, max_qubits_per_partition=1, max_dvars_per_partition=2
    )

    for dvar_values in product([0, 1], repeat=5):
        dvar_values = list(dvar_values)
        state = dvar_values_to_state(dvar_values, partitions)
        encoding_eval = np.real(state.expectation_value(operator)) + offset
        objective_eval = (
            problem.objective.evaluate(dvar_values) * problem.objective.sense.value
        )
        assert np.isclose(encoding_eval, objective_eval)


def test_adaptive_11p_31p():
    problem = get_random_maxcut_qp(degree=2, num_nodes=7, seed=1)

    operator, partitions, offset = encode(
        problem, max_qubits_per_partition=1, max_dvars_per_partition=3
    )

    for dvar_values in product([0, 1], repeat=7):
        dvar_values = list(dvar_values)
        state = dvar_values_to_state(dvar_values, partitions)
        encoding_eval = np.real(state.expectation_value(operator)) + offset
        objective_eval = (
            problem.objective.evaluate(dvar_values) * problem.objective.sense.value
        )
        assert np.isclose(encoding_eval, objective_eval)


# def test_adaptive_21p_31p():
#     problem = get_random_maxcut_qp(degree=1, num_nodes=16, seed=1)

#     operator, partitions, offset = encode(
#         problem, max_qubits_per_partition=1, max_dvars_per_partition=3
#     )

#     for dvar_values in product([0, 1], repeat=16):
#         dvar_values = list(dvar_values)
#         state = dvar_values_to_state(dvar_values, partitions)
#         encoding_eval = np.real(state.expectation_value(operator)) + offset
#         objective_eval = (
#             problem.objective.evaluate(dvar_values) * problem.objective.sense.value
#         )
#         assert np.isclose(encoding_eval, objective_eval)


def test_32p():
    problem = get_random_maxcut_qp(degree=1, num_nodes=6, seed=1)

    operator, partitions, offset = encode(
        problem, max_qubits_per_partition=2, max_dvars_per_partition=3
    )
    for dvar_values in product([0, 1], repeat=6):
        dvar_values = list(dvar_values)
        state = dvar_values_to_state(dvar_values, partitions)
        encoding_eval = np.real(state.expectation_value(operator)) + offset
        objective_eval = (
            problem.objective.evaluate(dvar_values) * problem.objective.sense.value
        )
        assert np.isclose(encoding_eval, objective_eval)


def test_adaptive_11p_32p():
    problem = get_random_maxcut_qp(degree=2, num_nodes=7, seed=1)

    operator, partitions, offset = encode(
        problem, max_qubits_per_partition=2, max_dvars_per_partition=3
    )
    for dvar_values in product([0, 1], repeat=7):
        dvar_values = list(dvar_values)
        state = dvar_values_to_state(dvar_values, partitions)
        encoding_eval = np.real(state.expectation_value(operator)) + offset
        objective_eval = (
            problem.objective.evaluate(dvar_values) * problem.objective.sense.value
        )
        assert np.isclose(encoding_eval, objective_eval)
