from itertools import product

import numpy as np

from qiskit.algorithms.minimum_eigen_solvers import NumPyMinimumEigensolver

from qiskit_optimization.problems.quadratic_program import QuadraticProgram

from qrao.multiqubit.encoding.problem import encode_problem, QuadraticProgramEncoding
from qrao.multiqubit.encoding.configuration import encode_configuration
from qrao.multiqubit.optimization import find_optimal_configuration

from qrao.utils import get_random_maxcut_qp


def verify_commutation(
    problem: QuadraticProgram,
    encoding: QuadraticProgramEncoding,
    num_variables: int,
):
    for configuration in product([0, 1], repeat=num_variables):
        state = encode_configuration(list(configuration), encoding.partitions)
        encoding_eval = np.real(state.primitive.expectation_value(encoding.operator))
        objective_eval = (
            problem.objective.evaluate(list(configuration))
            * problem.objective.sense.value
        )
        assert np.isclose(encoding_eval, objective_eval)


# 6 11ps
def test_11p():
    problem = get_random_maxcut_qp(degree=1, num_nodes=6, seed=1)
    encoding = encode_problem(
        problem, max_qubits_per_partition=1, max_variables_per_partition=1
    )
    verify_commutation(problem, encoding, 6)


# 3 21ps
def test_21p():
    problem = get_random_maxcut_qp(degree=2, num_nodes=6, seed=1)
    encoding = encode_problem(
        problem, max_qubits_per_partition=1, max_variables_per_partition=2
    )
    verify_commutation(problem, encoding, 6)


# 2 31ps
def test_31p():
    problem = get_random_maxcut_qp(degree=1, num_nodes=6, seed=1)
    encoding = encode_problem(
        problem, max_qubits_per_partition=1, max_variables_per_partition=3
    )
    verify_commutation(problem, encoding, 6)


# 21p 21p 11p
def test_adaptive_11p_21p():
    problem = get_random_maxcut_qp(degree=2, num_nodes=5, seed=1)
    encoding = encode_problem(
        problem, max_qubits_per_partition=1, max_variables_per_partition=2
    )
    verify_commutation(problem, encoding, 5)


# 31p 31p 11p
def test_adaptive_11p_31p():
    problem = get_random_maxcut_qp(degree=2, num_nodes=7, seed=1)
    encoding = encode_problem(
        problem, max_qubits_per_partition=1, max_variables_per_partition=3
    )
    verify_commutation(problem, encoding, 7)


# def test_adaptive_21p_31p():
#     problem = get_random_maxcut_qp(degree=1, num_nodes=16, seed=1)

#     encoding = encode(
#         problem, max_qubits_per_partition=1, max_variables_per_partition=3
#     )

#     for dvar_values in product([0, 1], repeat=16):
#         dvar_values = list(dvar_values)
#         state = dvar_values_to_state(dvar_values, partitions)
#         encoding_eval = np.real(state.expectation_value(operator)) + offset
#         objective_eval = (
#             problem.objective.evaluate(dvar_values) * problem.objective.sense.value
#         )
#         assert np.isclose(encoding_eval, objective_eval)


# 2 32ps
def test_32p():
    problem = get_random_maxcut_qp(degree=1, num_nodes=6, seed=1)
    encoding = encode_problem(
        problem, max_qubits_per_partition=2, max_variables_per_partition=3
    )
    verify_commutation(problem, encoding, 6)


# 32p 32p 11p
def test_adaptive_11p_32p():
    problem = get_random_maxcut_qp(degree=2, num_nodes=7, seed=1)
    encoding = encode_problem(
        problem, max_qubits_per_partition=2, max_variables_per_partition=3
    )
    verify_commutation(problem, encoding, 7)


def test_find_optimal_configuration():
    problem = get_random_maxcut_qp(degree=1, num_nodes=6, seed=1)
    encoding = encode_problem(
        problem, max_qubits_per_partition=1, max_variables_per_partition=1
    )
    optimal_configuration = find_optimal_configuration(
        encoding, NumPyMinimumEigensolver()
    )
    fval = problem.objective.evaluate(optimal_configuration)
    print(fval)
