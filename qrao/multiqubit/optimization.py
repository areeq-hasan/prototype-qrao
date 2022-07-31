from typing import Callable, List

from qiskit.algorithms import MinimumEigensolver

from .encoding.problem import QuadraticProgramEncoding
from .rounding import semideterministic_round


def find_optimal_configuration(
    encoding: QuadraticProgramEncoding,
    minimum_eigensolver: MinimumEigensolver,
    round_variable_values: Callable[[List[float]], List[int]] = semideterministic_round,
):
    return round_variable_values(
        [
            value[0]
            for value in minimum_eigensolver.compute_minimum_eigenvalue(
                encoding.operator.to_matrix_op(),
                aux_operators=[
                    variable.padded_operator.to_matrix_op()
                    for variable in encoding.variables
                ],
            ).aux_operator_eigenvalues
        ]
    )
