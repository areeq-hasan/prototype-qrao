from typing import List, Tuple

from dataclasses import dataclass
from functools import reduce

import numpy as np
import retworkx as rx

from qiskit.opflow import OperatorBase, StateFn, SummedOp, I
from qiskit_optimization.problems.quadratic_program import QuadraticProgram

from .partition_encoding import PARTITION_ENCODINGS


@dataclass
class Variable:
    index: int
    partition_index: int
    operator: OperatorBase


@dataclass
class Partition:
    variables: List[Variable]
    qubits: List[int]

    @property
    def num_variables(self):
        return len(self.variables)

    @property
    def num_qubits(self):
        return len(self.qubits)


def extract_terms(
    problem: QuadraticProgram, num_variables: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    sense: int = problem.objective.sense.value
    constant_term: float = problem.objective.constant * sense

    linear_terms: np.ndarray = np.zeros(num_variables)
    for i, coefficient in problem.objective.linear.to_dict().items():
        weight = coefficient * sense / 2
        linear_terms[i] -= weight
        constant_term += weight

    quadratic_terms: np.ndarray = np.zeros((num_variables, num_variables))
    for (i, j), coefficient in problem.objective.quadratic.to_dict().items():
        weight = coefficient * sense / 4
        if i == j:
            linear_terms[i] -= 2 * weight
            constant_term += 2 * weight
        else:
            quadratic_terms[i, j] += weight
            linear_terms[i] -= weight
            linear_terms[j] -= weight
            constant_term += weight

    return constant_term, linear_terms, quadratic_terms


def color(quadratic_terms: np.ndarray, num_variables: int) -> List[List[int]]:
    assert quadratic_terms.shape == (num_variables, num_variables)

    graph = rx.PyGraph()
    graph.add_nodes_from(range(num_variables))
    graph.add_edges_from_no_data(list(zip(*np.where(quadratic_terms != 0))))

    coloring = rx.graph_greedy_color(graph)
    return [
        [
            variable_idx
            for variable_idx, variable_color in coloring.items()
            if variable_color == color
        ]
        for color in range(max(coloring.values()) + 1)
    ]


def partition(
    coloring: List[List[int]],
    num_variables: int,
    max_qubits_per_partition: int,
    max_variables_per_partition: int,
) -> Tuple[List[Partition], List[Variable], int]:
    partitions = []
    variables = [None] * num_variables
    qubit_index = 0
    partition_index = 0
    for _, color_variable_indices in enumerate(coloring):
        for color_partition_index in range(
            0, len(color_variable_indices), max_variables_per_partition
        ):
            partition_variable_indices = color_variable_indices[
                color_partition_index : color_partition_index
                + max_variables_per_partition
            ]
            num_partition_variables = len(partition_variable_indices)
            num_partition_qubits = max(
                1, min(max_qubits_per_partition, num_partition_variables - 1)
            )
            partition_variables = []
            for variable_partition_index, variable_index in enumerate(
                partition_variable_indices
            ):
                var = Variable(
                    index=variable_index,
                    partition_index=partition_index,
                    operator=PARTITION_ENCODINGS[num_partition_variables][
                        num_partition_qubits
                    ].operators[variable_partition_index],
                )
                variables[variable_index] = var
                partition_variables.append(var)
            partition_qubits = list(
                range(qubit_index, qubit_index + num_partition_qubits)
            )
            partitions.append(Partition(partition_variables, partition_qubits))
            qubit_index += num_partition_qubits
            partition_index += 1
    num_qubits = qubit_index
    return partitions, variables, num_qubits


def pad_variable_operator(
    operator: OperatorBase, partition_qubits: List[int], num_qubits: int
) -> OperatorBase:
    return (
        (I ^ partition_qubits[0])
        ^ (operator)
        ^ (I ^ (num_qubits - partition_qubits[0] - len(partition_qubits)))
    )


def generate_problem_operator(
    variables: List[Variable],
    partitions: List[Partition],
    linear_terms: np.ndarray,
    quadratic_terms: np.ndarray,
    num_qubits: int,
) -> OperatorBase:
    operator_terms = []
    for variable_index, variable in enumerate(variables):
        coefficient = linear_terms[variable_index]
        if coefficient != 0:
            variable_partition = partitions[variable.partition_index]
            normalization = np.sqrt(variable_partition.num_variables)
            operator = pad_variable_operator(
                variable.operator, variable_partition.qubits, num_qubits
            )
            operator_terms.append(coefficient * normalization * operator)
    for variable_i_index, variable_i in enumerate(variables):
        for variable_j_index, variable_j in enumerate(variables):
            coefficient = quadratic_terms[variable_i_index][variable_j_index]
            if coefficient != 0:
                variable_i_partition = partitions[variable_i.partition_index]
                variable_j_partition = partitions[variable_j.partition_index]
                normalization = np.prod(
                    [
                        partition.num_variables ** (1 / (2**partition.num_qubits))
                        for partition in (variable_i_partition, variable_j_partition)
                    ]
                )
                operator = pad_variable_operator(
                    variable_i.operator, variable_i_partition.qubits, num_qubits
                ) @ pad_variable_operator(
                    variable_j.operator, variable_j_partition.qubits, num_qubits
                )
                operator_terms.append(coefficient * normalization * operator)
    return SummedOp(operator_terms)


def encode_problem(
    problem: QuadraticProgram,
    max_qubits_per_partition: int = 2,
    max_variables_per_partition: int = 3,
):
    num_variables = problem.get_num_vars()
    constant_term, linear_terms, quadratic_terms = extract_terms(problem, num_variables)
    coloring = color(quadratic_terms, num_variables)
    partitions, variables, num_qubits = partition(
        coloring, num_variables, max_qubits_per_partition, max_variables_per_partition
    )
    operator = generate_problem_operator(
        variables,
        partitions,
        linear_terms,
        quadratic_terms,
        num_qubits,
    )
    return operator, partitions, constant_term


def encode_partition_configuration(
    partition: Partition, configuration: List[int]
) -> StateFn:
    state_index = 0
    for value in configuration:
        state_index = (state_index << 1) | value
    return PARTITION_ENCODINGS[partition.num_variables][partition.num_qubits].states[
        state_index
    ]


def encode_configuration(
    configuration: List[int], partitions: List[Partition]
) -> StateFn:
    encoded_partition_configurations = [
        encode_partition_configuration(
            partition,
            [configuration[variables.index] for variables in partition.variables],
        )
        for partition in partitions
    ]
    return reduce(lambda x, y: x ^ y, encoded_partition_configurations)
