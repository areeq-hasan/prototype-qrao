from typing import Tuple, List

from functools import reduce
from dataclasses import dataclass

import numpy as np
import retworkx as rx

from qiskit.opflow import OperatorBase, MatrixOp, SummedOp, I, X, Y, Z
from qiskit.quantum_info import Statevector
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


p = 1.0 / 2.0 + np.sqrt(6) / 6.0

measures = [
    [
        np.array([np.sqrt(p), 0, np.sqrt((1 - p) / 2.0), np.sqrt((1 - p) / 2.0)]),
        np.array([0, -np.sqrt(p), -np.sqrt((1 - p) / 2.0), np.sqrt((1 - p) / 2.0)]),
    ],
    [
        np.array([-np.sqrt(p), -np.sqrt((1 - p) / 2.0), 0, np.sqrt((1 - p) / 2.0)]),
        np.array([0, np.sqrt((1 - p) / 2.0), np.sqrt(p), np.sqrt((1 - p) / 2.0)]),
    ],
    [
        np.array([-np.sqrt(p), np.sqrt((1 - p) / 2.0), np.sqrt((1 - p) / 2.0), 0]),
        np.array([0, -np.sqrt((1 - p) / 2.0), np.sqrt((1 - p) / 2.0), np.sqrt(p)]),
    ],
]

operators = [
    MatrixOp(np.outer(measure[0], measure[0]) - np.outer(measure[1], measure[1]))
    for measure in measures
]

ENCODING_TO_OPERATORS = {
    1: {1: [Z]},
    2: {1: [X, Z]},
    3: {1: [X, Y, Z], 2: operators},
}

ENCODING_TO_STATES = {
    1: {1: [Statevector(np.array([1.0, 0.0])), Statevector(np.array([0.0, 1.0]))]},
    2: {
        1: [
            [
                Statevector(np.array([-np.sin(np.pi / 8), np.cos(np.pi / 8)])),
                Statevector(np.array([np.cos(np.pi / 8), -np.sin(np.pi / 8)])),
            ],
            [
                Statevector(np.array([np.sin(np.pi / 8), np.cos(np.pi / 8)])),
                Statevector(np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)])),
            ],
        ]
    },
    3: {
        1: [
            [
                [
                    Statevector(
                        np.array(
                            [
                                -1 / np.sqrt(2 * (3 + np.sqrt(3)))
                                + 1j * 1 / np.sqrt(2 * (3 + np.sqrt(3))),
                                np.sqrt(1 - 1 / (3 + np.sqrt(3))),
                            ]
                        )
                    ),
                    Statevector(
                        np.array(
                            [
                                np.sqrt(1 - 1 / (3 + np.sqrt(3))),
                                -1 / np.sqrt(2 * (3 + np.sqrt(3)))
                                - 1j * 1 / np.sqrt(2 * (3 + np.sqrt(3))),
                            ]
                        )
                    ),
                ],
                [
                    Statevector(
                        np.array(
                            [
                                -1 / np.sqrt(2 * (3 + np.sqrt(3)))
                                - 1j * 1 / np.sqrt(2 * (3 + np.sqrt(3))),
                                np.sqrt(1 - 1 / (3 + np.sqrt(3))),
                            ]
                        )
                    ),
                    Statevector(
                        np.array(
                            [
                                np.sqrt(1 - 1 / (3 + np.sqrt(3))),
                                -1 / np.sqrt(2 * (3 + np.sqrt(3)))
                                + 1j * 1 / np.sqrt(2 * (3 + np.sqrt(3))),
                            ]
                        )
                    ),
                ],
            ],
            [
                [
                    Statevector(
                        np.array(
                            [
                                1 / np.sqrt(2 * (3 + np.sqrt(3)))
                                + 1j * 1 / np.sqrt(2 * (3 + np.sqrt(3))),
                                np.sqrt(1 - 1 / (3 + np.sqrt(3))),
                            ]
                        )
                    ),
                    Statevector(
                        np.array(
                            [
                                np.sqrt(1 - 1 / (3 + np.sqrt(3))),
                                1 / np.sqrt(2 * (3 + np.sqrt(3)))
                                - 1j * 1 / np.sqrt(2 * (3 + np.sqrt(3))),
                            ]
                        )
                    ),
                ],
                [
                    Statevector(
                        np.array(
                            [
                                1 / np.sqrt(2 * (3 + np.sqrt(3)))
                                - 1j * 1 / np.sqrt(2 * (3 + np.sqrt(3))),
                                np.sqrt(1 - 1 / (3 + np.sqrt(3))),
                            ]
                        )
                    ),
                    Statevector(
                        np.array(
                            [
                                np.sqrt(1 - 1 / (3 + np.sqrt(3))),
                                1 / np.sqrt(2 * (3 + np.sqrt(3)))
                                + 1j * 1 / np.sqrt(2 * (3 + np.sqrt(3))),
                            ]
                        )
                    ),
                ],
            ],
        ],
        2: [
            [
                [
                    Statevector(np.array([1.0, 0.0, 0.0, 0.0])),
                    Statevector((1.0 / np.sqrt(3.0)) * np.array([1.0, 1.0, 1.0, 0.0])),
                ],
                [
                    Statevector((1.0 / np.sqrt(3.0)) * np.array([1.0, -1.0, 0.0, 1.0])),
                    Statevector(np.array([0.0, 1.0, 0.0, 0.0])),
                ],
            ],
            [
                [
                    Statevector((1.0 / np.sqrt(3.0)) * np.array([-1.0, 0.0, 1.0, 1.0])),
                    Statevector(np.array([0.0, 0.0, 1.0, 0.0])),
                ],
                [
                    Statevector(np.array([0.0, 0.0, 0.0, 1.0])),
                    Statevector((1.0 / np.sqrt(3.0)) * np.array([0.0, 1.0, -1.0, 1.0])),
                ],
            ],
        ],
    },
}


@dataclass
class DecisionVariable:
    idx: int
    partition_idx: int
    operator: OperatorBase


@dataclass
class Partition:
    color: int
    dvars: List[DecisionVariable]
    qubits: List[int]

    @property
    def num_dvars(self):
        return len(self.dvars)

    @property
    def num_qubits(self):
        return len(self.qubits)


def extract_terms(
    problem: QuadraticProgram, num_dvars: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    sense: int = problem.objective.sense.value
    constant_term: float = problem.objective.constant * sense

    linear_terms: np.ndarray = np.zeros(num_dvars)
    for index, coefficient in problem.objective.linear.to_dict().items():
        weight = coefficient * sense / 2
        linear_terms[index] -= weight
        constant_term += weight

    quadratic_terms: np.ndarray = np.zeros((num_dvars, num_dvars))
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


def color(quadratic_terms: np.ndarray, num_dvars: int) -> List[List[int]]:
    assert quadratic_terms.shape == (num_dvars, num_dvars)

    graph = rx.PyGraph()
    graph.add_nodes_from(range(num_dvars))
    graph.add_edges_from_no_data(list(zip(*np.where(quadratic_terms != 0))))

    coloring = rx.graph_greedy_color(graph)
    return [
        [dvar_idx for dvar_idx, dvar_color in coloring.items() if dvar_color == color]
        for color in range(max(coloring.values()) + 1)
    ]


def partition(
    coloring: List[List[int]],
    max_qubits_per_partition: int,
    max_dvars_per_partition: int,
) -> Tuple[List[Partition], List[DecisionVariable], int]:
    partitions = []
    dvars = []
    qubit_idx = 0
    partition_idx = 0
    for color, color_dvar_ids in enumerate(coloring):
        for color_partition_idx in range(
            0, len(color_dvar_ids), max_dvars_per_partition
        ):
            partition_dvar_ids = color_dvar_ids[
                color_partition_idx : color_partition_idx + max_dvars_per_partition
            ]
            num_partition_dvars = len(partition_dvar_ids)
            num_partition_qubits = max(
                1, min(max_qubits_per_partition, num_partition_dvars - 1)
            )
            partition_dvars = [
                DecisionVariable(
                    idx=dvar_idx,
                    partition_idx=partition_idx,
                    operator=ENCODING_TO_OPERATORS[num_partition_dvars][
                        num_partition_qubits
                    ][dvar_partition_idx],
                )
                for dvar_partition_idx, dvar_idx in enumerate(partition_dvar_ids)
            ]
            partition_qubits = list(range(qubit_idx, qubit_idx + num_partition_qubits))
            dvars.extend(partition_dvars)
            partitions.append(Partition(color, partition_dvars, partition_qubits))
            qubit_idx += num_partition_qubits
            partition_idx += 1
    num_qubits = qubit_idx
    dvars = sorted(dvars, key=lambda dvar: dvar.idx)
    return partitions, dvars, num_qubits


def pad_dvar_operator(
    dvar_operator: OperatorBase, partition_qubits: List[int], num_qubits: int
) -> OperatorBase:
    return (
        (I ^ partition_qubits[0])
        ^ (dvar_operator)
        ^ (I ^ (num_qubits - partition_qubits[0] - len(partition_qubits)))
    )


def generate_operator(
    dvars: List[DecisionVariable],
    partitions: List[Partition],
    linear_terms: np.ndarray,
    quadratic_terms: np.ndarray,
    num_dvars: int,
    num_qubits: int,
) -> OperatorBase:
    op_terms = []

    # Add linear terms
    for dvar_idx, dvar in enumerate(dvars):
        coefficient = linear_terms[dvar_idx]
        if coefficient != 0:
            dvar_partition = partitions[dvar.partition_idx]
            normalization = np.sqrt(dvar_partition.num_dvars)
            dvar_op_term = pad_dvar_operator(
                dvar.operator, dvar_partition.qubits, num_qubits
            )
            op_terms.append(coefficient * normalization * dvar_op_term)

    # Add quadratic terms
    for dvar_i_idx, dvar_i in enumerate(dvars):
        for dvar_j_idx, dvar_j in enumerate(dvars):
            coefficient = quadratic_terms[dvar_i_idx][dvar_j_idx]
            if coefficient != 0:
                dvar_i_partition = partitions[dvar_i.partition_idx]
                dvar_j_partition = partitions[dvar_j.partition_idx]
                normalization = np.sqrt(
                    dvar_i_partition.num_dvars * dvar_j_partition.num_dvars
                )
                dvar_pair_op_term = pad_dvar_operator(
                    dvar_i.operator, dvar_i_partition.qubits, num_qubits
                ) @ pad_dvar_operator(
                    dvar_j.operator, dvar_j_partition.qubits, num_qubits
                )
                op_terms.append(coefficient * normalization * dvar_pair_op_term)
    return SummedOp(op_terms)


def encode(
    problem: QuadraticProgram,
    max_qubits_per_partition: int = 2,
    max_dvars_per_partition: int = 3,
):
    num_dvars = problem.get_num_vars()
    constant_term, linear_terms, quadratic_terms = extract_terms(problem, num_dvars)
    coloring = color(quadratic_terms, num_dvars)
    partitions, dvars, num_qubits = partition(
        coloring, max_qubits_per_partition, max_dvars_per_partition
    )
    operator = generate_operator(
        dvars,
        partitions,
        linear_terms,
        quadratic_terms,
        num_dvars,
        num_qubits,
    )
    return operator, partitions, constant_term


def dvar_values_to_partition_state(
    partition: Partition, dvar_values: List[int]
) -> Statevector:
    state = ENCODING_TO_STATES[partition.num_dvars][partition.num_qubits]
    for dvar_value in dvar_values:
        state = state[dvar_value]
    return state


def dvar_values_to_state(
    dvar_values: List[int], partitions: List[Partition]
) -> Statevector:
    return reduce(
        lambda x, y: x ^ y,
        [
            dvar_values_to_partition_state(
                partition, [dvar_values[dvar.idx] for dvar in partition.dvars]
            )
            for partition in partitions
        ],
    )
