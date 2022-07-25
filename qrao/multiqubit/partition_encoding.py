from typing import List

import warnings

import numpy as np

from qiskit.opflow import StateFn, OperatorBase, X, Y, Z, MatrixOp


class PartitionEncoding:
    def __init__(self, operators: List[OperatorBase], states: List[StateFn]):
        num_vars: int = len(operators)
        if num_vars < 1:
            raise ValueError("At least one variable must be encoded.")

        num_qubits: int = operators[0].num_qubits
        if num_vars < 1:
            raise ValueError("At least one qubit must be encoded onto.")

        if num_qubits > num_vars:
            warnings.warn("More qubits than variables.")
        simplification_factor = np.gcd(num_vars, num_qubits)
        if simplification_factor != 1:
            warnings.warn(
                f"Encoding can be simplified to represent {num_vars/simplification_factor}"
                f" using {num_qubits/simplification_factor}."
            )
        if num_vars > 4 ** (num_qubits) - 1:
            warnings.warn(
                f"{num_qubits} qubits can encode a maximum of {4**(num_qubits) - 1} variables."
            )

        if len(operators) != num_vars:
            raise ValueError("Each operator must correspond to a variable.")
        if not all(operator.num_qubits == num_qubits for operator in operators):
            raise ValueError(f"All operators must act on {num_qubits} qubits.")

        if len(states) != 2**num_vars:
            raise ValueError(
                "Each state must correspond to an configuration of variables."
            )
        if not all(state.num_qubits == num_qubits for state in states):
            raise ValueError(f"All states must represent {num_qubits} qubits.")

        for var_idx, operator in enumerate(operators):
            for configuration_idx, state in enumerate(states):
                assert np.isclose(
                    state.primitive.expectation_value(operator),
                    (1 / np.sqrt(num_vars))
                    * (
                        -1
                        if (configuration_idx & (1 << (num_vars - 1 - var_idx)))
                        else 1
                    ),
                )

        self._num_vars: int = num_vars
        self._num_qubits: int = num_qubits
        self._operators: List[OperatorBase] = operators
        self._states: List[StateFn] = states

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def compression_ratio(self) -> float:
        return self._num_vars / self._num_qubits

    @property
    def operators(self) -> List[OperatorBase]:
        return self._operators

    @property
    def states(self) -> List[StateFn]:
        return self._states


AMPLITUDE_CACHE = {
    1: {1: []},
    2: {1: [np.cos(np.pi / 8), np.sin(np.pi / 8)]},
    3: {
        1: [
            np.sqrt(1 - 1 / (3 + np.sqrt(3))),
            1 / np.sqrt(2 * (3 + np.sqrt(3))),
            1 / np.sqrt(2 * (3 - np.sqrt(3))),
            1 / np.sqrt(3 + np.sqrt(3)),
        ],
        2: [1 / np.sqrt(3)],
    },
}

PARTITION_ENCODINGS = {
    1: {
        1: PartitionEncoding(
            operators=[Z], states=[StateFn(np.array([1, 0])), StateFn(np.array([0, 1]))]
        )
    },
    2: {
        1: PartitionEncoding(
            operators=[X, Z],
            states=[
                StateFn(np.array([AMPLITUDE_CACHE[2][1][0], AMPLITUDE_CACHE[2][1][1]])),
                StateFn(np.array([AMPLITUDE_CACHE[2][1][1], AMPLITUDE_CACHE[2][1][0]])),
                StateFn(
                    np.array([-AMPLITUDE_CACHE[2][1][0], AMPLITUDE_CACHE[2][1][1]])
                ),
                StateFn(
                    np.array([-AMPLITUDE_CACHE[2][1][1], AMPLITUDE_CACHE[2][1][0]])
                ),
            ],
        )
    },
    3: {
        1: PartitionEncoding(
            operators=[X, Y, Z],
            states=[
                StateFn(
                    np.array(
                        [
                            AMPLITUDE_CACHE[3][1][0],
                            AMPLITUDE_CACHE[3][1][1] + AMPLITUDE_CACHE[3][1][1] * 1j,
                        ]
                    )
                ),
                StateFn(
                    np.array(
                        [
                            AMPLITUDE_CACHE[3][1][1] - AMPLITUDE_CACHE[3][1][1] * 1j,
                            AMPLITUDE_CACHE[3][1][0],
                        ]
                    )
                ),
                StateFn(
                    np.array(
                        [
                            -AMPLITUDE_CACHE[3][1][2] - AMPLITUDE_CACHE[3][1][2] * 1j,
                            -AMPLITUDE_CACHE[3][1][3],
                        ]
                    )
                ),
                StateFn(
                    np.array(
                        [
                            AMPLITUDE_CACHE[3][1][3],
                            AMPLITUDE_CACHE[3][1][2] - AMPLITUDE_CACHE[3][1][2] * 1j,
                        ]
                    )
                ),
                StateFn(
                    np.array(
                        [
                            -AMPLITUDE_CACHE[3][1][2] - AMPLITUDE_CACHE[3][1][2] * 1j,
                            AMPLITUDE_CACHE[3][1][3],
                        ]
                    )
                ),
                StateFn(
                    np.array(
                        [
                            -AMPLITUDE_CACHE[3][1][3],
                            AMPLITUDE_CACHE[3][1][2] - AMPLITUDE_CACHE[3][1][2] * 1j,
                        ]
                    )
                ),
                StateFn(
                    np.array(
                        [
                            AMPLITUDE_CACHE[3][1][0],
                            -AMPLITUDE_CACHE[3][1][1] - AMPLITUDE_CACHE[3][1][1] * 1j,
                        ]
                    )
                ),
                StateFn(
                    np.array(
                        [
                            -AMPLITUDE_CACHE[3][1][1] + AMPLITUDE_CACHE[3][1][1] * 1j,
                            AMPLITUDE_CACHE[3][1][0],
                        ]
                    )
                ),
            ],
        ),
        2: PartitionEncoding(
            operators=[
                MatrixOp(
                    np.array(
                        [
                            [
                                AMPLITUDE_CACHE[3][2][0],
                                0,
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                AMPLITUDE_CACHE[3][2][0] / 2,
                            ],
                            [
                                0,
                                AMPLITUDE_CACHE[3][2][0],
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                            ],
                            [
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                -AMPLITUDE_CACHE[3][2][0],
                                0,
                            ],
                            [
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                                0,
                                -AMPLITUDE_CACHE[3][2][0],
                            ],
                        ]
                    )
                ),
                MatrixOp(
                    np.array(
                        [
                            [
                                AMPLITUDE_CACHE[3][2][0],
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                0,
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                            ],
                            [
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                -AMPLITUDE_CACHE[3][2][0],
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                0,
                            ],
                            [
                                0,
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                AMPLITUDE_CACHE[3][2][0],
                                AMPLITUDE_CACHE[3][2][0] / 2,
                            ],
                            [
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                                0,
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                -AMPLITUDE_CACHE[3][2][0],
                            ],
                        ]
                    )
                ),
                MatrixOp(
                    np.array(
                        [
                            [
                                AMPLITUDE_CACHE[3][2][0],
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                                0,
                            ],
                            [
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                                -AMPLITUDE_CACHE[3][2][0],
                                0,
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                            ],
                            [
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                                0,
                                -AMPLITUDE_CACHE[3][2][0],
                                AMPLITUDE_CACHE[3][2][0] / 2,
                            ],
                            [
                                0,
                                -AMPLITUDE_CACHE[3][2][0] / 2,
                                AMPLITUDE_CACHE[3][2][0] / 2,
                                AMPLITUDE_CACHE[3][2][0],
                            ],
                        ]
                    )
                ),
            ],
            states=[
                StateFn(np.array([1.0, 0.0, 0.0, 0.0])),
                StateFn(AMPLITUDE_CACHE[3][2][0] * np.array([1.0, 1.0, 1.0, 0.0])),
                StateFn(AMPLITUDE_CACHE[3][2][0] * np.array([1.0, -1.0, 0.0, 1.0])),
                StateFn(np.array([0.0, 1.0, 0.0, 0.0])),
                StateFn(AMPLITUDE_CACHE[3][2][0] * np.array([-1.0, 0.0, 1.0, 1.0])),
                StateFn(np.array([0.0, 0.0, 1.0, 0.0])),
                StateFn(np.array([0.0, 0.0, 0.0, 1.0])),
                StateFn(AMPLITUDE_CACHE[3][2][0] * np.array([0.0, 1.0, -1.0, 1.0])),
            ],
        ),
    },
}
