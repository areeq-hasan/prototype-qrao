from typing import List

from functools import reduce

from qiskit.opflow import StateFn

from .problem import Partition
from .schemes import ENCODING_SCHEMES


def encode_partition_configuration(
    partition: Partition, configuration: List[int]
) -> StateFn:
    state_index = 0
    for value in configuration:
        state_index = (state_index << 1) | value
    return ENCODING_SCHEMES[partition.num_variables][partition.num_qubits].states[
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
