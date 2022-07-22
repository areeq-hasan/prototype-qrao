from itertools import product

import numpy as np

from qrao.utils import get_random_maxcut_qp

from qrao.multiqubit import encode, dvar_values_to_state


def test_multiqubit():
    problem = get_random_maxcut_qp(degree=3, num_nodes=8, seed=1)
    operator, partitions, offset = encode(
        problem, max_qubits_per_partition=1, max_dvars_per_partition=1
    )

    for dvar_values in product([0, 1], repeat=8):
        dvar_values = list(dvar_values)
        state = dvar_values_to_state(dvar_values, partitions)
        assert (
            np.real(state.expectation_value(operator)) + offset
            == problem.objective.evaluate(dvar_values) * problem.objective.sense.value
        )

    # 5 qubits
    # 01-01-0

    # map a bitstring to quantum state that is defined on all my qubits
