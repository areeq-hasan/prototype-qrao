from qrao.utils import get_random_maxcut_qp

from qrao.multiqubit import encode


def test_multiqubit():
    problem = get_random_maxcut_qp(degree=3, num_nodes=8, seed=1)
    operator, _ = encode(problem)

    # 5 qubits
    # 01-01-0

    # map a bitstring to quantum state that is defined on all my qubits
