import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer

"""
This code is an implementation of the (non-fault tolerant) 5-qubit-code, otherwise known as the [[5,1,3]] code or the perfect code.

Useful references:
https://web.mit.edu/8.371/www/lectures/lect06.pdf
https://www.lorentz.leidenuniv.nl/quantumcomputers/literature/preskill_7.pdf
https://arxiv.org/pdf/1010.3242.pdf
https://www.physics.unlv.edu/~bernard/MATH_book/Chap9/Notebook9_3.pdf
"""

def ket_0(U,qubits):
    """
    Initialise the 5 qubit register into logical ket 0 state.
    The logical vectors were derived using Wolfram Mathematica.
    """
    five_qubits = QuantumRegister(5)
    logical_ket0_vector = [1/4, 0, 0, 1/4, 0, -(1/4), 1/4, 0, 0, -(1/4), -(1/4), 0, 1/4, 0, 0, -(1/4), 0, 1/4, -(1/4), 0, -(1/4), 0, 0, -(1/4), 1/4, 0, 0, -(1/4), 0, -(1/4), -(1/4), 0]
    temp_U = QuantumCircuit(five_qubits)
    temp_U.initialize(logical_ket0_vector, range(5))
    U.append(temp_U.to_instruction(label="\\ket{0}\\textsubscript{\\textit{L}}"),qubits)


def ket_1(U,qubits):
    """
    Initialise the 5 qubit register into logical ket 1 state.
    The logical vectors were derived using Wolfram Mathematica.
    """
    five_qubits = QuantumRegister(5)
    logical_ket1_vector = [0, -(1/4), -(1/4), 0, -(1/4), 0, 0, 1/4, -(1/4), 0, 0, -(1/4), 0, -(1/4), 1/4, 0, -(1/4), 0, 0, 1/4, 0, -(1/4), -(1/4), 0, 0, 1/4, -(1/4), 0, 1/4, 0, 0, 1/4]
    temp_U = QuantumCircuit(five_qubits)
    temp_U.initialize(logical_ket1_vector, range(5))
    U.append(temp_U.to_instruction(label="\\ket{1}\\textsubscript{\\textit{L}}"),qubits)

def S0(U,ancillas,qubits):
    U.cz(ancillas[3],qubits[0])
    U.cx(ancillas[3],qubits[1])
    U.cx(ancillas[3],qubits[2])
    U.cz(ancillas[3],qubits[3])

def S1(U,ancillas,qubits):
    U.cx(ancillas[2],qubits[0])
    U.cx(ancillas[2],qubits[1])
    U.cz(ancillas[2],qubits[2])
    U.cz(ancillas[2],qubits[4])

def S2(U,ancillas,qubits):
    U.cx(ancillas[1],qubits[0])
    U.cz(ancillas[1],qubits[1])
    U.cz(ancillas[1],qubits[3])
    U.cx(ancillas[1],qubits[4])

def S3(U,ancillas,qubits):
    U.cz(ancillas[0],qubits[0])
    U.cz(ancillas[0],qubits[2])
    U.cx(ancillas[0],qubits[3])
    U.cx(ancillas[0],qubits[4])

def iterate_single_qubit_errors():
    """
    The expected output should be:
    Single qubit error  Syndrome
    X[0]                   1001
    X[1]                   0010
    X[2]                   0101
    X[3]                   1010
    X[4]                   0100
    Z[0]                   0110
    Z[1]                   1100
    Z[2]                   1000
    Z[3]                   0001
    Z[4]                   0011
    Y[0]                   1111
    Y[1]                   1110
    Y[2]                   1101
    Y[3]                   1011
    Y[4]                   0111
    """
    backend = Aer.get_backend('qasm_simulator')
    df_index = [f'X[{j}]' for j in range(5)] + [f'Z[{j}]' for j in range(5)] + [f'Y[{j}]' for j in range(5)]
    df_columns = ['Syndrome']

    syndromes = []

    n_ancillas = 4
    n_qubits = 5

    ancillas = QuantumRegister(n_ancillas, name='a')
    qubits = QuantumRegister(n_qubits, name='q')
    creg = ClassicalRegister(n_ancillas)

    for i in range(15):
        U = QuantumCircuit(ancillas, qubits, creg)
        ket_0(U,qubits)
        U.barrier() # Introduce error!
        if 0<=i<=4:
            U.x(qubits[i%5])
        elif 5<=i<=9:
            U.z(qubits[i%5])
        else: # 10<=i<=14
            U.y(qubits[i%5])
        U.barrier()
        U.h(ancillas)
        U.barrier()
        S0(U,ancillas,qubits)
        U.barrier()
        S1(U,ancillas,qubits)
        U.barrier()
        S2(U,ancillas,qubits)
        U.barrier()
        S3(U,ancillas,qubits)
        U.barrier()
        U.h(ancillas)
        U.barrier()
        U.measure(ancillas,creg)
        
        compiled_qc = transpile(U, backend)
        results = backend.run(compiled_qc, shots=128).result()
        counts = results.get_counts()
        assert len(counts) == 1 # assert that there is only one unique syndrome measurement for each single qubit error
        syndromes.append( list( counts.keys() )[0] )
    df = pd.DataFrame(syndromes, columns = df_columns, index = pd.Index(df_index,name='Single qubit error'))
    return df

df = iterate_single_qubit_errors()
print(df)
