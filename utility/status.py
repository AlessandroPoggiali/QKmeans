from qiskit import execute, Aer

"""
get_satevector: 
    
It returns the statevector associated to a quantum circuit

:circuit: quantum qiskit circuit

:return: statevector
"""
def get_satevector(circuit):
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend=backend, shots=1, memory=True)
    job_result = job.result()
    statevector = job_result.get_statevector(circuit)
    tol = 1e-16
    statevector.real[abs(statevector.real) < tol] = 0.0
    statevector.imag[abs(statevector.imag) < tol] = 0.0
    return statevector

"""
printUnitary: 
    
It prints the unitary matrix associaated to a quantum circuit

:circuit: quantum qiskit circuit
"""
def printUnitary(circuit):
    backend = Aer.get_backend('unitary_simulator')
    job = execute(circuit, backend=backend, shots=1, optimization_level=0)
    current_unitary = job.result().get_unitary(circuit, decimals=3)
    for row in current_unitary:
        column = ""
        for entry in row:
            column = column + str(entry.real) + " "
        print(column)