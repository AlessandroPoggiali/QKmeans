from qiskit import execute, Aer

def printSateVector(circuit):
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend=backend, shots=1, memory=True)
    job_result = job.result()
    sv = job_result.get_statevector(circuit)
    tol = 1e-16
    sv.real[abs(sv.real) < tol] = 0.0
    sv.imag[abs(sv.imag) < tol] = 0.0
    return sv

def printUnitary(circuit):
    backend = Aer.get_backend('unitary_simulator')
    job = execute(circuit, backend=backend, shots=1, optimization_level=0)
    current_unitary = job.result().get_unitary(circuit, decimals=3)
    for row in current_unitary:
        column = ""
        for entry in row:
            column = column + str(entry.real) + " "
        print(column)