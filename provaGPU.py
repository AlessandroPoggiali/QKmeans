import numpy as np
import math
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import datasets
from sklearn import metrics
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, transpile
from qiskit.providers.aer import StatevectorSimulator
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ
from qiskit.circuit.library import IntegerComparator

def indexing(circuit, Qregister, index):
    size = Qregister.size
    xored = index ^ (pow(2, size) - 1)
    j=1
    for k in (2**p for p in range(0, size)):
        if xored & k >= j:
            circuit.x(Qregister[j-1])
        j = j+1

def encodeVector(circuit, data, i, controls, rotationQubits, ancillaQubits):
    for j in range(len(data)): 
        # put the appropiate X gates on i qubits 
        indexing(circuit, i, j)
        # apply the controlled rotation
        #circuit.append(MCMT(RYGate(data[j]), len(controls), 1), controls[0:]+rotationQubits[0:])
        circuit.mcry(data[j], controls, rotationQubits, ancillaQubits)
        #circuit.mcry(2*np.arcsin(data[j]), controls, rotationQubits, ancillaQubits)
        #circuit.mcry(2*np.arcsin(data[j]/maxrow), controls, rotationQubits, ancillaQubits)
        # re-apply the appropiate X gates on i qubits
        indexing(circuit, i, j)
        circuit.barrier()
        
def encodeCentroids(circuit, data, i, controls, rotationQubit, ancillaQubits, c, cluster):
    # encode the cluster value putting the appropiate X gates on c qubits
    indexing(circuit, c, cluster)

    # encode centroid vector
    encodeVector(circuit, data, i, controls, rotationQubit, ancillaQubits)
    
    # re-apply the appropiate X gates on c qubits
    indexing(circuit, c, cluster)
    
    circuit.barrier()
    
def encodeVectors(circuit, data, i, controls, rotationQubit, ancillaQubits, qramindex, j):
    # put the appropiate X gates on qramindex qubits 
    indexing(circuit, qramindex, j)
    
    # encode centroid vector
    encodeVector(circuit, data, i, controls, rotationQubit, ancillaQubits)

    # re-apply the appropiate X gates on qramindex qubits
    indexing(circuit, qramindex, j)
    
    circuit.barrier()

def buildVectorsState(vectors, circuit, a, i, qramindex, r, q):
    for j in range(len(vectors.index)):
        vector = vectors.iloc[j]
        encodeVectors(circuit, vector, i, a[:]+i[:]+qramindex[:], r[0], q, qramindex, j)
    
def buildCentroidState(centroids, circuit, a, i, c, r, q):
    for cluster, centroid_vector in enumerate(centroids):
        encodeCentroids(circuit, centroid_vector, i, a[:]+i[:]+c[:], r[0], q, c, cluster)
 

def load_iris():
    df = datasets.load_iris(as_frame=True).frame
    # rename columns
    df.columns = ["f0","f1","f2","f3","class"]
    # drop class column
    df = df.drop('class', axis=1)
    
    scaler = StandardScaler()
    df.loc[:,:] = scaler.fit_transform(df.loc[:,:])
    df.loc[:,:] = normalize(df.loc[:,:])
    df = df.apply(lambda row: 2*np.arcsin(row), axis=1)
    
    return df

def computing_cluster_2(dataset, centroids):

    N = 150
    K = 3
    
    Aknn_qbits = 1                         # number of qbits for distance ancilla
    I_qbits = math.ceil(math.log(N,2))     # number of qubits needed to index the features
    if K == 1:
        C_qbits = 1
    else:
        C_qbits = math.ceil(math.log(K,2))     # number of qbits needed for indexing all centroids
    Rqram_qbits = 1                        # number of qbits for qram register
    Aqram_qbits = I_qbits + C_qbits + Aknn_qbits - 2 # number of qbits for qram ancillas
    max_qbits = I_qbits + C_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits
    #print("total qbits needed:  " + str(Tot_qbits))
    
    a = QuantumRegister(Aknn_qbits, 'a')   # ancilla qubit for distance
    i = QuantumRegister(I_qbits, 'i')      # feature index
    c = QuantumRegister(C_qbits, 'c')      # cluster
    r = QuantumRegister(1, 'r')            # rotation qubit for vector's features
    if Aqram_qbits > 0:
        q = QuantumRegister(Aqram_qbits, 'q')  # qram ancilla

    classical_bit = C_qbits + 2            # C_qbits for clusters + 1 for measuring ancilla + 1 for measuring register    
    outcome = ClassicalRegister(classical_bit, 'bit')  # for measuring
    
    cluster_assignment = []
    
    tot_execution_time = 0
    
    for index_v, vector in dataset.iterrows():
        print("record " + str(index_v))
        
        if Aqram_qbits > 0:
            circuit = QuantumCircuit(a, i, r, q, c, outcome)
        else:
            circuit = QuantumCircuit(a, i, r, c, outcome)
            q = None
        
        circuit.h(a)
        circuit.h(i)
        circuit.h(c)
        

        #--------------------- data vetcor encoding  -----------------------------#

        encodeVector(circuit, vector, i, a[:]+i[:], r[0], q)

        #-------------------------------------------------------------------------#

        circuit.x(a)
        circuit.barrier()

        #--------------- centroid vectors encoding -------------------------------#

        buildCentroidState(centroids, circuit, a, i, c, r, q)

        #----------------------------------------------------------------#
        circuit.measure(r, outcome[0])

        circuit.h(a) 

        circuit.measure(a, outcome[1])
            
        for b in range(C_qbits):
            circuit.measure(c[b], outcome[b+2])
        
        shots = 3*1024
       
        simulator = Aer.get_backend('statevector_simulator')
        simulator.set_options(device='GPU', cuStateVec_threshold=5)
        job = execute(circuit, simulator, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        #print("\nTotal counts are:",counts)
        #plot_histogram(counts)
    
        goodCounts = {k: counts[k] for k in counts.keys() if k.endswith('01')} # register 1 and ancilla 0
        cluster = max(goodCounts, key=goodCounts.get)
        cluster = int(cluster[:C_qbits], 2)
        
        if cluster >= K:
            cluster = 0 
        cluster_assignment.append(cluster)


if __name__ == "__main__":
    df = load_iris()
    centroids = df.sample(n=3, random_state=123).values
    computing_cluster_2(df,centroids)
