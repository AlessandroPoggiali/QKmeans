import numpy as np

def indexing(circuit, Qregister, index):
    size = Qregister.size
    xored = index ^ (pow(2, size) - 1)
    j=1
    for k in (2**p for p in range(0, size)):
        if xored & k >= j:
            circuit.x(Qregister[j-1])
        j = j+1

def encodeVector(circuit, data, i, controls, rotationQubits, ancillaQubits):
    for j in range(len(data)-1): # avoid cluster column
        # put the appropiate X gates on i qubits 
        indexing(circuit, i, j)
        # apply the controlled rotation
        #circuit.append(MCMT(RYGate(data[j]), len(controls), 1), controls[0:]+rotationQubits[0:])
        circuit.mcry(np.arcsin(data[j]), controls, rotationQubits, ancillaQubits)
        # re-apply the appropiate X gates on i qubits
        indexing(circuit, i, j)
        circuit.barrier()
        
def encodeCentroids(circuit, data, i, controls, rotationQubit, ancillaQubits, c, cluster, j):
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
    for j in range(len(centroids.index)):
        centroidVector = centroids.iloc[j]
        encodeCentroids(circuit, centroidVector, i, a[:]+i[:]+c[:], r[0], q, c, int(centroidVector['cluster']), j)
 