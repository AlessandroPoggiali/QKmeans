import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
#from qiskit.circuit.library import MCMT, RYGate
#from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_city
from dataset import Dataset
from utility import measures

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
        

'''
Compute all distances between vectors and centroids and assign cluster to every vector in many circuits
'''
def computing_cluster(data, centroids, M1, shots):
    
    N = len(data.columns) - 1           # number of features
    K = len(centroids)                     # number of cluster
    M = len(data)                       # number of records
       
    Aknn_qbits = 1                         # number of qbits for distance ancilla
    I_qbits = math.ceil(math.log(N,2))     # number of qubits needed to index the features
    C_qbits = math.ceil(math.log(K,2))     # number of qbits needed for indexing all centroids
    Rqram_qbits = 1                        # number of qbits for qram register
    QRAMINDEX_qbits = math.ceil(math.log(M1,2))     # number of qubits needed to index the qrams (i.e 'test' vectors)
    Aqram_qbits = I_qbits + C_qbits + QRAMINDEX_qbits + Aknn_qbits - 2 # number of qbits for qram ancillas
    if C_qbits > QRAMINDEX_qbits:
        Aqram_qbits = Aqram_qbits - QRAMINDEX_qbits
    else:
        Aqram_qbits = Aqram_qbits - C_qbits
    Tot_qbits = I_qbits + C_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits + QRAMINDEX_qbits
    
    a = QuantumRegister(Aknn_qbits, 'a')   # ancilla qubit for distance
    i = QuantumRegister(I_qbits, 'i')      # feature index
    c = QuantumRegister(C_qbits, 'c')      # cluster
    r = QuantumRegister(1, 'r')            # rotation qubit for vector's features
    qramindex = QuantumRegister(QRAMINDEX_qbits,'qramindex')     # index for qrams
    
    if Aqram_qbits > 0:
        q = QuantumRegister(Aqram_qbits, 'q')  # qram ancilla

    classical_bit = C_qbits + QRAMINDEX_qbits  + 2  # C_qbits for clusters + QRAMINDEX_qbits for qram + 1 for ancilla + 1 for register    
    outcome = ClassicalRegister(classical_bit, 'bit')  # for measuring
    
    n_circuits = int(M/M1)
    
    print("circuits needed:  " + str(n_circuits))
    print("total qbits needed for each circuit:  " + str(Tot_qbits))
    
    cluster_assignment = []
    
    for j in range(n_circuits):
    
        print("Circuit " + str(j+1) + "/" + str(n_circuits))
        
        vectors = data[j*M1:(j+1)*M1]
        
        if Aqram_qbits > 0:
            circuit = QuantumCircuit(a, i, r, q, c, qramindex, outcome)
        else:
            circuit = QuantumCircuit(a, i, r, c, qramindex, outcome)
            q = None

        circuit.h(a)
        circuit.h(i)
        circuit.h(c)
        circuit.h(qramindex)


        #--------------------- data vetcor encoding  -----------------------------#

        buildVectorsState(vectors, circuit, a, i, qramindex, r, q)

        #-------------------------------------------------------------------------#

        circuit.x(a)
        circuit.barrier()

        #--------------- centroid vectors encoding -------------------------------#

        buildCentroidState(centroids, circuit, a, i, c, r, q)

        #----------------------------------------------------------------#


        circuit.measure(r, outcome[0])

        circuit.h(a) 

        #return circuit

        circuit.measure(a, outcome[1])

        # measuring cluster bits
        for b in range(C_qbits):
            circuit.measure(c[b], outcome[b+2])

        # measuring qram bits
        for b in range(QRAMINDEX_qbits):
            circuit.measure(qramindex[b], outcome[b+2+C_qbits])

        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circuit, simulator, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        #print("\nTotal counts are:",counts)
        #plot_histogram(counts)
        # qram-classe-ancilla-registro
        goodCounts = {k: counts[k] for k in counts.keys() if k.endswith('01')} 
        #plot_histogram(goodCounts)

        for v in range(M1):
            strformat = "{0:0" + str(QRAMINDEX_qbits) + "b}"
            strbin = strformat.format(v)
            vi_counts = {k: goodCounts[k] for k in goodCounts.keys() if k.startswith(strbin)} 

            if len(vi_counts) == 0:
                cluster = 0
            else:
                count_sorted = sorted(((v,k) for k,v in vi_counts.items()))
                cluster = count_sorted[-1][1] # is the key related to the maximum count
                #cluster = max(vi_counts, key=vi_counts.get)
                cluster = int(cluster[QRAMINDEX_qbits:-2],2)

            #print('cluster prima ' + str(cluster))
            ## SEMPLICE EURISTICA: QUANDO RESTITUISCO UN CLUSTER CHE NON ESISTE ALLORA PRENDO IL SECONDO PIU ALTO 
            ind = 2
            while cluster >= K:
                if ind > len(count_sorted):
                    break
                cluster = count_sorted[-ind][1]
                cluster = int(cluster[QRAMINDEX_qbits:-2], 2)
                ind = ind + 1
            if cluster >= K:
                cluster = 0

            #print('cluster dopo ' + str(cluster))
            #print('------------------------------------------')

            cluster_assignment.append(cluster)
            #print("vector " + str(i))
            #print("Quantum assignment: " + str(cluster))    

        #df.loc[j*M1:(j+1)*M1-1,'cluster'] = cluster_assignment 
    return cluster_assignment
        

'''
Computes the new cluster centers as the mean of the records within the same cluster
'''
def computing_centroids_0(data, k):
    #data = data.reset_index(drop=True)
    
    series = []
    for i in range(k):
        series.append(data[data['cluster']==i].mean())
    centroids = pd.concat(series, axis=1).T
    
    # normalize centroid
    centroids.loc[:,centroids.columns[:-1]] = normalize(centroids.loc[:,centroids.columns[:-1]])    
    return centroids


'''
Computes the new cluster centers as matrix-vector products
'''
def computing_centroids(data, k):
    
    #data = data.reset_index(drop=True)
    Vt = data.drop('cluster', 1).T.values
    car_vectors = []
    for i in range(k):
        v = np.zeros(len(data))
        index = data[data['cluster']==i].index
        v[index] = 1
        v = v/len(index)
        car_vectors.append(v)
    
    # compute matrix-vector products
    newcentroids = []
    for i in range(k):
        c = Vt@car_vectors[i] 
        newcentroids.append(np.append(c, i))
    
    centroids = pd.DataFrame(data=newcentroids, columns=data.columns)
    # normalize centroid
    centroids.loc[:,centroids.columns[:-1]] = normalize(centroids.loc[:,centroids.columns[:-1]])    
    return centroids    
        

def check_condition(old_centroids, centroids):
    # ritorna true se i centroidi non cambiano piu di un tot false altrimenti
    return False

def QKMEANS(data, n, k, iterations):
    
    # select last n features
    #data = data.drop(columns=data.columns[:(len(data.columns)-2)])
    # add cluster column
    data['cluster'] = 0
    
    # select initial centroids
    centroids = data.sample(n=k)
    centroids['cluster'] = [x for x in range(k)]
    dataset.plot2Features(data, data.columns[0], data.columns[1], centroids)
    
    for i in range(iterations):
        
        old_centroids = centroids
        
        print("------------------ iteration " + str(i) + "------------------")
        print("Computing the distance between all vectors and all centroids and assigning the cluster to the vectors")
        cluster_assignment = computing_cluster(data, centroids, M1=len(data), shots=150000)
        data['cluster'] = cluster_assignment
        dataset.plot2Features(data, data.columns[0], data.columns[1], centroids, True)

        print("Computing new centroids")
        #centroids = computing_centroids_0(data, k)
        centroids = computing_centroids_0(data, k)

        if check_condition(old_centroids, centroids):
            break

        dataset.plot2Features(data, data.columns[0], data.columns[1], centroids, True, initial_space=True)

    
    dataset.plotOnCircle(data, centroids)
    print("SSE: " + str(measures.SSE(data, centroids)))
    
    



if __name__ == "__main__":
    
    #df = load_iris()
    #df = df.sample(n=4)
    
    #df = load_buddymove()
    
    dataset = Dataset('iris')
    df = dataset.df
    #df = dataset.load_iris()
    #df.head()
    
    ## df is a scaled and normalized dataset where in each entry is applied arcsin function and no class column is present
    QKMEANS(df, n=2, k=2, iterations=1)

