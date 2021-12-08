import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
#from qiskit.circuit.library import MCMT, RYGate
#from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_city
from dataset import Dataset
from QRAM import buildCentroidState, buildVectorsState
from utility import measures

class QKMeans():
    def __init__(self, conf):
        self.K = conf['K']
        self.dataset = Dataset(conf['dataset_name'])
        self.data = self.dataset.df
        self.data['cluster'] = 0
        self.N = self.dataset.N
        self.M = self.dataset.M
        self.centroids = self.data.sample(n=self.K)
        self.centroids['cluster'] = [x for x in range(self.K)]
        
    '''
    Compute all distances between vectors and centroids and assign cluster to every vector in many circuits
    '''
    def computing_cluster(self, M1, shots):
         
        N = self.N
        M = self.M
        K = self.K
        
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
            
            vectors = self.data[j*M1:(j+1)*M1]
            
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
    
            buildCentroidState(self.centroids, circuit, a, i, c, r, q)
    
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
    def computing_centroids_0(self):
        #data = data.reset_index(drop=True)
        
        series = []
        for i in range(self.K):
            series.append(self.data[self.data['cluster']==i].mean())
        self.centroids = pd.concat(series, axis=1).T
        
        # normalize centroid
        self.centroids.loc[:,self.centroids.columns[:-1]] = normalize(self.centroids.loc[:,self.centroids.columns[:-1]])    
    
    
    '''
    Computes the new cluster centers as matrix-vector products
    '''
    def computing_centroids(self, data, k):
        
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
            
    
    def check_condition(self, old_centroids, centroids):
        # ritorna true se i centroidi non cambiano piu di un tot false altrimenti
        return False
    
    def run(self, iterations):
        
        # select initial centroids
        #centroids = data.sample(n=k)
        #centroids['cluster'] = [x for x in range(k)]
        self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids)
        
        for i in range(iterations):
            
            self.old_centroids = self.centroids.copy()
            
            print("------------------ iteration " + str(i) + "------------------")
            print("Computing the distance between all vectors and all centroids and assigning the cluster to the vectors")
            cluster_assignment = self.computing_cluster(M1=len(self.data), shots=150000)
            self.data['cluster'] = cluster_assignment
            self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, True)
    
            print("Computing new centroids")
            #centroids = computing_centroids_0(data, k)
            self.computing_centroids_0()
    
            if self.check_condition(self.old_centroids, self.centroids):
                break
    
            self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, True, initial_space=True)
    
        
        self.dataset.plotOnCircle(self.data, self.centroids)
        print("SSE: " + str(measures.SSE(self.data, self.centroids)))
        
        
    


if __name__ == "__main__":
    
    conf = {"dataset_name": 'iris', "K": 2}
    
    QKMEANS = QKMeans(conf)
    QKMEANS.run(iterations=1)

