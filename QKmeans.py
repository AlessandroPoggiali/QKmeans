import numpy as np
import math
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
#from qiskit.circuit.library import MCMT, RYGate
#from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_city
from dataset import Dataset
from QRAM import buildCentroidState, buildVectorsState
from utility import measures

class QKMeans():
    def __init__(self, conf):
        
        self.K = conf['K']
        self.M1 = conf['M1']
        self.shots = (self.K + self.M1) * 1000
        self.dataset_name = conf['dataset_name']
        self.sc_tresh = conf['sc_tresh']
        self.max_iterations = conf['max_iterations']
        
        self.dataset = Dataset(self.dataset_name)
        self.data = self.dataset.df
        self.data['cluster'] = 0
        self.N = self.dataset.N
        self.M = self.dataset.M
        self.centroids = self.data.sample(n=self.K)
        self.centroids['cluster'] = [x for x in range(self.K)]
        self.initial_centroids = self.centroids.copy()
        self.old_centroids = None
        
        self.cluster_assignment = []
        
        self.ite = 0
        self.accs = []
        self.times = []
        
        
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
        
        a = QuantumRegister(Aknn_qbits, 'a')   # ancilla qubit for distance
        i = QuantumRegister(I_qbits, 'i')      # feature index
        c = QuantumRegister(C_qbits, 'c')      # cluster
        r = QuantumRegister(1, 'r')            # rotation qubit for vector's features
        
        n_circuits = math.ceil(M/M1)
        
        print("circuits needed:  " + str(n_circuits))
        
        cluster_assignment = []
        
        for j in range(n_circuits):
        
            print("Circuit " + str(j+1) + "/" + str(n_circuits))
            
            vectors = self.data[j*M1:(j+1)*M1]
            
            if j == n_circuits-1:
                M1 = M-M1*(n_circuits-1)
                
            print("vector to classify: " + str(M1))
                
            QRAMINDEX_qbits = math.ceil(math.log(M1,2))     # number of qubits needed to index the qrams (i.e 'test' vectors)
            
            if QRAMINDEX_qbits == 0: # se mi rimane solo un record da assegnare ad un cluster tengo comunque un qubit per qrama anche se non mi serve
                QRAMINDEX_qbits = 1
            
            Aqram_qbits = I_qbits + C_qbits + QRAMINDEX_qbits + Aknn_qbits - 2 # number of qbits for qram ancillas
            if C_qbits > QRAMINDEX_qbits:
                Aqram_qbits = Aqram_qbits - QRAMINDEX_qbits
            else:
                Aqram_qbits = Aqram_qbits - C_qbits
            Tot_qbits = I_qbits + C_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits + QRAMINDEX_qbits
            print("total qbits needed for this circuit:  " + str(Tot_qbits))
            
            qramindex = QuantumRegister(QRAMINDEX_qbits,'qramindex')     # index for qrams           
    
            classical_bit = C_qbits + QRAMINDEX_qbits  + 2  # C_qbits for clusters + QRAMINDEX_qbits for qram + 1 for ancilla + 1 for register    
            outcome = ClassicalRegister(classical_bit, 'bit')  # for measuring
                
            if Aqram_qbits > 0:
                q = QuantumRegister(Aqram_qbits, 'q')  # qram ancilla
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
        self.cluster_assignment = cluster_assignment
            
    
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
        self.centroids.loc[:, self.centroids.columns[:-1]] = self.dataset.normalize(self.centroids.loc[:,self.centroids.columns[:-1]])  

    
    '''
    Computes the new cluster centers as matrix-vector products
    '''
    def computing_centroids(self):
        
        #data = data.reset_index(drop=True)
        Vt = self.data.drop('cluster', 1).T.values
        car_vectors = []
        for i in range(self.K):
            v = np.zeros(len(self.data))
            index = self.data[self.data['cluster']==i].index
            v[index] = 1
            v = v/len(index)
            car_vectors.append(v)
        
        # compute matrix-vector products
        newcentroids = []
        for i in range(self.K):
            c = Vt@car_vectors[i] 
            newcentroids.append(np.append(c, i))
        
        self.centroids = pd.DataFrame(data=newcentroids, columns=self.data.columns)
        # normalize centroid
        #centroids.loc[:,centroids.columns[:-1]] = normalize(centroids.loc[:,centroids.columns[:-1]])    
        self.centroids.loc[:, self.centroids.columns[:-1]] = self.dataset.normalize(self.centroids.loc[:,self.centroids.columns[:-1]])
    
    
    def stop_condition(self):
        # ritorna true se i centroidi non cambiano piu di un tot false altrimenti

        if self.old_centroids is None:
            return False

        for i in range(len(self.centroids)):
            difference = np.linalg.norm(np.array(self.centroids.iloc[i][:-1])-np.array(self.old_centroids.iloc[i][:-1]))
            if difference > self.sc_tresh:
                return False
        
        return True
    
    
    def run(self):
        
        self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, initial_space=True)
        
        while not self.stop_condition():
            
            start = time.time()
            
            self.old_centroids = self.centroids.copy()
            
            print("------------------ iteration " + str(self.ite) + "------------------")
            print("Computing the distance between all vectors and all centroids and assigning the cluster to the vectors")
            self.computing_cluster(M1=self.M1, shots=self.shots)
            self.data['cluster'] = self.cluster_assignment
            #self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, True)
    
            print("Computing new centroids")
            #centroids = computing_centroids_0(data, k)
            self.computing_centroids_0()
    
            self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, assignment=True, initial_space=False)
            self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, assignment=True, initial_space=True)
            
            end = time.time()
            elapsed = end - start
            self.times.append(elapsed)
            
            acc = measures.check_accuracy(self.data, self.centroids)
            self.accs.append(acc)
            print("Accuracy: " + str(round(acc, 2)) + "%")
            
            
            self.ite = self.ite + 1
            if self.ite == self.max_iterations:
                break
        
    

    def print_result(self, filename=None):
        self.dataset.plotOnCircle(self.data, self.centroids)
        
        print("")
        print("---------------- QKMEANS RESULT ----------------")
        print("Iterations needed: " + str(self.ite) + "/" + str(self.max_iterations))
        
        avg_time = round(np.mean(self.times), 2)
        print("Average iteration time: " + str(avg_time) + " sec")
        
        avg_acc = round(np.mean(self.accs), 2)
        print("Average accuracy w.r.t classical assignment: " + str(avg_acc) + "%")
        
        SSE = measures.SSE(self.data, self.centroids)
        print("SSE: " + str(SSE))
        
        silhouette = metrics.silhouette_score(self.data.loc[:,self.data.columns[:-1]], self.data['cluster'], metric='euclidean')
        print("Silhouette score: " + str(silhouette))
    
        fig, ax = plt.subplots()
        ax.plot(range(self.ite), self.accs, marker="o")
        ax.set(xlabel='QKmeans iterations', ylabel='Accuracy',
               title='Accuracy w.r.t classical assignemnt')
        plt.show()
        
        if filename is not None:
            # stampa le cose anche su file 
            
            f = open(filename, 'a')
            dt = datetime.datetime.now().replace(microsecond=0)
            f.write("###### TEST " + str(dt) + " on " + str(self.dataset_name) + " dataset\n")
            f.write("## QKMEANS\n")
            f.write("# Parameters: K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", M1 = " + str(self.M1) + ", shots = " + str(self.shots) + "\n")
            f.write("# Iterations needed: " + str(self.ite) + "/" + str(self.max_iterations) + "\n")
            f.write('# Average iteration time: ' + str(avg_time) + 's \n')
            f.write('# Average accuracy w.r.t classical assignment: ' + str(avg_acc) + '% \n')
            f.write('# SSE: ' + str(SSE) + '\n')
            f.write('# Silhouette: ' + str(silhouette) + '\n')
            f.write("# Quantum kmeans assignment \n")
            f.write(str(self.cluster_assignment))            
            f.write("\n")
            #f.write('# Final centroids \n'
            #self.centroids.to_csv(f, index=False)
            f.close()
            



