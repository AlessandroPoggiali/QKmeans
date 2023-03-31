import numpy as np
import math
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, transpile
from qiskit.providers.aer import StatevectorSimulator
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ
from sklearn.preprocessing import StandardScaler, normalize
from qiskit.circuit.library import IntegerComparator
from QRAM import buildCentroidState, buildVectorsState, encodeVector
from utility import measures

font = {'size'   : 22}

plt.rc('font', **font)

class QKMeans():
    
    """
    QKMeans constructor: 
    
    :param dataset: dataset object
    :param conf: parameters configuration of the algorithm
    """
    def __init__(self, dataset, conf):

        self.dataset_name = conf['dataset_name']
        self.sc_tresh = conf['sc_tresh']
        self.max_iterations = conf['max_iterations']
        self.dataset = dataset
        self.data = self.dataset.df
        self.N = self.dataset.N
        self.M = self.dataset.M

        self.quantization = conf['quantization']
        self.K = conf['K']
        if self.quantization == 3:
            if conf['M1'] is None:
                self.M1 = self.M
            else: 
                self.M1 = conf['M1']
        else:
            self.M1 = None
                        
        if conf['shots'] is None:
            if self.M1 is None:
                if self.quantization == 1:
                    self.shots = 1024
                else:
                    self.shots = min(self.K * 1024, 500000)
            else:
                self.shots = min(self.K * self.M1 * 1024, 500000)
        else:
            self.shots = conf['shots']
    
        self.centroids = None
        self.old_centroids = None
        
        self.cluster_assignment = [0]*self.M
        
        self.max_qbits = 0
        self.n_circuits = 0
        
        self.ite = 0
        self.SSE_list = []
        self.silhouette_list = []
        self.similarity_list = []
        self.nm_info_list = []
        self.times = []
        self.execution_time_hw = []

        self.centroids_norm = []
        self.records_norm = []
        self.records_norm = dataset.original_df.apply(lambda row: np.linalg.norm(row), axis=1).tolist()    
            

    """
    computing_cluster_1: 
    
    Computes the quantum distances between all records and all centroids "sequentially"
        
    :provider (optional, default value=None): real quantum hardware
    """
    def computing_cluster_1(self, provider=None, check_prob=False):
    
        distances = []                         # list of distances: the i-th item is a list of distances between i-th vector and all centroids 
        
        N = self.N

        if check_prob:
            r1_list = []
            a0_list = []

        Rqram_qbits = 1                        # number of qbits for qram register
        Aknn_qbits = 1                         # number of qbits for distance ancilla
        I_qbits = math.ceil(math.log(N,2))     # number of qubits needed to index the features
        Aqram_qbits = I_qbits + Aknn_qbits - 2 # number of qbits for qram ancillas
        self.max_qbits = I_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits
        
        a = QuantumRegister(1, 'a')            # ancilla qubit for distance
        i = QuantumRegister(I_qbits, 'i')      # feature index
        r = QuantumRegister(1, 'r')            # rotation qubit for vector's features
        if Aqram_qbits > 0:
            q = QuantumRegister(Aqram_qbits, 'q')  # qram ancilla
    
        outcome = ClassicalRegister(2, 'bit')  # for measuring
        
        tot_execution_time = 0
    
        for index_v, vector in self.dataset.df.iterrows():
            centroid_distances = []            # list of distances between the current vector and all centroids
            for index_c in range(len(self.centroids)):
        
                if Aqram_qbits > 0:
                    circuit = QuantumCircuit(a, i, r, q, outcome)
                else:
                    circuit = QuantumCircuit(a, i, r, outcome)
                    q = None
        
                circuit.h(a)
                circuit.h(i)
    
                #-------------- states preparation  -----------------------#
    
                encodeVector(circuit, self.centroids[index_c], i, a[:]+i[:], r[0], q)  # sample vector encoding in QRAM
    
                circuit.x(a)
    
                encodeVector(circuit, vector, i, a[:]+i[:], r[0], q)    # centroid vector encoding in QRAM
    
                #----------------------------------------------------------#
                circuit.measure(r,outcome[0])
    
                circuit.h(a)
    
                circuit.measure(a,outcome[1])
    
                shots = self.shots

                if provider is not None:
                    large_enough_devices = provider.backends(filters=lambda x: x.configuration().n_qubits > 4 and not x.configuration().simulator)
                    backend = least_busy(large_enough_devices)
                    job = execute(circuit, backend, shots=shots)
                    job_monitor(job)
                    result = job.result()
                    execution_time = result.time_taken
                    tot_execution_time += execution_time
                    print("executed in: " + str(execution_time))
                    counts = result.get_counts(circuit)
                else:
                    simulator = Aer.get_backend('qasm_simulator')
                    #simulator.set_options(device='GPU')
                    job = execute(circuit, simulator, shots=shots)
                    result = job.result()
                    counts = result.get_counts(circuit)

                if check_prob:
                    r1 = {k: counts[k] for k in counts.keys() if k[1]=='1'}
                    r1_perc = (sum(r1.values())/self.shots)*100
                    r1_list.append(r1_perc)

                    a0 = {k: counts[k] for k in counts.keys() if k[0]=='0'}
                    a0_perc = (sum(a0.values())/self.shots)*100
                    a0_list.append(a0_perc)

                goodCounts = {k: counts[k] for k in counts.keys() & {'01','11'}}

                try:
                    n_p0 = goodCounts['01']
                except:
                    n_p0 = 0
            
                euclidian = 4-4*(n_p0/sum(goodCounts.values()))
                euclidian = math.sqrt(euclidian)
                if self.dataset.preprocessing == '2-norm':
                    r_norm = self.records_norm[index_v]
                    c_norm = self.centroids_norm[index_c] 
                    euclidean_2 = math.sqrt((r_norm-c_norm)**2+(euclidian**2)*(r_norm*c_norm))
                    euclidian = euclidean_2

                centroid_distances.append(euclidian)
                
            distances.append(centroid_distances)
        
        self.cluster_assignment = [(i.index(min(i))) for i in distances] # for each vector takes the closest centroid to perform the assignemnt
        
        if check_prob:
            return round(sum(r1_list)/len(r1_list),3),round(sum(a0_list)/len(a0_list),3)

        return tot_execution_time


    """
    computing_cluster_2: 
    
    Computes the quantum assignment of a record to the set of centroids for every record
        
    :provider (optional, default value=None): real quantum hardware
    """
    def computing_cluster_2(self, provider=None, check_prob=False):
        
        if check_prob:
            r1_list = []
            a0_list = []

        N = self.N
        K = self.K
        
        Aknn_qbits = 1                         # number of qbits for distance ancilla
        I_qbits = math.ceil(math.log(N,2))     # number of qubits needed to index the features
        if K == 1:
            C_qbits = 1
        else:
            C_qbits = math.ceil(math.log(K,2))     # number of qbits needed for indexing all centroids
        Rqram_qbits = 1                        # number of qbits for qram register
        Aqram_qbits = I_qbits + C_qbits + Aknn_qbits - 2 # number of qbits for qram ancillas
        self.max_qbits = I_qbits + C_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits
        
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
        
        for index_v, vector in self.dataset.df.iterrows():
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
    
            buildCentroidState(self.centroids, circuit, a, i, c, r, q)
    
            #----------------------------------------------------------------#
            circuit.measure(r, outcome[0])
    
            circuit.h(a) 
    
            circuit.measure(a, outcome[1])
             
            for b in range(C_qbits):
                circuit.measure(c[b], outcome[b+2])
            
            shots = self.shots

            if provider is not None:
                large_enough_devices = provider.backends(filters=lambda x: x.configuration().n_qubits > 4 and not x.configuration().simulator)
                backend = least_busy(large_enough_devices)
                job = execute(circuit, backend, shots=shots)
                job_monitor(job)
                result = job.result()
                execution_time = result.time_taken
                tot_execution_time += execution_time
                print("executed in: " + str(execution_time))
                counts = result.get_counts(circuit)
            else:
                simulator = Aer.get_backend('qasm_simulator')
                job = execute(circuit, simulator, shots=shots)
                result = job.result()
                counts = result.get_counts(circuit)
            
            if check_prob:
                r1 = {k: counts[k] for k in counts.keys() if k.endswith('1')}
                r1_perc = (sum(r1.values())/self.shots)*100
                r1_list.append(r1_perc)

                a0 = {k: counts[k] for k in counts.keys() if k[-2]=='0'}
                a0_perc = (sum(a0.values())/self.shots)*100
                a0_list.append(a0_perc)

            goodCounts = {k: counts[k] for k in counts.keys() if k.endswith('01')} # register 1 and ancilla 0
            cluster = max(goodCounts, key=goodCounts.get)
            cluster = int(cluster[:C_qbits], 2)
            
            if cluster >= K:
                cluster = 0 
            cluster_assignment.append(cluster)
             

        self.cluster_assignment = cluster_assignment 
        
        if check_prob:
            return round(sum(r1_list)/len(r1_list),3),round(sum(a0_list)/len(a0_list),3)

        return tot_execution_time
        
    
    """
    computing_cluster_3: 
    
    Compute the cluster assignment of every record to centroids
        
    :check_prob (optional, default value=None): if True the method returns the post selection probability
    """
    def computing_cluster_3(self, check_prob=False):
         
        if check_prob:
            r1_list = []
            a0_list = []
        
        N = self.N
        M = self.M
        K = self.K
        M1 = self.M1
        
        Aknn_qbits = 1                         # number of qbits for distance ancilla
        I_qbits = math.ceil(math.log(N,2))     # number of qubits needed to index the features
        if K == 1:
            C_qbits = 1
        else:
            C_qbits = math.ceil(math.log(K,2))     # number of qbits needed for indexing all centroids
        Rqram_qbits = 1                        # number of qbits for qram register
        
        a = QuantumRegister(Aknn_qbits, 'a')   # ancilla qubit for distance
        i = QuantumRegister(I_qbits, 'i')      # feature index
        c = QuantumRegister(C_qbits, 'c')      # cluster
        r = QuantumRegister(1, 'r')            # rotation qubit for vector's features
        
        self.n_circuits = math.ceil(M/M1)
      
  
        cluster_assignment = []
        
        for j in range(self.n_circuits):            
            vectors = self.data[j*M1:(j+1)*M1]
            
            if j == self.n_circuits-1:
                M1 = M-M1*(self.n_circuits-1)
                
            QRAMINDEX_qbits = math.ceil(math.log(M1,2))     # number of qubits needed to index the qrams (i.e 'test' vectors)
            
            if QRAMINDEX_qbits == 0: 
                QRAMINDEX_qbits = 1
            
            Aqram_qbits = I_qbits + C_qbits + QRAMINDEX_qbits + Aknn_qbits - 2 # number of qbits for qram ancillas
            if C_qbits > QRAMINDEX_qbits:
                Aqram_qbits = Aqram_qbits - QRAMINDEX_qbits
            else:
                Aqram_qbits = Aqram_qbits - C_qbits
            tot_qbits = I_qbits + C_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits + QRAMINDEX_qbits
            if j == 0:
                self.max_qbits = tot_qbits
            
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
            job = execute(circuit, simulator, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)

            
            if check_prob:
                r1 = {k: counts[k] for k in counts.keys() if k.endswith('1')}
                r1_perc = (sum(r1.values())/self.shots)*100
                r1_list.append(r1_perc)
                
                a0 = {k: r1[k] for k in r1.keys() if k.endswith('01')}
                a0_perc = (sum(a0.values())/(sum(r1.values()))*100)
                a0_list.append(a0_perc)
            
            
            goodCounts = {k: counts[k] for k in counts.keys() if k.endswith('01')} 
    
            for v in range(M1):
                strformat = "{0:0" + str(QRAMINDEX_qbits) + "b}"
                strbin = strformat.format(v)
                vi_counts = {k: goodCounts[k] for k in goodCounts.keys() if k.startswith(strbin)} 
    
                if len(vi_counts) == 0:
                    cluster = 0
                else:
                    count_sorted = sorted(((v,k) for k,v in vi_counts.items()))
                    cluster = count_sorted[-1][1] # is the key related to the maximum count
                    cluster = int(cluster[QRAMINDEX_qbits:-2],2)
    
                ind = 2
                while cluster >= K:
                    if ind > len(count_sorted):
                        break
                    cluster = count_sorted[-ind][1]
                    cluster = int(cluster[QRAMINDEX_qbits:-2], 2)
                    ind = ind + 1
                if cluster >= K:
                    cluster = 0
    
                cluster_assignment.append(cluster)
        self.cluster_assignment = cluster_assignment
        
        if check_prob:
            return round(sum(r1_list)/len(r1_list),3),round(sum(a0_list)/len(a0_list),3)
        

    """
    computing_centroids: 
        
    Computes the new cluster centers as the mean of the records within the same cluster
    """
    def computing_centroids(self):
        series = []
        old_series = []
        index_centroids = []
        index_oldcentroids = []
        for i in range(self.K):
            if i in self.cluster_assignment:
                series.append(self.dataset.original_df.loc[[index for index, n in enumerate(self.cluster_assignment) if n == i]].mean())
                index_centroids.append(i)
            else:
                old_centroid = pd.Series(self.centroids[i])
                old_centroid = old_centroid.rename(lambda x: "f" + str(x))
                old_series.append(old_centroid)##
                index_oldcentroids.append(i)
                
        if len(series)>0:
            df_centroids = pd.concat(series, axis=1).T
            df_centroids.index = index_centroids ## 
        else:
            df_centroids = pd.DataFrame(columns=self.dataset.original_df.columns)
        if len(old_series)>0:
            df_oldcentroids = pd.concat(old_series, axis=1).T##
            df_oldcentroids.index = index_oldcentroids ##
        else:
            df_oldcentroids = pd.DataFrame(columns=self.dataset.df.columns)
        df_centroids = self.dataset.preprocess(df_centroids)
        df_centroids = pd.concat([df_oldcentroids, df_centroids], axis=0, sort=False).sort_index()
        self.centroids_norm = df_centroids.apply(lambda row: np.linalg.norm(row), axis=1).tolist()

        self.centroids = df_centroids.values



    """
    stop_condition: 
        
    Checks if the algorithm have reached the stopping codition
    
    :return: True if the algorithm must terminate, False otherwise
    """
    def stop_condition(self):
        if self.old_centroids is None:
            return False

        if np.linalg.norm(self.centroids-self.old_centroids, ord='fro') >= self.sc_tresh:
            return False
        
        return True
    
   
    """
    run_shots: 
        
    Execute the quantum algorithm to check the postselection probabilities
    
    :initial_centroids: vectors chosen as inital centroids
    
    :return: [r1, a0]:
        - r1: postselection probability on the qubit |r>
        - a0: postselection probability on the qubit |a>
    """
    def run_shots(self, initial_centroids):
        self.centroids = initial_centroids
        print("theoretical postselection probability (r)" + str(1/2**(math.ceil(math.log(self.N,2)))))
        r1, a0 = self.computing_cluster_1(check_prob=True)
        print("r1: " + str(r1))
        print("a0: " + str(a0))
        return r1, a0
    

        
    """
    run: 
        
    It executes the algorithm 
    
    :initial_centroids (optional, default_value=None): vectors chosen as initial centroids
    :seed (optional, default value=123): seed to select randomly the initial centroids
    :real_hw (optional, default value=False): if True the algorithm will be executed on real quantum hardware
    """
    def run(self, initial_centroids=None, seed=123, real_hw=False):
        
        if initial_centroids is None:
            df_centroids = self.data.sample(n=self.K, random_state=seed)
        else:
            df_centroids = pd.DataFrame(initial_centroids, columns=self.dataset.original_df.columns)
        self.centroids_norm = df_centroids.apply(lambda row: np.linalg.norm(row), axis=1).tolist()
        df_centroids = self.dataset.preprocess(df_centroids)
        '''
        df_centroids = self.dataset.scale(df_centroids)
        #df_centroids = self.dataset.normalize(df_centroids)
        df_centroids = self.dataset.inv_stereo(df_centroids)
        #df_centroids.loc[:,:] = df_centroids.loc[:,:].apply(lambda row: 2*np.arcsin(row), axis=1)
        '''
        self.centroids = df_centroids.values        

        if real_hw:
            provider = IBMQ.load_account()
        else:
            provider = None
            
        while not self.stop_condition():
            start = time.time()
            
            self.old_centroids = self.centroids.copy()
            
            print("iteration: " + str(self.ite))
            if self.quantization == 1:
                hw_time = self.computing_cluster_1(provider)
                self.execution_time_hw.append(hw_time)
            elif self.quantization == 2:
                hw_time = self.computing_cluster_2(provider)
                self.execution_time_hw.append(hw_time)
            elif self.quantization == 3:
                self.computing_cluster_3()
            elif self.quantization == 4:
                self.computing_cluster_2_flagged()
            elif self.quantization == 5:
                self.computing_cluster_3_flagged()
            self.computing_centroids()

            end = time.time()
            elapsed = end - start
            self.times.append(elapsed)
            
            # computing measures
            sim = measures.check_similarity(self.data, self.centroids, self.cluster_assignment)
            self.similarity_list.append(sim)
            self.SSE_list.append(self.SSE())
            self.silhouette_list.append(self.silhouette())
            self.nm_info_list.append(self.nm_info())
            
            
            self.ite = self.ite + 1
            
            if self.ite == self.max_iterations:
                break
        
        
    """
    avg_ite_time: 
        
    Returns the average iteration time of the algorithm
    """
    def avg_ite_time(self):
        return round(np.mean(self.times), 2)
    
    """
    avg_ite_hw_time: 
        
    Returns the average iteration time of the algorithm execution on real hardware
    """
    def avg_ite_hw_time(self):
        if len(self.execution_time_hw) > 0:
            return round(np.mean(self.execution_time_hw), 2)
        else:
            return None
    
    """
    avg_sim: 
        
    Returns the average similarity of the cluster assignment produced by the algorithm
    """
    def avg_sim(self):
        return round(np.mean(self.similarity_list), 2)
    
    
    """
    SSE: 
        
    Returns the final Sum of Squared Error
    """
    def SSE(self):
        series = []
        for i in range(self.K):
            series.append(self.dataset.original_df.loc[[index for index, n in enumerate(self.cluster_assignment) if n == i]].mean())

        centroids = pd.concat(series, axis=1).T.values
        return round(measures.SSE(self.dataset.original_df, centroids, self.cluster_assignment), 3)
    
    
    """
    silhouette: 
        
    Returns the final Silhouette score
    """
    def silhouette(self):
        if len(set(self.cluster_assignment)) <= 1 :
            return None
        else:
            return round(metrics.silhouette_score(self.dataset.original_df, self.cluster_assignment, metric='euclidean'), 3)
    
    """
    vmeasure: 
        
    Returns the final v_measure
    """
    def vmeasure(self):
        if self.dataset.ground_truth is not None:
            return round(metrics.v_measure_score(self.dataset.ground_truth, self.cluster_assignment), 3)
        else:
            return None
    
    """
    nm_info: 
        
    Returns the final Normalized Mutual Info Score
    """
    def nm_info(self):
        if self.dataset.ground_truth is not None:
            return round(metrics.normalized_mutual_info_score(self.dataset.ground_truth, self.cluster_assignment), 3)
        else:
            return None
    
    """
    save_measure: 
        
    Write into file the measures per iteration
    
    :index: number associated to the algorithm execution
    """
    def save_measures(self, index):
        filename = "result/measures/" + str(self.dataset_name) + "_qkmeans_" + str(index) + ".csv"
        measure_df = pd.DataFrame({'similarity': self.similarity_list, 'SSE': self.SSE_list, 'silhouette': self.silhouette_list, 'nm_info': self.nm_info_list})
        pd.DataFrame(measure_df).to_csv(filename)
      
        
    """
    print_result: 
        
    Write into file the results of the algorithm
    
    :filename (optional, default value=None): name of the file where to save the results
    :process: process number which executed the algorithm 
    :index_conf: configuration number associated to the algorithm execution
    """
    def print_result(self, filename=None, process=0, index_conf=0):
        self.dataset.plotOnCircle(self.data, self.centroids, self.cluster_assignment)

        avg_time = self.avg_ite_time()
        print("Average iteration time: " + str(avg_time) + " sec")
        
        avg_sim = self.avg_sim()
        print("Average similarity w.r.t classical assignment: " + str(avg_sim) + "%")
        
        SSE = self.SSE()
        print("SSE: " + str(SSE))
        
        silhouette = self.silhouette()
        print("Silhouette score: " + str(silhouette))
        
        vm = self.vmeasure()
        print("Vmeasure: " + str(vm))
        
        nminfo = self.nm_info()
        print("Normalized mutual info score: " + str(nminfo))
    
        fig, ax = plt.subplots()
        ax.plot(range(self.ite), self.similarity_list, marker="o")
        ax.set(xlabel='QKmeans iterations', ylabel='Similarity w.r.t classical assignment')
        ax.set_title("K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", M1 = " + str(self.M1) + ", shots = " + str(self.shots))
        dt = datetime.datetime.now().replace(microsecond=0)
        fig.savefig("./plot/qkmeansSim_"+str(process)+"_"+str(index_conf)+".png")
        
        if filename is not None:
            f = open(filename, 'a')
            f.write("###### TEST " + str(process)+"_"+str(index_conf) + " on " + str(self.dataset_name) + " dataset\n")
            f.write("# Executed on " + str(dt) + "\n")
            f.write("## QKMEANS\n")
            f.write("# Parameters: VERSION = " + str(self.quantization) + "K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", M1 = " + str(self.M1) + ", shots = " + str(self.shots) + "\n")
            f.write("# Iterations needed: " + str(self.ite) + "/" + str(self.max_iterations) + "\n")
            f.write('# Average iteration time: ' + str(avg_time) + 's \n')
            f.write('# Average similarity w.r.t classical assignment: ' + str(avg_sim) + '% \n')
            f.write('# SSE: ' + str(SSE) + '\n')
            f.write('# Silhouette: ' + str(silhouette) + '\n')
            f.write('# Vmeasure: ' + str(vm) + '\n')
            f.write('# Normalized mutual info score: ' + str(nminfo) + '\n')
            f.write("# Quantum kmeans assignment \n")
            f.write(str(self.cluster_assignment))            
            f.write("\n")
            f.close()
       
    """
    print_params: 
        
    Prints the parameters configuration
    """
    def print_params(self, process=0, i=0):
        print("Process " + str(process) + " - configuration: " + str(i) + 
              "\nParameters: VERSION = " + str(self.quantization) + " K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", M1 = " + str(self.M1) + ", shots = " + str(self.shots) + "\n")

