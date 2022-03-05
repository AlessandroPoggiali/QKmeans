from QKmeans import QKMeans
from deltakmeans import DeltaKmeans
from itertools import product
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.cluster import KMeans, kmeans_plusplus
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn import metrics
import matplotlib.pyplot as plt
from utility import measures
from dataset import Dataset
import time
import datetime
import sys
import math

font = {'size'   : 22}

plt.rc('font', **font)


delta = 0.5
seed = 123
        
def par_test(params, dataset, algorithm='qkmeans', n_processes=2, seed=123):
    
    if algorithm != 'qkmeans':
        params['M1'] = [None]
        params['shots'] = [None]
        params['quantization'] = [None]
    
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]
    
    print(str(dataset.dataset_name) + " dataset test, total configurations: " + str(len(params_list)))
    
    list_chunks = np.array_split(params_list, n_processes)
    
    t = 0
    indexlist = [[0]*j for i,j in enumerate([len(x) for x in list_chunks])]
    for i,index in enumerate(indexlist):
        for j in range(len(index)):
            indexlist[i][j] = t
            t = t + 1
    
    if algorithm == "qkmeans":
        processes = [mp.Process(target=QKmeans_test, args=(dataset, chunk, i, seed, indexlist))  for i, chunk in enumerate(list_chunks)]
    elif algorithm == "kmeans":
        processes = [mp.Process(target=kmeans_test, args=(dataset, chunk, i, seed, indexlist))  for i, chunk in enumerate(list_chunks)]
    elif algorithm == "deltakmeans":
        processes = [mp.Process(target=delta_kmeans_test, args=(dataset, chunk, i, seed, indexlist))  for i, chunk in enumerate(list_chunks)]
    else: 
        print("ERROR: wrong algorithm parameter (use 'quantum', 'classical' or 'delta'")
        return

    for p in processes:
        p.start()

    for p in processes:
        p.join()
        print("process ", p, " terminated")

    print("Processes joined")

    filename = "result/" + str(params["dataset_name"][0]) + "_" + str(algorithm) + ".csv"
    f = open(filename, 'w')
    if algorithm == "qkmeans":
        f.write("index,date,q_v,K,M,N,M1,shots,n_circuits,max_qbits,n_ite,avg_ite_time,treshold,avg_similarity,SSE,silhouette,v_measure,nm_info\n")
        for i in range(len(processes)):
            f1_name  = "result/" + str(dataset.dataset_name) + "_qkmeans_" + str(i) + ".csv"
            f1 = open(f1_name, "r")
            f.write(f1.read())
            f1.close()
    elif algorithm == "kmeans":
        f.write("index,date,K,M,N,n_ite,avg_ite_time,treshold,SSE,silhouette,v_measure,nm_info\n")
        for i in range(len(processes)):
            f1_name  = "result/" + str(dataset.dataset_name) + "_kmeans_" + str(i) + ".csv"
            f1 = open(f1_name, "r")
            f.write(f1.read())
            f1.close()
    else:
        f.write("index,date,K,M,N,n_ite,avg_ite_time,treshold,avg_similarity,SSE,silhouette,v_measure,nm_info\n")
        for i in range(len(processes)):
            f1_name  = "result/" + str(dataset.dataset_name) + "_deltakmeans_" + str(i) + ".csv"
            f1 = open(f1_name, "r")
            f.write(f1.read())
            f1.close()
    
    f.close()


def QKmeans_test(dataset, chunk, n_chunk, seed, indexlist):
    
    filename  = "result/" + str(dataset.dataset_name) + "_qkmeans_" + str(n_chunk) + ".csv" 
    f = open(filename, "w")
    
    if len(chunk)==0:
        f = open(filename, 'w')
        f.close()
    
    for i, conf in enumerate(chunk):

        index = indexlist[n_chunk][i]
        dt = datetime.datetime.now().replace(microsecond=0)
        
        # execute quantum kmenas
        QKMEANS = QKMeans(dataset, conf)        
        if conf['random_init_centroids']:
            initial_centroids = dataset.df.sample(n=conf['K'], random_state=seed).values
        else:
            
            #original = dataset.load_dataset(dataset.dataset_name, preprocessing=False)
            #initial_centroids, indices = kmeans_plusplus(original.values, n_clusters=conf['K'], random_state=seed)
            #new_centroids = pd.DataFrame(initial_centroids, columns=dataset.df.columns)
            #preprocessed_centroids = dataset.scale(new_centroids)
            #preprocessed_centroids = dataset.normalize(preprocessed_centroids)
            initial_centroids, indices = kmeans_plusplus(dataset.df.values, n_clusters=conf['K'], random_state=seed)
            #initial_centroids = kmeans_plusplus_initializer(dataset.df.values, conf['K'], kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
        #dataset.plot2Features(QKMEANS.data, 'f0', 'f1', initial_centroids, initial_space=True, dataset_name=dataset.dataset_name, filename='plot/qinit_orig_'+str(index), conf=conf, algorithm='qkmeans')
        #dataset.plot2Features(QKMEANS.data, 'f0', 'f1', preprocessed_centroids.values, filename='plot/qinit_'+str(index), conf=conf, algorithm='qkmeans')
        
        filename_centroids = "result/initial_centroids/" + str(dataset.dataset_name) + "_qkmeans_" + str(index) + ".csv"
        centroids_df = pd.DataFrame(initial_centroids, columns=dataset.df.columns)
        pd.DataFrame(centroids_df).to_csv(filename_centroids)
        
        QKMEANS.print_params(n_chunk, i)
        
        QKMEANS.run(initial_centroids=initial_centroids)    
        
        
        f = open(filename, 'a')
        f.write(str(index) + ",")
        f.write(str(dt) + ",")
        f.write(str(conf['quantization']) + ",")
        f.write(str(QKMEANS.K) + ",")
        f.write(str(QKMEANS.M) + ",")
        f.write(str(QKMEANS.N) + ",")
        f.write(str(QKMEANS.M1) + ",")
        f.write(str(QKMEANS.shots) + ",")
        f.write(str(QKMEANS.n_circuits) + ",")
        f.write(str(QKMEANS.max_qbits) + ",")
        f.write(str(QKMEANS.ite) + ",")
        f.write(str(QKMEANS.avg_ite_time()) + ",")
        f.write(str(conf['sc_tresh']) + ",")
        f.write(str(QKMEANS.avg_sim()) + ",")
        f.write(str(QKMEANS.SSE()) + ",")
        f.write(str(QKMEANS.silhouette()) + ",")
        f.write(str(QKMEANS.vmeasure()) + ",")
        f.write(str(QKMEANS.nm_info()) + "\n")
        f.close()
        
        #QKMEANS.print_result(filename, n_chunk, i)
        filename_assignment = "result/assignment/" + str(dataset.dataset_name) + "_qkmeans_" + str(index) + ".csv"
        assignment_df = pd.DataFrame(QKMEANS.cluster_assignment, columns=['cluster'])
        pd.DataFrame(assignment_df).to_csv(filename_assignment)
        
        QKMEANS.save_measures(index)
        

def kmeans_test(dataset, chunk, n_chunk, seed, indexlist):
    
    filename  = "result/" + str(dataset.dataset_name) + "_kmeans_" + str(n_chunk) + ".csv" 
    f = open(filename, "w")
    
    if len(chunk)==0:
        f = open(filename, 'w')
        f.close()
    
    for i, conf in enumerate(chunk):
    
        index = indexlist[n_chunk][i]
        
        # execute classical kmeans
        data = dataset.df
        if conf['random_init_centroids']:
            initial_centroids = data.sample(n=conf['K'], random_state=seed).values
        else:
            initial_centroids, indices = kmeans_plusplus(data.values, n_clusters=conf['K'], random_state=seed)
            #initial_centroids = kmeans_plusplus_initializer(dataset.df.values, conf['K'], kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
        
        filename_centroids = "result/initial_centroids/" + str(dataset.dataset_name) + "_kmeans_" + str(index) + ".csv"
        centroids_df = pd.DataFrame(initial_centroids, columns=dataset.df.columns)
        pd.DataFrame(centroids_df).to_csv(filename_centroids)
        #dataset.plot2Features(data, 'f0', 'f1', initial_centroids, filename='plot/kinit_'+str(index), conf=conf, algorithm='kmeans')
        
        if conf['sc_tresh'] != 0:
            kmeans = KMeans(n_clusters=conf['K'], n_init=1, max_iter=conf['max_iterations'], init=initial_centroids, tol=conf['sc_tresh'])
        else:
            kmeans = KMeans(n_clusters=conf['K'], n_init=1, max_iter=conf['max_iterations'], init=initial_centroids)
        
        start = time.time()
        kmeans.fit(data)
        end = time.time()
        elapsed = end - start
        
        
        dt = datetime.datetime.now().replace(microsecond=0)
        
        #print("Iterations needed: " + str(kmeans.n_iter_) + "/" + str(conf['max_iterations']))
        avg_time = round((elapsed / kmeans.n_iter_), 2)
        #print('Average iteration time: ' + str(avg_time) + 's \n')
        #print('SSE kmeans %s' % kmeans.inertia_)
        sse = round(measures.SSE(data, kmeans.cluster_centers_, kmeans.labels_), 3)
        silhouette = round(metrics.silhouette_score(data, kmeans.labels_, metric='euclidean'), 3)
        if dataset.ground_truth is not None:
            vmeasure = round(metrics.v_measure_score(dataset.ground_truth, kmeans.labels_), 3)
            nm_info = round(metrics.normalized_mutual_info_score(dataset.ground_truth, kmeans.labels_), 3)
        else:
            vmeasure = None
            nm_info = None
        #print('Silhouette kmeans %s' % silhouette)
        f = open(filename, 'a')
        f.write(str(index) + ",")
        f.write(str(dt) + ",")
        f.write(str(conf['K']) + ",")
        f.write(str(dataset.M) + ",")
        f.write(str(dataset.N) + ",")
        f.write(str(kmeans.n_iter_) + ",")
        f.write(str(avg_time) + ",")
        f.write(str(conf['sc_tresh']) + ",")
        f.write(str(sse) + ",")
        f.write(str(silhouette) + ",")
        f.write(str(vmeasure) + ",")
        f.write(str(nm_info) + "\n")
        f.close()
        
        filename_assignment = "result/assignment/" + str(dataset.dataset_name) + "_kmeans_" + str(index) + ".csv"
        assignment_df = pd.DataFrame(kmeans.labels_, columns=['cluster'])
        pd.DataFrame(assignment_df).to_csv(filename_assignment)
        
        
def delta_kmeans_test(dataset, chunk, n_chunk, seed, indexlist):
    filename  = "result/" + str(dataset.dataset_name) + "_deltakmeans_" + str(n_chunk) + ".csv" 
    f = open(filename, "w")
    
    if len(chunk)==0:
        f = open(filename, 'w')
        f.close()
    
    for i, conf in enumerate(chunk):

        index = indexlist[n_chunk][i]
        dt = datetime.datetime.now().replace(microsecond=0)
        
        # execute delta kmenas
        deltakmeans = DeltaKmeans(dataset, conf, delta)        
        if conf['random_init_centroids']:
            initial_centroids = dataset.df.sample(n=conf['K'], random_state=seed).values
        else:
            initial_centroids, indices = kmeans_plusplus(dataset.df.values, n_clusters=conf['K'], random_state=seed)
            #initial_centroids = kmeans_plusplus_initializer(dataset.df.values, conf['K'], kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE).initialize()
        #dataset.plot2Features(deltakmeans.data, 'f0', 'f1', initial_centroids, filename='plot/deltainit_'+str(index), conf=conf, algorithm='deltameans')
        filename_centroids = "result/initial_centroids/" + str(dataset.dataset_name) + "_deltakmeans_" + str(index) + ".csv"
        centroids_df = pd.DataFrame(initial_centroids, columns=dataset.df.columns)
        pd.DataFrame(centroids_df).to_csv(filename_centroids)
        
        
        deltakmeans.print_params(n_chunk, i)
        deltakmeans.run(initial_centroids=initial_centroids) 

        f = open(filename, 'a')
        f.write(str(index) + ",")
        f.write(str(dt) + ",")
        f.write(str(deltakmeans.K) + ",")
        f.write(str(deltakmeans.M) + ",")
        f.write(str(deltakmeans.N) + ",")
        f.write(str(deltakmeans.ite) + ",")
        f.write(str(deltakmeans.avg_ite_time()) + ",")
        f.write(str(conf['sc_tresh']) + ",")
        f.write(str(deltakmeans.avg_sim()) + ",")
        f.write(str(deltakmeans.SSE()) + ",")
        f.write(str(deltakmeans.silhouette()) + ",")
        f.write(str(deltakmeans.vmeasure()) + ",")
        f.write(str(deltakmeans.nm_info()) + "\n")
        f.close()
        
        #QKMEANS.print_result(filename, n_chunk, i)
        filename_assignment = "result/assignment/" + str(dataset.dataset_name) +  "_deltakmeans_" + str(index) + ".csv"
        assignment_df = pd.DataFrame(deltakmeans.cluster_assignment, columns=['cluster'])
        pd.DataFrame(assignment_df).to_csv(filename_assignment)
        
        deltakmeans.save_measures(index)
        

def plot_initial_centroids(params, dataset, algorithm):
    if algorithm != 'qkmeans':
        params['M1'] = [None]
        params['shots'] = [None]
    
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]
        
    for i, params in enumerate(params_list):
        
        conf = {
            "dataset_name": params['dataset_name'],
            "K": params['K'],
            "M1": params['M1'],
            "sc_tresh": params['sc_tresh'],
            "max_iterations": params['max_iterations'] 
        }

        input_filename = "result/initial_centroids/" + str(dataset.dataset_name) + "_" + str(algorithm) + "_" + str(i) + ".csv"
        df_centroids = pd.read_csv(input_filename, sep=',')
        df_centroids = df_centroids.drop(df_centroids.columns[0], axis=1)
        output_filename = "plot/initial_centroids/" + str(dataset.dataset_name) + "_" + str(algorithm) + "_" + str(i) + ".png"
        dataset.plot2Features(dataset.df, dataset.df.columns[0], dataset.df.columns[1], df_centroids.values, filename=output_filename, conf=conf, algorithm=algorithm)
        

    
def plot_cluster(params, dataset, algorithm, seed):
    
    if algorithm != 'qkmeans':
        params['M1'] = [None]
        params['shots'] = [None]
    
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]
        
    for i, params in enumerate(params_list):
        
        conf = {
            "dataset_name": params['dataset_name'],
            "K": params['K'],
            "M1": params['M1'],
            "sc_tresh": params['sc_tresh'],
            "max_iterations": params['max_iterations'] 
        }

        input_filename = "result/assignment/" + str(dataset.dataset_name) + "_" + str(algorithm) + "_" + str(i) + ".csv"
        df_assignment = pd.read_csv(input_filename, sep=',')
        cluster_assignment = df_assignment['cluster']
        
        output_filename = "plot/cluster/" + str(dataset.dataset_name) + "_" + str(algorithm) + "_" + str(i) + ".png"
        dataset.plot2Features(dataset.df, dataset.df.columns[0], dataset.df.columns[1], cluster_assignment=cluster_assignment,
                              initial_space=True, dataset_name=dataset.dataset_name, seed=seed, filename=output_filename, conf=conf, algorithm=algorithm)
    
    
def shots_test():
    datasets = ['noisymoon','blobs','blobs2','aniso']
    datasets = ['noisymoon']
    for data in datasets:
        params = {
            'quantization': [3],
            'dataset_name': [data],
            'random_init_centroids': [False],
            'K': [2],
            'M1': [x for x in range(96,129)],
            'shots': [8192],
            'sc_tresh':  [0],
            'max_iterations': [1]
        }
        
        keys, values = zip(*params.items())
        params_list = [dict(zip(keys, v)) for v in product(*values)]
        
        filename = "result/probabilities/" + data + ".csv"

        probabilities = pd.DataFrame(columns=['M1','N','K','p(R=1)_theo','p(R=1)_1-norm','p(R=1)_inf-norm'])

        for i, params in enumerate(params_list):
        
            conf = {
                'quantization': params['quantization'],
                "dataset_name": params['dataset_name'],
                "random_init_centroids": params['random_init_centroids'],
                "K": params['K'],
                "M1": params['M1'],
                'shots': params['shots'],
                "sc_tresh": params['sc_tresh'],
                "max_iterations": params['max_iterations'] 
            }
            
            '''
            dataset = Dataset(data, '1-norm')
            QKMEANS = QKMeans(dataset, conf)      
            
            QKMEANS.print_params(0, i)
            print("1-norm")
            
            if conf['random_init_centroids']:
                initial_centroids = dataset.df.sample(n=conf['K'], random_state=seed).values
            else:
                initial_centroids, indices = kmeans_plusplus(dataset.df.values, n_clusters=conf['K'], random_state=seed)
            
            r1_1norm = QKMEANS.run_shots(initial_centroids=initial_centroids)
            '''
            dataset = Dataset(data, 'inf-norm')
            QKMEANS = QKMeans(dataset, conf)      
            
            QKMEANS.print_params(0, i)
            print("inf-norm")
            
            if conf['random_init_centroids']:
                initial_centroids = dataset.df.sample(n=conf['K'], random_state=seed).values
            else:
                initial_centroids, indices = kmeans_plusplus(dataset.df.values, n_clusters=conf['K'], random_state=seed)
            
            r1_infnorm = QKMEANS.run_shots(initial_centroids=initial_centroids)
            
            r1_1norm = 0
            r1_theo = 1/2**(math.ceil(math.log(dataset.N,2)))
            
            df1 = {'M1': conf['M1'], 'N': dataset.N, 'K': conf['K'], 'p(R=1)_theo': r1_theo ,'p(R=1)_1-norm': r1_1norm ,'p(R=1)_inf-norm': r1_infnorm }
            probabilities = probabilities.append(df1, ignore_index = True)
            
        pd.DataFrame(probabilities).to_csv(filename)


if __name__ == "__main__":
    
    shots_test()
    exit()
    
    if len(sys.argv) != 2:
        print("ERROR: type '" + str(sys.argv[0]) + " n_processes' to execute the test")
        exit()
   
    try:
        processes = int(sys.argv[1])
    except ValueError:
        print("ERROR: specify a positive integer for the number of processes")
        exit()
    
    if processes < 0:
        print("ERROR: specify a positive integer for the number of processes")
        exit()
        
   
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                IRIS DATASET TEST
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    
    params = {
        'quantization': [2],
        'dataset_name': ['iris'],
        'random_init_centroids': [False],
        'K': [3],
        'M1': [150],
        'shots': [8192],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
    
    dataset = Dataset('iris', '1-norm')
    
    print("---------------------- " + str(dataset.dataset_name) + " Test ----------------------\n")
    
    print("-------------------- Quantum Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="qkmeans", n_processes=processes, seed=seed)
    
    print("-------------------- Classical Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="kmeans", n_processes=processes, seed=seed)
    
    print("-------------------- Delta Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="deltakmeans", n_processes=processes, seed=seed)    
    
    plot_cluster(dict(params), dataset, algorithm='qkmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='deltakmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='kmeans', seed=seed)
    plot_initial_centroids(dict(params), dataset, algorithm='qkmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='deltakmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='kmeans')
    
    exit()

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                ANISO DATASET TEST
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''
    
    params = {
        'dataset_name': ['aniso'],
        'random_init_centroids': [False],
        'K': [3],
        'M1': [2,4,8,16,32,64,128,150],
        'shots': [None],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
    
    dataset = Dataset('aniso', 'z-norm')
    
    print("---------------------- " + str(dataset.dataset_name) + " Test ----------------------\n")
    
    print("-------------------- Quantum Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="qkmeans", n_processes=processes, seed=seed)
    
    print("-------------------- Classical Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="kmeans", n_processes=processes, seed=seed)
    
    print("-------------------- Delta Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="deltakmeans", n_processes=processes, seed=seed)    
    
    plot_cluster(dict(params), dataset, algorithm='qkmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='deltakmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='kmeans', seed=seed)
    plot_initial_centroids(dict(params), dataset, algorithm='qkmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='deltakmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='kmeans')
    
    '''
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                BLOBS DATASET TEST
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    

    params = {
        'quantization': [2],
        'dataset_name': ['blobs'],
        'random_init_centroids': [False],
        'K': [3],
        'M1': [128],
        'shots': [8192],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
     
    dataset = Dataset('blobs', 'inf-norm')
    
    print("---------------------- " + str(dataset.dataset_name) + " Test ----------------------\n")
    
    print("-------------------- Quantum Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="qkmeans", n_processes=processes, seed=seed)
        
    print("-------------------- Classical Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="kmeans", n_processes=processes, seed=seed)
    
    print("-------------------- Delta Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="deltakmeans", n_processes=processes, seed=seed)   
    
    plot_cluster(dict(params), dataset, algorithm='qkmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='deltakmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='kmeans', seed=seed)
    plot_initial_centroids(dict(params), dataset, algorithm='qkmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='deltakmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='kmeans')
    
    exit()
    
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                BLOBS2 DATASET TEST
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''

    params = {
        'dataset_name': ['blobs2'],
        'random_init_centroids': [False],
        'K': [3],
        'M1': [2,4,8,16,32,64,128,150],
        'shots': [None],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
     
    dataset = Dataset('blobs2', 'z-norm')
    
    print("---------------------- " + str(dataset.dataset_name) + " Test ----------------------\n")
    
    print("-------------------- Quantum Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="qkmeans", n_processes=processes, seed=seed)
        
    print("-------------------- Classical Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="kmeans", n_processes=processes, seed=seed)
    
    print("-------------------- Delta Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="deltakmeans", n_processes=processes, seed=seed)    
    
    plot_cluster(dict(params), dataset, algorithm='qkmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='deltakmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='kmeans', seed=seed)
    plot_initial_centroids(dict(params), dataset, algorithm='qkmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='deltakmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='kmeans')
      
    '''

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                NOISYMOON DATASET TEST
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    params = {
        'dataset_name': ['noisymoon'],
        'random_init_centroids': [False],
        'K': [2],
        'M1': [150],
        'shots': [50000],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
     
    dataset = Dataset('noisymoon', '1-norm')
    
    print("---------------------- " + str(dataset.dataset_name) + " Test ----------------------\n")
    
    print("-------------------- Quantum Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="qkmeans", n_processes=processes, seed=seed)
        
    print("-------------------- Classical Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="kmeans", n_processes=processes, seed=seed)
    
    print("-------------------- Delta Kmeans --------------------")
    par_test(dict(params), dataset, algorithm="deltakmeans", n_processes=processes, seed=seed)    
    
    plot_cluster(dict(params), dataset, algorithm='qkmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='deltakmeans', seed=seed)
    plot_cluster(dict(params), dataset, algorithm='kmeans', seed=seed)
    plot_initial_centroids(dict(params), dataset, algorithm='qkmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='deltakmeans')
    plot_initial_centroids(dict(params), dataset, algorithm='kmeans')