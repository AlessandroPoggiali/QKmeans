from QKmeans import QKMeans
from itertools import product
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import pair_confusion_matrix
from utility import measures
from dataset import Dataset
import time
import datetime

def test_iris():
    
    filename = 'result/iris.csv'
    
    params_iris = {

        'dataset_name': ['iris'],
        'K': [2],
        'M1': [4],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
    
    keys, values = zip(*params_iris.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]
    
    print("Iris dataset test, total configurations: " + str(len(params_list)))
    
    for i, params in enumerate(params_list):

        QKMEANS = None
        print("Configuration: " + str(i) +"\n")

        conf = {
            "dataset_name": params['dataset_name'],
            "K": params['K'],
            "M1": params['M1'],
            "sc_tresh": params['sc_tresh'],
            "max_iterations": params['max_iterations'] 
        }
    
        print("-------------------- QKMEANS --------------------")
    
        # execute quantum kmenas
        QKMEANS = QKMeans(conf)
        QKMEANS.print_params()
        QKMEANS.run()
        QKMEANS.print_result(filename, i)  
    
        print("")
        print("---------------- CLASSICAL KMEANS ----------------")
    
        # execute classical kmeans
        data = QKMEANS.data.loc[:,QKMEANS.data.columns[:-1]]
        initial_centroids = QKMEANS.initial_centroids.loc[:,QKMEANS.initial_centroids.columns[:-1]].values
        kmeans = KMeans(n_clusters=conf['K'], n_init=1, max_iter=conf['max_iterations'], init=initial_centroids)
        
        start = time.time()
        
        kmeans.fit(data)
        
        end = time.time()
        elapsed = end - start
        
        print("Iterations needed: " + str(kmeans.n_iter_) + "/" + str(conf['max_iterations']))
        avg_time = elapsed / kmeans.n_iter_
        print('Average iteration time: ' + str(avg_time) + 's \n')
        print('SSE kmeans %s' % kmeans.inertia_)
        silhouette = silhouette_score(data, kmeans.labels_, metric='euclidean')
        print('Silhouette kmeans %s' % silhouette)
        
        
        f = open(filename, 'a')
        f.write("## Classical KMEANS\n")
        f.write("# Iterations needed: " + str(kmeans.n_iter_) + "/" + str(conf['max_iterations']) + "\n")
        f.write('# Average iteration time: ' + str(avg_time) + 's \n')
        f.write('# SSE: ' + str(kmeans.inertia_) + '\n')
        f.write('# Silhouette: ' + str(silhouette) + '\n')
        f.write("# Classical Kmeans assignment\n")
        f.write(str(kmeans.labels_.tolist()))
        f.write("\n")

        print("")        
        print("------------- COMPARING THE TWO CLUSTERING ALGORITHM ------------")
        print("pair confision matrix")
        cm = pair_confusion_matrix(QKMEANS.cluster_assignment, kmeans.labels_.tolist())
        print(cm)

        f.write("## Comparing quantum with classical kmeans algorithm\n")
        f.write(str(cm))
        f.write("\n\n")
        f.close()
        
def par_test(params, dataset, algorithm='quantum', n_processes=2, seed=123):
    
    
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]
    
    print("Iris dataset test, total configurations: " + str(len(params_list)))
    
    list_chunks = np.array_split(params_list, n_processes)
    if algorithm == "qkmeans":
        processes = [mp.Process(target=QKmeans_test, args=(dataset, chunk, i, seed, n_processes))  for i, chunk in enumerate(list_chunks)]
    elif algorithm == "kmeans":
        processes = [mp.Process(target=kmeans_test, args=(dataset, chunk, i, seed, n_processes))  for i, chunk in enumerate(list_chunks)]
    elif algorithm == "deltakmeans":
        processes = [mp.Process(target=delta_kmeans_test, args=(dataset, chunk, i, seed, n_processes))  for i, chunk in enumerate(list_chunks)]
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
    f.write("index,date,K,M,N,M1,shots,n_circuits,max_qbits,n_ite,avg_ite_time,avg_similarity,SSE,silhouette,v_measure,nm_info\n")
    
    
    for i in range(len(processes)):
        f1_name  = "result/iris_qkmeans_" + str(i) + ".csv"
        f1 = open(f1_name, "r")
        f.write(f1.read())
        f1.close()
    f.close()


def QKmeans_test(dataset, chunk, n_chunk, seed, n_processes):
    
    filename  = "result/iris_qkmeans_" + str(n_chunk) + ".csv" 
    f = open(filename, "w")
    
    if len(chunk)==0:
        f = open(filename, 'w')
        f.close()
    
    for i, conf in enumerate(chunk):

        index = n_processes*n_chunk + i
        dt = datetime.datetime.now().replace(microsecond=0)
        
        QKMEANS = None
        
        # execute quantum kmenas
        QKMEANS = QKMeans(dataset, conf, seed)
        QKMEANS.print_params(n_chunk, i)
        QKMEANS.run()        
        
        f = open(filename, 'a')
        f.write(str(index) + ",")
        f.write(str(dt) + ",")
        f.write(str(QKMEANS.K) + ",")
        f.write(str(QKMEANS.M) + ",")
        f.write(str(QKMEANS.N) + ",")
        f.write(str(QKMEANS.M1) + ",")
        f.write(str(QKMEANS.shots) + ",")
        f.write(str(QKMEANS.n_circuits) + ",")
        f.write(str(QKMEANS.max_qbits) + ",")
        f.write(str(QKMEANS.ite) + ",")
        f.write(str(QKMEANS.avg_ite_time()) + ",")
        f.write(str(QKMEANS.avg_sim()) + ",")
        f.write(str(QKMEANS.SSE()) + ",")
        f.write(str(QKMEANS.silhouette()) + ",")
        f.write(str(QKMEANS.vmeasure()) + ",")
        f.write(str(QKMEANS.nm_info()) + "\n")

        #QKMEANS.print_result(filename, n_chunk, i)
        filename_assignment = "result/assignment/qkmeans_" + str(index) + ".csv"
        assignment_df = pd.DataFrame(QKMEANS.cluster_assignment, columns=['cluster'])
        pd.DataFrame(assignment_df).to_csv(filename_assignment)

def kmeans_test(dataset, chunk, n_chunk, seed, n_processes):
    
    filename  = "result/iris_kmeans_" + str(n_chunk) + ".csv" 
    
    if len(chunk)==0:
        f = open(filename, 'w')
        f.close()
    
    for i, conf in enumerate(chunk):
    
        # execute classical kmeans
        data = dataset.df
        initial_centroids = data.sample(n=conf['K'], random_state=seed).values
        kmeans = KMeans(n_clusters=conf['K'], n_init=1, max_iter=conf['max_iterations'], init=initial_centroids)
        
        start = time.time()
        
        kmeans.fit(data)
        
        end = time.time()
        elapsed = end - start
        
        #print("Iterations needed: " + str(kmeans.n_iter_) + "/" + str(conf['max_iterations']))
        avg_time = elapsed / kmeans.n_iter_
        #print('Average iteration time: ' + str(avg_time) + 's \n')
        #print('SSE kmeans %s' % kmeans.inertia_)
        sse = measures.SSE(data, kmeans.cluster_centers_, kmeans.labels_)
        silhouette = silhouette_score(data, kmeans.labels_, metric='euclidean')
        #print('Silhouette kmeans %s' % silhouette)
        '''
        f = open(filename, 'a')
        f.write("## Classical KMEANS\n")
        f.write("# Iterations needed: " + str(kmeans.n_iter_) + "/" + str(conf['max_iterations']) + "\n")
        f.write('# Average iteration time: ' + str(avg_time) + 's \n')
        f.write('# SSE: ' + str(sse) + '\n')
        f.write('# Silhouette: ' + str(silhouette) + '\n')
        f.write("# Classical Kmeans assignment\n")
        f.write(str(kmeans.labels_.tolist()))
        f.write("\n")

        f.close()
        '''
def delta_kmeans_test(dataset, chunk, n_chunk, seed, n_processes):
    return
    

if __name__ == "__main__":
    
    print("---------------------- Iris Test ----------------------")
    
    params = {
        'dataset_name': ['iris'],
        'K': [2],
        'M1': [2,4,8,16],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
    
    processes = 2
    seed = 123
    dataset = Dataset('iris')
    
    print("-------------------- Quantum Kmeans --------------------")
    par_test(params, dataset, algorithm="qkmeans", n_processes=processes, seed=seed)
    print("-------------------- Classical Kmeans --------------------")
    par_test(params, dataset, algorithm="kmeans", n_processes=processes, seed=seed)
    print("-------------------- Delta Kmeans --------------------")
    par_test(params, dataset, algorithm="deltakmeans", n_processes=processes, seed=seed)
    
    '''
    make_comparison() # qui per esempio mi calcolo tutte le pair confusion matrix a partire dagli assegnamenti classici quantistici e delta facendo un file dove per ogni configurazione ho due confusion matrix
    make_plot()
    '''
    