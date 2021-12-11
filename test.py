from QKmeans import QKMeans
from itertools import product
import numpy as np
import multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import pair_confusion_matrix
import time

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
        
def par_test_iris(n_processes=2):
    
    filename = 'result/iris.csv'
    
    params_iris = {

        'dataset_name': ['iris'],
        'K': [2,3,4,5],
        'M1': [2,4,8,16,32,64,128,150],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
    
    keys, values = zip(*params_iris.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]
    
    print("Iris dataset test, total configurations: " + str(len(params_list)))
    
    list_chunks = np.array_split(params_list, n_processes)
    processes = [mp.Process(target=test_iris_0, args=(chunk, i))  for i, chunk in enumerate(list_chunks)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
        print("process ", p, " terminated")

    print("Processes joined")
    
    f = open(filename, 'w')
    
    for i in range(len(processes)):
        f1_name  = "result/iris_" + str(i) + ".csv"
        f1 = open(f1_name, "r")
        f.write(f1.read())
        f.write("\n")
        f1.close()
    f.close()

def test_iris_0(chunk, n_chunk):
    
    filename  = "result/iris_" + str(n_chunk) + ".csv" 
    
    if len(chunk)==0:
        f = open(filename, 'w')
        f.close()
    
    for i, conf in enumerate(chunk):

        QKMEANS = None
    
        #print("-------------------- QKMEANS --------------------")
    
        # execute quantum kmenas
        QKMEANS = QKMeans(conf)
        QKMEANS.print_params(n_chunk, i)
        QKMEANS.run()        
        QKMEANS.print_result(filename, n_chunk, i)  
    
        #print("")
        #print("---------------- CLASSICAL KMEANS ----------------")
    
        # execute classical kmeans
        data = QKMEANS.data.loc[:,QKMEANS.data.columns[:-1]]
        initial_centroids = QKMEANS.initial_centroids.loc[:,QKMEANS.initial_centroids.columns[:-1]].values
        kmeans = KMeans(n_clusters=conf['K'], n_init=1, max_iter=conf['max_iterations'], init=initial_centroids)
        
        start = time.time()
        
        kmeans.fit(data)
        
        end = time.time()
        elapsed = end - start
        
        #print("Iterations needed: " + str(kmeans.n_iter_) + "/" + str(conf['max_iterations']))
        avg_time = elapsed / kmeans.n_iter_
        #print('Average iteration time: ' + str(avg_time) + 's \n')
        #print('SSE kmeans %s' % kmeans.inertia_)
        silhouette = silhouette_score(data, kmeans.labels_, metric='euclidean')
        #print('Silhouette kmeans %s' % silhouette)
        
        
        f = open(filename, 'a')
        f.write("## Classical KMEANS\n")
        f.write("# Iterations needed: " + str(kmeans.n_iter_) + "/" + str(conf['max_iterations']) + "\n")
        f.write('# Average iteration time: ' + str(avg_time) + 's \n')
        f.write('# SSE: ' + str(kmeans.inertia_) + '\n')
        f.write('# Silhouette: ' + str(silhouette) + '\n')
        f.write("# Classical Kmeans assignment\n")
        f.write(str(kmeans.labels_.tolist()))
        f.write("\n")

        #print("")        
        #print("------------- COMPARING THE TWO CLUSTERING ALGORITHM ------------")
        #print("pair confision matrix")
        cm = pair_confusion_matrix(QKMEANS.cluster_assignment, kmeans.labels_.tolist())
        #print(cm)

        f.write("## Comparing quantum with classical kmeans algorithm\n")
        f.write(str(cm))
        f.write("\n\n")
        f.close()

if __name__ == "__main__":
    
    par_test_iris(32)
    
    