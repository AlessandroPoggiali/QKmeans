from QKmeans import QKMeans
from deltakmeans import DeltaKmeans
from itertools import product
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import pair_confusion_matrix
import matplotlib.pyplot as plt
from utility import measures
from dataset import Dataset
import time
import datetime

delta = 0.5

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
        silhouette = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean')
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
        
def par_test(params, dataset, algorithm='qkmeans', n_processes=2, seed=123):
    
    
    keys, values = zip(*params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]
    
    print(str(dataset.dataset_name) + " dataset test, total configurations: " + str(len(params_list)))
    
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
    if algorithm == "qkmeans":
        f.write("index,date,K,M,N,M1,shots,n_circuits,max_qbits,n_ite,avg_ite_time,avg_similarity,SSE,silhouette,v_measure,nm_info\n")
        for i in range(len(processes)):
            f1_name  = "result/" + str(dataset.dataset_name) + "_qkmeans_" + str(i) + ".csv"
            f1 = open(f1_name, "r")
            f.write(f1.read())
            f1.close()
    elif algorithm == "kmeans":
        f.write("index,date,K,M,N,n_ite,avg_ite_time,SSE,silhouette,v_measure,nm_info\n")
        for i in range(len(processes)):
            f1_name  = "result/" + str(dataset.dataset_name) + "_kmeans_" + str(i) + ".csv"
            f1 = open(f1_name, "r")
            f.write(f1.read())
            f1.close()
    else:
        f.write("index,date,K,M,N,n_ite,avg_ite_time,SSE,silhouette,v_measure,nm_info\n")
        for i in range(len(processes)):
            f1_name  = "result/" + str(dataset.dataset_name) + "_deltakmeans_" + str(i) + ".csv"
            f1 = open(f1_name, "r")
            f.write(f1.read())
            f1.close()
    
    f.close()


def QKmeans_test(dataset, chunk, n_chunk, seed, n_processes):
    
    filename  = "result/" + str(dataset.dataset_name) + "_qkmeans_" + str(n_chunk) + ".csv" 
    f = open(filename, "w")
    
    if len(chunk)==0:
        f = open(filename, 'w')
        f.close()
    
    for i, conf in enumerate(chunk):

        index = n_processes*n_chunk + i
        dt = datetime.datetime.now().replace(microsecond=0)
        
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
        f.close()
        
        #QKMEANS.print_result(filename, n_chunk, i)
        filename_assignment = "result/assignment/" + str(dataset.dataset_name) + "_qkmeans_" + str(index) + ".csv"
        assignment_df = pd.DataFrame(QKMEANS.cluster_assignment, columns=['cluster'])
        pd.DataFrame(assignment_df).to_csv(filename_assignment)
        
        QKMEANS.save_similarities(index)

def kmeans_test(dataset, chunk, n_chunk, seed, n_processes):
    
    filename  = "result/" + str(dataset.dataset_name) + "_kmeans_" + str(n_chunk) + ".csv" 
    f = open(filename, "w")
    
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
        
        index = n_processes*n_chunk + i
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
        f.write(str(sse) + ",")
        f.write(str(silhouette) + ",")
        f.write(str(vmeasure) + ",")
        f.write(str(nm_info) + "\n")
        f.close()
        
        filename_assignment = "result/assignment/" + str(dataset.dataset_name) + "_kmeans_" + str(index) + ".csv"
        assignment_df = pd.DataFrame(kmeans.labels_, columns=['cluster'])
        pd.DataFrame(assignment_df).to_csv(filename_assignment)
        
        
def delta_kmeans_test(dataset, chunk, n_chunk, seed, n_processes):
    filename  = "result/" + str(dataset.dataset_name) + "_deltakmeans_" + str(n_chunk) + ".csv" 
    f = open(filename, "w")
    
    if len(chunk)==0:
        f = open(filename, 'w')
        f.close()
    
    for i, conf in enumerate(chunk):

        index = n_processes*n_chunk + i
        dt = datetime.datetime.now().replace(microsecond=0)
        
        # execute delta kmenas
        deltakmeans = DeltaKmeans(dataset, conf, delta, seed)
        deltakmeans.print_params(n_chunk, i)
        deltakmeans.run()        
        
        f = open(filename, 'a')
        f.write(str(index) + ",")
        f.write(str(dt) + ",")
        f.write(str(deltakmeans.K) + ",")
        f.write(str(deltakmeans.M) + ",")
        f.write(str(deltakmeans.N) + ",")
        f.write(str(deltakmeans.ite) + ",")
        f.write(str(deltakmeans.avg_ite_time()) + ",")
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
        
        deltakmeans.save_similarities(index)
    
def plot_similarity(params, dataset, algorithm):
    
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
        
        if algorithm == 'qkmeans':
            strfile = "qkmeansSim"
        elif algorithm == 'deltakmeans':
            strfile = "deltakmeansSim"
        
        input_filename = "result/similarity/" + str(dataset.dataset_name) + "_" + strfile + "_" + str(i) + ".csv"
        df_sim = pd.read_csv(input_filename, sep=',')
        
        
        fig, ax = plt.subplots()
        ax.plot(df_sim['similarity'], marker="o")
        ax.set(xlabel='QKmeans iterations', ylabel='Similarity w.r.t classical assignment')
        ax.set_title("K = " + str(conf["K"]) + ", M = " + str(dataset.M) + ", N = " + str(dataset.N) + ", M1 = " + str(conf["M1"]))
   
        #str_dt = str(dt).replace(" ", "_")
        fig.savefig("./plot/" + str(dataset.dataset_name) + "_" +  strfile + "_"+str(i) + ".png")

def plot_cluster(params, dataset, algorithm, seed):
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
        
        if algorithm == 'qkmeans':
            input_filename = "result/assignment/" + str(dataset.dataset_name) + "_qkmeans" + "_" + str(i) + ".csv"
            df_assignment = pd.read_csv(input_filename, sep=',')
            cluster_assignment = df_assignment['cluster']
            
            output_filename = "plot/cluster/" + str(dataset.dataset_name) + "_qkmeans" + "_" + str(i) + ".png"
            dataset.plot2Features(dataset.df, dataset.df.columns[0], dataset.df.columns[1], cluster_assignment=cluster_assignment,
                                  initial_space=True, dataset_name=dataset.dataset_name, seed=seed, filename=output_filename, conf=conf, algorithm=algorithm)
        elif algorithm == 'deltakmeans':
            input_filename = "result/assignment/" + str(dataset.dataset_name) + "_deltakmeans" + "_" + str(i) + ".csv"
            df_assignment = pd.read_csv(input_filename, sep=',')
            cluster_assignment = df_assignment['cluster']
            
            output_filename = "plot/cluster/"+ str(dataset.dataset_name) + "_deltakmeans" + "_" + str(i) + ".png"
            dataset.plot2Features(dataset.df, dataset.df.columns[0], dataset.df.columns[1], cluster_assignment=cluster_assignment,
                                  initial_space=True, dataset_name=dataset.dataset_name, seed=seed, filename=output_filename, conf=conf, algorithm=algorithm)
        elif algorithm == 'kmeans':
            input_filename = "result/assignment/" + str(dataset.dataset_name) + "_kmeans" + "_" + str(i) + ".csv"
            df_assignment = pd.read_csv(input_filename, sep=',')
            cluster_assignment = df_assignment['cluster']
            
            output_filename = "plot/cluster/" + str(dataset.dataset_name) + "_kmeans" + "_" + str(i) + ".png"
            dataset.plot2Features(dataset.df, dataset.df.columns[0], dataset.df.columns[1], cluster_assignment=cluster_assignment,
                                  initial_space=True, dataset_name=dataset.dataset_name, seed=seed, filename=output_filename, conf=conf, algorithm=algorithm)
        
        
    

if __name__ == "__main__":
    
    '''
    ANISO DATASET TEST
    '''
    params = {
        'dataset_name': ['aniso'],
        'K': [3],
        'M1': [2,4,8,16],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
     
    processes = 2
    seed = 123
    dataset = Dataset('aniso')
    
    print("---------------------- " + str(dataset.dataset_name) + " Test ----------------------\n")
    
    print("-------------------- Quantum Kmeans --------------------")
    par_test(params, dataset, algorithm="qkmeans", n_processes=processes, seed=seed)
    plot_similarity(params, dataset, algorithm='qkmeans')
    print("-------------------- Classical Kmeans --------------------")
    par_test(params, dataset, algorithm="kmeans", n_processes=processes, seed=seed)
    
    print("-------------------- Delta Kmeans --------------------")
    par_test(params, dataset, algorithm="deltakmeans", n_processes=processes, seed=seed)
    plot_similarity(params, dataset, algorithm='deltakmeans')

    
    plot_cluster(params, dataset, algorithm='qkmeans', seed=seed)
    plot_cluster(params, dataset, algorithm='deltakmeans', seed=seed)
    plot_cluster(params, dataset, algorithm='kmeans', seed=seed)

    '''
    make_comparison() # qui per esempio mi calcolo tutte le pair confusion matrix a partire dagli assegnamenti classici quantistici e delta facendo un file dove per ogni configurazione ho due confusion matrix
    '''
    
    '''
    BLOBS DATASET TEST
    '''
    params = {
        'dataset_name': ['blobs'],
        'K': [3],
        'M1': [2,4,8,16],
        'sc_tresh':  [0],
        'max_iterations': [10]
    }
     
    processes = 2
    seed = 123
    dataset = Dataset('blobs')
    
    print("---------------------- " + str(dataset.dataset_name) + " Test ----------------------\n")
    
    print("-------------------- Quantum Kmeans --------------------")
    par_test(params, dataset, algorithm="qkmeans", n_processes=processes, seed=seed)
    plot_similarity(params, dataset, algorithm='qkmeans')
    print("-------------------- Classical Kmeans --------------------")
    par_test(params, dataset, algorithm="kmeans", n_processes=processes, seed=seed)
    
    print("-------------------- Delta Kmeans --------------------")
    par_test(params, dataset, algorithm="deltakmeans", n_processes=processes, seed=seed)
    plot_similarity(params, dataset, algorithm='deltakmeans')

    
    plot_cluster(params, dataset, algorithm='qkmeans', seed=seed)
    plot_cluster(params, dataset, algorithm='deltakmeans', seed=seed)
    plot_cluster(params, dataset, algorithm='kmeans', seed=seed)
    