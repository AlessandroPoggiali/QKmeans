import numpy as np
import pandas as pd
import time
import datetime
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from utility import measures

font = {'size'   : 22}

plt.rc('font', **font)

class DeltaKmeans():
    
    def __init__(self, dataset, conf, delta):
        
        self.K = conf['K']
        self.dataset_name = conf['dataset_name']
        self.sc_tresh = conf['sc_tresh']
        self.max_iterations = conf['max_iterations']
        
        self.delta = delta
        
        self.dataset = dataset
        self.data = self.dataset.df
        self.N = self.dataset.N
        self.M = self.dataset.M
        self.centroids = None
        self.old_centroids = None
        
        self.cluster_assignment = [0]*self.M
        
        self.ite = 0
        self.SSE_list = []
        self.silhouette_list = []
        self.similarity_list = []
        self.nm_info_list = []
        self.times = []   
    
    def label_with_delta_squared(self): #given X (points) and centers: 2 numpy arrays
        X = self.data.values
        centers = self.centroids
        labels = []
        count = 0
        for dist_array in metrics.pairwise_distances(X, centers): #dist_array is the array of distances between on element Xi of X and each cluster! 
            dist_array = np.square(dist_array) #THIS IS THE SQUARED DISTANCE MODIFICATION!
            mindist = np.min(dist_array) #distance between Xi and its closest clusters
            normalmin = [np.argmin(dist_array)] # index of the clusters closest to Xi
            close_dist = set([dist for dist in dist_array if abs(dist - mindist) <= self.delta]) #array of all distance of dist_array if they are delta-close to mindist 
            deltamin = [i for i, item in enumerate(dist_array) if item in close_dist] #index of delta-close centers 
            deltachoice = random.choice(deltamin) #choose randomly one of the delta-close centers
            labels.append(deltachoice)
            if deltamin != normalmin:
                count+=1
        #print("DELTA K-MEANS: %d random choices of centers over %d"%(count,len(X)))
        return labels,count

    
    '''
    Computes the new cluster centers as the mean of the records within the same cluster
    '''
    def computing_centroids(self):
        #data = data.reset_index(drop=True)
        
        series = []
        for i in range(self.K):
            if i in self.cluster_assignment:
                series.append(self.data.loc[[index for index, n in enumerate(self.cluster_assignment) if n == i]].mean())
            else:
                old_centroid = pd.Series(self.centroids[i])
                old_centroid = old_centroid.rename(lambda x: "f" + str(x))
                series.append(old_centroid)
                
        df_centroids = pd.concat(series, axis=1).T
        self.centroids = self.dataset.normalize(df_centroids).values
    
    
    def stop_condition(self):
        # ritorna true se i centroidi non cambiano piu di un tot false altrimenti

        if self.old_centroids is None:
            return False

        for i in range(len(self.centroids)):
            difference = np.linalg.norm(self.centroids[i]-self.old_centroids[i])
            if difference > self.sc_tresh:
                return False
        
        return True
    
    
    def run(self, initial_centroids=None, seed=123):
        
        if initial_centroids is None:
            self.centroids = self.data.sample(n=self.K, random_state=seed).values
        else:
            self.centroids = initial_centroids
        
        #self.dataset.plot2Features(self.data, 'f0', 'f1', self.centroids, cluster_assignment=None, initial_space=True, dataset_name=self.dataset_name, seed=self.seed)
        while not self.stop_condition():
            
            start = time.time()
            
            self.old_centroids = self.centroids.copy()
            
            #print("iteration: " + str(self.ite))
            #print("Computing the distance between all vectors and all centroids and assigning the cluster to the vectors")
            self.cluster_assignment, count = self.label_with_delta_squared()
            #print(count)
            #self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, True)
    
            #print("Computing new centroids")
            #centroids = computing_centroids_0(data, k)
            self.computing_centroids()
    
            #self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, cluster_assignment=self.cluster_assignment, initial_space=False)
            #self.dataset.plot2Features(self.data, self.data.columns[0], self.data.columns[1], self.centroids, cluster_assignment=self.cluster_assignment, initial_space=True, dataset_name=self.dataset_name, seed=self.seed)
            
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
        
    def avg_ite_time(self):
        return round(np.mean(self.times), 2)
    
    def avg_sim(self):
        return round(np.mean(self.similarity_list), 2)
    
    def SSE(self):
        return round(measures.SSE(self.data, self.centroids, self.cluster_assignment), 3)
    
    def silhouette(self):
        if len(set(self.cluster_assignment)) <= 1 :
            return None
        else:
            return round(metrics.silhouette_score(self.data, self.cluster_assignment, metric='euclidean'), 3)
    
    def vmeasure(self):
        if self.dataset.ground_truth is not None:
            return round(metrics.v_measure_score(self.dataset.ground_truth, self.cluster_assignment), 3)
        else:
            return None
    
    def nm_info(self):
        if self.dataset.ground_truth is not None:
            return round(metrics.normalized_mutual_info_score(self.dataset.ground_truth, self.cluster_assignment), 3)
        else:
            return None
    
    def save_measures(self, index):
        filename = "result/measures/" + str(self.dataset_name) + "_deltakmeans_" + str(index) + ".csv"
        measure_df = pd.DataFrame({'similarity': self.similarity_list, 'SSE': self.SSE_list, 'silhouette': self.silhouette_list, 'nm_info': self.nm_info_list})
        pd.DataFrame(measure_df).to_csv(filename)
        
    def print_result(self, filename=None, process=0, index_conf=0):
        self.dataset.plotOnCircle(self.data, self.centroids, self.cluster_assignment)
        
        #print("")
        #print("---------------- QKMEANS RESULT ----------------")
        #print("Iterations needed: " + str(self.ite) + "/" + str(self.max_iterations))
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
        ax.set_title("K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N))
        #plt.show()
        dt = datetime.datetime.now().replace(microsecond=0)
        #str_dt = str(dt).replace(" ", "_")
        fig.savefig("./plot/deltakmeansSim_"+str(process)+"_"+str(index_conf)+".png")
        
        if filename is not None:
            # stampa le cose anche su file 
            
            f = open(filename, 'a')
            f.write("###### TEST " + str(process)+"_"+str(index_conf) + " on " + str(self.dataset_name) + " dataset\n")
            f.write("# Executed on " + str(dt) + "\n")
            f.write("## DELTAKMEANS\n")
            f.write("# Parameters: K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + "\n")
            f.write("# Iterations needed: " + str(self.ite) + "/" + str(self.max_iterations) + "\n")
            f.write('# Average iteration time: ' + str(avg_time) + 's \n')
            f.write('# Average similarity w.r.t classical assignment: ' + str(avg_sim) + '% \n')
            f.write('# SSE: ' + str(SSE) + '\n')
            f.write('# Silhouette: ' + str(silhouette) + '\n')
            f.write('# Vmeasure: ' + str(vm) + '\n')
            f.write('# Normalized mutual info score: ' + str(nminfo) + '\n')
            f.write("# Deltakmeans assignment \n")
            f.write(str(self.cluster_assignment))            
            f.write("\n")
            #f.write('# Final centroids \n'
            #self.centroids.to_csv(f, index=False)
            f.close()
            
    def print_params(self, process=0, i=0):
        print("Process " + str(process) + " - configuration: " + str(i) + 
              "\nParameters: K = " + str(self.K) + ", M = " + str(self.M) + ", N = " + str(self.N) + ", delta = " + str(self.delta) + "\n")


'''
if __name__ == "__main__":

    dataset = Dataset('noisymoon')
    conf = {
            "dataset_name": 'noisymoon',
            "K": 2,
            "M1": 10,
            "sc_tresh": 0,
            "max_iterations": 5
        }
    
    deltakmeans = DeltaKmeans(dataset, conf, 0.2, 10)
    deltakmeans.print_params()
    deltakmeans.run()
    deltakmeans.print_result("log/log.txt")
'''