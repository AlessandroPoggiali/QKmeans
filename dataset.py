import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
from utility import qdrawer


class Dataset:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.ground_truth = None
        self.scaler = StandardScaler()
        self.df = self.load_dataset(dataset_name)  
        self.N = len(self.df.columns)
        self.M = len(self.df)
        
    def scale(self, data):
        data.loc[:,:] = self.scaler.fit_transform(data.loc[:,:])
        return data
    
    def normalize(self, data):
        data.loc[:,:] = normalize(data.loc[:,:])
        return data
    

    def plot2Features(self, data, x, y, centroids=None, cluster_assignment=None, initial_space=False, dataset_name=None, seed=0):
        colors = ['b','g','r','c','m','y','k','w']    
        lw = 1
        plt.figure(figsize=(10,10))
        plt.xlabel(x)
        plt.ylabel(y)
        
        
        if initial_space:
            if dataset_name is not None:
                data = self.load_dataset(dataset_name, preprocessing=False)
                if cluster_assignment is not None:
                    series = []
                    for i in set(cluster_assignment):
                        series.append(data.loc[[index for index, n in enumerate(cluster_assignment) if n == i]].mean())
                    centroids = pd.concat(series, axis=1).T.values
                elif centroids is not None:
                    k = len(centroids)
                    centroids = data.sample(n=k, random_state=seed).values
            else:
                print("ERROR: unable to print in original features space")
                return
            
            # ti rileggi il dataset ci appiccichi il cluster assignment e via

            
        if centroids is not None:
            ind = 0 
            for cluster, c in enumerate(centroids):
                plt.plot(c[0],c[1],marker='*', color=colors[ind],markersize=30)
                centroid_name = "c" + str(ind)
                plt.annotate(centroid_name, (c[0],c[1]), fontsize=20)
                ind = ind + 1
                
        if cluster_assignment is not None:
            for cluster in set(cluster_assignment):
                X = data.loc[[index for index, n in enumerate(cluster_assignment) if n == cluster]]
                plt.scatter(X[x],X[y],color=colors[cluster],marker="o", linewidth=lw)
        else:
            plt.scatter(data[x],data[y],color='y',marker="o",linewidth=lw)
            
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    
    def plotOnCircle(self, data, centroids, cluster_assignment):  
        colors = ['b','g','r','c','m','y','k','w']      
        qdrawer.draw_qubit()
        for index, item in data.iterrows():
            cluster = cluster_assignment[index]
            qdrawer.draw_datapoint(item[0], item[1], color=colors[cluster])
        
        for cluster, c in enumerate(centroids):
            qdrawer.draw_datapoint(c[0], c[1], color=colors[cluster], centroid=True)
       
        
    def load_dataset(self, dataset_name, preprocessing=True):
        if dataset_name == 'iris':
            df = self.load_iris(preprocessing)
        elif dataset_name == 'buddy':
            df = self.load_buddymove(preprocessing)
        elif dataset_name == 'seeds':
            df = self.load_seeds(preprocessing)
        elif dataset_name == 'blobs':
            df = self.load_blobs(preprocessing)
        elif dataset_name == 'blobs2':
            df = self.load_blobs_2(preprocessing)
        elif dataset_name == 'noisymoon':
            df = self.load_noisymoon(preprocessing)
        elif dataset_name == 'aniso':
            df = self.load_aniso(preprocessing)
        else:
            print("ERROR: No dataset found")
        return df
           
    '''
    REAL DATA
    '''
    def load_iris(self, preprocessing=True):
        df = pd.read_csv("data/iris.csv", skipinitialspace=True, sep=',')
        # rename columns
        df.columns = ["f0","f1","f2","f3","class"]
        # drop class column
        df = df.drop('class', axis=1)
        #df = df.drop('f0', axis=1)
        #df = df.drop('f1', axis=1)
        
        df = df.sample(n=10)
        df.reset_index(drop=True, inplace=True)
        
        if preprocessing:
            df = self.scale(df)
            df = self.normalize(df)
        
        # Translate the vector coordinates in the rotation angle we have to apply to the QRAM register qbit
        #df.loc[:,"f0":"f3"] = df.loc[:,"f0":"f3"].apply(np.arcsin)
        
        return df
    
    
    def load_buddymove(self, preprocessing=True):
        df = pd.read_csv("data/buddymove_holidayiq.csv", skipinitialspace=True, sep=',')

        # drop userId column
        df = df.drop('User Id', 1)
                
        if preprocessing:
            df = self.scale(df)
            df = self.normalize(df)
        
        return df
    
    def load_seeds(self, preprocessing=True):
        df = pd.read_csv("data/seeds_dataset.txt", skipinitialspace=True, sep=',')
        # drop class column
        df = df.drop('class', 1)
                
        if preprocessing:
            df = self.scale(df)
            df = self.normalize(df)
        
        return df
    
    '''
    SYNTETIC DATA
    '''
    def load_noisymoon(self, preprocessing=True):
        x, y = datasets.make_moons(n_samples=1500, noise=0.05, random_state=170)
        df = pd.DataFrame(x, y, columns=["f0", "f1"])
        
        df['ground_truth'] = df.index
        df.reset_index(drop=True, inplace=True)
        self.ground_truth = df['ground_truth']
        df = df.drop('ground_truth', axis=1)
        
        if preprocessing:
            df = self.scale(df)
            df = self.normalize(df)
        return df
    
    def load_blobs(self, preprocessing=True):
        x, y = datasets.make_blobs(n_samples=150, random_state=8)
        df = pd.DataFrame(x, y, columns=["f0", "f1"])  
        
        df['ground_truth'] = df.index
        df.reset_index(drop=True, inplace=True)
        self.ground_truth = df['ground_truth']
        df = df.drop('ground_truth', axis=1)
        
        if preprocessing:
            df = self.scale(df)
            df = self.normalize(df)
        return df
        
    def load_aniso(self, preprocessing=True):
        x, y = datasets.make_blobs(n_samples=1500, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        df = pd.DataFrame(x, y, columns=["f0", "f1"])
        
        df['ground_truth'] = df.index
        df.reset_index(drop=True, inplace=True)
        self.ground_truth = df['ground_truth']
        df = df.drop('ground_truth', axis=1)
        
        if preprocessing:
            df = self.scale(df)
            df = self.normalize(df)
        return df
        
    def load_blobs_2(self, preprocessing=True):
        x, y = datasets.make_blobs(n_samples=1500, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        df = pd.DataFrame(x, y, columns=["f0", "f1"])
        
        df['ground_truth'] = df.index
        df.reset_index(drop=True, inplace=True)
        self.ground_truth = df['ground_truth']
        df = df.drop('ground_truth', axis=1)
        
        if preprocessing:
            df = self.scale(df)
            df = self.normalize(df)
        return df
        