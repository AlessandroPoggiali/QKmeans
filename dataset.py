import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
from utility import qdrawer


class Dataset:
    
    def __init__(self, dataset_name):
        self.scaler = StandardScaler()
        self.norms = pd.Series([])
        self.df = self.load_dataset(dataset_name)        
        self.N = len(self.df.columns)
        self.M = len(self.df)
        
    def scale(self, data, norms=True):
        data.loc[:,:] = self.scaler.fit_transform(data.loc[:,:])
        if norms:
            self.norms = data.apply(lambda row: np.linalg.norm(row), axis=1)
        return data
    
    def normalize(self, data):
        data.loc[:,:] = normalize(data.loc[:,:])
        return data
    
    def inverse_scale(self, data):
        data.loc[:,data.columns[:-1]] = self.scaler.inverse_transform(data.loc[:,data.columns[:-1]])
        return data
        
    def inverse_normalize(self, data):
        data.loc[:, data.columns[:-1]] = data.loc[:,data.columns[:-1]].mul(self.norms[data.index], axis=0)
        return data
        #return data.mul(self.norms[data.index], axis=0)   
    

    def plot2Features(self, data, x, y, centroids=None, assignment=False, initial_space=False):
        colors = ['b','g','r','c','m','y','k','w']    
        lw = 1
        plt.figure(figsize=(10,10))
        plt.xlabel(x)
        plt.ylabel(y)
        
        
        if initial_space:
            data = data.copy()
            data = self.inverse_normalize(data)
            data = self.inverse_scale(data)
            
            if centroids is not None:
                centroids = centroids.copy()
                centroids = self.inverse_normalize(centroids)
                centroids = self.inverse_scale(centroids)
                
            
        if centroids is not None:
            ind = 0 
            for index, c in centroids.iterrows():
                plt.plot(c[x],c[y],marker='*', color=colors[ind],markersize=30)
                centroid_name = "c" + str(ind)
                plt.annotate(centroid_name, (centroids.iloc[ind][x],centroids.iloc[ind][y]), fontsize=20)
                ind = ind + 1
                
        if assignment is True:
            for cluster in data['cluster'].unique():
                cluster = int(cluster)
                X = data[data["cluster"] == cluster]
                plt.scatter(X[x],X[y],color=colors[cluster],marker="o", linewidth=lw)
        else:
            plt.scatter(data[x],data[y],color='y',marker="o",linewidth=lw)
            
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    
    def plotOnCircle(self, data, centroids):  
        colors = ['b','g','r','c','m','y','k','w']      
        qdrawer.draw_qubit()
        for index, item in data.iterrows():
            cluster = int(item['cluster'])
            #qdrawer.draw_quantum_state(item[0], item[1], "v1", color=colors[cluster]) 
            qdrawer.draw_datapoint(item[0], item[1], color=colors[cluster])
        
        for index, item in centroids.iterrows():
            cluster = int(item['cluster'])
            qdrawer.draw_datapoint(item[0], item[1], color=colors[cluster], centroid=True)
       
        
    def load_dataset(self, dataset_name):
        if dataset_name == 'iris':
            df = self.load_iris()
        elif dataset_name == 'buddy':
            df = self.load_buddymove()
        elif dataset_name == 'seeds':
            df = self.load_seeds()
        else:
            print("ERROR: No dataset found")
        return df
            
    def load_iris(self):
        df = pd.read_csv("data/iris.csv", skipinitialspace=True, sep=',')
        # rename columns
        df.columns = ["f0","f1","f2","f3","class"]
        # drop class column
        df = df.drop('class', 1)
        df = df.drop('f0', 1)
        df = df.drop('f1', 1)
        
        df = df.sample(n=20)
        df.reset_index(drop=True, inplace=True)
        
        df = self.scale(df)
        df = self.normalize(df)
        
        # Translate the vector coordinates in the rotation angle we have to apply to the QRAM register qbit
        #df.loc[:,"f0":"f3"] = df.loc[:,"f0":"f3"].apply(np.arcsin)
        
        return df
    
    
    def load_buddymove(self):
        df = pd.read_csv("data/buddymove_holidayiq.csv", skipinitialspace=True, sep=',')

        # drop userId column
        df = df.drop('User Id', 1)
                
        df = self.scale(df)
        df = self.normalize(df)
        
        return df
    
    def load_seeds(self):
        df = pd.read_csv("data/seeds_dataset.txt", skipinitialspace=True, sep=',')
        # drop class column
        df = df.drop('class', 1)
                
        df = self.scale(df)
        df = self.normalize(df)
        
        return df