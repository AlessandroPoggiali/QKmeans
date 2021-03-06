import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
from utility import qdrawer
from sklearn.decomposition import PCA

font = {'size'   : 22}

plt.rc('font', **font)

n_samlpes = 1500

class Dataset:
    
    """
    Dataset constructor: 
    
    :param dataset_name: name of the dataset to load
    :param preprocessing (optional, default value=None): kind of preprocessing to apply
    """
    def __init__(self, dataset_name, preprocessing=None):
        self.dataset_name = dataset_name
        self.preprocessing = preprocessing
        self.ground_truth = None
        self.df = self.load_dataset(dataset_name)  
        self.N = len(self.df.columns)
        self.M = len(self.df)
        
    
    """
    scale: 
        
    Scale the data to have zero mean and standard deviation 1
    
    :data: dataframe of records
    
    :return: data
    """
    def scale(self, data):
        scaler = StandardScaler()
        data.loc[:,:] = scaler.fit_transform(data.loc[:,:])
        return data
    
    """
    normalize: 
        
    Normalize the data rows according to the preprocessing 
    
    :data: dataframe of records
    
    :return: data
    """
    def normalize(self, data):
        if self.preprocessing is None or self.preprocessing == '1-norm':
            data.loc[:,:] = normalize(data.loc[:,:])
        elif self.preprocessing == 'inf-norm':
            data = data.apply(lambda row : row/max(abs(row)), axis=1)
        else: 
            print("ERROR: wrong norm in input")
            exit()
        
        return data
    
    """
    plot2Features: 
        
    Plot two features of data 
    
    :data: dataframe of records
    :x: feature to plot on x axis 
    :y: feature to plot on y axies
    :centroids (optional, default value=None): numpy array of centroids
    :cluster_assignment (optional, default value=None): list of cluster assignment for every record
    :initial_space (optional, default value=False): if True plot the data in its original space
    :dataset_name (optional, default value=None): name of the dataset
    :seed (optional, default value=0): seed for random extraction
    :filename (optional, default value=None): name of the file where to save the plot
    :conf (optional, default value=None): parameters configuration of the algorithm which produced the plot
    :algorithm (optional, default value=None): name of the algorithm which produced the plot
    """
    def plot2Features(self, data, x, y, centroids=None, cluster_assignment=None, initial_space=False,
                      dataset_name=None, seed=0, filename=None, conf=None, algorithm=""):
        colors = ['b','g','r','c','m','y','k','w']    
        lw = 1
        plt.figure(figsize=(10,10))
        #plt.xlabel(x)
        #plt.ylabel(y)
        
        
        if initial_space:
            if dataset_name is not None:
                data = self.load_dataset(dataset_name, to_preprocess=False)
                if cluster_assignment is not None:
                    series = []
                    for i in set(cluster_assignment):
                        series.append(data.loc[[index for index, n in enumerate(cluster_assignment) if n == i]].mean())
                    centroids = pd.concat(series, axis=1).T.values

            else:
                print("ERROR: unable to print in original features space")
                return


            
        if centroids is not None:
            ind = 0 
            for cluster, c in enumerate(centroids):
                plt.plot(c[0],c[1],marker='*', color=colors[ind],markersize=40, markeredgecolor='k')
                centroid_name = "c" + str(ind)
                plt.annotate(centroid_name, (c[0],c[1]), fontsize=40)
                ind = ind + 1
                
        if cluster_assignment is not None:
            for cluster in set(cluster_assignment):
                X = data.loc[[index for index, n in enumerate(cluster_assignment) if n == cluster]]
                plt.scatter(X[x],X[y],color=colors[cluster],marker="o", linewidth=lw)
        else:
            plt.scatter(data[x],data[y],color='y',marker="o",linewidth=lw)
            
        plt.gca().set_aspect('equal', adjustable='box')
        
        if conf is not None:
            if algorithm == 'qkmeans':
                plt.title("K = " + str(conf["K"]) + ", M = " + str(self.M) + ", N = " + 
                          str(self.N) + ", M1 = " + str(conf["M1"]), fontdict = {'fontsize' : 30})
                plt.suptitle('q-k-means-q' + str(conf["quantization"]))
            elif algorithm == 'deltakmeans': 
                plt.title("K = " + str(conf["K"]) + ", M = " + str(self.M) + ", N = " + 
                          str(self.N) + ", delta = " + str(conf["delta"]), fontdict = {'fontsize' : 30})
                plt.suptitle('delta-k-means')
            else:
                plt.title("K = " + str(conf["K"]) + ", M = " + str(self.M) + ", N = " + 
                          str(self.N), fontdict = {'fontsize' : 30})
                plt.suptitle('k-means')
        
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
    
        # Clear the current axes.
        plt.cla() 
        # Clear the current figure.
        plt.clf() 
        # Closes all the figure windows.
        plt.close('all')
    
    """
    plotOnCircle: 
        
    Plot the data in a circumference
    
    :data: dataframe of records
    :centroids: numpy array of centroids
    :cluster_assignment: list of cluster assignment for every record
    """
    def plotOnCircle(self, data, centroids, cluster_assignment):  
        colors = ['b','g','r','c','m','y','k','w']      
        qdrawer.draw_qubit()
        for index, item in data.iterrows():
            cluster = cluster_assignment[index]
            qdrawer.draw_datapoint(item[0], item[1], color=colors[cluster])
        
        for cluster, c in enumerate(centroids):
            qdrawer.draw_datapoint(c[0], c[1], color=colors[cluster], centroid=True)
       
    
    """
    load_dataset: 
        
    It loads the dataset specified
    
    :data_name: name of the dataset to load
    :to_preprocess (optional, default value=True): if True apply preprocessing
    
    :return: df
    """
    def load_dataset(self, dataset_name, to_preprocess=True):
        if dataset_name == 'iris':
            df = self.load_iris(to_preprocess)
        if dataset_name == 'diabetes':
            df = self.load_diabetes(to_preprocess)
        elif dataset_name == 'buddy':
            df = self.load_buddymove(to_preprocess)
        elif dataset_name == 'seeds':
            df = self.load_seeds(to_preprocess)
        elif dataset_name == 'blobs':
            df = self.load_blobs(to_preprocess)
        elif dataset_name == 'blobs2':
            df = self.load_blobs_2(to_preprocess)
        elif dataset_name == 'blobs3':
            df = self.load_blobs_3(to_preprocess)
        elif dataset_name == 'noisymoon':
            df = self.load_noisymoon(to_preprocess)
        elif dataset_name == 'aniso':
            df = self.load_aniso(to_preprocess)
        else:
            print("ERROR: No dataset found")
        return df
           
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                REAL DATASET 
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    """
    load_iris: 
        
    It loads the iris dataset
    
    :to_preprocess (optional, default value=True): if True apply preprocessing
    
    :return: df
    """
    def load_iris(self, to_preprocess=True):
        df = datasets.load_iris(as_frame=True).frame
        # rename columns
        df.columns = ["f0","f1","f2","f3","class"]
        # drop class column
        df = df.drop('class', axis=1)
        
        #df = df.sample(n=8, random_state=123)
        #df.reset_index(drop=True, inplace=True)
        
        if to_preprocess:
            df = self.scale(df)
            df = self.normalize(df)
            df = df.apply(lambda row: 2*np.arcsin(row), axis=1)
        
        return df
    
    """
    load_iris: 
        
    It loads the iris dataset
    
    :to_preprocess (optional, default value=True): if True apply preprocessing
    
    :return: df
    """
    def load_diabetes(self, to_preprocess=True):
        df = datasets.load_diabetes(as_frame=True).frame
        # rename columns
        df.columns = ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","class"]
        # drop class column
        df = df.drop('class', axis=1)
        
        df = self.scale(df)
        pca = PCA(n_components=4)
        x = df.loc[:, :].values
        principalComponents = pca.fit_transform(x)
        df = pd.DataFrame(data = principalComponents, columns = ['f0','f1','f2','f3'])

        if to_preprocess:
            df = self.scale(df)
            df = self.normalize(df)
            df = df.apply(lambda row: 2*np.arcsin(row), axis=1)
        
        return df
    

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                SYNTHETIC DATASET 
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  
    """
    load_noisymoon: 
        
    It loads the noisymoon dataset
    
    :to_preprocess (optional, default value=True): if True apply preprocessing
    
    :return: df
    """
    def load_noisymoon(self, to_preprocess=True):
        x, y = datasets.make_moons(n_samples=n_samlpes, noise=0.05, random_state=170)
        df = pd.DataFrame(x, y, columns=["f0", "f1"])
        
        df['ground_truth'] = df.index
        df.reset_index(drop=True, inplace=True)
        self.ground_truth = df['ground_truth']
        df = df.drop('ground_truth', axis=1)
        
        if to_preprocess:
            df = self.scale(df)
            df = self.normalize(df)
            df = df.apply(lambda row: 2*np.arcsin(row), axis=1)
            
        return df
    
    """
    load_blobs: 
        
    It loads the blobs dataset
    
    :to_preprocess (optional, default value=True): if True apply preprocessing
    
    :return: df
    """
    def load_blobs(self, to_preprocess=True):
        x, y = datasets.make_blobs(n_samples=n_samlpes, random_state=8)
        df = pd.DataFrame(x, y, columns=["f0", "f1"])  
        
        df['ground_truth'] = df.index
        df.reset_index(drop=True, inplace=True)
        self.ground_truth = df['ground_truth']
        df = df.drop('ground_truth', axis=1)
        
        if to_preprocess:
            df = self.scale(df)
            df = self.normalize(df)
            df = df.apply(lambda row: 2*np.arcsin(row), axis=1)
            
        return df
      
    """
    load_aniso: 
        
    It loads the aniso dataset
    
    :to_preprocess (optional, default value=True): if True apply preprocessing
    
    :return: df
    """
    def load_aniso(self, to_preprocess=True):
        x, y = datasets.make_blobs(n_samples=n_samlpes, random_state=170)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x = np.dot(x, transformation)
        df = pd.DataFrame(x, y, columns=["f0", "f1"])
        
        df['ground_truth'] = df.index
        df.reset_index(drop=True, inplace=True)
        self.ground_truth = df['ground_truth']
        df = df.drop('ground_truth', axis=1)
        
        if to_preprocess:
            df = self.scale(df)
            df = self.normalize(df)
            df = df.apply(lambda row: 2*np.arcsin(row), axis=1)
            
        return df
      
    """
    load_blobs_2: 
        
    It loads the blobs2 dataset
    
    :to_preprocess (optional, default value=True): if True apply preprocessing
    
    :return: df
    """
    def load_blobs_2(self, to_preprocess=True):
        x, y = datasets.make_blobs(n_samples=n_samlpes, cluster_std=[1.0, 2.5, 0.5], random_state=170)
        df = pd.DataFrame(x, y, columns=["f0", "f1"])
        
        df['ground_truth'] = df.index
        df.reset_index(drop=True, inplace=True)
        self.ground_truth = df['ground_truth']
        df = df.drop('ground_truth', axis=1)
        
        if to_preprocess:
            df = self.scale(df)
            df = self.normalize(df)
            df = df.apply(lambda row: 2*np.arcsin(row), axis=1)
            
        return df
       
    """
    load_blobs_3: 
        
    It loads the blobs3 dataset
    
    :to_preprocess (optional, default value=True): if True apply preprocessing
    
    :return: df
    """
    def load_blobs_3(self, to_preprocess=True):
        x, y = datasets.make_blobs(n_samples=16, random_state=9, centers=2)
        df = pd.DataFrame(x, y, columns=["f0", "f1"])  
        
        df['ground_truth'] = df.index
        df.reset_index(drop=True, inplace=True)
        self.ground_truth = df['ground_truth']
        df = df.drop('ground_truth', axis=1)
        
        if to_preprocess:
            df = self.scale(df)
            df = self.normalize(df)
            df = df.apply(lambda row: 2*np.arcsin(row), axis=1)

        return df