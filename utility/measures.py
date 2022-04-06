import numpy as np

"""
SSE: 
    
It computes the Sum of Squared Error between records and centroids, given a certain assignment.

:data: dataframe of records
:centroids: numpy array of centroids
:assignment: list of cluster assignment for every record

:return: sse
"""
def SSE(data, centroids, assignment):
    sse = 0
    data = data.reset_index(drop=True)
    for index_v, v in data.iterrows():
        c = assignment[index_v]
        sse = sse + np.linalg.norm(np.array(data.iloc[index_v])-centroids[int(c)])**2
    return sse

"""
check_similarity: 
    
It checks similarity between quantum and classical assignment to cluster

:df: dataframe of records
:centroids: numpy array of centroids
:assignment: list of cluster assignment for every record

:return: similarity
"""
def check_similarity(df, centroids, assignemnt):
    error = 0
    for index_v, item_v in df.iterrows():
        dists = []
        for cluster, centroid in enumerate(centroids):
            dists.append(np.linalg.norm(np.array(item_v) - centroid))
        classical = dists.index(min(dists))
        if classical != assignemnt[index_v]:
            error = error + 1
    
    M = len(df)
    correct = M-error
    similarity = (correct/M)*100
    
    return similarity
