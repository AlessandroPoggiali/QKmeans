import numpy as np

def SSE(data, centroids, assignment):
    sse = 0
    data = data.reset_index(drop=True)
    for index_v, v in data.iterrows():
        #c = data.iloc[index_v]['cluster']
        c = assignment[index_v]
        sse = sse + np.linalg.norm(np.array(data.iloc[index_v])-centroids[int(c)])**2
    return sse

# method used to check classification accuracy between quantum and classical distances
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
    accuracy = (correct/M)*100
    
    return accuracy
