import numpy as np

#The SSE is defined as the sum of the squared Euclidean distances of each point to its closest centroid
def SSE(data, centroids):
    sse = 0
    data = data.reset_index(drop=True)
    for index_v, v in data.iterrows():
        c = data.iloc[index_v]['cluster']
        sse = sse + np.linalg.norm(np.array(data.iloc[index_v][:-1])-np.array(centroids.iloc[int(c)][:-1]))
    return sse

# method used to check classification accuracy between quantum and classical distances
def check_accuracy(df, centroids):
    error = 0
    for index_v, item_v in df.iterrows():
        dists = []
        for index_c, item_c in centroids.iterrows():
            dists.append(np.linalg.norm(np.array(item_v[:-1])-np.array(item_c[:-1])))
        classical = dists.index(min(dists))
        if classical != df.iloc[index_v]['cluster']:
            error = error + 1
    
    M = len(df)
    correct = M-error
    accuracy = (correct/M)*100
    print("Accuracy: " + str(round(accuracy,2)) + "%")
    return accuracy
    