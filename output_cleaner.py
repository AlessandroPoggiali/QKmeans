import os
import glob

files = glob.glob('./plot/cluster/*')
for f in files:
    os.remove(f)
    
files = glob.glob('./plot/initial_centroids/*')
for f in files:
    os.remove(f)
    
files = glob.glob('./result/assignment/*')
for f in files:
    os.remove(f)
    
files = glob.glob('./result/initial_centroids/*')
for f in files:
    os.remove(f)
    
files = glob.glob('./result/measures/*')
for f in files:
    os.remove(f)
    
files = glob.glob('./result/probabilities/*')
for f in files:
    os.remove(f)
    
files = glob.glob('./result/*.csv')
for f in files:
    os.remove(f)