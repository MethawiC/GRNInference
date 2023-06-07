from dynGENIE3 import *
import numpy as np

labels=np.load('labels.npy',allow_pickle=True)
dataset=np.load('dataset.npy',allow_pickle=True)
time_points=[[],[],[]]
for j in range(0,3):
    for i in range(0,12):
        time_points[j].append(i*120)
dataset=dataset.tolist()
for i in range(0,len(dataset)):
    dataset[i]=np.transpose(np.array(dataset[i],dtype=np.float64))
(VIM, alphas, prediction_score, stability_score, treeEstimators) = dynGENIE3(dataset,time_points, nthreads=8)

np.save('results/vim.npy',VIM)
np.save('results/alphas.npy',alphas)