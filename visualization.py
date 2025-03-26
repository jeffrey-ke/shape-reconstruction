#!/home/jeffk/miniconda3/envs/learning3d/bin/python3
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
features = torch.load("resnet_features.pth")
features = features.squeeze(1).numpy()
pca = PCA(n_components=2)
projection = pca.fit_transform(features)

examples = projection[[27, 506, 110, 103, 387]]
fig, ax = plt.subplots()
# and remember, ax can be passed around
origins = np.zeros_like(examples)
ax.quiver(origins[:,0], origins[:,1], examples[:,0], examples[:,1], angles='xy', scale_units='xy', scale=1, color='blue')
ax.set_xlim(examples[:,0].min() - 1, examples[:,0].max() + 1)
ax.set_ylim(examples[:,1].min() - 1, examples[:,1].max() + 1)
ax.grid(True)
plt.show()
