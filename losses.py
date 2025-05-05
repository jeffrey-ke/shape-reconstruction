import torch
from jutils.utils import pdb
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
)
from pytorch3d.ops import (
    knn_gather,
    knn_points,
)
import torch.nn as nn

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
    # now why doesn't binary cross entropy loss work?
    # it looks like after one iteration of optimization my tensors become accidentally 
    # a none probability. I'll clip them
    voxel_src = torch.clip(voxel_src, min=0.,max=1.)
    ce_loss = nn.BCELoss()
    # print somethings about voxel_src and voxel_tgt.. they should be probabilities...
    
    loss = ce_loss(voxel_src, voxel_tgt) 
    return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
    # find each point's nearest point
    # each point in src's nearest neighbor in tgt:
    #and chamfer loss needs to be a torch tensor
    dists1_2 = knn_points(p1=point_cloud_src, p2=point_cloud_tgt, K=1).dists[...,0]
    dists2_1 = knn_points(p1=point_cloud_tgt, p2=point_cloud_src, K=1).dists[...,0]
    _1 = torch.sum(dists2_1, dim=1)
    _2 = torch.sum(dists1_2, dim=1)
    loss = torch.mean(_1 + _2)
    return loss

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
    loss_laplacian = mesh_laplacian_smoothing(mesh_src)
    return loss_laplacian
