from torchvision import models as torchvision_models
from utils_nn import *
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
class Voxel3d(nn.Module):
    def __init__(self):
        super(Voxel3d, self).__init__()
        self.deconv3d_layers = nn.Sequential(
            nn.ConvTranspose3d(512, 128, 10, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 4, 2, 1),
        )
    def forward(self, x):
        x = x.view(-1, 512, 1, 1, 1)  
        x  = self.deconv3d_layers(x)
        return x    

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            pass
            # TODO:
            """
            layers = [32**2,32**3,"S"]
            decoder = create_decoder(layers)
            voxel_decoder = add_voxel_head(decoder, input_dim=32**3, x_size=32,y_size=32,z_size=32)
            """
            #self.decoder = voxel_decoder
            self.decoder = Voxel3d()
            """
            nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU(),
                nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            """
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 2048),
                nn.LeakyReLU(), 
                nn.Linear(2048, args.n_points * 3), 
            )        
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            self.decoder = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 2048),
                    nn.LeakyReLU(),
                    nn.Linear(2048, 4096),
                    nn.LeakyReLU(),
                    nn.Linear(4096, self.mesh_pred.verts_padded().shape[1] * 3),
                    nn.Tanh()
            )

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            return self.decoder(encoded_feat)

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat).view(-1, args.n_points, 3)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

