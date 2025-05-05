from torchvision import models as torchvision_models
import numpy as np
from jutils.utils import *
import clip
from utils_nn import *
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
class Point(nn.Module):
    def __init__(self, n_point):
        super(Point, self).__init__()
        self.layers =nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_point*3),
        )
        self.n_point = n_point
    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, self.n_point, 3)
        return x

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
        """
        self.encoder = clip
        """
        if args.arch == "resnet18":
            vision_model = torchvision_models.__dict__[args.arch](pretrained=not args.not_pretrained)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        elif args.arch == "clip":
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
            ])
            self.encoder, self.preprocess = clip.load("ViT-B/32", device=self.device)
        elif args.arch == "dino":
            self.transforms = transforms.Compose([
                transforms.Resize((224,224)),
            ])
            vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
            # going to need a resize in the preprocess
            self.encoder = vitb8
            self.last_layer = nn.Linear(768, 512)

        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # TODO:
            self.initial = nn.Sequential(
                nn.Linear(512, 2048)
            )
            self.decoder = nn.Sequential(
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
            #self.decoder = Voxel3d()
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = Point(self.n_point)
            # nn.Sequential(
            #     nn.Linear(512, 1024),
            #     nn.LeakyReLU(),
            #     nn.Linear(1024, 2048),
            #     nn.LeakyReLU(),
            #     nn.Linear(2048, 4096),
            #     nn.LeakyReLU(), 
            #     nn.Linear(4096, args.n_points * 3), 
            #     nn.Tanh()
            # )        
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
        if args.arch == "resnet18":
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        elif args.arch == "clip":
            topil = transforms.ToPILImage()
            images = [topil(img.cpu()) for img in images.permute(0,3,1,2)]
            images = [self.preprocess(img) for img in images]
            images = torch.tensor(np.stack(images), device=args.device)
            encoded_feat = self.encoder.encode_image(images).float()# b x 512
        elif args.arch == "dino":
            images = self.transforms(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images)
            encoded_feat = self.last_layer(encoded_feat)

        # call decoder
        if args.type == "vox":
            # TODO:
            initial = self.initial(encoded_feat)
            initial = initial.view(-1, 256, 2, 2, 2)
            return self.decoder(initial)
            # return self.decoder(encoded_feat)

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)#.view(-1, args.n_points, 3)
            if args.tanh is True:
                pointclouds_pred = nn.functional.tanh(pointclouds_pred)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          


class Pix2VoxDecoder(nn.Module):
    def __init__(self):
        self.decoder = nn.Sequential(
                nn.ConvTranspose3d(256, 128, 4, 2, 1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.ConvTranspose3d(128, 64, 4, 2, 1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.ConvTranspose3d(64, 32, 4, 2, 1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 8, 4, 2, 1),
                nn.BatchNorm3d(8),
                nn.ReLU(),
                nn.ConvTranspose3d(8, 1, 4, 2, 1),
                nn.Sigmoid(),
                )
    def forward(self, X):
        return self.decoder(X)

    """
    __init__:
        batch norm + relu:
    forward:
        input is a 256 channel 2^3 cube
        goes thru:
            convT kernel 4^3 128 channels stride of 2 and padding 1:
                (so the output size is going to be something like: (dout - 1) * 2 - 2 + 4 = 2dout - 2 - 2 + 4 = 2dout; a double
            bn
            relu
            convT same but 64 channels
            bn
            relu
            convT but 32 channels
            bn
            relu
            convT but 8 channels
            bn
            relu
            convT but 1 channel
            sigmoid
    """

