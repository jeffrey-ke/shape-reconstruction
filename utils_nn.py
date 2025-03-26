import torch
import torch.nn as nn
class VoxelHead(nn.Module):
    def __init__(self, body: nn.Module, input_dim, x_size, y_size, z_size):
        """ body is the rest of the neural netowrk: this class exists purely
        to add a voxel head output
        """
        super(VoxelHead,self).__init__()
        self.body = body
        self.input_dim = input_dim
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

    def forward(self, x):
        out = self.body(x)
        return out.view(-1, 1, self.x_size, self.y_size, self.z_size)

def create_decoder(layers, input_dim=512):
    """
        layers is a list. The first element is going to be the dimensionality of 
        that layer. Relu always follows after each layer

        if having a sigmoid after a relu is a problem,
        just add another special str tag: "R" for relu
        and only add a relu if you see an R
    """
    nn_layers = []
    last_dim = input_dim
    for i,tag in enumerate(layers):
        if isinstance(tag, int):
            nn_layers.append(nn.Linear(last_dim, tag))
            nn_layers.append(nn.ReLU())
            last_dim = tag
        elif isinstance(tag, str):
            if tag == "S":
                nn_layers.append(nn.Sigmoid())
        else:
            raise Exception("Check your input to nn_layers")
    return nn.Sequential(*nn_layers)

def add_voxel_head(body,input_dim, x_size, y_size, z_size):
    return VoxelHead(body,input_dim, x_size, y_size, z_size)

def save_checkpoint(step, model_dict, optim_dict, name):
    torch.save(
        {
            "step": step,
            "model_state_dict": model_dict,
            "optimizer_state_dict": optim_dict,
        },
        f"{name}.pth",
    )
    print("Saved checkpoint at {}.pth".format(name))
