import torch
import matplotlib.cm as cm  
import matplotlib.colors as mcolors 
import matplotlib.pyplot as plt
from pytorch3d.structures import (
    Meshes,
    Pointclouds,
)
import os
import imageio
import pytorch3d
import numpy as np
from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    AlphaCompositor,
    TexturesVertex,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    PointLights,
    look_at_view_transform, 
    FoVPerspectiveCameras
)
from pytorch3d.io import load_obj


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb


def load_cow_mesh(path="data/cow_mesh.obj"):
    """
    Loads vertices and faces from an obj file.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh (N_v, 3).
        faces (torch.Tensor): The faces of the mesh (N_f, 3).
    """
    vertices, faces, _ = load_obj(path)
    faces = faces.verts_idx
    return vertices, faces

def to_images(renders):
    return [(image[...,:3] * 255).int().cpu().numpy().astype(np.uint8) for image in renders]

def visualize_pointcloud(points, output_path, duration=0.05, steps=60, device="cuda"):
    points = points.to(device).squeeze(0)
    dists = torch.norm(points, dim=-1)  # Compute Euclidean distances
    dists_np = dists.detach().cpu().numpy()
    norm = mcolors.Normalize(vmin=dists_np.min(), vmax=dists_np.max())
    colormap = cm.get_cmap("plasma")
    colors_np = colormap(norm(dists_np))[:, :3]  # Get RGB values
    color = torch.tensor(colors_np, device=device, dtype=torch.float32)

    point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color]
    ).to(device)
    elev = torch.full((steps,), 10, device=device)
    azim = torch.linspace(0, 360, steps, device=device)
    R, T = look_at_view_transform(dist=2.0, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    renderer = get_points_renderer(image_size=512)
    images = renderer(point_cloud.extend(steps), cameras=cameras)
    images_np = (images[..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.mimsave(output_path, images_np, duration=duration, loop=0)


def gif_up(structure, dir, name, num_views=20, elev=10, dist=2):
    Rs, Ts = pytorch3d.renderer.look_at_view_transform(
                                                        dist=dist,
                                                         elev=elev,
                                                         azim=np.linspace(-180,180, num=num_views, endpoint=False))
    device = get_device()
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=Rs,
        T=Ts,
        fov=60,
        device=device
    )
    if isinstance(structure, Meshes):   
        renderer = get_mesh_renderer(image_size=512)
        type = "mesh"
    elif isinstance(structure, Pointclouds):
        renderer = get_points_renderer(image_size=512)
        type = "pc"
            
    renders = renderer(structure.extend(num_views), 
                        cameras=cameras, 
                        lights=PointLights(location=[[0,0,-3]], 
                        device=device)
                        )
    images = to_images(renders)
    imageio.mimsave("{}/{}.gif".format(dir, name), images, duration=10, loop=0)

def mesh_add_texture(mesh, color=[0.7,0.7,1]):
    verts = mesh.verts_list()[0].cpu()
    textures = torch.ones_like(verts).unsqueeze(0) * torch.tensor(color)
    mesh.textures = TexturesVertex(textures.to(get_device()))

def invert_voxels(src):
    with torch.no_grad():
        src[:] = 1. - src

def points_to_pointcloud(points):
    return create_p3d_pc(points)

def voxel_to_mesh(voxel):
    mesh = cubify(voxel, thresh=0.5)
    mesh_add_texture(mesh)
    return mesh

def render_voxels(src, tgt, src_path, tgt_path, elev=10, dist=10):
    src_mesh = voxel_to_mesh(src)
    dir, fn = os.path.split(src_path)
    gif_up(src_mesh, dir=dir, name=fn)
    if tgt is not None:
        tgt_mesh = cubify(tgt, thresh=0.5)
        mesh_add_texture(tgt_mesh)
        dir,fn = os.path.split(tgt_path)
        gif_up(tgt_mesh, dir=dir, name=fn)

def render(src, tgt, src_path, tgt_path, elev=10, dist=10):
    dir,fn = os.path.split(src_path)
    gif_up(src,dir=dir,name=fn, dist=dist)
    if tgt is not None:
        dir,fn = os.path.split(tgt_path)
        gif_up(tgt,dir=dir,name=fn, dist=dist)

def render_mesh(src, tgt, src_path, tgt_path, elev=10, dist=10):
    mesh_add_texture(src)
    if tgt is not None:
        mesh_add_texture(tgt)
    render(src, tgt, src_path, tgt_path, elev, dist)
    print("dist: ", dist)

def render_pc(src, tgt, src_path, tgt_path, elev=10, dist=10):
    render(src, tgt, src_path, tgt_path, elev, dist)

def create_p3d_pc(torch_pc, color=[0.7,0.7,1.]):
    textures = torch.ones_like(torch_pc).cpu() * torch.tensor(color)
    return Pointclouds(points=torch_pc, features=textures.to(get_device()).to(get_device()))

def image_up(image_np, dir, name):
    os.makedirs(dir, exist_ok=True)
    plt.imsave("{}/{}.png".format(dir, name), image_np)

def get_renderer(structure, image_size=512):
    if isinstance(structure, Meshes):
        return get_mesh_renderer(image_size=image_size)
    elif isinstance(structure, Pointclouds):
        return get_points_renderer(image_size=image_size)
    
def render_images(structure, path, dist=5, elev=10, n_images=1, image_size=512):
    Rs,Ts = look_at_view_transform(dist=dist,
                                 elev=elev,
                                 azim=np.linspace(-180,180,num=n_images)
                                 )
    renderer = get_renderer(structure)
    device = get_device()
    cameras = FoVPerspectiveCameras(
        R=Rs,
        T=Ts,
        fov=60,
        device=device
    )
    renders = renderer(structure.extend(n_images), 
                        cameras=cameras, 
                        lights=PointLights(location=[[0,0,-3]], 
                        device=device)
                        )
    images = to_images(renders)
    dir,name = os.path.split(path)
    if n_images > 1:
        imageio.mimsave("{}/{}.gif".format(dir, name), images, duration=10, loop=0)
    else:
        os.makedirs(dir, exist_ok=True)
        plt.imsave("{}/{}.png".format(dir, name), images[0])
    return images

def render_images_batched(structures, path, dist=2, elev=10, n_images=1, image_size=512):
    for idx,s in enumerate(structures):
        render_images(s, "{}{}".format(path, idx), dist, elev, n_images, image_size)

def get_time():
    from datetime import datetime
    now = datetime.now()
    filename_date = now.strftime("%B_%d_%I_%M_%p")
    return filename_date

class TensorCollector:
    def __init__(self, fname):
        self.fname = fname
        self.tensors = []
        self.shape = None

    def save_feature(self, tensor):
        tensor = tensor.detach().cpu()
        if len(list(tensor.shape)) >= 3:
            for t in tensor:
                print(f"Batched of shape {t.shape}")
                self.tensors.append(t)
                if self.shape is not None and self.shape != t.shape:
                    raise ValueError(f"My shape {self.shape} is not your shape {t.shape}")
                self.shape = t.shape
        else:
            if self.shape is not None and self.shape != tensor.shape:
                    raise ValueError(f"My shape {self.shape} is not your shape {t.shape}")
            self.shape = tensor.shape
            self.tensors.append(tensor)
            print("Saving")

    def store(self):
        print(f"Saving to {self.fname}.pth")
        batched = torch.stack(self.tensors)
        torch.save(batched, self.fname + ".pth")
