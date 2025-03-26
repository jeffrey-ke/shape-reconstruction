import argparse

from jutils.optim import *
import torchvision.transforms as transforms
from jutils.logger import *
from torchvision.transforms import ToPILImage
from jutils.utils import *
import importlib
import utils_nn
import utils
importlib.reload(utils_nn)
importlib.reload(utils)
from utils import *

from utils_nn import *
import time

import dataset_location
import losses
import torch
from model_original import SingleViewto3D

from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom import R2N2
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_iter", default=50000, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--min_loss_delta', default=4e-4, type=float)
    parser.add_argument('--max_patience', default=10, type=int)
    parser.add_argument('--tag', default="", type=str)
    return parser


def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def preprocess_clip(feed_dict, args, m_preprocess):
    images = feed_dict["images"].squeeze(1).permute((0,3,1,2))
    topil = ToPILImage()
    images = [topil(img.cpu()) for img in images]
    images = [m_preprocess(img) for img in images]
    images = torch.tensor(np.stack(images))
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type == "vox":
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type == "point":
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    return loss


def train_model(args):
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    train_loader = iter(loader)

    model = SingleViewto3D(args).float()
    #model_preprocess = model.preprocess
    model.to(args.device)
    model.train()
    # ============ preparing optimizer ... ============
    params = get_params(model)
    optimizer = torch.optim.Adam(params, lr=args.lr)  # to use with ViTs
    #lr_scheduler = cosine_anneal_schedule(optimizer, T_max=args.max_iter)
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
       checkpoint = torch.load(f"checkpoint_{args.type}{args.tag}.pth")
       model.load_state_dict(checkpoint["model_state_dict"])
       optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
       start_iter = checkpoint["step"]
       print(f"Succesfully loaded iter {start_iter}")

    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader) #the next batch (B,C,H,W), don't 
                                        # really get what we're squeezing here.

        images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time
        """

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        """
        # risk: images_gt might and not be the right shape
        prediction_3d = model(images_gt, args)
        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #lr_scheduler.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        writer = get_writer().add_scalar("Loss", loss_vis, get_step("loss"))
        if (step % args.save_freq) == 0 and step > 0:
            print(f"Saving checkpoint at step {step}")
            save_checkpoint(step, model.state_dict(), optimizer.state_dict(), name=f"checkpoint_{args.type}" + args.tag)
        print(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
            % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
        )

    save_checkpoint(step, model.state_dict(), optimizer.state_dict(), name=f"checkpoint_{args.type}" + args.tag)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
