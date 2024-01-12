import os
import logging
import time
import glob
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, classifier_defaults, args_to_dict
from guided_diffusion.utils import get_alpha_schedule
import random

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.linalg import orth
from pathlib import Path

from physics.ct import CT
from time import time
from utils import shrink, CG, clear, batchfy, _Dz, _DzT, get_beta_schedule



class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.args.image_folder = Path(self.args.image_folder)
        for t in ["input", "recon", "label"]:
            if t == "recon":
                (self.args.image_folder / t / "progress").mkdir(exist_ok=True, parents=True)
            else:
                (self.args.image_folder / t).mkdir(exist_ok=True, parents=True)
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        ckpt = os.path.join(self.args.exp, "vp", self.args.ckpt_load_name)
        
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        print(f"Model ckpt loaded from {ckpt}")
        model.to(self.device)
        model.eval()
        model = torch.nn.DataParallel(model)

        print('Run 3D DDS + DiffusionMBIR.',
            f'{self.args.T_sampling} sampling steps.',
            f'Task: {self.args.deg}.'
            )
        self.dds3d(model)
            
            
    def dds3d(self, model):
        args, config = self.args, self.config
        print(f"Dataset path: {self.args.dataset_path}")
        root = Path(self.args.dataset_path)
        
        # parameters to be moved to args
        Nview = self.args.Nview
        rho = self.args.rho
        lamb = self.args.lamb
        n_ADMM = 1
        n_CG = self.args.CG_iter
        
        # Specify save directory for saving generated samples
        save_root = Path(self.args.image_folder)
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['vol', 'input', 'recon', 'label']
        for t in irl_types:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)
        
        # read all data
        fname_list = os.listdir(root)
        fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
        all_img = []

        batch_size = 12
        print("Loading all data")
        for fname in fname_list:
            just_name = fname.split('.')[0]
            img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))
            h, w = img.shape
            img = img.view(1, 1, h, w)
            all_img.append(img)
        all_img = torch.cat(all_img, dim=0)
        x_orig = all_img
        print(f"Data loaded shape : {all_img.shape}")
        
        img_shape = (x_orig.shape[0], config.data.channels, config.data.image_size, config.data.image_size)
        
        if self.args.deg == "SV-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=True, circle=False, device=config.device)
        elif self.args.deg == "LA-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=False, circle=False, device=config.device)
        
        A = lambda z: A_funcs.A(z)
        Ap = lambda z: A_funcs.A_dagger(z)
        
        del_z = torch.zeros(img_shape, device=self.device)
        udel_z = torch.zeros(img_shape, device=self.device)
        
        x_orig = x_orig.to(self.device)
        y = A(x_orig)
        Apy = Ap(y)
        
        ATy = A_funcs.AT(y)
        def Acg_TV(x):
            return A_funcs.AT(A_funcs.A(x)) + rho * _DzT(_Dz(x))
        
        def ADMM(x, ATy, n_ADMM=n_ADMM):
            nonlocal del_z, udel_z
            for _ in range(n_ADMM):
                bcg_TV = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
                x = CG(Acg_TV, bcg_TV, x, n_inner=n_CG)
                del_z = shrink(_Dz(x) + udel_z, lamb / rho)
                udel_z = _Dz(x) - del_z + udel_z
            return x
        
        for idx in range(Apy.shape[0]):
            plt.imsave(str(self.args.image_folder / "input" / f"{str(idx).zfill(3)}.png"), clear(Apy[idx, ...]), cmap='gray')
            
        # init x_T
        x = torch.randn(
            x_orig.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        with torch.no_grad():
            skip = config.diffusion.num_diffusion_timesteps//args.T_sampling
            n = x.size(0)
            x0_preds = []
            xs = [x]
            
            # generate time schedule
            times = range(0, 1000, skip)
            times_next = [-1] + list(times[:-1])
            times_pair = zip(reversed(times), reversed(times_next))
            
            # reverse diffusion sampling
            for i, j in tqdm.tqdm(times_pair, total=len(times)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                
                at = compute_alpha(self.betas, t.long())
                at_next = compute_alpha(self.betas, next_t.long())
                
                xt = xs[-1].to('cuda')
                
                # 0 (optional). batchfy into sizes that fit into GPU
                xt_batch = batchfy(xt, 20)
                et_agg = list()
                for _, xt_batch_sing in enumerate(xt_batch):
                    t = torch.ones(xt_batch_sing.shape[0], device=self.device) * i
                    et_sing = model(xt_batch_sing, t)
                    et_agg.append(et_sing)
                et = torch.cat(et_agg, dim=0)

                if et.size(1) == 2:
                    et = et[:, :1]

                # 1. Tweedie
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                # 2. Data consistency (ADMM TV)
                x0_t_hat = ADMM(x0_t, ATy, n_ADMM=n_ADMM)

                eta = self.args.eta

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                # DDIM sampling
                if j != 0:
                    xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et
                # Final step
                else:
                    xt_next = x0_t_hat

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            x = xs[-1]
        
        for idx in range(x.shape[0]):
            x_sv = clear(x[idx, ...])
            plt.imsave(str(self.args.image_folder / "recon" / f"{str(idx).zfill(3)}.png"), x_sv, cmap='gray')
            x_orig_sv = clear(x_orig[idx, ...])
            plt.imsave(str(self.args.image_folder / "label" / f"{str(idx).zfill(3)}.png"), x_orig_sv, cmap='gray')
            
        np.save(str(self.args.image_folder / "recon" / "recon.npy"), x.detach().cpu().squeeze().numpy())
        np.save(str(self.args.image_folder / "recon" / "original.npy"), x_orig.detach().cpu().squeeze().numpy())


        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
