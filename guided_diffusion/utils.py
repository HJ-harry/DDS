import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_alpha_schedule(type="linear", start=0.4, end=0.015, total=1000):
    if type == "linear":
        schedule = np.linspace(start, end, total)
    elif type == "const":
        assert start == end, f"For const schedule, start and end should match. Got start:{start}, end:{end}"
        schedule = np.full(total, start)
    return np.flip(schedule)


def CG(A_fn, b_cg, x, n_inner=10, eps=1e-8):
    r = b_cg - A_fn(x)
    p = r.clone()
    rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)
    for _ in range(n_inner):
        Ap = A_fn(p)
        a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

        x += a * p
        r -= a * Ap

        rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
        
        if torch.sqrt(rs_new) < eps:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

# x0_t_hat = x0_t - A_funcs.A_pinv(
#                     A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
#                 ).reshape(*x0_t.size())
# returns vectorized A^T(A(x))


def Acg(x, A_func):
    x_vec = x.reshape(x.size(0), -1)
    tmp = A_func.At(A_func.A(x_vec))
    return tmp.reshape(*x.size())


def clear_color(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))


def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return x


def normalize_np(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= np.min(img)
  img /= np.max(img)
  return img


def clip(img):
    return torch.clip(img, -1.0, 1.0)