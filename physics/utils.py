import torch 
try:
    import torch_radon
    from torch_radon.utils import _normalize_shape, _unnormalize_shape
except ModuleNotFoundError:
    pass 
import numpy as np
import torch.nn.functional as F


def filter_sinogram(sinogram, filter_name="ramp"):
    """
    From https://torch-radon.readthedocs.io/en/latest/_modules/torch_radon.html#Radon
    """
    # Pad sinogram to improve accuracy
    sinogram, old_sinogram = _normalize_shape(sinogram, d=2)
    size = sinogram.size(2)
    n_angles = sinogram.size(1)
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    pad = padded_size - size
    padded_sinogram = F.pad(sinogram.float(), (0, pad, 0, 0))
    # TODO should be possible to use onesided=True saving memory and time
    sino_fft = torch.fft.fft(padded_sinogram)
    # get filter and apply
    fourier_filters = torch_radon.FourierFilters()
    f = fourier_filters.get(padded_size, filter_name=filter_name, device=sinogram.device)
    filtered_sino_fft = sino_fft * f.view(-1, 1, padded_size)
    # Inverse fft
    filtered_sinogram = torch.fft.ifft(filtered_sino_fft)
    # pad removal and rescaling
    filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))

    return _unnormalize_shape(filtered_sinogram, old_sinogram).real.to(dtype=sinogram.dtype)