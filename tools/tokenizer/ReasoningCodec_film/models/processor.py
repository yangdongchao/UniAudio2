import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp

class SampleProcessor(torch.nn.Module):
    def project_sample(self, x: torch.Tensor):
        """Project the original sample to the 'space' where the diffusion will happen."""
        return x

    def return_sample(self, z: torch.Tensor):
        """Project back from diffusion space to the actual sample space."""
        return z

class Feature2DProcessor(SampleProcessor):
    def __init__(self, counts, sum_x, sum_x2, dim: int = 8, power_std: tp.Union[float, tp.List[float], torch.Tensor] = 1.):
        ''' we fix the statistical prior
        '''
        super().__init__()
        self.dim = dim
        self.power_std = power_std
        self.counts = counts
        self.sum_x = sum_x
        self.sum_x2 = sum_x2

        self.mean = self.sum_x / self.counts
        self.std =  (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()

    @property
    def target_std(self):
        return 1

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        x = (x - self.mean.view(1, -1, self.dim).contiguous()) * rescale.view(1, -1, self.dim).contiguous()
        return x
    
    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        rescale = (self.std / self.target_std) ** self.power_std
        x = x * rescale.view(1, -1, self.dim).contiguous() + self.mean.view(1, -1, self.dim).contiguous()
        return x

