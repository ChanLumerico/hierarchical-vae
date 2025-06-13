import torch
import torchvision

import matplotlib.pyplot as plt

from hvae import Encoder, Decoder, HierarchicalVAE


input_dim = 784
latent_dim = 20
sample_size = 64


with torch.serialization.safe_globals([HierarchicalVAE]):
    model: HierarchicalVAE = torch.load("hvae/out/2_hier_vae.pth", weights_only=False)


with torch.no_grad():
    sample_size = 64
    z = torch.randn(sample_size, latent_dim)

    h = z
    for dec in reversed(model.decoders):
        h = dec(h)

    gen_img = h.view(sample_size, 1, 28, 28)

grid_img = torchvision.utils.make_grid(gen_img, nrow=8, padding=2, normalize=True)

plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.tight_layout()
plt.savefig("generated")
