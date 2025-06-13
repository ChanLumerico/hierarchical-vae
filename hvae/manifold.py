import torch
import torchvision
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from hvae import Encoder, Decoder, HierarchicalVAE


latent_dim = 20
grid_size = 16
traversal_range = 3.0
dim_x, dim_y = 0, 2


with torch.serialization.safe_globals([HierarchicalVAE]):
    model: HierarchicalVAE = torch.load("hvae/out/2_hier_vae.pth", weights_only=False)
    model.eval()


xs = torch.linspace(-traversal_range, traversal_range, grid_size)
ys = torch.linspace(-traversal_range, traversal_range, grid_size)
mesh = torch.stack(torch.meshgrid(xs, ys), -1)
flat = mesh.view(-1, 2)

z_full = torch.zeros(flat.size(0), latent_dim)
z_full[:, dim_x] = flat[:, 0]
z_full[:, dim_y] = flat[:, 1]


with torch.no_grad():
    h = z_full
    for dec in reversed(model.decoders):
        h = dec(h)
    imgs = h.view(-1, 1, 28, 28)


img_grid = make_grid(imgs, nrow=grid_size, pad_value=1.0)
gray_np = img_grid[0].numpy()

plt.figure(figsize=(6, 6))
plt.imshow(gray_np, cmap="gray")
plt.axis("off")
plt.title("Manifold Traverse over 2-D Latent Space")
plt.tight_layout()
plt.savefig("manifold.png")
