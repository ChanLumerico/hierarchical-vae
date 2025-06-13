import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

from hvae import Encoder, Decoder, HierarchicalVAE


input_dim = 784
latent_dim = 20
sample_size = 64
batch_size = 256


with torch.serialization.safe_globals([HierarchicalVAE]):
    model: HierarchicalVAE = torch.load("hvae/out/2_hier_vae.pth", weights_only=False)
    model.eval()


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))]
)
mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=batch_size, shuffle=False)


zs = []
labels = []
with torch.no_grad():
    for x, y in loader:
        h = x
        # pass through all encoders in the ModuleList
        for enc in model.encoders:
            mu, sigma = enc(h)
            h = mu  # use the mean for t-SNE
        z_top = h  # shape [batch_size, latent_dim]
        zs.append(z_top.numpy())
        labels.append(y.numpy())

zs = np.concatenate(zs, axis=0)  # shape [N, 20]
labels = np.concatenate(labels, axis=0)

# 6) run t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
z2d = tsne.fit_transform(zs)  # [N, 2]

plt.figure(figsize=(8, 6))
sc = plt.scatter(z2d[:, 0], z2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.8)
cbar = plt.colorbar(sc, ticks=range(10))
cbar.set_label("True digit label")
plt.clim(-0.5, 9.5)
plt.title(f"t-SNE of {latent_dim}-D Latent Space")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("tsne")
