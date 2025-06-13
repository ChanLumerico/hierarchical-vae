import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.linear(x))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(latent_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.use_sigmoid = use_sigmoid

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.linear_1(z))
        h = self.linear_2(h)
        return torch.sigmoid(h) if self.use_sigmoid else h


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(sigma)
    return mu + eps * sigma


class HierarchicalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        use_bce: bool = True,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "Number of layers must be >= 1"
        self.num_layers = num_layers
        self.use_bce = use_bce

        # Build encoders and decoders dynamically
        dims = [input_dim] + [latent_dim] * (num_layers - 1)
        self.encoders = nn.ModuleList(
            [Encoder(dims[i], hidden_dim, latent_dim) for i in range(num_layers)]
        )
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # decoder for z1 -> x
                self.decoders.append(
                    Decoder(latent_dim, hidden_dim, input_dim, use_sigmoid=True)
                )
            else:
                # decoder for z_{i+1} -> z_i
                self.decoders.append(Decoder(latent_dim, hidden_dim, latent_dim))

    def get_loss(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Encoding pass
        mus, sigmas, zs = [], [], []
        h = x
        for enc in self.encoders:
            mu, sigma = enc(h)
            z = reparameterize(mu, sigma)
            mus.append(mu)
            sigmas.append(sigma)
            zs.append(z)
            h = z

        # Decoding pass
        x_hat = self.decoders[0](zs[0])
        z_hats = [None] * (self.num_layers - 1)
        for level in range(self.num_layers, 1, -1):
            idx = level - 1
            z_hats[idx - 1] = self.decoders[idx](zs[idx])

        # Reconstruction loss
        if self.use_bce:
            L_recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
        else:
            L_recon = F.mse_loss(x_hat, x, reduction="sum")

        # KL divergence
        # Top-level prior N(0,I)
        mu_T, sigma_T = mus[-1], sigmas[-1]
        L_kl = -torch.sum(1 + torch.log(sigma_T.pow(2)) - mu_T.pow(2) - sigma_T.pow(2))

        # Intermediate levels prior N(z_hat, I)
        for i in range(self.num_layers - 1):
            mu_i, sigma_i = mus[i], sigmas[i]
            z_hat_i = z_hats[i]
            L_kl += -torch.sum(
                1 + torch.log(sigma_i.pow(2)) - (mu_i - z_hat_i).pow(2) - sigma_i.pow(2)
            )

        return (L_recon + L_kl) / batch_size


if __name__ == "__main__":
    # Hyperparameters
    input_dim = 784
    hidden_dim = 100
    latent_dim = 20
    num_layers = 2
    epochs = 30
    learning_rate = 1e-3
    batch_size = 64

    # Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))]
    )
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Model & optimizer
    model = HierarchicalVAE(input_dim, hidden_dim, latent_dim, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        desc = f"Epoch {epoch+1}/{epochs}"
        with tqdm(dataloader, desc=desc, unit="batch") as pbar:
            for x, _ in pbar:
                optimizer.zero_grad()
                loss = model.get_loss(x)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                loss_sum += loss.item()
                cnt += 1
                pbar.set_postfix(avg_loss=loss_sum / cnt)

    # Save model & plot
    torch.save(model, f"hvae/out/{num_layers}_hier_vae.pth")

    plt.figure()
    plt.plot(losses, lw=1, label="ELBO Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{num_layers}-Hierarchy VAE on MNIST")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result")
    plt.show()
