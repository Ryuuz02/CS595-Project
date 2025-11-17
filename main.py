import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        MNIST(root="data", train=True, download=True)
        MNIST(root="data", train=False, download=True)

    def setup(self, stage=None):
        self.train = MNIST(root="data", train=True, transform=self.transform)
        self.test = MNIST(root="data", train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

class SimpleMNISTNet(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
class GossipNode:
    def __init__(self, model, dataloader, id):
        self.model = model
        self.id = id
        self.dataloader = dataloader
        self.trainer = pl.Trainer(
            max_epochs=5,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            devices=1,
            accelerator="cuda",   # real experiments can use GPU
        )

    def local_train(self):
        self.trainer.fit(self.model, self.dataloader)

    def get_weights(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, new_state):
        self.model.load_state_dict(new_state)

    def gossip_with(self, other, alpha=0.5):
        """Average this node's weights with another node's weights."""
        w1 = self.get_weights()
        w2 = other.get_weights()
        mixed = {}
        for k in w1.keys():
            mixed[k] = alpha * w1[k] + (1 - alpha) * w2[k]
        self.set_weights(mixed)
        other.set_weights(mixed)

def make_nodes(num_nodes=5, batch_size=64):
    data = MNISTDataModule(batch_size=batch_size)
    data.prepare_data()
    data.setup()

    nodes = []
    for i in range(num_nodes):
        model = SimpleMNISTNet()
        # each node gets its own dataloader slice
        loader = DataLoader(
            torch.utils.data.Subset(data.train, range(i*10000, (i+1)*10000)),
            batch_size=batch_size,
            shuffle=True,
        )
        nodes.append(GossipNode(model, loader, id=i))
    return nodes


import random

def run_gossip_simulation(nodes, rounds=20):
    for r in range(rounds):
        print(f"\n--- Round {r+1} ---")

        # Each node trains locally
        for node in nodes:
            node.local_train()

        # Gossip: randomly pair up nodes
        random.shuffle(nodes)
        for i in range(0, len(nodes), 2):
            if i+1 < len(nodes):
                nodes[i].gossip_with(nodes[i+1], alpha=0.5)

if __name__ == "__main__":
    num_nodes = 4
    nodes = make_nodes(num_nodes=num_nodes, batch_size=64)
    run_gossip_simulation(nodes, rounds=10)
