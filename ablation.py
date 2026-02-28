import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import copy

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float() 
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

# data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# SGD

class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                p.sub_(d_p, alpha=lr)

        return loss


# NewtonSchulzSGD
class NewtonSchulzSGD(Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad

                if len(d_p.shape) == 2:
                    orthogonal_d_p = newtonschulz5(d_p).to(p.dtype)
                    p.sub_(orthogonal_d_p, alpha=lr)
                else:
                    p.sub_(d_p, alpha=lr)

        return loss


# Momentum
class Momentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
            
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(d_p)
                p.sub_(buf, alpha=lr)

        return loss


# Muon
class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
            
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(d_p)
                if len(p.shape) == 2:
                    orthogonal_update = newtonschulz5(buf).to(p.dtype)
                    p.sub_(orthogonal_update, alpha=lr)
                else:
                    p.sub_(buf, alpha=lr)

        return loss


# train

def train_optimizer(optimizer_class, opt_name, model, train_loader, device, epochs=5, **opt_kwargs):
    print(f"\n 开始训练: {opt_name}")
    
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 记录每个 Epoch 的平均 Loss
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"  Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}")
        
    return epoch_losses

if __name__ == "__main__":
    epochs_to_run = 8 
    
    base_model = SimpleMLP().to(device)
    
    experiments = {
        "1. Standard SGD": (SGD, {"lr": 0.05}),
        "2. Orthogonal SGD (NS)": (NewtonSchulzSGD, {"lr": 0.05}),
        "3. Momentum SGD": (Momentum, {"lr": 0.05, "momentum": 0.9}),
        "4. Muon (Momentum + NS)": (Muon, {"lr": 0.05, "momentum": 0.9})
    }
    
    results = {}
    
    for name, (opt_class, kwargs) in experiments.items():
        model_copy = copy.deepcopy(base_model)
        losses = train_optimizer(opt_class, name, model_copy, train_loader, device, epochs=epochs_to_run, **kwargs)
        results[name] = losses

    plt.figure(figsize=(10, 6))
    styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (name, losses) in enumerate(results.items()):
        plt.plot(range(1, epochs_to_run + 1), losses, label=name, 
                 linestyle=styles[i], color=colors[i], linewidth=2.5, marker='o')

    plt.title('Optimizer Ablation Study on MNIST (SimpleMLP)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Training Loss (Cross Entropy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    print("\n训练完成，生成对比图表...")
    plt.tight_layout()
    plt.show()