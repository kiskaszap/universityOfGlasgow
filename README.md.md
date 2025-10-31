# End‑to‑end PyTorch Examples — One‑file Markdown

This document contains **three fully runnable PyTorch examples**, each with:
- Model definition
- Optimizer + loss
- Epoch training loop
- Metrics printing
- Matplotlib loss curve

Copy any code block to a `.py` file and run it **inside your conda env**.

> Quick setup (inside terminal):
> ```bash
> conda create --name pytorch_env -y
> conda activate pytorch_env
> conda install pytorch torchvision torchaudio -c pytorch -y
> pip install matplotlib
> ```
---

## 1) Linear Regression — MSELoss + Adam (synthetic data)

```python
# linear_regression_example.py
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1) Reproducibility
torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

# 2) Synthetic dataset: y = 3*x + 2 + noise
N = 500
X = np.random.uniform(-5.0, 5.0, size=(N, 1)).astype(np.float32)
true_w = 3.0
true_b = 2.0
noise = np.random.normal(0, 0.8, size=(N, 1)).astype(np.float32)
y = true_w * X + true_b + noise

X_t = torch.from_numpy(X)  # [N, 1]
y_t = torch.from_numpy(y)  # [N, 1]

# 3) Model
class LinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)  # linear layer

    def forward(self, x):
        return self.fc(x)

model = LinearRegressor()

# 4) Loss + Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# 5) Training loop
EPOCHS = 30
batch_size = 64
loss_history = []

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N)
    epoch_loss = 0.0

    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        xb = X_t[idx]
        yb = y_t[idx]

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= math.ceil(N / batch_size)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.6f}")

# 6) Inspect learned parameters
w_learned = model.fc.weight.item()
b_learned = model.fc.bias.item()
print(f\"Learned w ≈ {w_learned:.3f}, b ≈ {b_learned:.3f} (true w={true_w}, b={true_b})\")

# 7) Plot loss
plt.figure()
plt.plot(loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Linear Regression — Training Loss')
plt.grid(True)
plt.show()
```

---

## 2) Binary Classification — Sigmoid + BCELoss + SGD (synthetic)

```python
# binary_classification_example.py
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1) Reproducibility
torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

# 2) Synthetic 2D blobs (two classes)
N_PER_CLASS = 400
mean0 = np.array([-2.0, -2.0])
mean1 = np.array([ 2.0,  2.0])
cov = np.array([[1.0, 0.2],
                [0.2, 1.0]])

X0 = np.random.multivariate_normal(mean0, cov, size=N_PER_CLASS).astype(np.float32)
X1 = np.random.multivariate_normal(mean1, cov, size=N_PER_CLASS).astype(np.float32)
X = np.vstack([X0, X1])
y = np.vstack([np.zeros((N_PER_CLASS,1),dtype=np.float32),
               np.ones((N_PER_CLASS,1),dtype=np.float32)])

# Shuffle
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

X_t = torch.from_numpy(X)  # [N,2]
y_t = torch.from_numpy(y)  # [N,1]

# 3) Model: 2 -> 8 -> 1
class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = BinaryNet()

# 4) Loss + Optimizer (BCELoss + SGD)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 5) Train
EPOCHS = 15
batch_size = 64
N = len(X_t)
loss_history = []

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N)
    epoch_loss = 0.0

    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        xb = X_t[idx]
        yb = y_t[idx]

        optimizer.zero_grad()
        probs = model(xb)           # [B,1] in (0,1)
        loss = criterion(probs, yb) # BCE expects probabilities
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= math.ceil(N / batch_size)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.6f}")

# 6) Accuracy
with torch.no_grad():
    probs = model(X_t)
    preds = (probs >= 0.5).float()
    acc = (preds.eq(y_t).float().mean().item()) * 100.0
print(f\"Train accuracy: {acc:.2f}%\")


# 7) Plot loss
plt.figure()
plt.plot(loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.title('Binary Classification — Training Loss')
plt.grid(True)
plt.show()
```

---

## 3) MNIST — Multiclass Classification (ReLU + CrossEntropyLoss)

```python
# mnist_mlp_example.py
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1) Reproducibility
torch.manual_seed(1337)
random.seed(1337)

# 2) Data
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

# 3) Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # x: [B,1,28,28] -> flatten
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # CrossEntropyLoss expects raw logits
        return x

model = MLP()

# 4) Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 5) Train
EPOCHS = 5
loss_history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    loss_history.append(epoch_loss)
    print(f\"Epoch {epoch:02d} | Loss: {epoch_loss:.4f}\")

# 6) Evaluate
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        logits = model(images)
        _, preds = torch.max(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
acc = 100.0 * correct / total
print(f\"Test accuracy: {acc:.2f}%\")


# 7) Plot training loss
plt.figure()
plt.plot(loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('CrossEntropy Loss')
plt.title('MNIST MLP — Training Loss')
plt.grid(True)
plt.show()
```

---

### Notes
- All three scripts are **standalone**. Save each block to its own `.py` file and run with `python file.py` inside your environment.
- For GPU usage, add:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  # then move your tensors/images with .to(device)
  ```
