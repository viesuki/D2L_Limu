import os
import time
import copy
import torch
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from torch import nn
from IPython import display
from torch.utils.data import Dataset
from matplotlib_inline import backend_inline
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CosineAnnealingLR


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')
        
class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(5.5, 3.5)):
        """Defined in :numref:`sec_utils`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




class MyDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, dataf=None, transform=None, train=True):
        if csv_file:
            self.data = pd.read_csv(csv_file)
        else:
            self.data = dataf
        self.img_dir = img_dir
        self.transform = transform
        self.train = train   # train / test

        if self.train:
            classes = sorted(self.data.iloc[:, 1].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        else:
            self.class_to_idx = None
            self.idx_to_class = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx, 0])  # filename
        img_path = os.path.join(self.img_dir, img_name)        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)        
            
        if self.train:
            # train dataset with labels
            label = str(self.data.iloc[idx, 1])  # label
            label = self.class_to_idx[label]  # convert to index
            return image, label
        else:
            # test dataset without labels
            return image, img_name





def evaluate_loss_gpu(net, data_iter, loss, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
        net.to(device)        
    metric = Accumulator(2)  # Sum of losses, no. of examples

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(loss(net(X), y) * y.numel(), y.numel())

    return metric[0] / metric[1]

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]




def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def train_image_valid(net, train_iter, valid_iter, num_epochs, lr, device):
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 
                                    'valid loss', 'valid acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None, None))
        valid_l = evaluate_loss_gpu(net, valid_iter, loss)
        valid_acc = evaluate_accuracy_gpu(net, valid_iter)
        animator.add(epoch + 1, (None, None, valid_l, valid_acc))
    print(f'train loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'valid loss {valid_l:.3f}, valid acc {valid_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    

def train_image(net, train_iter, valid_iter, num_epochs, lr, device):
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train acc', 'valid acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (train_l, train_acc, None, None))
        valid_l = evaluate_loss_gpu(net, valid_iter, loss)
        valid_acc = evaluate_accuracy_gpu(net, valid_iter)
        animator.add(epoch + 1, (train_acc, valid_acc))
        print(f'train loss {train_l:.3f}, train acc {train_acc:.3f}, '
                f'valid loss {valid_l:.3f}, valid acc {valid_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')



def best_model_ada(net, train_iter, valid_iter, num_epochs, lr, device,
               patience=7, monitor='acc'):
    """
    训练并返回验证集表现最优的模型
    Args:
        net: 神经网络
        train_iter: 训练 DataLoader
        valid_iter: 验证 DataLoader
        num_epochs: 最大迭代轮数
        lr: 学习率
        device: cuda/cpu
        patience: early stopping 容忍的轮数
        monitor: 'acc' 或 'loss'，监控哪个指标做early stopping
    """
    # net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()

    best_state = None
    best_metric = -float("inf") if monitor == 'acc' else float("inf")
    epochs_no_improve = 0
    timer = Timer()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1.25],
                        legend=['train loss','train acc', 'valid loss','valid acc'])
    # torch.backends.cudnn.benchmark = True # accelerate training

    for epoch in range(num_epochs):
        # ---- Train ----
        net.train()
        train_los, train_acc, total = 0.0, 0, 0
        for X, y in train_iter:
            timer.start()
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            y_hat = net(X)
            los = loss(y_hat, y)
            los.backward()
            optimizer.step()            
            timer.stop()

            train_los += los.item() * X.size(0)
            train_acc += (y_hat.argmax(1) == y).sum().item()
            total += X.size(0)

        scheduler.step()
        train_los /= total
        train_acc = train_acc / total

        # ---- Validation ----
        net.eval()
        valid_los = evaluate_loss_gpu(net, valid_iter, loss)
        valid_acc = evaluate_accuracy_gpu(net, valid_iter)
        # if epoch < 5: warmup.step()
        # else: scheduler.step()

        print('Current learning rate:', scheduler.get_last_lr())
        print(f"Epoch {epoch+1:03d}: "
              f"train_loss={train_los:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={valid_los:.4f}, val_acc={valid_acc:.4f}")        

        # ---- Save best ----
        current_metric = valid_acc if monitor == 'acc' else valid_los
        is_better = (current_metric > best_metric) if monitor == 'acc' else (current_metric < best_metric)
        if is_better:
            best_metric = current_metric
            best_state = copy.deepcopy(net.state_dict())
            epochs_no_improve = 0
            print("  ✅ best model updated")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  ⏹️ Early stopping triggered")
                break

        animator.add(epoch + 1, (train_los, train_acc, valid_los, valid_acc))

    print(f'{total * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    print('Current learning rate:', scheduler.get_last_lr())
    print(f"Epoch {epoch+1:03d}: "
              f"train_loss={train_los:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={valid_los:.4f}, val_acc={valid_acc:.4f}")   
    
    # ---- Load best ----
    if best_state is not None:
        net.load_state_dict(best_state)

    return net




