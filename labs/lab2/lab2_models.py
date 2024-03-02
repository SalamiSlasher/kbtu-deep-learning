from time import time
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class MLP(nn.Module):
    def __init__(self, hidden_sizes: list[int], output_size=10, activation_func=nn.ReLU,
                 dropout_prob=None, use_batchnorm=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.LazyLinear(hidden_sizes[0]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i]))

            layers.append(activation_func())

            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def add_optimizer(self, optim_func, param_dict=dict()):
        self.optimizer = optim_func(self.model.parameters(), **param_dict)

    def accuracy(self, model, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        return correct / total


def train(self, model,  num_epochs, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transform, download=True)

    # Split into train and validation
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # Set dataloaders
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Set loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    # One epoch = one pass over training dataset
    losses, train_accs, val_accs, times = [], [], [], []
    for epoch in range(num_epochs):
        begin = time()
        total_correct_train = 0
        total_samples_train = 0
        self.model.train()
        running_loss = 0.0

        # Iterating by mini-batches
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28 * 28)

            # Set all gradients to 0
            self.optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss function
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient step
            self.optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct_train += (predicted == labels).sum().item()
            total_samples_train += labels.size(0)

            if (i+1) % (len(train_loader) // 10) == 0:
                print('Epoch [{}/{}], step [{}/{}],train loss: {:.4f}, train accuracy: {:.2%}'.format(
                    epoch+1, num_epochs, i+1, len(train_loader),
                    running_loss/(i+1), total_correct_train / total_samples_train)
                )
        val_acc = self.accuracy(val_loader)
        losses.append(running_loss/len(train_loader))
        train_accs.append(total_correct_train / total_samples_train)
        val_accs.append(val_acc)
        times.append(time() - begin)
        print(f"Validation accuracy: {val_acc:.2%}")
        print(f"Epoch {epoch + 1} took {times[-1]:.3f} s")
    test_acc = self.accuracy(test_loader)
    print(f"Test accuracy: {test_acc:.2%}")
    return losses, train_accs, val_accs, test_acc, sum(times)