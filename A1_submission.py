"""
TODO: Finish and submit your code for logistic regression and hyperparameter search.
"""
import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 10
batch_size_train = 200
batch_size_test = 1000
learning_rate = 1e-3
momentum = 0.5
log_interval = 100
validation_interval = 3
weight_decay = 1e-5  # reduce overfitting

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class LogisticRegressionModel(nn.Module):
    def __init__(self, dimensions):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(dimensions, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_dataset(dataset_name):
    dataset = None
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    else:
        raise AssertionError(f"Invalid dataset: {dataset_name}")
    return dataset

def check_dataloader(loader):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    print(example_targets.shape)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

def load_data(dataset_name):
    dataset = load_dataset(dataset_name)
    training_set, validation_set = random_split(dataset, [48000, 12000])
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size_train, shuffle=True)
    print("--- Check train loader:")
    check_dataloader(train_loader)
    print("--- Check validation loader:")
    check_dataloader(validation_loader)
    return train_loader, validation_loader

def train(epoch, data_loader, model, device, optimizer):
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item())
            )

def eval(data_loader, model, device):
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss += F.cross_entropy(output, target, reduction='sum').item()
    loss /= len(data_loader.dataset)
    print('set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

def logistic_regression(dataset_name, device):
    # TODO: implement logistic regression here
    logistic_regression_model
    if dataset_name == "MNIST":
        logistic_regression_model = LogisticRegressionModel(28*28).to(device)
    else:
        logistic_regression_model = LogisticRegressionModel(32 * 32 * 3).to(device)
    optimizer = optim.Adam(logistic_regression_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader, validation_loader = load_data(dataset_name)
    eval(validation_loader, logistic_regression_model, device)
    for epoch in range(1, n_epochs + 1):
        train(epoch, train_loader, logistic_regression_model, device, optimizer)
        if epoch % validation_interval == 0:
            eval(validation_loader, logistic_regression_model, device)
    results = dict(
        model=logistic_regression_model
    )
    return results

def tune_hyper_parameter(dataset_name, target_metric, device):
    # TODO: implement logistic regression hyper-parameter tuning here
    best_params = best_metric = None
    return best_params, best_metric