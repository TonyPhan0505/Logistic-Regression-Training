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

batch_size_test = 1000
log_interval = 100
validation_interval = 3

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

def load_datasets(dataset_name):
    train_dataset = None
    test_dataset = None
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    else:
        raise AssertionError(f"Invalid dataset: {dataset_name}")
    return train_dataset, test_dataset

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

def load_data(dataset_name, batch_size_train):
    train_dataset, test_dataset = load_datasets(dataset_name)
    training_set, validation_set = random_split(train_dataset, [48000, 12000])
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)
    print("--- Check train loader:")
    check_dataloader(train_loader)
    print("--- Check validation loader:")
    check_dataloader(validation_loader)
    print("--- Check test loader:")
    check_dataloader(test_loader)
    return train_loader, validation_loader, test_loader

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

def eval(data_loader, model, device, set_name):
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
    acc = 100. * correct / len(data_loader.dataset)
    print(set_name + ' set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), acc))
    return acc, loss

def logistic_regression(dataset_name, device, learning_rate = 1e-3, batch_size_train = 200, weight_decay = 1e-5, n_epochs = 20):
    # TODO: implement logistic regression here
    logistic_regression_model = None
    if dataset_name == "MNIST":
        logistic_regression_model = LogisticRegressionModel(28*28).to(device)
    else:
        logistic_regression_model = LogisticRegressionModel(32 * 32 * 3).to(device)
    optimizer = optim.Adam(logistic_regression_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader, validation_loader, test_loader = load_data(dataset_name, batch_size_train)
    acc = loss = 0
    eval(validation_loader, logistic_regression_model, device, "Validation")
    for epoch in range(1, n_epochs + 1):
        train(epoch, train_loader, logistic_regression_model, device, optimizer)
        if epoch % validation_interval == 0:
            acc, loss = eval(validation_loader, logistic_regression_model, device, "Validation")
    eval(test_loader, logistic_regression_model, device, "Test")
    results = dict(
        model=logistic_regression_model,
        accuracy=acc,
        loss=loss
    )
    return results

def tune_hyper_parameter(dataset_name, target_metric, device):
    # TODO: implement logistic regression hyper-parameter tuning here
    best_params = best_metric = None
    if target_metric == "acc":
        hyperparameter_grid = {
            'learning_rate': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'batch_size_train': [200, 300, 400, 500, 600, 700],
            'weight_decay': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-1]
        }
        learningRateIndex = 0
        batchSizeTrainIndex = 0
        weightDecayIndex = 0
        currentAccuracy = 0
        while learningRateIndex <= 5:
            while batchSizeTrainIndex <= 5:
                while weightDecayIndex <= 5:
                    learningRate = hyperparameter_grid['learning_rate'][learningRateIndex]
                    batchSizeTrain = hyperparameter_grid['batch_size_train'][batchSizeTrainIndex]
                    weightDecay = hyperparameter_grid['weight_decay'][weightDecayIndex]
                    results = logistic_regression(dataset_name, device, learningRate, batchSizeTrain, weightDecay)
                    accuracy = results['accuracy']
                    if accuracy > currentAccuracy:
                        currentAccuracy = accuracy
                        best_metric = accuracy
                        best_params = {
                            'learning_rate': learningRate,
                            'batch_size_train': batchSizeTrain,
                            'weight_decay': weightDecay
                        }
                    weightDecayIndex += 1
                batchSizeTrainIndex += 1
            learningRateIndex += 1
    else:
        hyperparameter_grid = {
            'learning_rate': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'n_epochs': [20, 30, 40, 50, 60, 70],
            'weight_decay': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-1]
        }
        learningRateIndex = 0
        nEpochsIndex = 0
        weightDecayIndex = 0
        currentLoss = float('inf')
        while learningRateIndex <= 5:
            while nEpochsIndex <= 5:
                while weightDecayIndex <= 5:
                    learningRate = hyperparameter_grid['learning_rate'][learningRateIndex]
                    nEpochs = hyperparameter_grid['n_epochs'][nEpochsIndex]
                    weightDecay = hyperparameter_grid['weight_decay'][weightDecayIndex]
                    results = logistic_regression(dataset_name, device, learningRate, 200, weightDecay, nEpochs)
                    loss = results['loss']
                    if loss < currentLoss:
                        currentLoss = loss
                        best_metric = loss
                        best_params = {
                            'learning_rate': learningRate,
                            'n_epochs': nEpochs,
                            'weight_decay': weightDecay
                        }
                    weightDecayIndex += 1
                nEpochsIndex += 1
            learningRateIndex += 1
    return best_params, best_metric