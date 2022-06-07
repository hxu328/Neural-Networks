import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training=True):
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=custom_transform)
    test_set = datasets.MNIST('./data', train=False, transform=custom_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)
    if training:
        return train_loader
    if not training:
        return test_loader


def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get inputs
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            if i % len(train_loader) == len(train_loader)-1:
                correct = 0
                total = 0
                for points, correct_labels in train_loader:
                    outs = model(points)
                    pred = torch.max(outs, 1)[1]
                    total += correct_labels.size(0)
                    temp = (pred == correct_labels).sum()
                    correct += temp.item()
                accuracy = (correct/total) * 100
                loss = running_loss / len(train_loader)
                print('Train Epoch: {}   Accuracy: {}/{}({:.2f}%)  Loss: {:.3f}'.format(
                    epoch, correct, total, accuracy, loss
                ))


def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred = torch.max(outputs, 1)[1]
            total += labels.size(0)
            temp = (pred == labels).sum()
            correct += temp.item()
        loss = running_loss / len(test_loader)
        accuracy = 100 * correct / total
        if show_loss:
            print('Average loss: {:.4f}'.format(loss))
            print('Accuracy: {:.2f}%'.format(accuracy))
        else:
            print('Accuracy: {:.2f}%'.format(accuracy))


def predict_label(model, test_images, index):
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    image = test_images[index]
    image = image.unsqueeze(0)

    output = model(image)
    prob = F.softmax(output, dim=1)
    prob = prob.squeeze(0)
    prob = prob.tolist()

    s = np.array(prob)
    sort_index = np.argsort(s)
    sort_index = sort_index.tolist()
    first_index = sort_index[len(sort_index) - 1]
    second_index = sort_index[len(sort_index) - 2]
    third_index = sort_index[len(sort_index) - 3]
    first = prob[first_index] * 100
    second = prob[second_index] * 100
    third = prob[third_index] * 100

    print('{}: {:.2f}%'.format(class_names[first_index], first))
    print('{}: {:.2f}%'.format(class_names[second_index], second))
    print('{}: {:.2f}%'.format(class_names[third_index], third))


if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()
