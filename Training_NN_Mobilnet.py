import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

# importing dataset
train_data_folder = './train'
val_data_folder = './validation'
writer = SummaryWriter('runs/tensor_board_mobilnet_v3_small')
training_model = models.mobilenet_v3_small(pretrained=True)

train_transforms = transforms.Compose([transforms.RandomCrop(90), transforms.RandomRotation(15),
                                         transforms.ToTensor()])

training_dataset = torchvision.datasets.ImageFolder(root=train_data_folder, transform=train_transforms)
validation_dataset = torchvision.datasets.ImageFolder(root=val_data_folder, transform=train_transforms)

'''To show the images loaded from the ImageFolder above
def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)
    batch = next(iter(loader))
    # images output (batch_size, channels, height, width)
    images, labels = batch

    # the grid nrow determines the images per row
    # grid.shape gives (no of channels, grid height, grid width)
    grid = torchvision.utils.make_grid(images, nrow=4)
    # figure size is the size of the display image
    plt.figure(figsize=(5, 5))
    # the following will syntax will output grid.shape into (grid height, grid width, no of channels)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
    print('labels: ', labels)


show_transformed_images(train_dataset)
'''

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=32, shuffle=True)
# original_loader = torch.utils.data.DataLoader(original_dataset, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_imageset = iter(training_loader)
training_images, _ = next(training_imageset)


def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    best_acc = 0
    for epoch in range(n_epochs):
        print("Epoch number %d" % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            # unpack data, labels are the classifications
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # total = cumulative no of images in the dataset
            total += labels.size(0)
            # print(f"labels: ", labels)
            optimizer.zero_grad()
            outputs = model(images)
            print('output shape', outputs.shape)

            _, predicted = torch.max(outputs.data, 1)
            # the predicted will output torch.Size([32]), i.e. 1d, and is use for target labels or predictions
            print('training predicted ', predicted)
            print(predicted.shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * running_correct / total
        writer.add_scalar('Training loss', loss, epoch)
        writer.add_scalar('Training accuracy', epoch_acc, epoch)
        writer.close()

        print("- Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))

        test_dataset_acc = evaluate_model_on_test_set(model, test_loader, epoch)
        if test_dataset_acc > best_acc:
            best_acc = test_dataset_acc
            save_checkpoint(model, epoch, optimizer, best_acc)

    print("Finished")
    return model


def evaluate_model_on_test_set(model, test_loader, epoch):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print('testing predicted ', predicted)
            print(predicted.shape)
            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    epoch_acc = 100.0 * predicted_correctly_on_epoch / total
    writer.add_scalar('Validation accuracy', epoch_acc, epoch)
    writer.close()
    print("- Validation dataset. Got %d out of %d images correctly (%.3f%%)"
          % (predicted_correctly_on_epoch, total, epoch_acc))

    return epoch_acc


def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best accuracy': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, '20211026_mobilnet_v3_small.pth.tar')


num_features = training_model.classifier[-1].in_features
number_of_classes = 651
training_model_fc = nn.Linear(num_features, number_of_classes)
training_model.classifier[-1] = training_model_fc
training_model = training_model.to(device)
loss_fn = nn.CrossEntropyLoss()
# momentum = accelerate the gradient vectors in the right direction, leading to faster converging
# weight decay = help with prevent overfitting
optimizer = optim.SGD(training_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.003)

writer.add_graph(training_model, training_images.to(device))
writer.close()

train_nn(training_model, training_loader, validation_loader, loss_fn, optimizer, 100)

checkpoint = torch.load('20211026_mobilnet_v3_small.pth.tar')
print(checkpoint['epoch'])
print(checkpoint['best accuracy'])

num_features = training_model.classifier[-1].in_features
number_of_classes = 651
training_model_fc = nn.Linear(num_features, number_of_classes)
training_model.classifier[-1] = training_model_fc
training_model.load_state_dict(checkpoint['model'])

torch.save(training_model.state_dict(), '20211026_mobilnet_v3_small.pth')
