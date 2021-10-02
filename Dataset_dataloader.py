import os
import torchvision.transforms as transforms
import torchvision
import torch


training_dataset_path = './train'
val_dataset_path = './validation'

datasets_transforms = transforms.Compose([transforms.ToTensor()])

training_dataset = torchvision.datasets.ImageFolder(root=training_dataset_path, transform=datasets_transforms)
validation_dataset = torchvision.datasets.ImageFolder(root=val_dataset_path, transform=datasets_transforms)

training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=3000, shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=3000, shuffle=False)

for images, _ in training_loader:
    tr_image_count_in_a_batch = images.size(0)
    print("training images" + str(tr_image_count_in_a_batch))

for images, _ in validation_loader:
    tr_image_count_in_a_batch = images.size(0)
    print("validation images" + str(tr_image_count_in_a_batch))