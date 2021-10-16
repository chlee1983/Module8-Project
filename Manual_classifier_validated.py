import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import torch.nn as nn
import torchvision.models as models
import glob
import cv2
import os
import numpy as np

label_directory = 'validation'

model = models.resnet18()
num_features = model.fc.in_features
number_of_classes = 651
model.fc = nn.Linear(num_features, number_of_classes)
model.load_state_dict(torch.load('best_model_modified_train.pth'))
image_transforms = transforms.Compose([transforms.RandomCrop(90), transforms.ToTensor()])


def find_classes(mapping_dir):
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(mapping_dir) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {mapping_dir}.")
    # class_to_idx is a dictionary
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    # for class_labels, class_idx in class_to_idx.items():
    #     print('ground truth: ', class_labels, '\tindex: ', class_idx)
    return class_to_idx


mapping_results = find_classes(label_directory)


def classify(model, image_transforms, image_path, mapping_results):
    model.eval()
    image = Image.open(image_path)
    image = image_transforms(image)
    # unsqueeze function will add a dimension of 1 representing a batch size of 1
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    # the print below will print out the index, i.e the internal label generated by ImageFolder
    print(list(mapping_results.keys())[list(mapping_results.values()).index(predicted)])
    # print(predicted.item())


path = r'D:/Git/Module8 Project/train/5/*.jpg'
for file in glob.glob(path):
    classify(model, image_transforms, file, mapping_results)












