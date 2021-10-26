import torchvision.models as models
import torch.nn as nn

'''how to use vgg models'''
vgg_model = models.vgg19_bn()
# print(vgg_model)
# print(vgg_model.classifier[-1])
num_features = vgg_model.classifier[-1].in_features
num_classes = 651
vgg_model_fc = nn.Linear(num_features, num_classes)
vgg_model.classifier[-1] = vgg_model_fc
# print(vgg_model.classifier)

'''how to use resnet models'''
resnet_model = models.resnet18()
#print(resnet_model)
num_features = resnet_model.fc.in_features
number_classes = 651
resnet_model.fc = nn.Linear(num_features, number_classes)
# print(resnet_model)

'''how to use densenet models'''
densenet121_model = models.densenet121()
# print(densenet121_model)
print('before ', densenet121_model.classifier)
num_features = densenet121_model.classifier.in_features
number_classes = 651
densenet121_model_fc = nn.Linear(num_features, number_classes)
densenet121_model.classifier = densenet121_model_fc
print('after ', densenet121_model.classifier)

'''how to use alexnet'''
alexnet_model = models.alexnet()
# print(alexnet_model)
# print(alexnet_model.classifier)
# print(alexnet_model.classifier[-1])
num_features = alexnet_model.classifier[-1].in_features
num_classes = 651
alexnet_model_fc = nn.Linear(num_features, num_classes)
alexnet_model.classifier[-1] = alexnet_model_fc
# print(alexnet_model.classifier)
# print(alexnet_model.classifier[-1])

