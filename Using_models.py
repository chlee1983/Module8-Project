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
densenet_model = models.densenet121()
# print(densenet121_model)
# print('before ', densenet_model.classifier)
num_features = densenet_model.classifier.in_features
number_classes = 651
densenet_model_fc = nn.Linear(num_features, number_classes)
densenet_model.classifier = densenet_model_fc
# print('after ', densenet_model.classifier)

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

'''how to use googlenet'''
googlenet_model = models.googlenet()

'''how to use inception_v3'''
inception_v3_model = models.inception_v3()
# print(inception_v3_model.fc)
num_features = inception_v3_model.fc.in_features
number_classes = 651
inception_v3_model_fc = nn.Linear(num_features, number_classes)
inception_v3_model.fc = inception_v3_model_fc
# print(inception_v3_model.fc)

'''how to use mnasnet'''
mnasnet_model = models.mnasnet0_75()

'''how to use mobilnet_v3_small'''
mobilnet_model_v3s = models.mobilenet_v3_small()
# print(mobilnet_model_v3s)
num_features = mobilnet_model_v3s.classifier[-1].in_features
num_classes = 651
mobilnet_model_fc = nn.Linear(num_features, num_classes)
mobilnet_model_v3s.classifier[-1] = mobilnet_model_fc

'''how to use squeezenet'''
squeeze_model = models.squeezenet1_1()
# print(squeeze_model)

'''how to use shufflenet'''
shufflenet_model = models.shufflenet_v2_x2_0()
print(shufflenet_model)
