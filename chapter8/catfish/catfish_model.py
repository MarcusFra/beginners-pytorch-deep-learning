import torch.nn as nn # new statement
from torchvision import models

CatfishClasses = ["cat","fish"]

CatfishModel = models.resnet50() # ResNet50
CatfishModel.fc = nn.Sequential(nn.Linear(CatfishModel.fc.in_features,500), # transfer_model
                  nn.ReLU(),
                  nn.Dropout(), nn.Linear(500,2))

def load_catfish_model():
  return CatfishModel