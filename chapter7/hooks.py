import numpy as np
import torchvision
import torch.utils.data


def print_hook(module, input, output): # self, module, input, output): # noch zeigen im Debug mode: input und output
    # ##kein self
    # vertauschen - dim input 2048, dim output 1000
    print(f"Shape of input is {input[0].shape}") #np.array(input) ## [0]

model = torchvision.models.resnet50()
hook_ref = model.fc.register_forward_hook(print_hook)
model(torch.rand([1,3,224,224]))
hook_ref.remove()
model(torch.rand([1,3,224,224]))

