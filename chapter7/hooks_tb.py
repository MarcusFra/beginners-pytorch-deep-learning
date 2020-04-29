import statistics
from functools import partial
from statistics import mean, stdev
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import torch.utils.data

writer = SummaryWriter()

def send_stats(i, module, input, output):
    writer.add_scalar(f"{i}-mean", output.data.mean())
    writer.add_scalar(f"{i}-stdev", output.data.stdev())

model = torchvision.models.resnet50()
hook_ref = model.fc.register_forward_hook(send_stats)
model(torch.rand([1,3,224,224]))
hook_ref.remove()
model(torch.rand([1,3,224,224]))