import torch
import torchvision.models

model = torchvision.models.alexnet()
model.eval()
traced_model = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
print(traced_model)
print(traced_model.code)

torch.jit.save(traced_model, "traced_model")