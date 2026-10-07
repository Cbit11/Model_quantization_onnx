import torch 
import torch.nn as nn 
from model import final_model
from torch.export import Dim
device = 'cpu'
model = final_model(dims = 96,depths= [1,2], mlp_ratio= 0.6, window_size= [8, 4], num_classes= 10, drop_rate= 0.6, drop_path_rate= 0.4, attn_drop_rate= 0.6, num_heads= [6,6]).to(device)
checkpoint = torch.load("/home/cj/models/quantization_project/checkpoint_432.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
dummy_input = torch.randn(2, 3, 32, 32)      # not 1
batch = Dim("batch", min=1, max=256)
torch.onnx.export(
    model,
    (dummy_input,),
    "mamba_vision_cifar10.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    dynamic_shapes={"x": {0: batch}},        # "x" must match forward(self, x)
    dynamo=True,
)