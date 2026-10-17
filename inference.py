import onnx
import onnxruntime as ort
import numpy as np
import torch
from model import final_model
from onnx import numpy_helper
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
])
test_dataset  = datasets.CIFAR10(root="/home/cj/models/quantization_project", train=False, download=True, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = final_model(dims=96, depths=[1,2], mlp_ratio=0.6, window_size=[8,4],
                    num_classes=10, drop_rate=0.6, drop_path_rate=0.4,
                    attn_drop_rate=0.6, num_heads=[6,6]).to(device)
checkpoint = torch.load("/home/cj/models/quantization_project/ckpts_2/last.pth",
                        map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

sess = ort.InferenceSession("mamba_vision_cifar10.onnx",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ORT providers:", sess.get_providers())     
in_name = sess.get_inputs()[0].name

all_preds_pt, all_preds_ort, all_labels = [], [], []
loss_pt = loss_ort = 0.0
max_diff = 0.0
loss_fn = torch.nn.CrossEntropyLoss()
with torch.no_grad():
    for vinputs, vlabels in test_loader:
        # PyTorch on the GPU
        out_pt = model(vinputs.to(device)).cpu()                    

        
        out_ort = torch.from_numpy(
            sess.run(None, {in_name: vinputs.numpy()})[0])         
        loss_pt  += loss_fn(out_pt,  vlabels).item()              
        loss_ort += loss_fn(out_ort, vlabels).item()
        max_diff  = max(max_diff, (out_pt - out_ort).abs().max().item())

        all_preds_pt.append(out_pt.argmax(dim=1))
        all_preds_ort.append(out_ort.argmax(dim=1))
        all_labels.append(vlabels)
pt  = torch.cat(all_preds_pt).numpy()
ort_p = torch.cat(all_preds_ort).numpy()
labels = torch.cat(all_labels).numpy()
n = len(test_loader)

print(f"PyTorch: loss {loss_pt/n:.4f}  acc {accuracy_score(labels, pt)*100:.2f}%")
print(f"ORT:     loss {loss_ort/n:.4f}  acc {accuracy_score(labels, ort_p)*100:.2f}%")
print(f"max |logit diff|: {max_diff:.2e}   prediction agreement: {(pt == ort_p).mean()*100:.2f}%")