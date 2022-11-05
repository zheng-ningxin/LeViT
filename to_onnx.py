import torch
from model_vit import ExtremelyFastViT_M0, ExtremelyFastViT_M5, ExtremelyFastViT_M1, ExtremelyFastViT_M2, ExtremelyFastViT_M3, ExtremelyFastViT_M4

batchsize = 2048
data = torch.rand(2048, 3, 224, 224)
net5 = ExtremelyFastViT_M5(fuse=False, pretrained=False)
net4 = ExtremelyFastViT_M4(fuse=False, pretrained=False)
net3 = ExtremelyFastViT_M3(fuse=False, pretrained=False)
net2 = ExtremelyFastViT_M2(fuse=False, pretrained=False)
net1 = ExtremelyFastViT_M1(fuse=False, pretrained=False)
net0 = ExtremelyFastViT_M0(fuse=False, pretrained=False)
torch.onnx.export(net0, data, 'net0.onnx')
torch.onnx.export(net1, data, 'net1.onnx')
torch.onnx.export(net2, data, 'net2.onnx')
torch.onnx.export(net3, data, 'net3.onnx')
torch.onnx.export(net4, data, 'net4.onnx')
torch.onnx.export(net5, data, 'net5.onnx')
