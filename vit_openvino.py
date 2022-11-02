from model_vit import ExtremelyFastViT_M5
from openvino.inference_engine import IECore
import time
import numpy as np
 
def measure_time(model, dummy_input, runtimes=200):
    times = []
    for runtime in range(runtimes):
        start = time.time()
        re = model.infer(dummy_input)
        end = time.time()
        times.append(end-start)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean*1000, std*1000

ie = IECore()
net = ie.read_network('vit_bs16.onnx')
exec_net_onnx = ie.load_network(network=net, device_name="CPU")

dummy_input = {}
for name in net.input_info:
    dummy_input[name] = np.random.random(size=net.input_info[name].tensor_desc.dims).astype(np.float32)

print(measure_time(exec_net_onnx, dummy_input))   