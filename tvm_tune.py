#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /tvm_tune.py
# \brief: 
# Author: raphael hao
import onnx
import numpy as np
import tvm
import tvm.relay as relay
from tvm import auto_scheduler
from tvm.contrib import graph_executor
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()
model = args.model
serial_onnx_model = onnx.load(f"{model}.onnx")
# fusion_onnx_model = onnx.load("fusion_thor_model.onnx")

serial_mod, serial_params = relay.frontend.from_onnx(serial_onnx_model)
# fusion_mod, fusion_params = relay.frontend.from_onnx(fusion_onnx_model)
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = f"serial_{model}_tune.log"

serial_tasks, serial_task_weights = auto_scheduler.extract_tasks(serial_mod["main"], serial_params, target)

for idx, task in enumerate(serial_tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
# import ipdb; ipdb.set_trace()
task_num = len(serial_tasks)
def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=4, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(serial_tasks, serial_task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000*task_num,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)
run_tuning()

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(serial_mod, target=target, params=serial_params)

# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
import ipdb; ipdb.set_trace()
# data_1 = tvm.nd.array((np.random.uniform(size=(1, 3, 270, 180))).astype(dtype))
# data_2 = tvm.nd.array((np.random.uniform(size=(1, 9, 1080, 720))).astype(dtype))
# module.set_input("input.1", data_1)
# module.set_input("input.21", data_2)

# # Evaluate
# print("Evaluate inference time cost...")
# print(module.benchmark(dev, repeat=3, min_repeat_ms=500))