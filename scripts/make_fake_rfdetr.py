"""Build a tiny ONNX file with RF-DETR-Seg's I/O contract, for testing the
ORT backend without real weights. Output does not depend on image content.

Query 0: person (logit +4) box cx=.5 cy=.5 w=.25 h=.5, mask = upper box
Query 1: class 2 only (not a person)
Query 2: person with very low score (filtered)
"""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

S, Q, C, M = 64, 3, 91, 16
dets = np.array([[[.5, .5, .25, .5], [.2, .2, .1, .1], [.8, .8, .1, .1]]], np.float32)
labels = np.full((1, Q, C), -10.0, np.float32)
labels[0, 0, 1] = 4.0
labels[0, 1, 2] = 4.0
labels[0, 2, 1] = -5.0
masks = np.full((1, Q, M, M), -1.0, np.float32)
masks[0, 0, 4:12, 6:10] = 1.0  # whole person silhouette inside box (box = cells 6..10 x 4..12)

nodes, inits = [], []
inits.append(numpy_helper.from_array(np.zeros((1,), np.float32), "zero"))
nodes.append(helper.make_node("ReduceSum", ["input"], ["s"], keepdims=0))
nodes.append(helper.make_node("Mul", ["s", "zero"], ["z"]))
outs = []
for name, arr in [("dets", dets), ("labels", labels), ("masks", masks)]:
    inits.append(numpy_helper.from_array(arr, name + "_c"))
    nodes.append(helper.make_node("Add", [name + "_c", "z"], [name]))
    outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(arr.shape)))
g = helper.make_graph(nodes, "fake_rfdetr",
                      [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, S, S])],
                      outs, inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, sys.argv[1])
print("wrote", sys.argv[1])
