"""Tiny ONNX with PP-OCR det's I/O contract for tests: dynamic [1,3,H,W]
in, [1,1,H,W] out = sigmoid(mean over channels). Bright pixels read as text."""
import sys
import onnx
from onnx import helper, TensorProto

nodes = [
    helper.make_node("ReduceMean", ["x"], ["m"], keepdims=1, axes=[1]),
    helper.make_node("Sigmoid", ["m"], ["sigmoid_0.tmp_0"]),
]
g = helper.make_graph(
    nodes, "fake_ppocr",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
    [helper.make_tensor_value_info("sigmoid_0.tmp_0", TensorProto.FLOAT, [1, 1, "h", "w"])],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, sys.argv[1])
print("wrote", sys.argv[1])
