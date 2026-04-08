import onnx
import onnx.numpy_helper
import numpy as np
import onnxruntime as ort

model = onnx.load("mainProgram/src/AIModel/epoch40.onnx")

# Print the output tensor info
for output in model.graph.output:
    print(output)

session = ort.InferenceSession("mainProgram/src/AIModel/epoch40.onnx")
dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
outputs = session.run(None, {"images": dummy_input})
print("Min:", outputs[0].min())
print("Max:", outputs[0].max())
print("Shape:", outputs[0].shape)

confs = outputs[0][0, :, 4]  # all 300 confidence values
print("Conf min:", confs.min())
print("Conf max:", confs.max())
print("Conf > 0.5:", (confs > 0.5).sum())
print("Conf > 0.9:", (confs > 0.9).sum())