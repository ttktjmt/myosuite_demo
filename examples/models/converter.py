from stable_baselines3 import PPO
import torch
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"{CUR_DIR = }")
filename = 'baseline' # Change this to the filename of the model you want to convert



model = PPO.load(os.path.join(CUR_DIR, filename + '.zip'), device='cpu')
print('Model loaded successfully')

# Export the policy model to ONNX
dummy_input = torch.randn(1, model.policy.observation_space.shape[0])  # Example input
torch.onnx.export(
    model.policy,
    dummy_input,
    os.path.join(CUR_DIR, filename + ".onnx"),
    export_params=True,
    opset_version=11,  # Ensure compatibility with TensorFlow.js
    input_names=["obs"],  # Input name
    output_names=["action"],  # Output name
    dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}}
)
print(f'Saved model in SavedModel format')