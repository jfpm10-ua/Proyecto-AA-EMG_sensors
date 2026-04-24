import kagglehub

# Download latest version
path = kagglehub.dataset_download("kyr7plus/emg-4")

print("Path to dataset files:", path)