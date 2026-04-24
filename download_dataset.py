import kagglehub
import os
import shutil

dataset_name = "kyr7plus/emg-4"

download_path = kagglehub.dataset_download(dataset_name)

target_path = "./data"

if os.path.exists(target_path):
    shutil.rmtree(target_path)

shutil.move(download_path, target_path)

print("Ruta final de los archivos del dataset:", target_path)