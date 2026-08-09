import os
import json
import subprocess
import pytest
from PIL import Image

def test_cli_bridge_train_and_predict_flow(tmp_path):
    # Setup test images
    img1_path = tmp_path / "photo1.jpg"
    img2_path = tmp_path / "photo2.jpg"
    img3_path = tmp_path / "photo3.jpg"
    
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(img1_path)
    Image.new("RGB", (100, 100), color=(0, 255, 0)).save(img2_path)
    Image.new("RGB", (100, 100), color=(0, 0, 255)).save(img3_path)
    
    dir_file = tmp_path / "temp_directory.txt"
    labels_file = tmp_path / "temp_labels.json"
    files_file = tmp_path / "temp_files.json"
    
    dir_file.write_text(str(tmp_path), encoding="utf-8")
    labels_file.write_text(json.dumps({"photo1.jpg": "keep", "photo2.jpg": "trash"}), encoding="utf-8")
    files_file.write_text(json.dumps([str(img1_path), str(img2_path), str(img3_path)]), encoding="utf-8")
    
    py = os.getenv("USERPROFILE") + r"\AppData\Local\pypoetry\Cache\virtualenvs\photo-app-rBz6-pE0-py3.12\Scripts\python.exe"
    script = os.getenv("LOCALAPPDATA") + r"\darktable\lua\derush\cli_bridge.py"
    
    # 1. Test TRAIN
    cmd_train = [py, script, "train", "--directory-file", str(dir_file), "--labels-file", str(labels_file), "--files-file", str(files_file)]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True)
    assert res_train.returncode == 0, f"Train failed with stderr: {res_train.stderr}"
    
    train_data = json.loads(res_train.stdout)
    assert train_data.get("status") == "success"
    assert train_data["result"]["n_samples"] == 2
    assert train_data["result"]["n_keep"] == 1
    assert train_data["result"]["n_trash"] == 1
    
    # 2. Test PREDICT
    cmd_predict = [py, script, "predict", "--directory-file", str(dir_file), "--files-file", str(files_file)]
    res_predict = subprocess.run(cmd_predict, capture_output=True, text=True)
    assert res_predict.returncode == 0, f"Predict failed with stderr: {res_predict.stderr}"
    
    predict_data = json.loads(res_predict.stdout)
    assert predict_data.get("status") == "success"
    assert "predictions" in predict_data
    assert len(predict_data["predictions"]) > 0
