# Qiskit Image Anomaly Detection

This project implements a quantum computer vision model for anomaly detection.

## Dataset

The MVTec Anomaly Detection (MVTec AD) dataset is used for training and testing the model.

- **Source:** [MVTec AD Dataset Downloads](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)

## System, Environment, and Dependencies

- **System:** Windows 11 x64
- **Environment:** CPython 3.13.3
- **Dependencies:**

  ```text
  numpy==2.2.6
  opencv-python==4.11.0.86
  qiskit==2.0.1
  qiskit-aer==0.17.0
  scikit-learn==1.6.1
  torch==2.7.0+cu128
  torchvision==0.22.0+cu128
  tqdm==4.67.1
  ```

Note: The specified system, environment, and dependencies have been tested and are known to work.

## How to Run

1. Ensure all dependencies listed in `requirements.txt` are installed in your Python environment.
2. Navigate to the project's root directory in your terminal.
3. Execute the main script using the following command:

   ```cmd
   python src/main.py
   ```
