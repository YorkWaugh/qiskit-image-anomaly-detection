import numpy as np

# FRQI and Quantum Setup
IMG_SIZE = 16
NUM_POSITION_QUBITS = int(np.log2(IMG_SIZE * IMG_SIZE))
INTENSITY_QUBIT_INDEX = NUM_POSITION_QUBITS
NUM_ANCILLA_QUBITS = 0
if NUM_POSITION_QUBITS > 2:
    NUM_ANCILLA_QUBITS = NUM_POSITION_QUBITS - 2
TOTAL_QUBITS = NUM_POSITION_QUBITS + 1 + NUM_ANCILLA_QUBITS

# MVTec Dataset
MVTEC_BASE_PATH = "./mvtec_anomaly_detection"  # Relative to the execution directory
CATEGORY = "bottle"  # Example category

# DataLoader
BATCH_SIZE = 16

# OneClassSVM Parameters
OC_SVM_NU = 0.1
OC_SVM_KERNEL = "rbf"
OC_SVM_GAMMA = "auto"

# Results Directory
RESULTS_DIR = "result"  # Relative to the execution directory
