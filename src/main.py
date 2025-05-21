import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from qiskit_aer import AerSimulator
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import from local modules
import config
from data_loader import load_mvtec_data
from quantum_utils import (
    create_frqi_circuit,
    add_feature_extraction_ansatz,
    get_quantum_features,
)


if __name__ == "__main__":
    print(
        f"Image size for FRQI: {config.IMG_SIZE}x{config.IMG_SIZE} (Total qubits: {config.TOTAL_QUBITS})"
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved in: {os.path.abspath(config.RESULTS_DIR)}")

    try:
        train_dataset, test_dataset = load_mvtec_data(
            config.MVTEC_BASE_PATH, config.CATEGORY, config.IMG_SIZE
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(
            "Please ensure the MVTec dataset is correctly placed and paths are correct."
        )
        exit()

    if (
        not train_dataset
        or not test_dataset
        or len(train_dataset) == 0
        or len(test_dataset) == 0
    ):
        print("Dataset loading failed or datasets are empty. Exiting.")
        exit()

    train_batch_size = min(config.BATCH_SIZE, len(train_dataset))
    test_batch_size = min(config.BATCH_SIZE, len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    quantum_simulator = AerSimulator()

    train_quantum_features_list = []
    print("Extracting features from training set...")
    with torch.no_grad():
        for images_frqi, _ in tqdm(
            train_loader, desc="Training Set Feature Extraction"
        ):
            images_frqi_rescaled = (images_frqi + 1.0) / 2.0
            images_frqi_flat = images_frqi_rescaled.view(images_frqi.size(0), -1)
            batch_q_features = []
            for img_idx in range(images_frqi_flat.size(0)):
                single_image_pixels = images_frqi_flat[img_idx].cpu().numpy()
                try:
                    qc_frqi = create_frqi_circuit(single_image_pixels)
                    qc_full = add_feature_extraction_ansatz(qc_frqi)
                    features = get_quantum_features(qc_full, quantum_simulator)
                    batch_q_features.append(features)
                except Exception as e:
                    print(
                        f"Error processing quantum features for training image {img_idx}: {e}"
                    )
                    batch_q_features.append(np.zeros(config.TOTAL_QUBITS))
            train_quantum_features_list.append(np.array(batch_q_features))

    if train_quantum_features_list:
        train_features_final = np.concatenate(train_quantum_features_list, axis=0)
        print(
            f"Extracted {train_features_final.shape[0]} training quantum features. Shape: {train_features_final.shape}"
        )
    else:
        print("Error: No quantum features were produced for training.")
        exit()

    test_quantum_features_list = []
    test_labels_list = []
    print("Extracting features from test set...")
    with torch.no_grad():
        for (
            images_frqi,
            labels_batch,
        ) in tqdm(test_loader, desc="Test Set Feature Extraction"):
            test_labels_list.append(labels_batch.numpy())

            images_frqi_rescaled = (images_frqi + 1.0) / 2.0
            images_frqi_flat = images_frqi_rescaled.view(images_frqi.size(0), -1)
            batch_q_features = []
            for img_idx in range(images_frqi_flat.size(0)):
                single_image_pixels = images_frqi_flat[img_idx].cpu().numpy()
                try:
                    qc_frqi = create_frqi_circuit(single_image_pixels)
                    qc_full = add_feature_extraction_ansatz(qc_frqi)
                    features = get_quantum_features(qc_full, quantum_simulator)
                    batch_q_features.append(features)
                except Exception as e:
                    print(
                        f"Error processing quantum features for test image {img_idx}: {e}"
                    )
                    batch_q_features.append(np.zeros(config.TOTAL_QUBITS))
            test_quantum_features_list.append(np.array(batch_q_features))

    test_labels = np.concatenate(test_labels_list, axis=0)

    if test_quantum_features_list:
        test_features_final = np.concatenate(test_quantum_features_list, axis=0)
        print(
            f"Extracted {test_features_final.shape[0]} test quantum features. Shape: {test_features_final.shape}"
        )
    else:
        print("Error: No quantum features were produced for testing.")
        exit()

    scaler = StandardScaler()
    train_features_final_scaled = scaler.fit_transform(train_features_final)
    test_features_final_scaled = scaler.transform(test_features_final)
    print("Quantum features scaled using StandardScaler.")

    print(
        f"Training OneClassSVM with nu={config.OC_SVM_NU}, kernel='{config.OC_SVM_KERNEL}', gamma='{config.OC_SVM_GAMMA}'..."
    )
    oc_svm = OneClassSVM(
        nu=config.OC_SVM_NU, kernel=config.OC_SVM_KERNEL, gamma=config.OC_SVM_GAMMA
    )
    oc_svm.fit(train_features_final_scaled)
    print("OneClassSVM training finished.")

    anomaly_scores = -oc_svm.decision_function(test_features_final_scaled)

    predicted_svm_labels = oc_svm.predict(test_features_final_scaled)
    mapped_predictions = np.where(predicted_svm_labels == -1, 1, 0)

    auc = roc_auc_score(test_labels, anomaly_scores)
    accuracy = accuracy_score(test_labels, mapped_predictions)
    precision = precision_score(test_labels, mapped_predictions, zero_division=0)
    recall = recall_score(test_labels, mapped_predictions, zero_division=0)
    f1 = f1_score(test_labels, mapped_predictions, zero_division=0)

    print("--- Final Evaluation on Test Set (using OneClassSVM) ---")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    training_summary = {
        "category": config.CATEGORY,
        "frqi_img_size": config.IMG_SIZE,
        "batch_size": config.BATCH_SIZE,
        "total_qubits_for_frqi": config.TOTAL_QUBITS,
        "anomaly_detection_model": "OneClassSVM",
        "oc_svm_parameters": {
            "nu": config.OC_SVM_NU,
            "kernel": config.OC_SVM_KERNEL,
            "gamma": config.OC_SVM_GAMMA,
        },
        "feature_scaler": "StandardScaler",
        "final_test_results": {
            "auc": auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }

    summary_filepath = os.path.join(
        config.RESULTS_DIR, f"{config.CATEGORY}_training_summary.json"
    )
    with open(summary_filepath, "w") as f:
        json.dump(training_summary, f, indent=4)
    print(f"Training summary saved to: {summary_filepath}")

    detailed_predictions = []

    num_samples = len(test_dataset.image_paths)
    if not (
        len(test_labels) == num_samples
        and len(mapped_predictions) == num_samples
        and len(anomaly_scores) == num_samples
    ):
        print(
            "Warning: Mismatch in lengths of test data arrays. Detailed predictions might be incomplete or misaligned."
        )
        num_samples = min(
            len(test_dataset.image_paths),
            len(test_labels),
            len(mapped_predictions),
            len(anomaly_scores),
        )

    for i in range(num_samples):
        detailed_predictions.append(
            {
                "image_path": test_dataset.image_paths[i],
                "true_label": int(test_labels[i]),
                "predicted_label": int(mapped_predictions[i]),
                "anomaly_score": float(anomaly_scores[i]),
            }
        )

    detailed_predictions_filepath = os.path.join(
        config.RESULTS_DIR, f"{config.CATEGORY}_test_predictions_detailed.json"
    )
    with open(detailed_predictions_filepath, "w") as f:
        json.dump(detailed_predictions, f, indent=4)
    print(f"Detailed test predictions saved to: {detailed_predictions_filepath}")

    print("Script finished.")
