import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from data_loader import DataLoader
from image_encoding import ImageEncoder
from quantum_algorithms import QuantumAnomalyDetection
from classical_components import ClassicalPostprocessor
from utils import visualize_data, save_results
from tqdm import tqdm


def main():
    print("Starting Quantum Image Anomaly Detection Pipeline for UCSD Dataset...")

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    # http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
    # Root directory of the UCSD dataset
    ucsd_data_root_dir = "./UCSD_Anomaly_Dataset.v1p2"
    # Specify UCSDped1 or UCSDped2
    ucsd_dataset_name = "UCSDped1"
    # Number of normal frames from training set for profile learning
    num_train_normal_samples = 200
    # Number of normal frames from test set for evaluation
    num_eval_test_normal_samples = 20
    # Number of anomaly frames from test set for evaluation
    num_eval_test_anomaly_samples = 20
    # Target image size for encoding
    img_size_for_encoding = (32, 32)
    num_pixels = img_size_for_encoding[0] * img_size_for_encoding[1]
    num_qubits_needed = int(np.ceil(np.log2(num_pixels)))
    shots = 4096
    result_base_dir = "./result_ucsd"

    # --- Clean up previous results directory ---
    if os.path.exists(result_base_dir):
        print(f"Cleaning up existing results directory: {result_base_dir}")
        try:
            shutil.rmtree(result_base_dir)
            print(f"Successfully removed {result_base_dir}")
        except OSError as e:
            print(f"Error removing directory {result_base_dir}: {e}")
            print("Please check permissions or close any open files in the directory.")

    # --- 1. Load and Preprocess Data ---
    print(f"\n--- Step 1: Loading and Preprocessing {ucsd_dataset_name} Data ---")

    try:
        loader = DataLoader(
            root_dir=ucsd_data_root_dir, dataset_name=ucsd_dataset_name, device=device
        )
        (x_train_orig_frames, y_train_orig_labels), (
            x_test_orig_frames,
            y_test_orig_labels,
            test_gt_info,
        ) = loader.load_data(image_size=img_size_for_encoding)
    except FileNotFoundError as e:
        print(f"Error initializing DataLoader or loading data: {e}")
        print(
            "Please ensure the UCSD dataset is correctly placed and paths are correct."
        )
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    if x_train_orig_frames.numel() == 0:
        print("No training frames were loaded. Exiting.")
        return

    print(
        f"Loading {num_train_normal_samples} normal frames from TRAINING set ({ucsd_dataset_name}) for profile learning."
    )
    profile_train_normal_images, _ = loader.preprocess_data(
        x_train_orig_frames,
        y_train_orig_labels,
        num_samples=num_train_normal_samples,
        target_label=0,
        image_size=img_size_for_encoding,
        is_training_data=True,
    )
    if profile_train_normal_images.shape[0] == 0:
        print(f"No normal frames found in training set for profile. Exiting.")
        return
    print(
        f"Profile normal frames (train set) shape: {profile_train_normal_images.shape}, Device: {profile_train_normal_images.device}"
    )
    if profile_train_normal_images.shape[0] > 0:
        pass

    print(
        f"Loading {num_eval_test_normal_samples} normal frames from TEST set ({ucsd_dataset_name}) for evaluation."
    )
    eval_test_normal_images, eval_test_normal_labels_numeric = loader.preprocess_data(
        x_test_orig_frames,
        y_test_orig_labels,
        num_samples=num_eval_test_normal_samples,
        target_label=0,
        image_size=img_size_for_encoding,
    )
    if eval_test_normal_images.shape[0] == 0:
        print(f"No normal frames found in test set for evaluation.")
    print(
        f"Evaluation normal frames (test set) shape: {eval_test_normal_images.shape}, Device: {eval_test_normal_images.device}"
    )
    if eval_test_normal_images.shape[0] > 0:
        pass

    print(
        f"Loading {num_eval_test_anomaly_samples} anomaly frames from TEST set ({ucsd_dataset_name}) for evaluation."
    )
    eval_test_anomaly_images, eval_test_anomaly_labels_numeric = loader.preprocess_data(
        x_test_orig_frames,
        y_test_orig_labels,
        num_samples=num_eval_test_anomaly_samples,
        target_label=1,
        image_size=img_size_for_encoding,
    )
    if eval_test_anomaly_images.shape[0] == 0:
        print(
            f"No anomaly frames found in test set for evaluation (or _gt folders missing/misinterpreted)."
        )
    print(
        f"Evaluation anomaly frames (test set) shape: {eval_test_anomaly_images.shape}, Device: {eval_test_anomaly_images.device}"
    )
    if eval_test_anomaly_images.shape[0] > 0:
        pass

    all_images_to_process_list = []
    ground_truth_string_labels_list = []
    ground_truth_numeric_labels_list = []

    if eval_test_normal_images.shape[0] > 0:
        all_images_to_process_list.append(eval_test_normal_images)
        ground_truth_string_labels_list.extend(
            ["normal"] * eval_test_normal_images.shape[0]
        )
        ground_truth_numeric_labels_list.extend([0] * eval_test_normal_images.shape[0])

    if eval_test_anomaly_images.shape[0] > 0:
        all_images_to_process_list.append(eval_test_anomaly_images)
        ground_truth_string_labels_list.extend(
            ["anomaly"] * eval_test_anomaly_images.shape[0]
        )
        ground_truth_numeric_labels_list.extend([1] * eval_test_anomaly_images.shape[0])

    if not all_images_to_process_list:
        print(
            "No images available for evaluation (neither normal nor anomaly selected from test set). Exiting."
        )
        return

    all_images_to_process = torch.cat(all_images_to_process_list, dim=0)
    ground_truth_string_labels = ground_truth_string_labels_list
    ground_truth_numeric_labels = torch.tensor(
        ground_truth_numeric_labels_list, dtype=torch.long, device=device
    )

    print(
        f"\nTotal images for evaluation: {all_images_to_process.shape[0]}, Device: {all_images_to_process.device}"
    )
    unique_str_labels, counts_str_labels = np.unique(
        ground_truth_string_labels, return_counts=True
    )
    print("Evaluation set composition (selected samples):")
    for label, count in zip(unique_str_labels, counts_str_labels):
        print(f"  {label}: {count}")

    # --- 2. Initialize Quantum Components ---
    print(f"\n--- Step 2: Initializing Quantum Components ---")
    encoder = ImageEncoder(image_size=img_size_for_encoding, device=device)
    qad = QuantumAnomalyDetection(num_qubits=num_qubits_needed, shots=shots)
    classical_postprocessor = ClassicalPostprocessor(random_state=42)

    # --- 3. Learn Normal Profile ---
    print(f"\n--- Step 3: Learning Normal Profile ---")
    if profile_train_normal_images.shape[0] > 0:
        print(
            f"Encoding {profile_train_normal_images.shape[0]} normal images for profile..."
        )
        profile_train_normal_images_device = profile_train_normal_images.to(device)

        encoded_profile_circuits = [
            encoder.amplitude_encode(img.flatten())
            for img in profile_train_normal_images_device
        ]
        print(
            f"Learning profile from {len(encoded_profile_circuits)} encoded circuits..."
        )
        qad.learn_normal_profile(encoded_profile_circuits)
        print("Normal profile learned.")

    else:
        print(
            "Skipping profile learning as no normal training images were loaded/processed."
        )
        pass

    if not qad.normal_profile:
        print(
            "Error: Normal profile is empty. Cannot proceed with anomaly detection. Exiting."
        )
        return

    # --- 4. Process EVALUATION Images and Detect Anomalies ---
    print("\n--- Step 4: Processing Evaluation Set and Detecting Anomalies ---")
    all_anomaly_scores = []
    detection_results_summary = []

    if len(all_images_to_process) == 0:
        print("No images in the evaluation set to process. Skipping detection.")
    else:
        for i, img_data_tensor in enumerate(all_images_to_process):
            true_label_for_print = ground_truth_string_labels[i]

            if img_data_tensor.numel() != num_pixels:
                print(
                    f"Error: Flattened evaluation image size {img_data_tensor.numel()} does not match expected {num_pixels}. Skipping."
                )
                all_anomaly_scores.append(-1.0)
                detection_results_summary.append(
                    f"Eval Image {i+1} (True: {true_label_for_print}): Error in processing, size mismatch."
                )
                continue
            try:
                img_data_tensor_device = img_data_tensor.to(device)
                encoded_qc = encoder.amplitude_encode(img_data_tensor_device.flatten())

                full_qc = qad.create_feature_map_circuit(encoded_qc)
                counts = qad.run_quantum_circuit(full_qc)
                anomaly_score = qad.analyze_results(counts)
                all_anomaly_scores.append(anomaly_score)

            except Exception as e:
                print(
                    f"Error processing evaluation image {i+1} (True: {true_label_for_print}): {e}"
                )
                all_anomaly_scores.append(-1.0)
                detection_results_summary.append(
                    f"Eval Image {i+1} (True: {true_label_for_print}): Error - {e}"
                )
                continue

    print("\n--- Fitting Classifier with Evaluation Data ---")
    valid_scores_for_fitting_indices = [
        idx for idx, score in enumerate(all_anomaly_scores) if score >= 0
    ]

    if valid_scores_for_fitting_indices:
        scores_for_fitting_list = [
            all_anomaly_scores[idx] for idx in valid_scores_for_fitting_indices
        ]
        scores_for_fitting = np.array(scores_for_fitting_list)

        labels_for_fitting_tensor = ground_truth_numeric_labels[
            torch.tensor(
                valid_scores_for_fitting_indices,
                device=ground_truth_numeric_labels.device,
            )
        ]
        labels_for_fitting = labels_for_fitting_tensor.cpu().numpy()

        if len(scores_for_fitting) > 0 and len(np.unique(labels_for_fitting)) >= 2:
            classical_postprocessor.fit(scores_for_fitting, labels_for_fitting)
        elif len(scores_for_fitting) == 0:
            print(
                "Warning: No valid scores from evaluation set to fit the classifier. Skipping fitting."
            )
        else:
            print(
                f"Warning: Not enough classes ({len(np.unique(labels_for_fitting))}) in the evaluation data (with valid scores) to fit the classifier. Unique labels found: {np.unique(labels_for_fitting)}. Skipping fitting."
            )
    else:
        print(
            "Warning: No valid scores available from evaluation set to fit the classifier. Skipping fitting."
        )

    # --- 5. Classical Post-processing ---
    print("\n--- Step 5: Classical Post-processing of Anomaly Scores ---")
    classifier_info = classical_postprocessor.get_classifier_info()
    print(
        f"Classical Postprocessor Info: Type={classifier_info['type']}, Model={classifier_info['model']}"
    )
    if classifier_info["type"] == "classifier":
        mean_val = classifier_info.get("scaler_mean", ["N/A"])
        mean_val = (
            mean_val[0]
            if isinstance(mean_val, np.ndarray) and len(mean_val) > 0
            else mean_val
        )
        scale_val = classifier_info.get("scaler_scale", ["N/A"])
        scale_val = (
            scale_val[0]
            if isinstance(scale_val, np.ndarray) and len(scale_val) > 0
            else scale_val
        )

        mean_str = (
            f"{mean_val:.4f}"
            if isinstance(mean_val, (float, np.floating))
            else str(mean_val)
        )
        scale_str = (
            f"{scale_val:.4f}"
            if isinstance(scale_val, (float, np.floating))
            else str(scale_val)
        )
        print(f"  (Scaler: Mean={mean_str}, Scale={scale_str})")
    elif classifier_info["type"] == "unfitted":
        print("  (Classifier has not been fitted)")

    os.makedirs(result_base_dir, exist_ok=True)
    detected_true_dir = os.path.join(
        result_base_dir, "detected_anomaly_true_positive_or_false_positive"
    )
    detected_false_dir = os.path.join(
        result_base_dir, "detected_anomaly_false_negative_or_true_negative"
    )
    os.makedirs(detected_true_dir, exist_ok=True)
    os.makedirs(detected_false_dir, exist_ok=True)

    valid_scores_indices = [
        idx for idx, score in enumerate(all_anomaly_scores) if score >= 0
    ]

    actual_labels_for_metric = []
    predicted_labels_for_metric = []

    if not valid_scores_indices:
        print(
            "No valid anomaly scores from evaluation set to process for classification."
        )
    else:
        valid_scores_np = np.array(
            [all_anomaly_scores[idx] for idx in valid_scores_indices]
        )

        detected_anomalies_flags = classical_postprocessor.process_scores(
            valid_scores_np
        )

        processed_score_idx = 0
        for i in range(len(all_images_to_process)):
            current_img_data_tensor = all_images_to_process[i]
            current_true_label_str = ground_truth_string_labels[i]
            current_true_label_numeric = ground_truth_numeric_labels[i].item()
            current_raw_score = all_anomaly_scores[i]

            if current_raw_score >= 0:
                is_predicted_as_anomaly = detected_anomalies_flags[processed_score_idx]

                actual_labels_for_metric.append(current_true_label_numeric)
                predicted_labels_for_metric.append(1 if is_predicted_as_anomaly else 0)

                score_str_for_filename = f"{current_raw_score:.4f}".replace(".", "p")
                filename = f"eval_img_{i+1}_true_{current_true_label_str}_pred_{'anomaly' if is_predicted_as_anomaly else 'normal'}_score_{score_str_for_filename}.png"

                save_dir = (
                    detected_true_dir if is_predicted_as_anomaly else detected_false_dir
                )
                save_image_path = os.path.join(save_dir, filename)

                try:
                    plt.imsave(
                        save_image_path,
                        current_img_data_tensor.cpu().numpy(),
                        cmap="gray",
                    )
                except Exception as e_save:
                    print(f"Error saving image {filename} to {save_dir}: {e_save}")

                result_str = f"Image {i+1} (True: {current_true_label_str}): Score={current_raw_score:.4f}, Predicted: {'Anomaly' if is_predicted_as_anomaly else 'Normal'}"
                print(result_str)
                detection_results_summary.append(result_str)
                processed_score_idx += 1
            else:
                pass

    # --- Calculate and Print Accuracy ---
    accuracy_value_str = "N/A (no successfully processed samples for metrics)"
    print(f"\n--- Evaluation Metrics ---")
    if actual_labels_for_metric:
        correct_predictions = sum(
            1
            for actual, pred in zip(
                actual_labels_for_metric, predicted_labels_for_metric
            )
            if actual == pred
        )
        total_processed_for_metric = len(actual_labels_for_metric)
        if total_processed_for_metric > 0:
            accuracy = correct_predictions / total_processed_for_metric
            accuracy_value_str = (
                f"{accuracy:.4f} ({correct_predictions}/{total_processed_for_metric})"
            )
            print(f"Accuracy on evaluated samples: {accuracy_value_str}")
        else:
            print(
                "No successfully processed samples with valid scores to calculate accuracy."
            )
    else:
        print("No successfully processed samples to calculate accuracy.")

    # --- 6. Save Results ---
    print("\n--- Step 6: Saving Results ---")
    results_filename = os.path.join(result_base_dir, "detection_summary_ucsd.txt")
    config_summary = [
        f"Configuration for {ucsd_dataset_name}:",
        f"  Dataset Root: {ucsd_data_root_dir}",
        f"  Dataset Name: {ucsd_dataset_name}",
        f"  Profile Normal Samples (from Train set): {num_train_normal_samples} (selected from {x_train_orig_frames.shape[0]} available)",
        f"  Evaluation Normal Samples (from Test set): {num_eval_test_normal_samples} (selected from {torch.sum(y_test_orig_labels == 0).item()} available)",
        f"  Evaluation Anomaly Samples (from Test set): {num_eval_test_anomaly_samples} (selected from {torch.sum(y_test_orig_labels == 1).item()} available)",
        f"  Image Size for Encoding: {img_size_for_encoding}",
        f"  Number of Qubits: {num_qubits_needed}",
        f"  Shots per Circuit: {shots}",
    ]
    classifier_info_summary = classical_postprocessor.get_classifier_info()
    classifier_log_str = f"  Classifier Info: Type={classifier_info_summary['type']}, Model={classifier_info_summary['model']}"
    if classifier_info_summary["type"] == "classifier":
        mean_val_sum = classifier_info_summary.get("scaler_mean", ["N/A"])
        mean_val_sum = (
            mean_val_sum[0]
            if isinstance(mean_val_sum, np.ndarray) and len(mean_val_sum) > 0
            else mean_val_sum
        )
        scale_val_sum = classifier_info_summary.get("scaler_scale", ["N/A"])
        scale_val_sum = (
            scale_val_sum[0]
            if isinstance(scale_val_sum, np.ndarray) and len(scale_val_sum) > 0
            else scale_val_sum
        )

        mean_str_sum = (
            f"{mean_val_sum:.4f}"
            if isinstance(mean_val_sum, (float, np.floating))
            else str(mean_val_sum)
        )
        scale_str_sum = (
            f"{scale_val_sum:.4f}"
            if isinstance(scale_val_sum, (float, np.floating))
            else str(scale_val_sum)
        )
        classifier_log_str += (
            f" (Scaler Mean={mean_str_sum}, Scaler Scale={scale_str_sum})"
        )
    elif classifier_info_summary["type"] == "unfitted":
        classifier_log_str += " (Classifier has not been fitted)"
    config_summary.append(classifier_log_str + "\n")

    config_summary.append(f"Evaluation Set Composition (Selected for Processing):")
    for label, count in zip(unique_str_labels, counts_str_labels):
        config_summary.append(f"  {label}: {count}")
    config_summary.append(f"\nAccuracy on evaluated samples: {accuracy_value_str}")
    config_summary.append("\nDetection Results:")

    save_results(config_summary + detection_results_summary, results_filename)
    print(f"Results saved to {results_filename}")

    print(
        f"\nQuantum Image Anomaly Detection Pipeline for {ucsd_dataset_name} Finished."
    )


if __name__ == "__main__":
    main()
