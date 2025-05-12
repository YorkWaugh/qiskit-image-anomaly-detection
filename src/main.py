import os
import numpy as np
import matplotlib.pyplot as plt
import shutil  # Added for directory cleanup
from data_loader import DataLoader
from image_encoding import ImageEncoder
from quantum_algorithms import QuantumAnomalyDetection
from classical_components import ClassicalPostprocessor
from utils import visualize_data, save_results


def main():
    print("Starting Quantum Image Anomaly Detection Pipeline...")

    # --- Configuration ---
    mnist_data_dir = "./mnist_data"
    num_train_normal_samples = 250
    num_eval_test_normal_samples = 20
    num_eval_test_anomaly_samples = 20
    normal_digit = 7
    anomaly_digit = 1
    img_size_for_encoding = (16, 16)
    num_pixels = img_size_for_encoding[0] * img_size_for_encoding[1]
    num_qubits_needed = int(np.ceil(np.log2(num_pixels)))
    shots = 1024
    anomaly_threshold = 0.36
    result_base_dir = "./result"  # Define result_base_dir earlier for cleanup

    # --- Clean up previous results directory ---
    if os.path.exists(result_base_dir):
        print(f"Cleaning up existing results directory: {result_base_dir}")
        try:
            shutil.rmtree(result_base_dir)
            print(f"Successfully removed {result_base_dir}")
        except OSError as e:
            print(f"Error removing directory {result_base_dir}: {e}")
            print("Please check permissions or close any open files in the directory.")
            # Optionally, decide if the script should exit if cleanup fails
            # return

    # --- 1. Load and Preprocess Data ---
    print("\n--- Step 1: Loading and Preprocessing MNIST Data ---")
    if not os.path.exists(mnist_data_dir):
        os.makedirs(mnist_data_dir)

    loader = DataLoader(root_dir=mnist_data_dir)
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = loader.load_data()

    print(
        f"Loading {num_train_normal_samples} normal images (digit '{normal_digit}') from TRAINING set for profile learning."
    )
    profile_train_normal_images, _ = loader.preprocess_data(
        x_train_orig,
        y_train_orig,
        num_samples=num_train_normal_samples,
        target_digit=normal_digit,
        image_size=img_size_for_encoding,
    )
    if profile_train_normal_images.size == 0:
        print(
            f"No images found for normal digit {normal_digit} in training set for profile. Exiting."
        )
        return
    print(
        f"Profile normal images (train set) shape: {profile_train_normal_images.shape}"
    )
    if profile_train_normal_images.shape[0] > 0:
        visualize_data(
            profile_train_normal_images[0],
            title=f"Example Profile Normal Image (Digit {normal_digit}, Train Set, {img_size_for_encoding})",
        )

    print(
        f"Loading {num_eval_test_normal_samples} normal images (digit '{normal_digit}') from TEST set for evaluation."
    )
    eval_test_normal_images, _ = loader.preprocess_data(
        x_test_orig,
        y_test_orig,
        num_samples=num_eval_test_normal_samples,
        target_digit=normal_digit,
        image_size=img_size_for_encoding,
    )
    if eval_test_normal_images.size == 0:
        print(
            f"No images found for normal digit {normal_digit} in test set for evaluation."
        )
    print(f"Evaluation normal images (test set) shape: {eval_test_normal_images.shape}")
    if eval_test_normal_images.shape[0] > 0:
        visualize_data(
            eval_test_normal_images[0],
            title=f"Example Evaluation Normal Image (Digit {normal_digit}, Test Set, {img_size_for_encoding})",
        )

    print(
        f"Loading {num_eval_test_anomaly_samples} anomaly images (digit '{anomaly_digit}') from TEST set for evaluation."
    )
    eval_test_anomaly_images, _ = loader.preprocess_data(
        x_test_orig,
        y_test_orig,
        num_samples=num_eval_test_anomaly_samples,
        target_digit=anomaly_digit,
        image_size=img_size_for_encoding,
    )
    if eval_test_anomaly_images.size == 0:
        print(
            f"No images found for anomaly digit {anomaly_digit} in test set for evaluation."
        )
    print(
        f"Evaluation anomaly images (test set) shape: {eval_test_anomaly_images.shape}"
    )
    if eval_test_anomaly_images.shape[0] > 0:
        visualize_data(
            eval_test_anomaly_images[0],
            title=f"Example Evaluation Anomaly Image (Digit {anomaly_digit}, Test Set, {img_size_for_encoding})",
        )

    all_images_to_process_list = []
    eval_labels_list = []

    if eval_test_normal_images.shape[0] > 0:
        all_images_to_process_list.append(eval_test_normal_images)
        eval_labels_list.extend(
            [f"test_normal_digit{normal_digit}"] * len(eval_test_normal_images)
        )

    if eval_test_anomaly_images.shape[0] > 0:
        all_images_to_process_list.append(eval_test_anomaly_images)
        eval_labels_list.extend(
            [f"test_anomaly_digit{anomaly_digit}"] * len(eval_test_anomaly_images)
        )

    if not all_images_to_process_list:
        print("No images available for evaluation. Exiting.")
        return

    all_images_to_process = np.concatenate(all_images_to_process_list, axis=0)
    ground_truth_labels = eval_labels_list
    print(f"\nTotal images for evaluation: {len(all_images_to_process)}")
    unique_labels, counts = np.unique(ground_truth_labels, return_counts=True)
    print("Evaluation set composition:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")

    # --- 2. Initialize Encoders and Quantum Components ---
    print(
        f"\n--- Step 2: Initializing Quantum Components ({num_qubits_needed} qubits) ---"
    )
    image_encoder = ImageEncoder(image_size=img_size_for_encoding)
    q_anomaly_detector = QuantumAnomalyDetection(
        num_qubits=num_qubits_needed, shots=shots
    )
    classical_postprocessor = ClassicalPostprocessor()

    # --- 2.5 Learn Normal Profile ---
    print("\n--- Step 2.5: Learning Normal Profile ---")
    encoded_profile_circuits = []
    if profile_train_normal_images.shape[0] > 0:
        for i, img_data in enumerate(profile_train_normal_images):
            flattened_img = img_data.flatten()
            if flattened_img.size != num_pixels:
                print(
                    f"Error: Flattened profile normal image size {flattened_img.size} does not match expected {num_pixels}. Skipping for profile."
                )
                continue
            try:
                encoded_qc = image_encoder.amplitude_encode(flattened_img)
                encoded_profile_circuits.append(encoded_qc)
            except Exception as e:
                print(
                    f"Error encoding profile normal image {i+1} for profile learning: {e}"
                )

        if encoded_profile_circuits:
            q_anomaly_detector.learn_normal_profile(encoded_profile_circuits)
            print(
                f"Normal profile learned using {len(encoded_profile_circuits)} training images."
            )
        else:
            print(
                "No profile normal images were successfully encoded. Skipping normal profile learning. Anomaly detection will likely fail or be inaccurate."
            )
            q_anomaly_detector.normal_profile = {}
    else:
        print(
            "No profile normal images available to learn profile. Anomaly detection might be less effective or fail."
        )
        q_anomaly_detector.normal_profile = {}

    if not q_anomaly_detector.normal_profile:
        print(
            "Error: Normal profile is empty. Cannot proceed with anomaly detection. Exiting."
        )
        return

    # --- 3. Process EVALUATION Images and Detect Anomalies ---
    print("\n--- Step 3: Processing Evaluation Set and Detecting Anomalies ---")
    all_anomaly_scores = []
    detection_results_summary = []

    if len(all_images_to_process) == 0:
        print("No images in the evaluation set to process. Skipping detection.")
    else:
        for i, img_data in enumerate(all_images_to_process):
            true_label_for_print = ground_truth_labels[i]

            flattened_img = img_data.flatten()
            if flattened_img.size != num_pixels:
                print(
                    f"Error: Flattened evaluation image size {flattened_img.size} does not match expected {num_pixels}. Skipping."
                )
                all_anomaly_scores.append(-1.0)
                detection_results_summary.append(
                    f"Eval Image {i+1} ({true_label_for_print}): Error in processing, size mismatch."
                )
                continue

            try:
                encoded_qc = image_encoder.amplitude_encode(flattened_img)
                full_qc = q_anomaly_detector.create_feature_map_circuit(
                    encoded_qc.copy()
                )
                counts = q_anomaly_detector.run_quantum_circuit(full_qc)
                anomaly_score = q_anomaly_detector.analyze_results(counts)
                all_anomaly_scores.append(anomaly_score)

            except Exception as e:
                print(f"Error processing evaluation image {i+1}: {e}")
                all_anomaly_scores.append(-1.0)
                detection_results_summary.append(
                    f"Eval Image {i+1} ({true_label_for_print}): Error - {e}"
                )
                continue

    # --- 4. Classical Post-processing ---
    print("\n--- Step 4: Classical Post-processing of Anomaly Scores ---")
    print(f"Anomaly Threshold: {anomaly_threshold}")

    os.makedirs(result_base_dir, exist_ok=True)
    detected_true_dir = os.path.join(result_base_dir, "detected_anomaly_true")
    detected_false_dir = os.path.join(result_base_dir, "detected_anomaly_false")
    os.makedirs(detected_true_dir, exist_ok=True)
    os.makedirs(detected_false_dir, exist_ok=True)

    valid_scores_for_classification = [
        score for score in all_anomaly_scores if score >= 0
    ]

    actual_labels_for_metric = []
    predicted_labels_for_metric = []

    if not valid_scores_for_classification:
        print(
            "No valid anomaly scores from evaluation set to process for classification."
        )
    else:
        detected_anomalies_flags = classical_postprocessor.process_scores(
            np.array(valid_scores_for_classification), threshold=anomaly_threshold
        )

        processed_score_idx = 0
        for i in range(len(all_images_to_process)):
            current_img_data = all_images_to_process[i]
            current_true_label_str = ground_truth_labels[i]
            current_raw_score = all_anomaly_scores[i]

            if current_raw_score >= 0:
                is_predicted_as_anomaly = detected_anomalies_flags[processed_score_idx]

                actual_labels_for_metric.append(
                    1 if "anomaly" in current_true_label_str else 0
                )
                predicted_labels_for_metric.append(1 if is_predicted_as_anomaly else 0)

                score_str_for_filename = f"{current_raw_score:.4f}".replace(".", "p")
                filename = f"eval_img_{i+1}_true_{current_true_label_str}_pred_{'anomaly' if is_predicted_as_anomaly else 'normal'}_score_{score_str_for_filename}.png"

                save_dir = (
                    detected_true_dir if is_predicted_as_anomaly else detected_false_dir
                )
                save_image_path = os.path.join(save_dir, filename)

                try:
                    plt.imsave(save_image_path, current_img_data, cmap="gray")
                except Exception as e_save:
                    print(f"Error saving image {filename} to {save_dir}: {e_save}")

                result_str = f"Image {i+1} (True: {current_true_label_str}): Score={current_raw_score:.4f}, Predicted: {'Anomaly' if is_predicted_as_anomaly else 'Normal'}"
                print(result_str)
                detection_results_summary.append(result_str)
                processed_score_idx += 1

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

    # --- 5. Save Results ---
    print("\n--- Step 5: Saving Results ---")
    results_filename = "./detection_summary.txt"
    config_summary = [
        f"Configuration:",
        f"  Normal Digit for Profile (Train): {normal_digit}, Anomaly Digit for Eval: {anomaly_digit}",
        f"  Profile Normal Samples (Train): {num_train_normal_samples}",
        f"  Evaluation Normal Samples (Test): {num_eval_test_normal_samples}",
        f"  Evaluation Anomaly Samples (Test): {num_eval_test_anomaly_samples}",
        f"  Image Size for Encoding: {img_size_for_encoding}",
        f"  Number of Qubits: {num_qubits_needed}",
        f"  Shots per Circuit: {shots}",
        f"  Anomaly Threshold: {anomaly_threshold}\n",
        f"Evaluation Set Composition:",
    ]
    for label, count in zip(unique_labels, counts):
        config_summary.append(f"  {label}: {count}")
    config_summary.append(f"\nAccuracy on evaluated samples: {accuracy_value_str}")
    config_summary.append("\nDetection Results:")

    save_results(config_summary + detection_results_summary, results_filename)
    print(f"Results saved to {results_filename}")

    print("\nQuantum Image Anomaly Detection Pipeline Finished.")


if __name__ == "__main__":
    main()
