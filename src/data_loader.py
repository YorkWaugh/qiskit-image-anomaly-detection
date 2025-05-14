import os
import glob
from PIL import Image
import numpy as np
import torch
import re
import imageio


class DataLoader:
    def __init__(self, root_dir, dataset_name="UCSDped1", device=None):
        """
        Initializes the DataLoader.
        Args:
            root_dir (str): The root directory where the UCSD dataset is located.
                           Example: "./UCSD_Anomaly_Dataset.v1p2"
            dataset_name (str): "UCSDped1" or "UCSDped2".
            device (torch.device): The device to load tensors onto.
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(self.root_dir, self.dataset_name)
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        self.gt_mat_file_path = os.path.join(
            self.dataset_path, "Test", f"{self.dataset_name}.m"
        )
        self.device = device if device is not None else torch.device("cpu")

    def _parse_gt_mat_file(self):
        """
        Parses the .m file containing ground truth frame ranges for anomalous test videos.
        Returns a dictionary where keys are test folder names (e.g., "Test003")
        and values are lists of anomalous frame numbers (1-indexed).
        Returns None if parsing fails or file not found.
        """
        if not os.path.exists(self.gt_mat_file_path):
            return None

        parsed_m_file_info = {}
        try:
            with open(self.gt_mat_file_path, "r") as f:
                content = f.read()

            test_path_for_listing = os.path.join(self.dataset_path, "Test")
            all_test_folders_in_fs = sorted(
                [
                    d
                    for d in os.listdir(test_path_for_listing)
                    if os.path.isdir(os.path.join(test_path_for_listing, d))
                    and d.startswith("Test")
                    and not d.endswith("_gt")
                ]
            )

            gt_assignments = re.findall(
                r"TestVideoFile\{(\d+)\}\.gt_frame\s*=\s*\[([^\]]*)\];", content
            )

            if not gt_assignments:
                return None

            for index_str, ranges_str in gt_assignments:
                try:
                    folder_index = int(index_str)
                    folder_name = f"Test{folder_index:03d}"

                    if folder_name not in all_test_folders_in_fs:
                        continue

                    anomalous_frames_for_folder = []
                    if ranges_str.strip():
                        for part in ranges_str.split(","):
                            part = part.strip()
                            if not part:
                                continue
                            if ":" in part:
                                try:
                                    start_frame, end_frame = map(int, part.split(":"))
                                    anomalous_frames_for_folder.extend(
                                        list(range(start_frame, end_frame + 1))
                                    )
                                except ValueError:
                                    print(
                                        f"Warning: Could not parse range '{part}' in .m file for {folder_name}. Skipping range."
                                    )
                                    continue
                            else:
                                try:
                                    anomalous_frames_for_folder.append(int(part))
                                except ValueError:
                                    print(
                                        f"Warning: Could not parse frame number '{part}' in .m file for {folder_name}. Skipping frame."
                                    )
                                    continue
                    parsed_m_file_info[folder_name] = sorted(
                        list(set(anomalous_frames_for_folder))
                    )

                except ValueError:
                    print(
                        f"Warning: Could not parse TestVideoFile index '{index_str}' from .m file. Skipping entry."
                    )
                    continue

            if not parsed_m_file_info:
                return None

            return parsed_m_file_info

        except Exception as e:
            print(f"Error parsing ground truth .m file {self.gt_mat_file_path}: {e}")
            return None

    def _load_frames_from_folder(self, folder_path, image_size):
        """Loads all .tif image frames from a given folder and preprocesses them."""
        frames_np_list = []
        # Ensure folder_path is a string, as Path objects might not be directly usable by glob in all contexts
        frame_files = sorted(glob.glob(os.path.join(str(folder_path), "*.tif")))
        if not frame_files:
            return torch.empty(
                (0, *image_size), dtype=torch.float32, device=self.device
            )

        for frame_file in frame_files:
            try:
                # Use imageio.imread for TIFF files
                img_array_raw = imageio.imread(frame_file)

                # Ensure image is grayscale (L mode equivalent)
                if img_array_raw.ndim == 3 and img_array_raw.shape[2] > 1:
                    if img_array_raw.shape[2] == 3:  # RGB
                        img_array_gray = np.dot(
                            img_array_raw[..., :3], [0.2989, 0.5870, 0.1140]
                        )
                    elif img_array_raw.shape[2] == 4:  # RGBA
                        img_array_gray = np.dot(
                            img_array_raw[..., :3], [0.2989, 0.5870, 0.1140]
                        )
                    else:
                        img_array_gray = img_array_raw[..., 0]
                elif img_array_raw.ndim == 2:  # Already grayscale
                    img_array_gray = img_array_raw
                else:
                    print(
                        f"Warning: Frame {frame_file} has unexpected dimensions {img_array_raw.shape}. Skipping."
                    )
                    continue

                # Convert to PIL Image for resizing to maintain consistency with previous logic
                img_pil = Image.fromarray(img_array_gray.astype(np.uint8), mode="L")
                img_resized = img_pil.resize(
                    (image_size[1], image_size[0]), Image.Resampling.LANCZOS
                )
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                frames_np_list.append(img_array)
            except Exception as e:
                print(
                    f"Error loading or processing frame {frame_file} with imageio: {e}"
                )

        if not frames_np_list:
            return torch.empty(
                (0, *image_size), dtype=torch.float32, device=self.device
            )

        frames_tensor = torch.from_numpy(np.array(frames_np_list)).to(self.device)
        return frames_tensor

    def load_data(self, image_size=(16, 16)):
        """
        Loads training and testing frames.
        Returns: (train_frames, train_labels), (test_frames, test_labels, test_gt_info)
                 All frames and labels are PyTorch tensors. test_gt_info is a dict.
        """
        train_frames_list = []
        train_path = os.path.join(self.dataset_path, "Train")
        if os.path.exists(train_path):
            for train_folder in sorted(os.listdir(train_path)):
                folder_abs_path = os.path.join(train_path, train_folder)
                if os.path.isdir(folder_abs_path):
                    frames = self._load_frames_from_folder(folder_abs_path, image_size)
                    if frames.shape[0] > 0:
                        train_frames_list.append(frames)
        else:
            print(f"Warning: Training path not found: {train_path}")

        if not train_frames_list:
            train_frames = torch.empty(
                (0, *image_size), dtype=torch.float32, device=self.device
            )
            train_labels = torch.tensor([], dtype=torch.long, device=self.device)
        else:
            train_frames = torch.cat(train_frames_list, dim=0)
            train_labels = torch.zeros(
                len(train_frames), dtype=torch.long, device=self.device
            )

        detailed_gt_frames_from_m_file = self._parse_gt_mat_file()

        test_frames_list = []
        test_labels_list = []
        test_gt_info = {}

        test_path = os.path.join(self.dataset_path, "Test")
        if os.path.exists(test_path):
            test_folders = sorted(
                [
                    d
                    for d in os.listdir(test_path)
                    if os.path.isdir(os.path.join(test_path, d))
                    and d.startswith("Test")
                    and not d.endswith("_gt")
                ]
            )

            for test_folder in test_folders:
                folder_abs_path = os.path.join(test_path, test_folder)
                frames = self._load_frames_from_folder(folder_abs_path, image_size)
                num_frames_in_folder = len(frames)

                current_folder_labels = torch.zeros(
                    num_frames_in_folder, dtype=torch.long, device=self.device
                )
                folder_gt_data = {
                    "frames_count": num_frames_in_folder,
                    "anomalous_frames_indices": [],
                    "source_of_gt": "normal_default",
                }

                if num_frames_in_folder == 0:
                    test_gt_info[test_folder] = folder_gt_data
                    continue

                test_frames_list.append(frames)
                anomalous_frame_indices_0based = []

                if (
                    detailed_gt_frames_from_m_file
                    and test_folder in detailed_gt_frames_from_m_file
                ):
                    anomalous_frames_1based = detailed_gt_frames_from_m_file[
                        test_folder
                    ]
                    folder_gt_data["source_of_gt"] = ".m file"
                    if anomalous_frames_1based:
                        anomalous_frame_indices_0based = sorted(
                            list(
                                set(
                                    [
                                        f_idx - 1
                                        for f_idx in anomalous_frames_1based
                                        if 0 < f_idx <= num_frames_in_folder
                                    ]
                                )
                            )
                        )
                        if (
                            not anomalous_frame_indices_0based
                            and anomalous_frames_1based
                        ):
                            pass

                elif os.path.exists(os.path.join(test_path, test_folder + "_gt")):
                    anomalous_frame_indices_0based = list(range(num_frames_in_folder))
                    folder_gt_data["source_of_gt"] = "_gt folder existence (fallback)"

                if anomalous_frame_indices_0based:
                    current_folder_labels[anomalous_frame_indices_0based] = 1
                    folder_gt_data["anomalous_frames_indices"] = (
                        anomalous_frame_indices_0based
                    )

                test_labels_list.append(current_folder_labels)
                test_gt_info[test_folder] = folder_gt_data
        else:
            print(f"Warning: Test path not found: {test_path}")

        if not test_frames_list:
            test_frames = torch.empty(
                (0, *image_size), dtype=torch.float32, device=self.device
            )
            test_labels = torch.tensor([], dtype=torch.long, device=self.device)
        else:
            test_frames = torch.cat(test_frames_list, dim=0)
            test_labels = torch.cat(test_labels_list, dim=0)

        return (train_frames, train_labels), (test_frames, test_labels, test_gt_info)

    def preprocess_data(
        self,
        images,
        labels,
        num_samples=None,
        target_label=None,
        image_size=None,
        is_training_data=False,
    ):
        if images.device != self.device:
            images = images.to(self.device)
        if labels.device != self.device:
            labels = labels.to(self.device)

        if target_label is not None:
            target_indices = torch.where(labels == target_label)[0]
            if len(target_indices) == 0:
                actual_dims_for_empty = None
                if images.ndim > 1:
                    actual_dims_for_empty = images.shape[1:]
                elif image_size:
                    actual_dims_for_empty = image_size
                else:
                    actual_dims_for_empty = (1, 1)
                final_empty_images_shape = (0, *actual_dims_for_empty)
                return torch.empty(
                    final_empty_images_shape, dtype=images.dtype, device=self.device
                ), torch.tensor([], dtype=labels.dtype, device=self.device)

            filtered_images = images[target_indices]
            filtered_labels = labels[target_indices]
        else:
            filtered_images = images
            filtered_labels = labels

        if (
            num_samples is not None
            and num_samples > 0
            and len(filtered_images) > num_samples
        ):
            indices = torch.randperm(len(filtered_images), device=self.device)[
                :num_samples
            ]
            selected_images = filtered_images[indices]
            selected_labels = filtered_labels[indices]
        else:
            selected_images = filtered_images
            selected_labels = filtered_labels

        return selected_images, selected_labels
