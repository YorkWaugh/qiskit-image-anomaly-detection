import os
import glob
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


# https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
class MVTecDataset(Dataset):
    def __init__(self, image_paths, labels, transform_frqi=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform_frqi = transform_frqi

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise FileNotFoundError(f"Image not found or could not be read: {img_path}")

        label = self.labels[idx]

        if self.transform_frqi:
            img_transformed = self.transform_frqi(image_gray)
        else:
            # This case should ideally not happen if transform is always applied for FRQI
            img_transformed = image_gray

        return img_transformed, label


def load_mvtec_data(base_path, category, img_size_frqi):
    print(
        f"Loading MVTec dataset for category: {category} with FRQI image size: {img_size_frqi}x{img_size_frqi}"
    )
    train_image_paths = []
    train_labels = []
    test_image_paths = []
    test_labels = []

    transform_frqi = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size_frqi, img_size_frqi)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_good_path = os.path.join(base_path, category, "train", "good")
    for img_file in glob.glob(os.path.join(train_good_path, "*.png")):
        train_image_paths.append(img_file)
        train_labels.append(0)

    test_base_path = os.path.join(base_path, category, "test")

    test_good_path = os.path.join(test_base_path, "good")
    for img_file in glob.glob(os.path.join(test_good_path, "*.png")):
        test_image_paths.append(img_file)
        test_labels.append(0)

    anomaly_types = [
        d
        for d in os.listdir(test_base_path)
        if os.path.isdir(os.path.join(test_base_path, d)) and d != "good"
    ]
    for anomaly_type in anomaly_types:
        anomaly_path = os.path.join(test_base_path, anomaly_type)
        for img_file in glob.glob(os.path.join(anomaly_path, "*.png")):
            test_image_paths.append(img_file)
            test_labels.append(1)

    print(
        f"Found {len(train_image_paths)} training images and {len(test_image_paths)} testing images."
    )

    train_dataset = MVTecDataset(
        train_image_paths,
        train_labels,
        transform_frqi=transform_frqi,
    )
    test_dataset = MVTecDataset(
        test_image_paths,
        test_labels,
        transform_frqi=transform_frqi,
    )

    return train_dataset, test_dataset
