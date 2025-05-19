from torchvision import datasets
import numpy as np
from PIL import Image


class DataLoader:
    def __init__(self, dataset_name="MNIST", root_dir="./data"):
        self.dataset_name = dataset_name
        self.root_dir = root_dir

    def load_data(self):
        if self.dataset_name == "MNIST":
            train_dataset = datasets.MNIST(
                root=self.root_dir, train=True, download=True
            )
            test_dataset = datasets.MNIST(
                root=self.root_dir, train=False, download=True
            )

            x_train = train_dataset.data.numpy()
            y_train = train_dataset.targets.numpy()
            x_test = test_dataset.data.numpy()
            y_test = test_dataset.targets.numpy()
            return (x_train, y_train), (x_test, y_test)
        else:
            raise ValueError("Unsupported dataset. Please use 'MNIST'.")

    def preprocess_data(
        self,
        images,
        labels,
        num_samples=None,
        target_digit=None,
        image_size=(8, 8),
        normalize=True,
    ):
        if not isinstance(images, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("Images and labels must be numpy arrays.")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("Images and labels must have the same number of samples.")

        current_images = images
        current_labels = labels

        if target_digit is not None:
            indices = np.where(current_labels == target_digit)[0]
            if len(indices) == 0:
                return np.array([]), np.array([])
            current_images = current_images[indices]
            current_labels = current_labels[indices]

        if num_samples is not None and num_samples < len(current_images):
            indices = np.random.choice(len(current_images), num_samples, replace=False)
            current_images = current_images[indices]
            current_labels = current_labels[indices]

        processed_images_list = []
        for img_array in current_images:
            img = Image.fromarray(img_array.astype(np.uint8))
            resized_img = img.resize(image_size, Image.Resampling.LANCZOS)
            processed_img_array = np.array(resized_img)
            if normalize:
                processed_img_array = processed_img_array / 255.0
            processed_images_list.append(processed_img_array)

        if not processed_images_list:
            return np.array([]), np.array([])

        return (
            np.array(processed_images_list),
            current_labels,
        )
