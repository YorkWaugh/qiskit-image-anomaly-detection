import numpy as np
from qiskit import QuantumCircuit


class ImageEncoder:
    def __init__(self, image_size=(8, 8)):
        self.image_size = image_size
        self.num_pixels = image_size[0] * image_size[1]
        self.num_qubits = int(np.ceil(np.log2(self.num_pixels)))

    def _normalize_image(self, image_vector):
        norm = np.linalg.norm(image_vector)
        if norm == 0:
            return np.ones(len(image_vector)) / np.sqrt(len(image_vector))
        return image_vector / norm

    def amplitude_encode(self, image_data):
        if image_data.ndim != 1:
            raise ValueError("Image data must be a 1D array (flattened).")

        required_length = 2**self.num_qubits

        if len(image_data) > required_length:
            raise ValueError(
                f"Flattened image data length ({len(image_data)}) exceeds encoding capacity ({required_length}) for {self.num_qubits} qubits."
            )

        padded_image_vector = np.zeros(required_length, dtype=float)
        padded_image_vector[: len(image_data)] = image_data

        normalized_vector = self._normalize_image(padded_image_vector)

        qc = QuantumCircuit(self.num_qubits)
        qc.prepare_state(normalized_vector, list(range(self.num_qubits)))
        return qc

    def encode_rgb(self, image_rgb):
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("RGB image must be a 3D array with 3 color channels.")

        grayscale_image = (
            0.299 * image_rgb[:, :, 0]
            + 0.587 * image_rgb[:, :, 1]
            + 0.114 * image_rgb[:, :, 2]
        )

        flattened_grayscale = grayscale_image.flatten()

        if len(flattened_grayscale) != self.num_pixels:
            raise ValueError(
                f"Input image dimensions ({grayscale_image.shape}) after grayscale conversion "
                f"do not produce the expected number of pixels ({self.num_pixels} from {self.image_size}). "
                "Ensure the input image is resized to match the encoder's configuration or vice-versa."
            )

        return self.amplitude_encode(flattened_grayscale)

    def encode_hsi(self, image_hsi):
        print(
            "Warning: HSI encoding is not fully implemented. Using a grayscale representation as a fallback."
        )
        if image_hsi.ndim == 3 and image_hsi.shape[2] == 3:
            intensity_channel = image_hsi[:, :, 2]
            flattened_intensity = intensity_channel.flatten()
            if len(flattened_intensity) != self.num_pixels:
                raise ValueError(
                    f"Input HSI image dimensions do not match encoder's expected pixel count ({self.num_pixels})."
                )
            return self.amplitude_encode(flattened_intensity)
        elif image_hsi.ndim == 2:  # If a single channel (e.g., intensity) is passed
            flattened_image = image_hsi.flatten()
            if len(flattened_image) != self.num_pixels:
                raise ValueError(
                    f"Input HSI (single channel) image dimensions do not match encoder's expected pixel count ({self.num_pixels})."
                )
            return self.amplitude_encode(flattened_image)
        else:
            raise ValueError(
                "Unsupported HSI image format for this simplified encoding."
            )

    def encode_lab(self, image_lab):
        print(
            "Warning: LAB encoding is not fully implemented. Using a grayscale representation as a fallback."
        )
        if image_lab.ndim == 3 and image_lab.shape[2] == 3:
            lightness_channel = image_lab[:, :, 0]
            flattened_lightness = lightness_channel.flatten()
            if len(flattened_lightness) != self.num_pixels:
                raise ValueError(
                    f"Input LAB image dimensions do not match encoder's expected pixel count ({self.num_pixels})."
                )
            return self.amplitude_encode(flattened_lightness)
        elif image_lab.ndim == 2:
            flattened_image = image_lab.flatten()
            if len(flattened_image) != self.num_pixels:
                raise ValueError(
                    f"Input LAB (single channel) image dimensions do not match encoder's expected pixel count ({self.num_pixels})."
                )
            return self.amplitude_encode(flattened_image)
        else:
            raise ValueError(
                "Unsupported LAB image format for this simplified encoding."
            )
