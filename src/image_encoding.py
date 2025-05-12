import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class ImageEncoder:
    def __init__(self, image_size=(8, 8)):
        self.image_size = image_size
        self.num_pixels = image_size[0] * image_size[1]
        # Number of qubits required for amplitude encoding
        self.num_qubits = int(np.ceil(np.log2(self.num_pixels)))

    def _normalize_image(self, image_vector):
        """Normalizes the image vector so that its L2 norm is 1."""
        norm = np.linalg.norm(image_vector)
        if norm == 0:
            # Avoid division by zero for blank images.
            # Return a vector representing a uniform superposition.
            return np.ones(len(image_vector)) / np.sqrt(len(image_vector))
        return image_vector / norm

    def amplitude_encode(self, image_data):
        """
        Encodes a flattened and normalized grayscale image into a quantum state
        using amplitude encoding.
        Assumes image_data is a 1D numpy array (flattened image).
        The length of image_data can be less than 2^num_qubits, in which case it will be padded.
        """
        if image_data.ndim != 1:
            raise ValueError("Image data must be a 1D array (flattened).")

        # The state vector for amplitude encoding must have 2^num_qubits elements.
        required_length = 2**self.num_qubits

        if len(image_data) > required_length:
            raise ValueError(
                f"Flattened image data length ({len(image_data)}) exceeds encoding capacity ({required_length}) for {self.num_qubits} qubits."
            )

        # Pad with zeros if image_data is shorter than 2^num_qubits
        padded_image_vector = np.zeros(required_length, dtype=float)
        padded_image_vector[: len(image_data)] = image_data

        # Normalize for amplitude encoding
        normalized_vector = self._normalize_image(padded_image_vector)

        # Create a quantum circuit
        qc = QuantumCircuit(self.num_qubits)
        qc.prepare_state(normalized_vector, list(range(self.num_qubits)))
        return qc

    def encode_rgb(self, image_rgb):
        """
        Encodes an RGB image by first converting it to grayscale.
        Assumes image_rgb is a 3D numpy array (height, width, 3).
        The grayscale image is then encoded using amplitude encoding.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("RGB image must be a 3D array with 3 color channels.")

        # Convert RGB to grayscale using the luminosity method
        # Y = 0.299 R + 0.587 G + 0.114 B
        grayscale_image = (
            0.299 * image_rgb[:, :, 0]
            + 0.587 * image_rgb[:, :, 1]
            + 0.114 * image_rgb[:, :, 2]
        )

        # Flatten and normalize
        flattened_grayscale = grayscale_image.flatten()

        # Check if the flattened image matches the expected number of pixels for encoding
        if len(flattened_grayscale) != self.num_pixels:
            # This indicates a mismatch between the ImageEncoder's configured image_size
            # and the actual size of the image being passed.
            # For a robust implementation, resizing should happen before this step.
            raise ValueError(
                f"Input image dimensions ({grayscale_image.shape}) after grayscale conversion "
                f"do not produce the expected number of pixels ({self.num_pixels} from {self.image_size}). "
                "Ensure the input image is resized to match the encoder's configuration or vice-versa."
            )

        return self.amplitude_encode(flattened_grayscale)

    def encode_hsi(self, image_hsi):
        """Placeholder for HSI encoding. Currently falls back to grayscale.
        If a 3-channel HSI-like image is provided, uses the Intensity channel.
        If a 2D array (single channel) is provided, it's treated as grayscale.
        """
        print(
            "Warning: HSI encoding is not fully implemented. Using a grayscale representation as a fallback."
        )
        # Fallback: treat as grayscale
        if image_hsi.ndim == 3 and image_hsi.shape[2] == 3:
            # Assuming Intensity is the 3rd channel (index 2)
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
        """Placeholder for LAB encoding. Currently falls back to grayscale.
        If a 3-channel LAB-like image is provided, uses the L* (lightness) channel.
        If a 2D array (single channel) is provided, it's treated as grayscale.
        """
        print(
            "Warning: LAB encoding is not fully implemented. Using a grayscale representation as a fallback."
        )
        # Fallback: treat as grayscale
        if image_lab.ndim == 3 and image_lab.shape[2] == 3:
            # L* channel (lightness) is the 1st channel (index 0)
            lightness_channel = image_lab[:, :, 0]
            flattened_lightness = lightness_channel.flatten()
            if len(flattened_lightness) != self.num_pixels:
                raise ValueError(
                    f"Input LAB image dimensions do not match encoder's expected pixel count ({self.num_pixels})."
                )
            return self.amplitude_encode(flattened_lightness)
        elif image_lab.ndim == 2:  # If a single channel (e.g., L*) is passed
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
