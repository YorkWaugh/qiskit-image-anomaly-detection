import numpy as np
import torch
from qiskit import QuantumCircuit


class ImageEncoder:
    def __init__(self, image_size=(8, 8), device=None):
        self.image_size = image_size
        self.num_pixels = image_size[0] * image_size[1]
        self.num_qubits = int(np.ceil(np.log2(self.num_pixels)))
        self.device = device if device is not None else torch.device("cpu")

    def _normalize_image(self, image_tensor: torch.Tensor):
        """Normalize tensor to unit L2 norm."""
        image_tensor_float64 = image_tensor.to(dtype=torch.float64)
        norm = torch.linalg.norm(image_tensor_float64)
        if norm == 0:
            num_elements = image_tensor_float64.numel()
            if num_elements == 0:
                return torch.tensor([], dtype=torch.float64, device=self.device)
            return torch.ones(
                num_elements, dtype=torch.float64, device=self.device
            ) / torch.sqrt(
                torch.tensor(num_elements, dtype=torch.float64, device=self.device)
            )
        return image_tensor_float64 / norm

    def amplitude_encode(self, image_data_tensor: torch.Tensor):
        """Amplitude-encode grayscale tensor with padding and log1p transform."""
        if not isinstance(image_data_tensor, torch.Tensor):
            raise TypeError(
                f"Input image_data must be a PyTorch tensor. Got {type(image_data_tensor)}"
            )

        image_data_tensor = image_data_tensor.to(self.device)

        if image_data_tensor.ndim == 2:
            flat_image_tensor = image_data_tensor.flatten()
        elif image_data_tensor.ndim == 1:
            flat_image_tensor = image_data_tensor
        else:
            raise ValueError(
                f"Image data tensor must be 1D or 2D. Got {image_data_tensor.ndim}D."
            )

        required_length = 2**self.num_qubits

        if flat_image_tensor.numel() > required_length:
            raise ValueError(
                f"Flattened image data length ({flat_image_tensor.numel()}) exceeds encoding capacity ({required_length}) for {self.num_qubits} qubits."
            )

        padded_image_tensor = torch.zeros(
            required_length, dtype=torch.float64, device=self.device
        )
        padded_image_tensor[: flat_image_tensor.numel()] = flat_image_tensor.to(
            dtype=torch.float64
        )

        # apply log1p transform to non-negative pixels
        processed_padded_tensor = torch.log1p(torch.clamp(padded_image_tensor, min=0.0))

        normalized_tensor = self._normalize_image(processed_padded_tensor)

        normalized_vector_np = normalized_tensor.cpu().numpy()

        qc = QuantumCircuit(self.num_qubits)
        qc.prepare_state(normalized_vector_np, list(range(self.num_qubits)))
        return qc

    def encode_rgb(self, image_rgb_tensor: torch.Tensor):
        """Convert RGB to grayscale and amplitude-encode."""
        if not isinstance(image_rgb_tensor, torch.Tensor):
            raise TypeError("RGB image must be a PyTorch tensor.")
        image_rgb_tensor = image_rgb_tensor.to(self.device)

        if image_rgb_tensor.ndim == 3:
            if image_rgb_tensor.shape[2] == 3:
                r_channel, g_channel, b_channel = (
                    image_rgb_tensor[:, :, 0],
                    image_rgb_tensor[:, :, 1],
                    image_rgb_tensor[:, :, 2],
                )
            elif image_rgb_tensor.shape[0] == 3:
                r_channel, g_channel, b_channel = (
                    image_rgb_tensor[0, :, :],
                    image_rgb_tensor[1, :, :],
                    image_rgb_tensor[2, :, :],
                )
            else:
                raise ValueError(
                    "RGB image tensor must have 3 color channels in shape (H,W,C) or (C,H,W)."
                )
        else:
            raise ValueError("RGB image tensor must be 3D.")

        grayscale_image_tensor = (
            0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
        )

        if grayscale_image_tensor.numel() != self.num_pixels:
            raise ValueError(
                f"Input image dimensions ({grayscale_image_tensor.shape}) after grayscale conversion "
                f"do not produce the expected number of pixels ({self.num_pixels} from {self.image_size}). "
                "Ensure the input image is resized to match the encoder's configuration."
            )
        return self.amplitude_encode(grayscale_image_tensor)

    def encode_hsi(self, image_hsi_tensor: torch.Tensor):
        """Encode HSI using intensity channel or grayscale."""
        print("Warning: using HSI intensity channel")
        if not isinstance(image_hsi_tensor, torch.Tensor):
            raise TypeError("HSI image must be a PyTorch tensor.")
        image_hsi_tensor = image_hsi_tensor.to(self.device)

        if image_hsi_tensor.ndim == 3 and (
            image_hsi_tensor.shape[2] == 3 or image_hsi_tensor.shape[0] == 3
        ):
            if image_hsi_tensor.shape[2] == 3:
                intensity_channel = image_hsi_tensor[:, :, 2]
            else:
                intensity_channel = image_hsi_tensor[2, :, :]
            if intensity_channel.numel() != self.num_pixels:
                raise ValueError(
                    f"HSI Intensity channel size mismatch: expected {self.num_pixels}, got {intensity_channel.numel()}"
                )
            return self.amplitude_encode(intensity_channel)
        elif image_hsi_tensor.ndim == 2:
            if image_hsi_tensor.numel() != self.num_pixels:
                raise ValueError(
                    f"HSI single channel size mismatch: expected {self.num_pixels}, got {image_hsi_tensor.numel()}"
                )
            return self.amplitude_encode(image_hsi_tensor)
        else:
            raise ValueError("Unsupported HSI tensor format.")

    def encode_lab(self, image_lab_tensor: torch.Tensor):
        """Encode LAB using L* channel or grayscale."""
        print("Warning: using LAB lightness channel")
        if not isinstance(image_lab_tensor, torch.Tensor):
            raise TypeError("LAB image must be a PyTorch tensor.")
        image_lab_tensor = image_lab_tensor.to(self.device)

        if image_lab_tensor.ndim == 3 and (
            image_lab_tensor.shape[2] == 3 or image_lab_tensor.shape[0] == 3
        ):
            if image_lab_tensor.shape[2] == 3:
                lightness_channel = image_lab_tensor[:, :, 0]
            else:
                lightness_channel = image_lab_tensor[0, :, :]
            if lightness_channel.numel() != self.num_pixels:
                raise ValueError(
                    f"LAB L* channel size mismatch: expected {self.num_pixels}, got {lightness_channel.numel()}"
                )
            return self.amplitude_encode(lightness_channel)
        elif image_lab_tensor.ndim == 2:
            if image_lab_tensor.numel() != self.num_pixels:
                raise ValueError(
                    f"LAB single channel size mismatch: expected {self.num_pixels}, got {image_lab_tensor.numel()}"
                )
            return self.amplitude_encode(image_lab_tensor)
        else:
            raise ValueError("Unsupported LAB tensor format.")


if __name__ == "__main__":
    print("Testing ImageEncoder with PyTorch...")
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using test device: {test_device}")

    encoder_2x2 = ImageEncoder(image_size=(2, 2), device=test_device)
    img_2x2_tensor = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4]], device=test_device, dtype=torch.float32
    )
    print(f"Input 2x2 tensor:\n{img_2x2_tensor}")
    try:
        qc_2x2 = encoder_2x2.amplitude_encode(img_2x2_tensor)
        print("Successfully encoded 2x2 image.")
    except Exception as e:
        print(f"Error encoding 2x2 image: {e}")
        import traceback

        traceback.print_exc()

    img_3_pixels_tensor = torch.tensor(
        [0.5, 0.5, 0.5], device=test_device, dtype=torch.float32
    )
    print(f"\nInput 3-pixel tensor (for 2x2 encoder):\n{img_3_pixels_tensor}")
    try:
        qc_3_pixels = encoder_2x2.amplitude_encode(img_3_pixels_tensor)
        print("Successfully encoded 3-pixel image with padding.")
    except Exception as e:
        print(f"Error encoding 3-pixel image: {e}")
        traceback.print_exc()

    img_blank_tensor = torch.zeros((2, 2), device=test_device, dtype=torch.float32)
    print(f"\nInput blank 2x2 tensor:\n{img_blank_tensor}")
    try:
        qc_blank = encoder_2x2.amplitude_encode(img_blank_tensor)
        print("Successfully encoded blank image.")
    except Exception as e:
        print(f"Error encoding blank image: {e}")
        traceback.print_exc()

    encoder_4x4 = ImageEncoder(image_size=(4, 4), device=test_device)
    img_4x4_tensor = torch.rand((4, 4), device=test_device, dtype=torch.float32)
    print(
        f"\nInput 4x4 random tensor (sum of elements: {torch.sum(img_4x4_tensor).item()})"
    )
    try:
        qc_4x4 = encoder_4x4.amplitude_encode(img_4x4_tensor)
        print("Successfully encoded 4x4 image.")
    except Exception as e:
        print(f"Error encoding 4x4 image: {e}")
        traceback.print_exc()

    print("\nTesting RGB encoding...")
    img_rgb_hwc_tensor = torch.rand((2, 2, 3), device=test_device, dtype=torch.float32)
    try:
        qc_rgb_hwc = encoder_2x2.encode_rgb(img_rgb_hwc_tensor)
        print("Successfully encoded RGB (H,W,C) image.")
    except Exception as e:
        print(f"Error encoding RGB (H,W,C) image: {e}")
        traceback.print_exc()

    img_rgb_chw_tensor = torch.rand((3, 2, 2), device=test_device, dtype=torch.float32)
    try:
        qc_rgb_chw = encoder_2x2.encode_rgb(img_rgb_chw_tensor)
        print("Successfully encoded RGB (C,H,W) image.")
    except Exception as e:
        print(f"Error encoding RGB (C,H,W) image: {e}")
        traceback.print_exc()
