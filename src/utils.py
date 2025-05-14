import numpy as np  # Import numpy at the module level


def save_results(results, filename):
    """Saves a list of strings to a text file, with each string on a new line.

    Args:
        results (list[str]): A list of strings to be saved.
        filename (str): The name of the file to save the results to.
    """
    try:
        with open(filename, "w") as f:
            for result in results:
                f.write(f"{result}\n")
        print(f"Results saved to {filename}")
    except IOError as e:
        print(f"Error saving results to {filename}: {e}")


def visualize_data(image_array, title="Image"):
    """Displays a single 2D numpy array as a grayscale image using matplotlib.

    Args:
        image_array (np.ndarray): The 2D numpy array representing the image.
        title (str, optional): The title for the image plot. Defaults to "Image".
    """
    import matplotlib.pyplot as plt  # Import here to make matplotlib optional at module level

    if not isinstance(image_array, np.ndarray):
        print(f"Cannot visualize. Expected a numpy array, but got {type(image_array)}.")
        return
    if image_array.ndim != 2:
        print(
            f"Cannot visualize. Expected a 2D numpy array, but got {image_array.ndim} dimensions (shape: {image_array.shape}). Title: {title}"
        )
        return
    if image_array.size == 0:
        print(f"Cannot visualize empty image data (array size is 0). Title: {title}")
        return

    try:
        plt.imshow(image_array, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()
    except ImportError:
        print(
            "matplotlib is not installed. Cannot visualize image. Please install it: pip install matplotlib"
        )
    except Exception as e:
        print(f"Error during visualization of '{title}': {e}")
