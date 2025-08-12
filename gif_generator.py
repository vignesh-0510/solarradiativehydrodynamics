import os
import re
from PIL import Image


def create_gif_from_array(array, folder_path, file_name, duration=100):
    """
    Create and save a GIF from a numpy array of shape (H, W, T).
    Args:
        array: numpy array of shape (H, W, T)
        folder_path: folder to save the GIF
        file_name: name of the GIF file
        duration: frame duration in ms
    """
    import numpy as np
    import matplotlib.pyplot as plt

    output_gif = os.path.join(folder_path, file_name)
    images = []
    H, W, T = array.shape
    # array = array.astype(np.uint8)  # Convert to uint8 for image representation
    for t in range(T):
        img = array[:, :, t]
        # Normalize to 0-1 for colormap
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        # Apply viridis colormap
        img_color = plt.get_cmap('viridis')(img_norm)
        # Convert to uint8 RGB
        img_rgb = (img_color[:, :, :3] * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_rgb)
        images.append(img_pil)
    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to {output_gif}")
    else:
        print("No images found in array.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create GIF from image{index}.png files in a folder.")
    parser.add_argument('folder', type=str, help='Folder containing image{index}.png files')
    parser.add_argument('output', type=str, help='Output GIF file path')
    parser.add_argument('--duration', type=int, default=100, help='Frame duration in ms')
    args = parser.parse_args()
    create_gif_from_folder(args.folder, args.output, args.duration)
