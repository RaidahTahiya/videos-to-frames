import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import pandas as pd
import glob2
import os, fnmatch

from mtcnn.mtcnn import MTCNN
from skimage import measure

def extract_multiply_videos(input_filenames, image_path_infile):
    i = 1

    for input_filename in input_filenames:
        cap = cv2.VideoCapture(input_filename)

        if not cap.isOpened():
            print(f"Error opening file {input_filename}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imwrite(os.path.join(image_path_infile, f"{i}.jpg"), frame)
            cv2.imshow('Frame', frame)

            key = cv2.waitKey(30)
            if key == ord('q'):
                break

            i += 1

        cap.release()

    cv2.destroyAllWindows()


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compare_images(imageA, imageB, title="Comparison"):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    fig = plt.figure(title)
    plt.suptitle(f"MSE: {m:.2f}, SSIM: {s:.2f}")

    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    # Folder to save frames
    image_folder = Path("./frames")
    image_folder.mkdir(parents=True, exist_ok=True)

    # Get all .mp4 files from the videos folder
    video_folder = Path("./videos")
    videos = [video_folder / f for f in os.listdir(video_folder) if fnmatch.fnmatch(f, "*.mp4")]

    # Extract frames and show them
    extract_multiply_videos(videos, str(image_folder))

    # Compare two frames (grayscale)
    frame1 = cv2.imread(str(image_folder / "1.jpg"), cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(str(image_folder / "2.jpg"), cv2.IMREAD_GRAYSCALE)
    compare_images(frame1, frame2, "Frame Comparison")