import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def show_first_frame(video_path: Path, title: str = "First Frame") -> None:
    """
    Display the first frame of a video file using OpenCV and Matplotlib.
    :param video_path:
    :type video_path: pathlib.Path
    :param title: Title for the displayed frame
    :type title: str
    :return:
    """
    cap = cv2.VideoCapture(str(video_path))
    # Read the first frame
    ret, frame = cap.read()
    cap.release()

    # Check if frame is grayscale (single channel)
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        # Already grayscale, no conversion needed
        plt.imshow(frame, cmap="gray")
    else:
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)

    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Example usage
    video_path = Path("/test/data/sampled.mp4")
    show_first_frame(video_path)
