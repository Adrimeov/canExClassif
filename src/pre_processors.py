import ffmpeg
import os
from pathlib import Path
import logging
import cv2

logger = logging.getLogger(__name__)
FFMPEG_LOG_LEVEL = "quiet"


def concat_videos(video_list: list[Path], output_path: Path) -> Path:
    """
    Concatenate a list of videos and removing audio.
    :param video_list: A list of absolute path to the videos to be concatenated.
    :type video_list: List[pathlib.Path]
    :param output_path: An absolute path to the desired output concatenated video.
    :type output_path: pathlib.Path

    :return: The full path of the resulting concatenated videos
    :rtype: pathlib.Path
    """
    # TODO I could use a memory stream instead of writing to disk
    # Create a temporary file list for FFmpeg concat demuxer
    list_file_name = output_path.parent / "video_list.txt"
    with open(str(list_file_name), "w") as f:
        for video_path in video_list:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File not found: {video_path}")
            f.write(f"file '{str(video_path)}'\n")

    # Run FFmpeg concat using input list
    ffmpeg.input(str(list_file_name), format="concat", safe=0).output(
        str(output_path), vcodec="copy", an=None, loglevel=FFMPEG_LOG_LEVEL
    ).run(overwrite_output=True)

    logger.info(f"Concatenated video saved to {output_path}")
    return output_path


def crop_video(
    input_path: Path,
    output_path: Path,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
) -> Path:
    """
    Crop video using FFmpeg. See https://video.stackexchange.com/questions/4563/how-can-i-crop-a-video-with-ffmpeg

    :param input_path: An absolute path to the input video file to be cropped.
    :type input_path: pathlib.Path
    :param output_path: An absolute path to the desired output cropped video.
    :type output_path: pathlib.Path
    :param crop_x: The starting x-coordinate (from the left) of the crop region.
    :type crop_x: int
    :param crop_y: The starting y-coordinate (from the top) of the crop region.
    :type crop_y: int
    :param crop_w: The width of the crop region.
    :type crop_w: int
    :param crop_h: The height of the crop region.
    :type crop_h: int

    :return: The full path of the resulting cropped video.
    :rtype: pathlib.Path
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    (
        ffmpeg.input(str(input_path))
        .output(
            str(output_path),
            vf=f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        )
        .run(overwrite_output=True)
    )

    logger.info(f"Cropped video (no audio) saved to {output_path}")
    return output_path


def reduce_fps(input_path: Path, output_path: Path, target_fps: int):
    """
    Reduce the frame rate of a video to target_fps.

    :param input_path: An absolute path to the input video file to be cropped.
    :type input_path: pathlib.Path
    :param output_path: An absolute path to the desired output cropped video.
    :type output_path: pathlib.Path
    :param target_fps: The desired fps
    :type target_fps: int

    :return: The full path of the resulting cropped video.
    :rtype: pathlib.Path

    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    (
        ffmpeg.input(str(input_path))
        .filter("fps", fps=target_fps, round="down")
        .output(str(output_path), loglevel=FFMPEG_LOG_LEVEL)
        .run(overwrite_output=True)
    )
    logger.info(f"Video saved at {target_fps} FPS: {output_path}")
    return output_path


def to_grayscale(input_path: Path, output_path: Path) -> Path:
    """
    Convert a video to grayscale using FFmpeg.
    :param input_path: An absolute path to the input video file to be converted.
    :type input_path: pathlib.Path
    :param output_path: An absolute path to the desired output grayscale video.
    :type output_path: pathlib.Path
    :return: The full path of the resulting grayscale video.
    :rtype: pathlib.Path
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    (
        ffmpeg.input(str(input_path))
        .output(str(output_path), vf="format=gray", loglevel=FFMPEG_LOG_LEVEL)
        .run(overwrite_output=True)
    )
    logger.info(f"Grayscale video saved to: {output_path}")
    return output_path


def reduce_resolution(
    input_path: Path, output_path: Path, width: int, height: int
) -> Path:
    """
    Reduce the resolution of a video using FFmpeg.

    :param input_path: Path to the input video.
    :type input_path: pathlib.Path
    :param output_path:  where the resized video will be saved.
    :type output_path: pathlib.Path
    :param width: Target width of the output video.
    :type width: int
    :param height: Target height of the output video.
    :type height: int
    :return: Path to the resized video.
    :rtype: pathlib.Path
    """

    (
        ffmpeg.input(str(input_path))
        .output(
            str(output_path), vf=f"scale={width}:{height}", loglevel=FFMPEG_LOG_LEVEL
        )
        .run(overwrite_output=True)
    )
    logger.info(f"Resolution reduced to {width}x{height}: {output_path}")
    return output_path


def extract_frames(input_path: Path, output_dir: Path, fps: int = 24) -> list[Path]:
    """
    Extract frames from a video at a specified frame rate.

    :param input_path: Path to the input video.
    :type input_path: pathlib.Path
    :param output_dir: Directory where extracted frames will be saved.
    :type output_dir: pathlib.Path
    :param fps: Desired fps extraction rate (default is 30).
    :type fps: int
    :return: List of paths to the extracted frames.
    :rtype: list[pathlib.Path]
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    (
        ffmpeg.input(str(input_path))
        .filter("fps", fps=fps)  # This is the default fps in label studio
        .output(str(output_dir / "frame_%d.png"), loglevel=FFMPEG_LOG_LEVEL)
        .run(overwrite_output=True)
    )

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths


def flip_frame_vertically(input_img_path: Path, output_img_path: Path) -> Path:
    """
    Flip image vertically using OpenCV.

    :param input_img_path: Path to the input image.
    :type input_img_path: pathlib.Path
    :param output_img_path: Path where the flipped image will be saved.
    :type output_img_path: pathlib.Path
    :return: output_img_path
    :rtype: pathlib.Path
    """
    img = cv2.imread(str(input_img_path))
    if img is None:
        raise ValueError(f"Could not read image: {input_img_path}")

    flipped_img = cv2.flip(img, 1)  # 1 → flip around vertical axis
    cv2.imwrite(str(output_img_path), flipped_img)
    logger.info(f"Flipped {input_img_path.name} on vertical axis.")

    return output_img_path
