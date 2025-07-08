from src import const
from src import annotations as ann
from src import data_loader as dl
from src import model
from src import pre_processors as ppr
from tqdm import trange
import torch
import time
import json
import logging

logger = logging.getLogger(__name__)


def run_full_pipeline():
    # Raw to clean data processing.
    concat_videos = ppr.concat_videos(
        const.RAW_FILES, const.CLEAN_DATA_DIR / "concatenated.mp4"
    )
    crop_video = ppr.crop_video(
        concat_videos,
        const.CLEAN_DATA_DIR / "cropped.mp4",
        crop_x=750,
        crop_y=200,
        crop_w=1400,
        crop_h=750,
    )

    grayscale_vid = ppr.to_grayscale(
        input_path=crop_video,
        output_path=const.NORMALIZED_DATA_DIR / "grayscale_vid.mp4",
    )
    resized_vid = ppr.reduce_resolution(
        input_path=grayscale_vid,
        output_path=const.NORMALIZED_DATA_DIR / "resized_vid.mp4",
        width=const.FINAL_IMAGE_SIZE_W,
        height=const.FINAL_IMAGE_SIZE_H,
    )

    # Extract frames from the resized video and save them to the curated data directory.
    # Use const.NORMALIZED_DATA_DIR / "resized_vid.mp4" to start the pipeline from here.
    _ = ppr.extract_frames(
        input_path=resized_vid,
        output_dir=const.CURATED_DATA_DIR / "frames",
    )

    # Annotations processing.
    annotations_df, annotations_file_path = ann.generate_annotations_csv(
        const.LSTUDIO_ANNOTATIONS, const.CURATED_DATA_DIR / "frames"
    )

    # Custom dataset generator/iterator for torch model ingestion
    splitter = dl.DatasetSplitter(annotations_file_path)

    _train_iterator = dl.DatasetIterator(splitter.get_training_set())
    _valid_iterator = dl.DatasetIterator(splitter.get_test_set())
    params = {"batch_size": 128, "shuffle": True, "num_workers": 1}
    train_iterator = torch.utils.data.DataLoader(_train_iterator, **params)
    valid_iterator = torch.utils.data.DataLoader(_valid_iterator, **params)

    # Model to train.
    a_model = model.AlexNet(
        input_shape=(const.FINAL_IMAGE_SIZE_W, const.FINAL_IMAGE_SIZE_H)
    )

    EPOCHS = 5

    # Initialize result storage
    result = {}
    best_valid_loss = float("inf")
    training_losts = []
    validation_losts = []
    training_accuracies = []
    validation_accuracies = []

    # Training loop
    for epoch in trange(EPOCHS, desc="Epochs"):
        start_time = time.monotonic()

        train_loss, train_acc = a_model.train_alex(train_iterator)
        valid_loss, valid_acc = a_model.evaluate_alex(valid_iterator)

        training_losts.append(train_loss)
        validation_losts.append(valid_loss)
        training_accuracies.append(train_acc)
        validation_accuracies.append(valid_acc)

        # Model checkpoint
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(a_model.state_dict(), "canEx-integration-model.pt")

        end_time = time.monotonic()

        epoch_mins, epoch_secs = a_model.epoch_time(start_time, end_time)

        logger.info(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        logger.info(
            f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%"
        )
        logger.info(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%"
        )
        result = {
            "train_loss": training_losts,
            "valid_loss": validation_losts,
            "train_acc": training_accuracies,
            "valid_acc": validation_accuracies,
        }
        with open(const.CURATED_DATA_DIR / "result.json", "w") as f:
            json.dump(result, f)
    return result


if __name__ == "__main__":
    results = run_full_pipeline()
