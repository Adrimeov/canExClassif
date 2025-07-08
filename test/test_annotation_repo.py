from src import annotations
from src import const

TEST_ANNOTATIONS_FILE = const.TEST_DATA_DIR / "annotations.json"
TEST_FRAMES_DIR = const.TEST_DATA_DIR / "frames"


def test_annotation_repo():
    annotations.generate_annotations_csv(TEST_ANNOTATIONS_FILE, TEST_FRAMES_DIR)
