import pandas

from src.data_loader import DatasetIterator, DatasetSplitter
from src import const


def test_split_dataset():
    split_dataset = DatasetSplitter(csv_path=const.TEST_CSV_ANNOTATIONS)
    assert type(split_dataset.test) is pandas.DataFrame
    assert type(split_dataset.train) is pandas.DataFrame


def test_data_loader():
    split_dataset = DatasetSplitter(csv_path=const.TEST_CSV_ANNOTATIONS)
    loader = DatasetIterator(split_dataset.get_training_set())
    assert len(loader) > 0, "Loader should not be empty"
