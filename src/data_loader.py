import pandas
import pandas as pd
import torchvision
from torch.utils.data.dataset import Dataset
from pathlib import Path
from src import const
import torch
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """
    Split a dataset from a CSV file into training and test sets using sklearn's train_test_split.

    :param csv_path: path to the csv file containing the dataset annotations with columns "path" and "label".
    :type csv_path: pathlib.Path
    """

    def __init__(self, csv_path: Path):
        self.data = pd.read_csv(csv_path)
        self.train, self.test = train_test_split(
            self.data, random_state=const.SEED, test_size=0.4
        )
        # self.test, self.validation = train_test_split(self.test, random_state=const.SEED, test_size=0.5)

    def get_training_set(self) -> pd.DataFrame:
        """
        :return: Return the training set as a Pandas DataFrame.
        :rtype: pandas.DataFrame
        """
        return self.train

    def get_test_set(self):
        """
        :return: Return the test set as a Pandas DataFrame.
        :rtype: pandas.DataFrame
        """
        return self.test


class DatasetIterator(Dataset):
    """
    Implementation of a custom dataset loader that reads images and labels from dataframe columns.

    :param dataset: dataset annotations with columns "path" and "label".
    :type dataset: pandas.DataFrame
    """

    def __init__(self, dataset: pandas.DataFrame):
        self.frame_path = dataset[const.PATH_COL]
        self.labels = dataset[const.LABEL_COL]

    def __getitem__(self, index):
        label = torch.tensor(self.labels.iloc[index], dtype=torch.long)
        normalized_tensor = (
            torchvision.io.decode_image(
                self.frame_path.iloc[index], mode=torchvision.io.ImageReadMode.GRAY
            ).float()  # Add batch dimension
            / 255.0
        )
        return normalized_tensor, label

    def __len__(self):
        return len(self.labels)
