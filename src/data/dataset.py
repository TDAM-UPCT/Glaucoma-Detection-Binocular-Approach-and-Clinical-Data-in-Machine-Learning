"""
author: Oleksandr Kovalyk-Borodyak

Module for loading and manipulating the dataset.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from numpy.typing import ArrayLike
from scipy.io import loadmat  # type: ignore

# from xgboost import DMatrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FILE_DIR, "..", "..", "data")


class PapilaTorchSE(Dataset):
    def __init__(self, data: ArrayLike, labels: ArrayLike, transform: "transforms.Compose" = None) -> None:  # type: ignore
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)  # type: ignore

    def __getitem__(self, idx: int) -> Tuple[ArrayLike, ArrayLike]:
        if torch.is_tensor(idx):
            idx = idx.tolist()  # type: ignore

        sample = self.data[idx]  # type: ignore
        label = self.labels[idx]  # type: ignore

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class PapilaTorchBE(Dataset):
    def __init__(self, data: ArrayLike, labels: ArrayLike, transform: "transforms.Compose" = None) -> None:  # type: ignore
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)  # type: ignore

    def __getitem__(self, index) -> Any:
        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.data[index]  # type: ignore
        label = self.labels[index]  # type: ignore

        if self.transform:
            sample_0 = self.transform(sample[0, ...])  # type: ignore
            sample_1 = self.transform(sample[1, ...])  # type: ignore
            sample = torch.stack([sample_0, sample_1], dim=0)  # type: ignore
        return sample, label


class PapilaTorchTrainSE(PapilaTorchSE):
    ...


class PapilaTorchTestSE(PapilaTorchSE):
    ...


class PapilaTorchTrainBE(PapilaTorchBE):
    ...


class PapilaTorchTestBE(PapilaTorchBE):
    ...


@dataclass
class PapilaOFI(ABC):
    hpot_path: str = os.path.join(
        DATA_DIR, "mat_db", "PAPILA_10.4_HPOT.mat"
    )  # type: ignore
    kfold_path: str = os.path.join(
        DATA_DIR, "mat_db", "PAPILAv10.4.mat"
    )  # type: ignore
    train_transform: "transforms.Compose" = None  # type: ignore
    test_transform: "transforms.Compose" = None  # type: ignore
    seed: int = 2021

    def _load_data(self, data_type: str, sample_num: int):
        data = loadmat(self.kfold_path)
        X = data["X"]
        y = data["L"][0]
        pat_id = data["patID"][0]
        del data
        # eye_id = data['eyeID'][0] # 1 = right, 0 = left
        sample = sorted(
            os.listdir(os.path.join(DATA_DIR, "splits", data_type)),
            key=lambda x: int(x.split("_")[2]),
        )[sample_num]

        df = pd.read_csv(
            os.path.join(DATA_DIR, "splits", data_type, sample),
            header=[0],
            index_col=None,
        )
        sample_num = sample.split("_")[2]  # type: ignore
        seed = int(sample.split(".")[0].split("_")[-1])
        # print(f'|{sample_num}|', seed)
        for fold in range(1, 6):
            pat_ids = df.T.loc[str(fold), :].values
            mat_indexes = np.where(np.isin(pat_id, pat_ids))[0]  # type: ignore
            X_fold, y_fold = X[mat_indexes], y[mat_indexes]
            # print(X_fold.shape, y_fold.shape)
            yield X_fold, y_fold, sample_num, seed

    def _get_splits(
        self, x: ArrayLike, y: ArrayLike, seed
    ) -> tuple[ArrayLike, ArrayLike]:
        result_x = []
        result_y = []
        np.random.seed(seed)
        for i in np.arange(start=0, stop=x.shape[0], step=2):  # type: ignore
            if np.random.rand() > 0.5:
                result_x.append(x[i])  # type: ignore
                result_y.append(y[i])  # type: ignore
            else:
                result_x.append(x[i + 1])  # type: ignore
                result_y.append(y[i + 1])  # type: ignore
        return np.array(result_x), np.array(result_y)

    @abstractmethod
    def get_kfold(self) -> Generator[tuple[Any, Any], Any, None]:
        raise NotImplementedError

    @abstractmethod
    def get_hopt(self) -> Generator[tuple[Any, Any], Any, None]:
        raise NotImplementedError


@dataclass
class PapilaOFISE(PapilaOFI):
    def get_kfold(
        self, sample_num
    ) -> Generator[tuple[PapilaTorchTrainSE, PapilaTorchTestSE], Any, None]:
        train_data = self._load_data(data_type="train", sample_num=sample_num)
        test_data = self._load_data(data_type="test", sample_num=sample_num)
        for (x_train, y_train, _, seed), (x_test, y_test, _, _) in zip(
            train_data, test_data
        ):
            x_train, y_train = self._get_splits(x_train, y_train, seed)
            x_test, y_test = self._get_splits(x_test, y_test, seed)

            print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # type: ignore

            yield (
                PapilaTorchTrainSE(x_train, y_train, transform=self.train_transform),
                PapilaTorchTestSE(x_test, y_test, transform=self.test_transform),
            )

    def get_hopt(
        self,
    ) -> Generator[tuple[PapilaTorchTrainSE, PapilaTorchTestSE], Any, None]:
        data = loadmat(self.hpot_path)

        x_train = data["x_train"]
        y_train = data["y_train"][0]
        x_test = data["x_test"]
        y_test = data["y_test"][0]

        x_train, y_train = self._get_splits(x_train, y_train)  # type: ignore
        x_test, y_test = self._get_splits(x_test, y_test)  # type: ignore

        yield (
            PapilaTorchTrainSE(x_train, y_train, transform=self.train_transform),
            PapilaTorchTestSE(x_test, y_test, transform=self.test_transform),
        )


@dataclass
class PapilaOFIBE(PapilaOFI):
    def get_kfold(
        self, sample_num: int
    ) -> Generator[tuple[PapilaTorchTrainBE, PapilaTorchTestBE], Any, None]:
        train_data = self._load_data(data_type="train", sample_num=sample_num)
        test_data = self._load_data(data_type="test", sample_num=sample_num)
        for (x_train, y_train, _, _), (x_test, y_test, _, _) in zip(
            train_data, test_data
        ):
            x_train = x_train.reshape(-1, 2, *x_train.shape[1:])
            x_test = x_test.reshape(-1, 2, *x_test.shape[1:])
            y_train = y_train.reshape(-1, 2).max(axis=1)
            y_test = y_test.reshape(-1, 2).max(axis=1)

            print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

            yield (
                PapilaTorchTrainBE(x_train, y_train, transform=self.train_transform),
                PapilaTorchTestBE(x_test, y_test, transform=self.test_transform),
            )

    def get_hopt(
        self,
    ) -> Generator[tuple[PapilaTorchTrainBE, PapilaTorchTestBE], Any, None]:
        data = loadmat(self.hpot_path)

        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        x_train = x_train.reshape(-1, 2, *x_train.shape[1:])
        x_test = x_test.reshape(-1, 2, *x_test.shape[1:])
        y_train = y_train.reshape(-1, 2).max(axis=1)
        y_test = y_test.reshape(-1, 2).max(axis=1)

        yield (
            PapilaTorchTrainBE(x_train, y_train, transform=self.train_transform),
            PapilaTorchTestBE(x_test, y_test, transform=self.test_transform),
        )


############################################################################################################
# TABULAR DATASETS
############################################################################################################


@dataclass
class PapilaTabular(ABC):
    train_path: str = None  # type: ignore
    test_path: str = None  # type: ignore
    # k_train_path: str = None # type: ignore
    # k_test_path: str = None # type: ignore
    data_path: str = None  # type: ignore
    seed: int = 2021

    @abstractmethod
    def _get_splits(
        self, data: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def _load_test_train_data(self, data_type: str, sample_num: int, fold: int):
        data = pd.read_excel(
            os.path.join(DATA_DIR, self.data_path), index_col=[0], header=[0]
        )
        data.index = data.index.astype(int)
        sample = sorted(
            os.listdir(os.path.join(DATA_DIR, "splits", data_type)),
            key=lambda x: int(x.split("_")[2]),
        )[sample_num]

        df = pd.read_csv(
            os.path.join(DATA_DIR, "splits", data_type, sample),
            header=[0],
            index_col=None,
        )
        sample_num = sample.split("_")[2]  # type: ignore
        seed = int(sample.split(".")[0].split("_")[-1])

        pat_ids = df.T.loc[str(fold), :].values
        mat_indexes = np.where(np.isin(data.index, pat_ids))[0]  # type: ignore
        return data.iloc[mat_indexes, :], sample_num, seed

    def _load_data(
        self, sample: int, fold: int = 0
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        assert fold in [0, 1, 2, 3, 4, 5], "fold must be in [0, 1, 2, 3, 4, 5]"

        if fold == 0:
            clinical_test = pd.read_excel(
                os.path.join(DATA_DIR, self.test_path), index_col=[0], header=[0]
            )
            clinical_train = pd.read_excel(
                os.path.join(DATA_DIR, self.train_path), index_col=[0], header=[0]
            )

            x_train, y_train = self._get_splits(clinical_train, seed=self.seed)
            x_test, y_test = self._get_splits(clinical_test, seed=self.seed)

            return x_train, x_test, y_train, y_test

        else:
            clinical_test, _, seed = self._load_test_train_data(
                data_type="test", sample_num=sample, fold=fold
            )
            clinical_train, _, _ = self._load_test_train_data(
                data_type="train", sample_num=sample, fold=fold
            )

            x_train, y_train = self._get_splits(clinical_train, seed=seed)
            x_test, y_test = self._get_splits(clinical_test, seed=seed)

            return x_train, x_test, y_train, y_test

    def get_kfold(
        self, sample
    ) -> Generator[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Any, None]:
        for i in range(1, 6):
            x_train, x_test, y_train, y_test = self._load_data(sample=sample, fold=i)

            # dtrain = DMatrix(data=x_train, label=y_train)
            # dtest = DMatrix(data=x_test, label=y_test)

            yield x_train, x_test, y_train, y_test

    def get_hopt(
        self,
    ) -> Generator[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Any, None]:
        x_train, x_test, y_train, y_test = self._load_data(sample=-1)

        # dtrain = DMatrix(data=x_train, label=y_train)
        # dtest = DMatrix(data=x_test, label=y_test)

        yield x_train, x_test, y_train, y_test


@dataclass
class PapilaTabularSingleEye(PapilaTabular):
    def _get_splits(
        self, data: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = data.iloc[:, :-3]
        y = data.loc[:, ["tag_od", "tag_os"]]

        od_index = [i for i in x.columns if ("os" not in i)]
        os_index = [i for i in x.columns if ("od" not in i)]

        x_od = x[od_index]
        x_os = x[os_index]
        y_od = y["tag_od"]
        y_os = y["tag_os"]

        new_columns_names = [i.replace("_od", "") for i in x_od.columns]

        x_od.columns = new_columns_names
        x_os.columns = new_columns_names
        y_od.columns = ["labels"]
        y_os.columns = ["labels"]

        x = []
        y = []

        np.random.seed(seed)

        for i in range(len(x_od)):
            if np.random.rand() > 0.5:
                x.append(x_od.iloc[i].values)
                y.append(y_od.iloc[i])
            else:
                x.append(x_os.iloc[i].values)
                y.append(y_os.iloc[i])

        x = pd.DataFrame(x, columns=new_columns_names, index=x_od.index)
        y = pd.DataFrame(y, columns=["labels"], index=x_od.index)

        return x, y

    def get_hopt_v2(
        self, dataname: str
    ) -> Generator[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Any, None]:
        for i in range(10):
            train_x_y = pd.read_csv(
                os.path.join(DATA_DIR, "xgboost_hopt", f"train_{dataname}_{i}.csv"),
                index_col=0,
            )
            test_x_y = pd.read_csv(
                os.path.join(DATA_DIR, "xgboost_hopt", f"test_{dataname}_{i}.csv"),
                index_col=0,
            )
            y_test = test_x_y[["Y"]]
            y_train = train_x_y[["Y"]]

            x_train = train_x_y.drop(["Y", "2Y", "eyeID"], axis=1)
            x_test = test_x_y.drop(["Y", "2Y", "eyeID"], axis=1)

            yield x_train, x_test, y_train, y_test


@dataclass
class PapilaTabularBothEyes(PapilaTabular):
    def _get_splits(
        self, data: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = data.iloc[:, :-3]
        y = data.loc[:, ["labels"]]

        return x, y

    def get_hopt_v2(
        self, dataname: str
    ) -> Generator[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Any, None]:
        for i in range(10):
            train_x_y = pd.read_csv(
                os.path.join(DATA_DIR, "xgboost_hopt", f"train_{dataname}_{i}.csv"),
                index_col=0,
            )
            test_x_y = pd.read_csv(
                os.path.join(DATA_DIR, "xgboost_hopt", f"test_{dataname}_{i}.csv"),
                index_col=0,
            )

            def prepara_dataset(df):
                df_od = df[df.eyeID == "od"]
                df_os = df[df.eyeID == "os"]

                df_od_os = df_od.merge(
                    df_os, left_index=True, right_index=True, suffixes=("_od", "_os")
                )

                labels_od_2y = df_od_os[["2Y_od"]]
                labels_os_2y = df_od_os[["2Y_os"]]

                assert np.array_equal(labels_od_2y.values, labels_os_2y.values)

                labels = df_od_os[["2Y_od"]]
                df_od_os = df_od_os.drop(
                    [
                        "eyeID_od",
                        "eyeID_os",
                        "2Y_od",
                        "2Y_os",
                        "Y_od",
                        "Y_os",
                    ],
                    axis=1,
                )
                if dataname == "clinical":
                    df_od_os = df_od_os.drop(
                        [
                            "Age_os",
                            "Gender_os",
                        ],
                        axis=1,
                    )
                    df_od_os = df_od_os.rename(
                        {"Age_od": "Age", "Gender_od": "Gender"}, axis=1
                    )

                return df_od_os, labels

            x_train, y_train = prepara_dataset(train_x_y)
            x_test, y_test = prepara_dataset(test_x_y)

            yield x_train, x_test, y_train, y_test


@dataclass
class ClinicalSE(PapilaTabularSingleEye):
    test_path: str = os.path.join("xlsx_db", "hpot_clinical_test.xlsx")
    train_path: str = os.path.join("xlsx_db", "hpot_clinical_test.xlsx")
    # k_test_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'clinical_kfold', f'clinical_test_{i}_fold.xlsx') for i in range(1, 6)])
    # k_train_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'clinical_kfold', f'clinical_train_{i}_fold.xlsx') for i in range(1, 6)])
    data_path: str = os.path.join("xlsx_db", "clinical_data.xlsx")


@dataclass
class ClinicalBE(PapilaTabularBothEyes):
    test_path: str = os.path.join("xlsx_db", "hpot_clinical_test.xlsx")
    train_path: str = os.path.join("xlsx_db", "hpot_clinical_train.xlsx")
    data_path: str = os.path.join("xlsx_db", "clinical_data.xlsx")
    # k_test_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'clinical_kfold', f'clinical_test_{i}_fold.xlsx') for i in range(1, 6)])
    # k_train_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'clinical_kfold', f'clinical_train_{i}_fold.xlsx') for i in range(1, 6)])


@dataclass
class MorphologicalE1SE(PapilaTabularSingleEye):
    test_path: str = os.path.join("xlsx_db", "hpot_morph_exp_1_test.xlsx")
    train_path: str = os.path.join("xlsx_db", "hpot_morph_exp_1_train.xlsx")
    data_path: str = os.path.join("xlsx_db", "morph_data_exp_1.xlsx")
    # k_test_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'morph_exp_1_kfold', f'mf_exp_1_test_{i}_fold.xlsx') for i in range(1, 6)])
    # k_train_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'morph_exp_1_kfold', f'mf_exp_1_train_{i}_fold.xlsx') for i in range(1, 6)])


@dataclass
class MorphologicalE2SE(PapilaTabularSingleEye):
    test_path: str = os.path.join("xlsx_db", "hpot_morph_exp_2_test.xlsx")
    train_path: str = os.path.join("xlsx_db", "hpot_morph_exp_2_train.xlsx")
    # k_test_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'morph_exp_2_kfold', f'mf_exp_2_test_{i}_fold.xlsx') for i in range(1, 6)])
    # k_train_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'morph_exp_2_kfold', f'mf_exp_2_train_{i}_fold.xlsx') for i in range(1, 6)])
    data_path: str = os.path.join("xlsx_db", "morph_data_exp_2.xlsx")


@dataclass
class MorphologicalE1BE(PapilaTabularBothEyes):
    test_path: str = os.path.join("xlsx_db", "hpot_morph_exp_1_test.xlsx")
    train_path: str = os.path.join("xlsx_db", "hpot_morph_exp_1_train.xlsx")
    data_path: str = os.path.join("xlsx_db", "morph_data_exp_1.xlsx")
    # k_test_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'morph_exp_1_kfold', f'mf_exp_1_test_{i}_fold.xlsx') for i in range(1, 6)])
    # k_train_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'morph_exp_1_kfold', f'mf_exp_1_train_{i}_fold.xlsx') for i in range(1, 6)])


@dataclass
class MorphologicalE2BE(PapilaTabularBothEyes):
    test_path: str = os.path.join("xlsx_db", "hpot_morph_exp_2_test.xlsx")
    train_path: str = os.path.join("xlsx_db", "hpot_morph_exp_2_train.xlsx")
    # k_test_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'morph_exp_2_kfold', f'mf_exp_2_test_{i}_fold.xlsx') for i in range(1, 6)])
    # k_train_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'morph_exp_2_kfold', f'mf_exp_2_train_{i}_fold.xlsx') for i in range(1, 6)])
    data_path: str = os.path.join("xlsx_db", "morph_data_exp_2.xlsx")


############################################################################################################
# Clinica + Mprphological
############################################################################################################


@dataclass
class ClinicalPlusMorphological:
    morphological_train_path: str = os.path.join(
        "xlsx_db", "hpot_morph_exp_1_train.xlsx"
    )
    morphological_test_path: str = os.path.join("xlsx_db", "hpot_morph_exp_1_test.xlsx")
    # morphological_k_test_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'morph_exp_2_kfold', f'mf_exp_2_test_{i}_fold.xlsx') for i in range(1, 6)])
    # morphological_k_train_path: Tuple[str] =  tuple([os.path.join('xlsx_db', 'morph_exp_2_kfold', f'mf_exp_2_train_{i}_fold.xlsx') for i in range(1, 6)])

    clinical_train_path: str = os.path.join("xlsx_db", "hpot_clinical_train.xlsx")
    clinical_test_path: str = os.path.join("xlsx_db", "hpot_clinical_test.xlsx")
    # clinical_k_test_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'clinical_kfold', f'clinical_test_{i}_fold.xlsx') for i in range(1, 6)])
    # clinical_k_train_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'clinical_kfold', f'clinical_train_{i}_fold.xlsx') for i in range(1, 6)])
    clinical_kfold_path: str = os.path.join("xlsx_db", "clinical_data.xlsx")
    morphological_kfold_path: str = os.path.join("xlsx_db", "morph_data_exp_1.xlsx")

    seed: int = 2021

    def _load_test_train_data(
        self, data_type: str, sample_num: int, fold: int, data_path: str
    ):
        data = pd.read_excel(
            os.path.join(DATA_DIR, data_path), index_col=[0], header=[0]
        )
        data.index = data.index.astype(int)
        sample = sorted(
            os.listdir(os.path.join(DATA_DIR, "splits", data_type)),
            key=lambda x: int(x.split("_")[2]),
        )[sample_num]

        df = pd.read_csv(
            os.path.join(DATA_DIR, "splits", data_type, sample),
            header=[0],
            index_col=None,
        )
        sample_num = sample.split("_")[2]
        seed = int(sample.split(".")[0].split("_")[-1])

        pat_ids = df.T.loc[str(fold), :].values
        mat_indexes = np.where(np.isin(data.index, pat_ids))[0]
        return data.iloc[mat_indexes, :], sample_num, seed

    def _get_splits(
        self, data: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def _load_data(
        self, sample, fold: int = 0
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert fold in [0, 1, 2, 3, 4, 5], "fold must be in [0, 1, 2, 3, 4, 5]"

        if fold == 0:
            morphological_train = pd.read_excel(
                os.path.join(DATA_DIR, self.morphological_train_path),
                index_col=[0],
                header=[0],
            )
            morphological_test = pd.read_excel(
                os.path.join(DATA_DIR, self.morphological_test_path),
                index_col=[0],
                header=[0],
            )

            clinical_test = pd.read_excel(
                os.path.join(DATA_DIR, self.clinical_test_path),
                index_col=[0],
                header=[0],
            )
            clinical_train = pd.read_excel(
                os.path.join(DATA_DIR, self.clinical_train_path),
                index_col=[0],
                header=[0],
            )

            train = pd.concat([morphological_train, clinical_train], axis=1)
            test = pd.concat([morphological_test, clinical_test], axis=1)

            x_train, y_train = self._get_splits(train)
            x_test, y_test = self._get_splits(test)

            return x_train, x_test, y_train, y_test

        else:
            clinical_test, _, seed = self._load_test_train_data(
                data_type="test",
                sample_num=sample,
                fold=fold,
                data_path=self.clinical_kfold_path,
            )
            clinical_train, _, _ = self._load_test_train_data(
                data_type="train",
                sample_num=sample,
                fold=fold,
                data_path=self.clinical_kfold_path,
            )

            morphological_test, _, _ = self._load_test_train_data(
                data_type="test",
                sample_num=sample,
                fold=fold,
                data_path=self.morphological_kfold_path,
            )
            morphological_train, _, _ = self._load_test_train_data(
                data_type="train",
                sample_num=sample,
                fold=fold,
                data_path=self.morphological_kfold_path,
            )

            train = pd.concat([morphological_train, clinical_train], axis=1)
            test = pd.concat([morphological_test, clinical_test], axis=1)

            x_train, y_train = self._get_splits(train, seed=seed)
            x_test, y_test = self._get_splits(test, seed=seed)

            return x_train, x_test, y_train, y_test

    def get_kfold(
        self, sample: int
    ) -> Generator[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Any, None]:
        for i in range(1, 6):
            x_train, x_test, y_train, y_test = self._load_data(sample=sample, fold=i)

            # dtrain = DMatrix(data=x_train, label=y_train)
            # dtest = DMatrix(data=x_test, label=y_test)

            yield x_train, x_test, y_train, y_test

    def get_hopt(
        self,
    ) -> Generator[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Any, None]:
        x_train, x_test, y_train, y_test = self._load_data()

        # dtrain = DMatrix(data=x_train, label=y_train)
        # dtest = DMatrix(data=x_test, label=y_test)

        yield x_train, x_test, y_train, y_test


@dataclass
class ClinicalPlusMorphologicalSE(ClinicalPlusMorphological):
    def _get_splits(
        self, data: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        y = data.loc[:, ["tag_od", "tag_os"]]
        y = y.iloc[:, [1, 2]]
        x = data.loc[
            :,
            [
                i
                for i in data.columns
                if ("tag_od" != i) and ("tag_os" != i) and ("labels" != i)
            ],
        ]

        od_index = [i for i in x.columns if ("os" not in i)]
        os_index = [i for i in x.columns if ("od" not in i)]

        x_od = x[od_index]
        x_os = x[os_index]
        y_od = y["tag_od"]
        y_os = y["tag_os"]

        new_columns_names = [i.replace("_od", "") for i in x_od.columns]

        x_od.columns = new_columns_names
        x_os.columns = new_columns_names
        y_od.columns = ["labels"]
        y_os.columns = ["labels"]

        x = []
        y = []

        np.random.seed(seed)

        for i in range(len(x_od)):
            if np.random.rand() > 0.5:
                x.append(x_od.iloc[i].values)
                y.append(y_od.iloc[i])
            else:
                x.append(x_os.iloc[i].values)
                y.append(y_os.iloc[i])

        x = pd.DataFrame(x, columns=new_columns_names, index=x_od.index)
        y = pd.DataFrame(y, columns=["labels"], index=x_od.index)

        return x, y


@dataclass
class ClinicalPlusMorphologicalBE(ClinicalPlusMorphological):
    def _get_splits(
        self, data: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = data.loc[
            :,
            [
                i
                for i in data.columns
                if ("tag_od" != i) and ("tag_os" != i) and ("labels" != i)
            ],
        ]
        y = data.loc[:, ["labels"]].iloc[:, [-1]]

        return x, y


############################################################################################################
# Clinica + OFI
############################################################################################################


@dataclass
class ClinicalPlusOFI:
    data_path: str = os.path.join(DATA_DIR, "xlsx_db")
    clinical_train_path: str = os.path.join("xlsx_db", "hpot_clinical_train.xlsx")
    clinical_test_path: str = os.path.join("xlsx_db", "hpot_clinical_test.xlsx")
    # clinical_k_test_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'clinical_kfold', f'clinical_test_{i}_fold.xlsx') for i in range(1, 6)])
    # clinical_k_train_path: Tuple[str] = tuple([os.path.join('xlsx_db', 'clinical_kfold', f'clinical_train_{i}_fold.xlsx') for i in range(1, 6)])
    clinical_kfold_path: str = os.path.join("xlsx_db", "clinical_data.xlsx")
    seed: int = 2021

    def _load_test_train_data(
        self, data_type: str, sample_num: int, fold: int, data_path: str
    ):
        data = pd.read_excel(
            os.path.join(DATA_DIR, data_path), index_col=[0], header=[0]
        )
        data.index = data.index.astype(int)
        sample = sorted(
            os.listdir(os.path.join(DATA_DIR, "splits", data_type)),
            key=lambda x: int(x.split("_")[2]),
        )[sample_num]

        df = pd.read_csv(
            os.path.join(DATA_DIR, "splits", data_type, sample),
            header=[0],
            index_col=None,
        )
        sample_num = sample.split("_")[2]
        seed = int(sample.split(".")[0].split("_")[-1])

        pat_ids = df.T.loc[str(fold), :].values
        mat_indexes = np.where(np.isin(data.index, pat_ids))[0]
        return data.iloc[mat_indexes, :], sample_num, seed

    def _get_ofi_predictions(
        self,
        data_type: str,
        sample_num: int,
        fold: int,
        experiment_id: str,
        parent_run_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mlflow_tracking_uri = "http://localhost:5003"

        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9007"
        os.environ["AWS_ACCESS_KEY_ID"] = "masoud"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "Strong#Pass#2022"

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        client = MlflowClient()
        child_runs = client.search_runs(
            experiment_ids=experiment_id,
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' and params.sample = '{sample_num + 1}'",
        )
        for run in child_runs:
            with tempfile.TemporaryDirectory() as tmpdir:
                client.download_artifacts(run.info.run_id, "predictions", f"{tmpdir}")
                df = (
                    pd.read_csv(
                        f"{tmpdir}/predictions/predictions.csv", index_col=1, header=0
                    )
                    .drop(["Unnamed: 0", "true_labels"], axis=1)
                    .sort_index()
                )

                df.columns = ("OFI-H", "OFI-G")

        sample = sorted(
            os.listdir(os.path.join(DATA_DIR, "splits", data_type)),
            key=lambda x: int(x.split("_")[2]),
        )[sample_num]

        df_samples = pd.read_csv(
            os.path.join(DATA_DIR, "splits", data_type, sample),
            header=[0],
            index_col=None,
        )

        pat_ids = df_samples.T.loc[str(fold), :].values
        mat_indexes = np.where(np.isin(df.index, pat_ids))[0]
        assert (
            df.iloc[mat_indexes, :].index.tolist() == pat_ids.tolist()
        ), f"patient ids are not the same {df.iloc[mat_indexes, :].index.tolist()} != {pat_ids.tolist()}"
        return df.iloc[mat_indexes, :]

    def _get_splits(
        self, data: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def _load_data(
        self, sample: int, fold: int = 0
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert fold in [0, 1, 2, 3, 4, 5], "fold must be in [0, 1, 2, 3, 4, 5]"

        if fold == 0:
            # TODO: currently not working
            clinical_test = pd.read_excel(
                os.path.join(DATA_DIR, self.clinical_test_path),
                index_col=[0],
                header=[0],
            )
            clinical_train = pd.read_excel(
                os.path.join(DATA_DIR, self.clinical_train_path),
                index_col=[0],
                header=[0],
            )

            ofi_train, ofi_test = self._get_ofi_predictions_hopt(
                train_index=clinical_train.index.tolist(),
                test_index=clinical_test.index.tolist(),
            )

            train = pd.concat([ofi_train, clinical_train], axis=1)
            test = pd.concat([ofi_test, clinical_test], axis=1)

            x_train, y_train = self._get_splits(train, seed=self.seed)
            x_test, y_test = self._get_splits(test, seed=self.seed)

            return x_train, x_test, y_train, y_test

        else:
            clinical_test, _, seed = self._load_test_train_data(
                data_type="test",
                sample_num=sample,
                fold=fold,
                data_path=self.clinical_kfold_path,
            )
            clinical_train, _, _ = self._load_test_train_data(
                data_type="train",
                sample_num=sample,
                fold=fold,
                data_path=self.clinical_kfold_path,
            )
            # ofi_train, ofi_test = self._get_ofi_predictions(train_index=clinical_train.index.tolist(), test_index=clinical_test.index.tolist())
            ofi_test = self._get_ofi_predictions(
                data_type="test", sample_num=sample, fold=fold
            )
            ofi_train = self._get_ofi_predictions(
                data_type="train", sample_num=sample, fold=fold
            )
            train = pd.concat([ofi_train, clinical_train], axis=1)
            test = pd.concat([ofi_test, clinical_test], axis=1)

            x_train, y_train = self._get_splits(train, seed=seed)
            x_test, y_test = self._get_splits(test, seed=seed)

            return x_train, x_test, y_train, y_test

    def get_kfold(
        self, sample
    ) -> Generator[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Any, None]:
        for i in range(1, 6):
            x_train, x_test, y_train, y_test = self._load_data(fold=i, sample=sample)

            # dtrain = DMatrix(data=x_train, label=y_train)
            # dtest = DMatrix(data=x_test, label=y_test)

            yield x_train, x_test, y_train, y_test

    def get_hopt(
        self, sample=0
    ) -> Generator[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Any, None]:
        x_train, x_test, y_train, y_test = self._load_data(sample)

        yield x_train, x_test, y_train, y_test


@dataclass
class ClinicalPlusOFISE(ClinicalPlusOFI):
    def _get_ofi_predictions_hopt(
        self, train_index: List[int], test_index: List[int]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(
            f"{self.data_path}/ResNet50_predictions_single.csv", index_col=0
        )
        sample = np.random.choice(np.arange(df.shape[-1]), size=1, replace=True)
        df = df.iloc[:, sample]
        df.columns = ["ofi"]
        return df.loc[train_index, :], df.loc[test_index, :]

    def _get_splits(
        self, data: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = data.iloc[:, :-3]
        y = data.loc[:, ["tag_od", "tag_os"]]

        od_index = [i for i in x.columns if ("os" not in i)]
        os_index = [i for i in x.columns if ("od" not in i)]

        x_od = x[od_index]
        x_os = x[os_index]
        y_od = y["tag_od"]
        y_os = y["tag_os"]

        new_columns_names = [i.replace("_od", "") for i in x_od.columns]

        x_od.columns = new_columns_names
        x_os.columns = new_columns_names
        y_od.columns = ["labels"]
        y_os.columns = ["labels"]

        x = []
        y = []

        np.random.seed(seed)

        for i in range(len(x_od)):
            if np.random.rand() > 0.5:
                x.append(x_od.iloc[i].values)
                y.append(y_od.iloc[i])
            else:
                x.append(x_os.iloc[i].values)
                y.append(y_os.iloc[i])

        x = pd.DataFrame(x, columns=new_columns_names, index=x_od.index)
        y = pd.DataFrame(y, columns=["labels"], index=x_od.index)

        return x, y

    def _get_ofi_predictions(
        self, data_type: str, sample_num: int, fold: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        experiment_id, parent_run_id = ("28", "808a02da232f47e3896e4d83bf427594")
        return super()._get_ofi_predictions(
            data_type=data_type,
            sample_num=sample_num,
            fold=fold,
            experiment_id=experiment_id,
            parent_run_id=parent_run_id,
        )


@dataclass
class ClinicalPlusOFIBE(ClinicalPlusOFI):
    def _get_ofi_predictions(
        self, data_type: str, sample_num: int, fold: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        experiment_id, parent_run_id = ("30", "089dc4fd4f5d4d5cb1e322cd1dbc521a")
        return super()._get_ofi_predictions(
            data_type=data_type,
            sample_num=sample_num,
            fold=fold,
            experiment_id=experiment_id,
            parent_run_id=parent_run_id,
        )

    def _get_ofi_predictions_hopt(
        self, train_index: List[int], test_index: List[int]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(f"{self.data_path}/ResNet50_predictions_both.csv", index_col=0)
        sample = np.random.choice(np.arange(df.shape[-1]), size=1, replace=True)
        df = df.iloc[:, sample]
        df.columns = ["ofi"]
        return df.loc[train_index, :], df.loc[test_index, :]

    def _get_splits(
        self, data: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = data.iloc[:, :-3]
        y = data.loc[:, ["labels"]]

        return x, y

