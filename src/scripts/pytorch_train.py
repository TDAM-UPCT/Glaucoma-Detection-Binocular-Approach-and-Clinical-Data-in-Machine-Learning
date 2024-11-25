import os
import sys
import tempfile
from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import ArrayLike
from ray import air, tune
from ray.air import session
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from scipy.io import loadmat
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torchvision import transforms
from torchvision.models import VisionTransformer

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(SRC_DIR, "data")
sys.path.append(SRC_DIR)

from src.data.dataset import PapilaOFI


class ModelOPS:
    def __init__(self, model, device='cpu') -> None:
        self.model = model
        self.device = device

    def metrics(self, y_true: ArrayLike, y_pred: ArrayLike, threshold: float = 0.5) -> Tuple[float, float, float]:


        # _, _, thresholds = roc_curve(y_true, y_pred)
        # weights = np.array([0.48702595, 1.89147287])
        # sample_weights = [weights[0] if i == 0 else weights[1] for i in y_true]
        m_roc_auc = roc_auc_score(y_true, y_pred) # sample_weight=sample_weights, average="weighted")

        y_pred = [1 if i > threshold else 0 for i in y_pred]

        m_precision = precision_score(y_true, y_pred, average="binary")
        m_recall = recall_score(y_true, y_pred, average="binary")

        return m_precision, m_recall, m_roc_auc

    def _train_epoch(self, model, data_loader, optimizer, criterion):
        print("Training...")
        model.train()
        running_loss = 0.0
        y_true = []
        y_pred = []

        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            outputs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs[:, 1])

        train_loss = running_loss / len(data_loader)

        m_precision, m_recall, m_roc_auc = self.metrics(y_true, y_pred)

        return train_loss, m_precision, m_recall, m_roc_auc

    def _validate_epoch(self, model, data_loader, criterion):
        print("Validating...")
        model.eval()
        val_running_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in data_loader:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                # Calculate loss
                loss = criterion(outputs, labels.long())
                val_running_loss += loss.item()
                outputs = F.softmax(outputs, dim=1).cpu().numpy()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs[:, 1])

        m_precision, m_recall, m_roc_auc = self.metrics(y_true, y_pred)
        val_running_loss = val_running_loss / len(data_loader)

        return val_running_loss, m_precision, m_recall, m_roc_auc

    def inference(self, model, data_loader):

        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in data_loader:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Calculate loss
                outputs = model(inputs)
                # outputs = F.softmax(outputs, dim=1).cpu().numpy()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())


        # m_precision, m_recall, m_roc_auc = self.metrics(y_true, y_pred)

        return y_true, y_pred

    def fit(self, config, train_dataset, val_dataset, hopt: bool = True, debug: bool = False):

        train_loader = torch.utils.data.DataLoader(train_dataset, # type: ignore
                                                   batch_size=config["batch_size"],
                                                   shuffle=True,
                                                   num_workers=2)
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(val_dataset, # type: ignore
                                                    batch_size=config["batch_size"],
                                                    shuffle=False,
                                                    num_workers=2)


        model = self.model(num_classes=config["num_classes"],
                                          pretrained=config["pretrained"],
                                          hidden_units=config["hidden_units"],
                                          hidden_layers=config["hidden_layers"],
                                          dropout=config["dropout"],
                                          freeze_layers=config["freeze_layers"])

        if torch.cuda.is_available():
            self.device = "cuda:0"
            # if torch.cuda.device_count() > 1:
            #     model = nn.DataParallel(model)

        model.to(self.device)

        class_weights = torch.tensor(config["class_weights"]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config["label_smoothing"])
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        # Start training
        for epoch in range(config["epochs"]):

            print(f"[INFO]: Epoch {epoch + 1} of {config['epochs']}")

            # TRAIN LOOP
            train_epoch_loss, train_precision, train_recall, train_roc_auc = self._train_epoch(model=model,
                                                                                               data_loader=train_loader,
                                                                                               optimizer=optimizer,
                                                                                               criterion=criterion)
            if val_dataset is not None:
                # VALIDATION LOOP
                val_epoch_loss, val_precision, val_recall, val_roc_auc = self._validate_epoch(model=model,
                                                                                            data_loader=val_loader,
                                                                                            criterion=criterion)
            if debug:
                print("Train Loss: {:.4f}".format(train_epoch_loss),
                      "Train precision: {:.4f}".format(train_precision),
                      "Train recall: {:.4f}".format(train_recall),
                      "Train roc_auc: {:.4f}".format(train_roc_auc),
                      sep=" : ")
                print("Val Loss: {:.4f}".format(val_epoch_loss),
                      "Val precision: {:.4f}".format(val_precision),
                      "Val recall: {:.4f}".format(val_recall),
                      "Val roc_auc: {:.4f}".format(val_roc_auc),
                      sep=" : ")

            if hopt:
                session.report({"loss": train_epoch_loss,
                                "train_precision": train_precision,
                                "train_recall": train_recall,
                                "train_roc_auc": train_roc_auc,
                                "val_loss": val_epoch_loss,
                                "val_precision": val_precision,
                                "val_recall": val_recall,
                                "val_roc_auc": val_roc_auc})

        if not hopt:
            self.trained_model = model



        print("Finished Training")

class Experiment:

    def __init__(self, model_ops: ModelOPS, dataset: PapilaOFI, search_field: Dict = None) -> None:

        self.model_ops = model_ops
        self.config = search_field
        self.dataset = dataset

    def run_kfold(self, num_samples: int = 100, mode: str = ''):

        mlflow.set_experiment(f"KFOLD-{mode}-{self.config['model_name']}")

        with mlflow.start_run(run_name=f"KFOLD-{mode}-{self.config['model_name']}"):
            mlflow.log_params(self.config)

            test_precision = []
            test_recall = []
            test_auc = []

            for i in range(num_samples):
                print(f"[INFO]: Sample {i + 1} of {num_samples}")
                data_generator = self.dataset.get_kfold(i)

                true_labels = []
                pred_labels = []
                pat_ids = []

                with mlflow.start_run(nested=True):
                    # Log config
                    # mat_db = loadmat(os.path.join(DATA_DIR, 'mat_db', 'PAPILA_10.4.mat'))
                    sample = sorted(os.listdir(os.path.join(DATA_DIR, 'splits', 'test')),
                         key=lambda x: int(x.split('_')[2]))[i]
                    print(i, sample, sep=" : ")
                    df = pd.read_csv(os.path.join(DATA_DIR, 'splits', 'test', sample), header=0)

                    for j, (train_dataset, val_dataset) in enumerate(data_generator, start=1):
                        print(f"[INFO]: Fold {j} of 5")
                        self.model_ops.fit(config=self.config, train_dataset=train_dataset, val_dataset=None, hopt=False)
                        trained_model = self.model_ops.trained_model



                        test_loader = torch.utils.data.DataLoader(val_dataset, # type: ignore
                                                                batch_size=1,
                                                                shuffle=False,
                                                                num_workers=1)

                        y_true, y_pred = self.model_ops.inference(model=trained_model, data_loader=test_loader)

                        true_labels.extend(y_true)
                        pred_labels.extend(y_pred)
                        print(df.shape)
                        print(df.iloc[:, j - 1].values)
                        # pat_ids.append(mat_db[f'pat_id_test_{j}'][0][::2])
                        pat_ids.append(df.iloc[:, j - 1].values)


                    true_labels = np.array(true_labels)
                    pred_labels = np.array(pred_labels)
                    pat_ids = np.concatenate(pat_ids)

                    m_precision, m_recall, m_roc_auc = self.model_ops.metrics(true_labels, pred_labels[:, 1])

                    mlflow.log_metric("precision", m_precision)
                    mlflow.log_metric("recall", m_recall)
                    mlflow.log_metric("roc_auc", m_roc_auc)
                    mlflow.log_param("sample", i + 1)

                    test_precision.append(m_precision)
                    test_recall.append(m_recall)
                    test_auc.append(m_roc_auc)

                    pred_labels_class_0 = pred_labels[:, 0]
                    pred_labels_class_1 = pred_labels[:, 1]

                    print(pat_ids.shape, true_labels.shape, pred_labels_class_0.shape, pred_labels_class_1.shape)
                    df = pd.DataFrame({"pat_id": pat_ids,
                                       "true_labels": true_labels,
                                       "pred_labels_class_0": pred_labels_class_0,
                                       "pred_labels_class_1": pred_labels_class_1})

                    with tempfile.TemporaryDirectory() as tmpdir:
                        df.to_csv(os.path.join(tmpdir, "predictions.csv"))
                        mlflow.log_artifacts(tmpdir, "predictions")

            mlflow.log_metric("test_precision", np.mean(test_precision))
            mlflow.log_metric("test_precision_std", np.std(test_precision))
            mlflow.log_metric("test_recall", np.mean(test_recall))
            mlflow.log_metric("test_recall_std", np.std(test_recall))
            mlflow.log_metric("test_auc", np.mean(test_auc))
            mlflow.log_metric("test_auc_std", np.std(test_auc))

    def run_hopt_search(self, max_num_epochs: int, grace_period: int, reduction_factor: int, num_samples: int, gpus_per_trial: float, mode: str = '') -> None:

        scheduler = ASHAScheduler(max_t=max_num_epochs,
                                  grace_period=grace_period,
                                  reduction_factor=reduction_factor)
        algo = HyperOptSearch()

        data_generator = self.dataset.get_hopt()
        for train_dataset, val_dataset in data_generator:

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(trainable=self.model_ops.fit,
                                        train_dataset=train_dataset,
                                        val_dataset=val_dataset),
                    resources = {"cpu": 0, "gpu": gpus_per_trial}
                ),
                tune_config=tune.TuneConfig(
                    metric="val_loss",
                    mode="min",
                    scheduler=scheduler,
                    num_samples=num_samples,
                    search_alg=algo,
                ),
                run_config=air.RunConfig(
                    name="mlflow",
                    callbacks=[
                        MLflowLoggerCallback(
                            tracking_uri=mlflow_tracking_uri,
                            experiment_name=f"HPOT-{mode}-{self.config['model_name']}", # here
                            save_artifact=True,
                        )
                    ],
                ),
                param_space=self.config,
            )

            result = tuner.fit()

    def debug(self) -> None:
        data_generator = self.dataset.get_hopt()
        for train_dataset, val_dataset in data_generator:
            self.model_ops.fit(config=self.config, train_dataset=train_dataset, val_dataset=val_dataset, hopt=False, debug=True)

def _load_config(run_id: str) -> Dict:

        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
        config = mlflow_client.get_run(run_id=run_id).data.to_dictionary()['params']
        del mlflow_client
        config["epochs"] = int(config["epochs"])
        config["batch_size"] = int(config["batch_size"])
        config["num_classes"] = int(config["num_classes"])
        # config["smote"] = True if config["smote"] == "True" else False
        config["pretrained"] = True if config["pretrained"] == "True" else False
        config["lr"] = float(config["lr"])
        config["weight_decay"] = float(config["weight_decay"])
        config["dropout"] = float(config["dropout"])
        config["hidden_units"] = int(config["hidden_units"])
        config["hidden_layers"] = int(config["hidden_layers"])
        config["freeze_layers"] = int(config["freeze_layers"])
        config["class_weights"] = [float(i) for i in config["class_weights"].strip("()").split(",")]
        config["label_smoothing"] = float(config["label_smoothing"])
        return config

def _general_hopt_experiment(models: Dict, config: Dict, dataset: PapilaOFI) -> None:

    model_ops = ModelOPS(model=models[config["model_name"]])

    experiment = Experiment(model_ops=model_ops,
                            dataset=dataset,
                            search_field=config)

    experiment.run_hopt_search(max_num_epochs=1000,
                                grace_period=300,
                                reduction_factor=2,
                                num_samples=200,
                                gpus_per_trial=0.5,
                                mode=config["data_mode"])

def _general_kfold_experiment(config: Dict, models: Dict,  dataset: PapilaOFI) -> None:

    model_ops = ModelOPS(model=models[config["model_name"]])

    experiment = Experiment(model_ops=model_ops,
                            dataset=dataset,
                            search_field=config)

    experiment.run_kfold(num_samples=100, mode=config["data_mode"])

def _general_debug_experiment(models: Dict, config: Dict, dataset: PapilaOFI) -> None:

    model_ops = ModelOPS(model=models[config["model_name"]])

    experiment = Experiment(model_ops=model_ops,
                            dataset=dataset,
                            search_field=config)

    experiment.debug()

def cnn_experiments(experiment_mode: str,
                    train_transform: "transforms.Compose",
                    val_transform: "transforms.Compose",
                    config: Dict = None,
                    run_id: str = None) -> None:
    """
    Run CNN experiments.

    Parameters:
    -----------
    experiment_mode: str one from the list ["hopt", "kfold", "debug"].
    train_transform: transforms.Compose
    val_transform: transforms.Compose
    config: Dict ony for "hopt" mode
    run_id: str only for "kfold" mode
    """

    assert (config is not None) and (run_id is not None), "Either config or run_id must be provided"
    assert experiment_mode in ["hopt", "kfold", "debug"]

    if config is None or experiment_mode == "kfold":

        assert run_id is not None, "In kfold `run_id` must be provided"

        config = _load_config(run_id=run_id)

    assert config["data_mode"] in ["single", "both"], "data_mode must be one from the list ['single', 'both']"

    if config["data_mode"] == "single":

        from src.data.dataset import PapilaOFISE
        from src.models.cnn import (VGG16, DenseNet121, InceptionV3,
                                    MobileNetV3, ResNet50)

        models = {"resnet50": ResNet50,
                  "densenet121": DenseNet121,
                  "inceptionv3": InceptionV3,
                  "mobilenetv3": MobileNetV3,
                  "vgg16": VGG16}

        if experiment_mode == "hopt":
            dataset = PapilaOFISE(train_transform=train_transform,
                                  test_transform=val_transform)

            _general_hopt_experiment(models=models, config=config, dataset=dataset)

        elif experiment_mode == "debug":

            dataset = PapilaOFISE(train_transform=train_transform,
                                  test_transform=val_transform)

            _general_debug_experiment(models=models, config=config, dataset=dataset)

        else:

            dataset = PapilaOFISE(train_transform=train_transform,
                                  test_transform=val_transform)

            _general_kfold_experiment(config=config, models=models, dataset=dataset)

    else:

        from src.data.dataset import PapilaOFIBE
        from src.models.cnn import (DenseNet121BothEyes, InceptionV3BothEyes,
                                    MobileNetV3BothEyes, ResNet50BothEyes,
                                    VGG16BothEyes)

        models = {"resnet50BE": ResNet50BothEyes,
                  "densenet121BE": DenseNet121BothEyes,
                  "inceptionv3BE": InceptionV3BothEyes,
                  "mobilenetv3BE": MobileNetV3BothEyes,
                  "vgg16BE": VGG16BothEyes}


        if experiment_mode == "hopt":

            dataset = PapilaOFIBE(train_transform=train_transform,
                                  test_transform=val_transform)

            _general_hopt_experiment(models=models, config=config, dataset=dataset)

        elif experiment_mode == "debug":

            dataset = PapilaOFIBE(train_transform=train_transform,
                                  test_transform=val_transform)
            _general_debug_experiment(models=models, config=config, dataset=dataset)

        else:

            dataset = PapilaOFIBE(train_transform=train_transform,
                                  test_transform=val_transform)
            _general_kfold_experiment(config=config, models=models, dataset=dataset)

# def vit_experiments(eye_mode: str,
#                     experiment_mode: str,
#                     config: Dict,
#                     train_transform: "transforms.Compose",
#                     val_transform: "transforms.Compose",
#                     run_id: str = None) -> None:

#     assert eye_mode in ["single", "both", "debug"]
#     assert experiment_mode in ["hopt", "kfold", "debug"]

#     if eye_mode == "single":

#         from src.data.dataset import PapilaOFISE
#         # from src.models.vit import (ViT_B_16_SE, ViT_B_32_SE, ViT_L_16_SE,
#         #                             ViT_L_32_SE)

#         # models = {"vit_b_16": ViT_B_16_SE,
#         #           "vit_b_32": ViT_B_32_SE,
#         #           "vit_l_16": ViT_L_16_SE,
#         #           "vit_l_32": ViT_L_32_SE}


#         if experiment_mode == "hopt":
#             dataset = PapilaOFISE(train_transform=train_transform,
#                                   test_transform=val_transform)
#             _general_hopt_experiment(models=models, config=config, dataset=dataset)
#         elif experiment_mode == "debug":

#             dataset = PapilaOFISE(train_transform=train_transform,
#                                   test_transform=val_transform)
#             _general_debug_experiment(models=models, config=config, dataset=dataset)


#         else:
#             dataset = PapilaOFISE(train_transform=train_transform,
#                                   test_transform=val_transform)
#             _general_kfold_experiment(run_id=run_id, models=models, dataset=dataset)

#     else:
#         from src.data.dataset import PapilaOFIBE
#         # from src.models.vit import (ViT_B_16_BE, ViT_B_32_BE, ViT_L_16_BE,
#         #                             ViT_L_32_BE)

#         # models = {"vit_b_16_BE": ViT_B_16_BE,
#         #           "vit_b_32_BE": ViT_B_32_BE,
#         #           "vit_l_16_BE": ViT_L_16_BE,
#         #           "vit_l_32_BE": ViT_L_32_BE}

#         if experiment_mode == "hpot":

#             dataset = PapilaOFIBE(train_transform=train_transform,
#                                   test_transform=val_transform)
#             _general_hopt_experiment(models=models, config=config, dataset=dataset)
#         elif experiment_mode == "debug":

#             dataset = PapilaOFIBE(train_transform=train_transform,
#                                   test_transform=val_transform)
#             _general_debug_experiment(models=models, config=config, dataset=dataset)

#         else:

#             dataset = PapilaOFIBE(train_transform=train_transform,
#                                   test_transform=val_transform)
#             _general_kfold_experiment(run_id=run_id, models=models, dataset=dataset)
