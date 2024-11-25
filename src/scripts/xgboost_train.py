import os
import sys
import tempfile

import warnings
from typing import Any, Dict, List, Tuple, Optional

import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from numpy.typing import ArrayLike

from ray import air, tune
from ray.air import session
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.metrics import precision_score, recall_score, roc_auc_score


# Silence all warnings
warnings.filterwarnings("ignore")

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(SRC_DIR)
from src.data.dataset import (
    ClinicalPlusMorphologicalBE,
    ClinicalPlusMorphologicalSE,
    ClinicalPlusOFIBE,
    ClinicalPlusOFISE,
    PapilaTabular,
    PapilaTabularBothEyes,
    PapilaTabularSingleEye,
)

mlflow_tracking_uri = "http://localhost:5003"


class ModelOPS:
    def metrics(
        self, y_true: ArrayLike, y_pred: ArrayLike, threshold: float = 0.5
    ) -> Tuple[float, float, float]:
        # _, _, thresholds = roc_curve(y_true, y_pred)

        y_pred = y_pred[:, 1]  # type: ignore

        # weights = np.array([0.48702595, 1.89147287])

        # sample_weights = [weights[0] if i == 0 else weights[1] for i in y_true]
        # sample_weight=sample_weights, average="weighted")
        m_roc_auc = float(
            roc_auc_score(
                y_true,
                y_pred,
            )
        )

        y_pred = [1 if i > threshold else 0 for i in y_pred]  # type: ignore

        m_precision = float(
            precision_score(
                y_true,
                y_pred,
            )
        )  # average="weighted")
        m_recall = float(
            recall_score(
                y_true,
                y_pred,
            )
        )  # average="weighted")

        return m_precision, m_recall, m_roc_auc

    def inference(
        self, model: xgb.Booster, ddata: xgb.DMatrix
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        X = ddata.get_data().toarray()
        y = ddata.get_label()

        return y, model.predict_proba(X), model.predict(X)  # type: ignore

    def fit(
        self,
        config: Dict,
        X_train,
        y_train,
        X_test,
        y_test,
        hopt: bool = True,
        kfold_search: bool = False,
        debug: bool = False,
    ) -> Tuple[float, float] | Tuple[ArrayLike, ArrayLike, xgb.Booster]:
        raw_model = xgb.XGBClassifier(
            sampling_method="uniform",
            objective=config["objective"],
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            max_leaves=config["max_leaves"],
            learning_rate=config["learning_rate"],
            booster=config["booster"],
            gamma=config["gamma"],
            min_child_weight=config["min_child_weight"],
            max_delta_step=config["max_delta_step"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            colsample_bylevel=config["colsample_bylevel"],
            reg_lambda=config["reg_lambda"],
            reg_alpha=config["reg_alpha"],
            scale_pos_weight=config["scale_pos_weight"],
            missing=np.nan,
        )
        model = raw_model.fit(
            X=X_train,
            y=y_train.values,
            eval_set=[(X_train, y_train.values), (X_test, y_test.values)],
            eval_metric=config["eval_metric"],
            early_stopping_rounds=config["early_stopping_rounds"],
            sample_weight=xgb.DMatrix(X_train, y_train).get_weight(),
            verbose=config["verbose"],
        )

        evals_result = model.evals_result()
        train_predictions = model.predict_proba(X_train)
        test_predictions = model.predict_proba(X_test)
        if debug:
            return train_predictions, test_predictions, model

        if kfold_search:
            val_loss = evals_result["validation_1"]["logloss"][-1]
            train_loss = evals_result["validation_0"]["logloss"][-1]
            return train_loss, val_loss
        else:
            self.trained_model = model
            return train_predictions, test_predictions, model
        # TODO: check this
        if hopt:
            train_precision, train_recall, train_roc_auc = self.metrics(
                y_train.values, train_predictions
            )
            val_precision, val_recall, val_roc_auc = self.metrics(
                y_test.values, test_predictions
            )
            train_loss = evals_result["validation_0"]["logloss"][-1]
            val_loss = evals_result["validation_1"]["logloss"][-1]
            session.report(
                {
                    "train_loss": train_loss,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_roc_auc": train_roc_auc,
                    "val_loss": val_loss,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_roc_auc": val_roc_auc,
                }
            )
        else:
            self.trained_model = model


class Experiment:
    def __init__(
        self, model_ops: ModelOPS, dataset: PapilaTabular, config: Dict
    ) -> None:
        self.model_ops = model_ops
        self.config = config
        self.dataset = dataset

    def run_kfold(
        self, num_samples: int = 100, mode: str = "", datasource: str = ""
    ) -> None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(f"KFOLD-{mode}-{self.config['model_name']}-{datasource}")

        with mlflow.start_run(run_name=f"{self.config['model_name']}-{mode}"):
            mlflow.log_params(self.config)

            test_precision, test_recall, test_roc_auc = [], [], []
            shap_values_per_sample: List[Any] = []

            for i in range(num_samples):
                print(f"[INFO]: Sample {i+1}/{num_samples}")
                data_generator = self.dataset.get_kfold(i)

                true_labels, pred_labels, pred_proba, pat_ids = [], [], [], []
                shap_values_per_fold: List[Any] = []
                X_test_per_fold: List[pd.DataFrame] = []  # type: ignore

                with mlflow.start_run(nested=True):
                    for j, (x_train, x_test, y_train, y_test) in enumerate(
                        data_generator, start=1
                    ):
                        print(f"[INFO]: Fold {j} of 5")
                        self.model_ops.fit(
                            config=self.config,
                            X_train=x_train,
                            X_test=x_test,
                            y_test=y_test,
                            y_train=y_train,
                            hopt=False,
                        )

                        y_true, y_proba, y_pred = self.model_ops.inference(
                            model=self.model_ops.trained_model,  # type: ignore
                            ddata=xgb.DMatrix(x_test, label=y_test),
                        )
                        true_labels.extend(y_true)  # type: ignore
                        pred_labels.extend(y_pred)  # type: ignore
                        pred_proba.extend(y_proba)  # type: ignore
                        pat_ids.extend(x_test.index.tolist())  # type: ignore

                        X_test_per_fold.append(x_test)  # type: ignore

                        # START SHAP CALCULATIONS

                        expleainer = shap.TreeExplainer(self.model_ops.trained_model)  # type: ignore
                        shap_values = expleainer.shap_values(
                            x_test
                        )  # 2D array same shape as x_test
                        if isinstance(x_test, pd.DataFrame):
                            # TODO: check if this is the right way to do it
                            shape_values = pd.DataFrame(
                                shap_values, columns=x_test.columns, index=x_test.index
                            )
                            shap_values_per_fold.append(shape_values)
                        else:
                            raise Exception("x_test must be a pandas dataframe")

                        # END SHAP CALCULATIONS

                    # END - For loop for folds
                    X_test_per_fold: pd.DataFrame = pd.concat(
                        X_test_per_fold  # type: ignore
                    ).sort_index()  # type: ignore
                    shap_values_per_fold = pd.concat(shap_values_per_fold).sort_index()  # type: ignore
                    shap_values_per_sample.append(shap_values_per_fold.values)  # type: ignore

                    true_labels = np.array(true_labels)
                    pred_labels = np.array(pred_labels)
                    pred_proba = np.array(pred_proba)
                    pat_ids = np.array(pat_ids)

                    m_precision, m_recall, m_roc_auc = self.model_ops.metrics(
                        true_labels, pred_proba
                    )

                    # START MLFLOW LOGGING FOR KFOLD

                    mlflow.log_metrics(
                        {
                            "precision": m_precision,
                            "recall": m_recall,
                            "roc_auc": m_roc_auc,
                        }
                    )

                    mlflow.log_param("sample", i + 1)

                    test_precision.append(m_precision)
                    test_recall.append(m_recall)
                    test_roc_auc.append(m_roc_auc)

                    mlflow.xgboost.log_model(self.model_ops.trained_model, "model")  # type: ignore

                    pred_labels_class_0 = pred_proba[:, 0]
                    pred_labels_class_1 = pred_proba[:, 1]

                    print(
                        pat_ids.shape,
                        true_labels.shape,
                        pred_labels_class_0.shape,
                        pred_labels_class_1.shape,
                    )
                    df = pd.DataFrame(
                        {
                            "pat_id": pat_ids,
                            "true_labels": true_labels,
                            "pred_labels": pred_labels,
                            "pred_proba_class_0": pred_labels_class_0,
                            "pred_proba_class_1": pred_labels_class_1,
                        }
                    )

                    with tempfile.TemporaryDirectory() as tmpdir:
                        df.to_csv(os.path.join(tmpdir, "predictions.csv"))
                        mlflow.log_artifacts(tmpdir, "predictions")

                    # END - MLFLOW LOGGING FOR KFOLD

            # START - MLFLOW LOGGING FOR SAMPLEWICE METRICS
            shap_values_per_sample = np.array(shap_values_per_sample)  # type: ignore

            mean_shap_values_per_sample = np.mean(shap_values_per_sample, axis=0)
            print(mean_shap_values_per_sample)
            mean_shap_values_per_sample = pd.DataFrame(
                mean_shap_values_per_sample,
                columns=X_test_per_fold.columns,  # type: ignore
                index=X_test_per_fold.index,  # type: ignore
            )  # type: ignore
            print(mean_shap_values_per_sample)
            with tempfile.TemporaryDirectory() as tmpdir:
                mean_shap_values_per_sample.to_csv(
                    os.path.join(tmpdir, "shap_values.csv")
                )
                X_test_per_fold.to_csv(os.path.join(tmpdir, "X_test.csv"))  # type: ignore
                mlflow.log_artifacts(tmpdir, "shap_values")

            mlflow.log_metrics(
                {
                    "test_precision": np.mean(test_precision),
                    "test_precision_std": np.std(test_precision),
                    "test_recall": np.mean(test_recall),
                    "test_recall_std": np.std(test_recall),
                    "test_roc_auc": np.mean(test_roc_auc),
                    "test_roc_auc_std": np.std(test_roc_auc),
                }  # type: ignore
                # ignore because of mlflow typing
            )
            # END - MLFLOW LOGGING FOR SAMPLEWICE METRICS

    def run_kfold_search(
        self,
        max_num_epochs: int,
        grace_period: int,
        reduction_factor: int,
        num_samples: int,
        cpus_per_trial: float,
        mode: str = "",
        datasource: str = "",
    ) -> None:
        # data_generator = list(self.dataset.get_hopt_v2(datasource))
        data_generator = list(self.dataset.get_kfold(0))

        def train_model(config):
            l_val_loss = []
            l_train_loss = []
            for x_train, x_test, y_train, y_test in data_generator:
                train_loss, val_loss = self.model_ops.fit(  # type: ignore
                    config=config,
                    X_train=x_train,
                    y_train=y_train,
                    X_test=x_test,
                    y_test=y_test,
                    kfold_search=True,
                    hopt=False,
                    debug=False,
                )
                l_val_loss.append(val_loss)
                l_train_loss.append(train_loss)

            session.report(
                {"val_loss": np.mean(l_val_loss), "train_loss": np.mean(l_train_loss)}
            )

        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )

        algo = HyperOptSearch()

        tuner = tune.Tuner(
            tune.with_resources(
                train_model,
                resources={"cpu": cpus_per_trial, "gpu": 0},
            ),
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                scheduler=scheduler,
                search_alg=algo,
                num_samples=num_samples,
            ),
            run_config=air.RunConfig(
                name="mlflow",
                callbacks=[
                    MLflowLoggerCallback(
                        tracking_uri=mlflow_tracking_uri,
                        experiment_name=f"HOPT-KFOLD-{mode}-{self.config['model_name']}-{datasource}",
                        save_artifact=True,
                    )
                ],
            ),
            param_space=self.config,
        )

        results = tuner.fit()

    def run_hopt_search(
        self,
        max_num_epochs: int,
        grace_period: int,
        reduction_factor: int,
        num_samples: int,
        cpus_per_trial: float,
        mode: str = "",
        datasource: str = "",
    ) -> None:
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )
        algo = HyperOptSearch()

        data_generator = self.dataset.get_hopt()

        for x_train, x_test, y_train, y_test in data_generator:
            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(
                        self.model_ops.fit,
                        X_train=x_train,
                        y_train=y_train,
                        X_test=x_test,
                        y_test=y_test,
                    ),
                    resources={"cpu": cpus_per_trial, "gpu": 0},
                ),
                tune_config=tune.TuneConfig(
                    metric="val_loss",
                    mode="min",
                    scheduler=scheduler,
                    search_alg=algo,
                    num_samples=num_samples,
                ),
                run_config=air.RunConfig(
                    name="mlflow",
                    callbacks=[
                        MLflowLoggerCallback(
                            tracking_uri=mlflow_tracking_uri,
                            experiment_name=f"HOPT-{mode}-{self.config['model_name']}-{datasource}",
                            save_artifact=True,
                        )
                    ],
                ),
                param_space=self.config,
            )

            results = tuner.fit()

    def debug(self) -> None:
        data_generator = self.dataset.get_hopt()
        for x_train, x_test, y_train, y_test in data_generator:
            train_predictions, test_predictions, model = self.model_ops.fit(  # type: ignore
                config=self.config,
                X_test=x_test,
                y_test=y_test,
                X_train=x_train,
                y_train=y_train,
                debug=True,
            )

            y_train_true, y_train_pred = self.model_ops.inference(  # type: ignore
                model=model, ddata=xgb.DMatrix(x_test, label=y_test)
            )
            y_test_true, y_test_pred = self.model_ops.inference(  # type: ignore
                model=model, ddata=xgb.DMatrix(x_train, label=y_train)
            )

            train_precision, train_recall, train_roc_auc = self.model_ops.metrics(
                y_train_true, y_train_pred
            )
            test_precision, test_recall, test_roc_auc = self.model_ops.metrics(
                y_test_true, y_test_pred
            )

            print(y_train_true, y_train_pred)
            print(y_test_true, y_test_pred)
            print(y_train_true.shape, y_train_pred.shape)  # type: ignore
            print(y_test_true.shape, y_test_pred.shape)  # type: ignore

            print(train_precision, train_recall, train_roc_auc)
            print(test_precision, test_recall, test_roc_auc)


def _load_config(run_id: str) -> Dict[str, int | float | str]:
    mlflow_client = mlflow.tracking.MlflowClient(mlflow_tracking_uri)

    config: Dict[str, Any] = mlflow_client.get_run(run_id).data.params  # type: ignore

    del mlflow_client

    # converting to the right type
    config["n_estimators"] = int(config["n_estimators"])
    config["colsample_bytree"] = float(config["colsample_bytree"])
    config["colsample_bylevel"] = float(config["colsample_bylevel"])
    config["max_depth"] = int(config["max_depth"])
    config["max_leaves"] = int(config["max_leaves"])
    config["learning_rate"] = float(config["learning_rate"])
    config["gamma"] = float(config["gamma"])
    config["min_child_weight"] = float(config["min_child_weight"])
    config["max_delta_step"] = float(config["max_delta_step"])
    config["subsample"] = float(config["subsample"])
    config["reg_lambda"] = float(config["reg_lambda"])
    config["reg_alpha"] = float(config["reg_alpha"])
    config["scale_pos_weight"] = float(config["scale_pos_weight"])
    config["early_stopping_rounds"] = (
        int(config["early_stopping_rounds"])
        if config["early_stopping_rounds"] != "None"
        else None
    )
    config["verbose"] = 0  # True if config["verbose"] == "True" else False
    return config


def _general_hopt_experiment(
    config: Dict[str, int | float | str], dataset: PapilaTabular
) -> None:
    model_ops = ModelOPS()

    experiment = Experiment(model_ops=model_ops, dataset=dataset, config=config)

    experiment.run_hopt_search(
        max_num_epochs=1000,
        grace_period=100,
        reduction_factor=2,
        num_samples=10000,
        cpus_per_trial=1,
        mode=str(config["data_mode"]),
        datasource=str(config["data_source"]),
    )


def _general_kfold_experiment(run_id: str, dataset: PapilaTabular) -> None:
    config = _load_config(run_id)

    model_ops = ModelOPS()

    experiment = Experiment(model_ops=model_ops, dataset=dataset, config=config)

    experiment.run_kfold(
        num_samples=100,
        mode=str(config["data_mode"]),
        datasource=str(config["data_source"]),
    )


def _general_debug_experiment(
    config: Dict[str, float | int | str], dataset: PapilaTabular
) -> None:
    model_ops = ModelOPS()

    experiment = Experiment(model_ops=model_ops, dataset=dataset, config=config)

    experiment.debug()


def run_experiment(dataset_name: str, experiment_mode: str, config: Optional[Dict[str, float | int | str]] = None, run_id: str = None) -> None:
    assert (config is not None) or (
        run_id is not None
    ), "Either config or run_id must be provided"
    assert experiment_mode in ["kfold", "hopt", "debug"]

    if experiment_mode == "kfold":
        assert run_id is not None, "run_id must be provided"
        config = _load_config(run_id)

    assert config["data_mode"] in [
        "single",
        "both",
    ], "data_mode must be either single or both"

    if config["data_mode"] == "single":
        from src.data.dataset import (
            ClinicalPlusMorphologicalSE,
            ClinicalPlusOFISE,
            ClinicalSE,
            MorphologicalE1SE,
            MorphologicalE2SE,
        )

        _datasets = {
            "clinical": ClinicalSE,
            "morphological_e1": MorphologicalE1SE,
            "morphological_e2": MorphologicalE2SE,
            "clinical_morphological_e2": ClinicalPlusMorphologicalSE,
            "clinical_morphological_e1": ClinicalPlusMorphologicalSE,
            "clinical_ofi": ClinicalPlusOFISE,
        }

        dataset = _datasets[dataset_name]()

        if experiment_mode == "hopt":
            _general_hopt_experiment(config=config, dataset=dataset)  # type: ignore

        elif experiment_mode == "debug":
            _general_debug_experiment(config=config, dataset=dataset)  # type: ignore

        else:
            _general_kfold_experiment(run_id=run_id, dataset=dataset)  # type: ignore

    else:
        from src.data.dataset import (
            ClinicalBE,
            ClinicalPlusMorphologicalBE,
            ClinicalPlusOFIBE,
            MorphologicalE1BE,
            MorphologicalE2BE,
        )

        _datasets = {
            "clinical": ClinicalBE,
            "morphological_e1": MorphologicalE1BE,
            "morphological_e2": MorphologicalE2BE,
            "clinical_morphological_e2": ClinicalPlusMorphologicalBE,
            "clinical_morphological_e1": ClinicalPlusMorphologicalBE,
            "clinical_ofi": ClinicalPlusOFIBE,
        }

        dataset = _datasets[dataset_name]()

        if experiment_mode == "hopt":
            _general_hopt_experiment(config=config, dataset=dataset)  # type: ignore

        elif experiment_mode == "debug":
            _general_debug_experiment(config=config, dataset=dataset)  # type: ignore

        else:
            _general_kfold_experiment(run_id=run_id, dataset=dataset)  # type: ignore


if __name__ == "__main__":
    search_space = {
        # No model params
        "data_source": "clinical_ofi"
        "data_mode": "both",
        "model_name": "xgboost",
        # Model params
        # No tunable params
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "early_stopping_rounds": None,
        "verbose": False,
        "booster": "gbtree",
        # Tunable params
        # No tuned params
        "max_leaves": 0,
        "max_delta_step": 0,
        "n_estimators": 180,
        "subsample": 1,
        "colsample_bylevel": 1,
        "scale_pos_weight": 3.47,
        # ---
        # Tuned params
        "learning_rate": tune.uniform(0.0001, 0.1),
        "max_depth": tune.qrandint(3, 5, 1),
        "gamma": tune.uniform(0.001, upper=4),
        "reg_lambda": tune.uniform(0, 1),
        "reg_alpha": tune.quniform(1, 40, 1),
        "colsample_bytree": tune.uniform(0.5, 1),
        "min_child_weight": tune.quniform(0, 10, 1),
    }

    if search_space["data_mode"] == "single":
        dataset = PapilaTabularBothEyes()
    else:
        dataset = PapilaTabularSingleEye()

    model_ops = ModelOPS()

    experiment = Experiment(model_ops=model_ops, dataset=dataset, config=search_space)
    experiment.run_kfold_search(
        max_num_epochs=1000,
        grace_period=100,
        reduction_factor=2,
        num_samples=1000,
        cpus_per_trial=1,
        mode=str(search_space["data_mode"]),
        datasource=str(search_space["data_source"]),
    )
    # data_sources = ["clinical", "morphological_e1", "morphological_e2"]
    # SE
    # run_ids = [
    #     # ofi
    #     # no ofi
    #     # "2631cf20619343a2bb4ad42687e4ca4f",  #    "00d30514102a492cbf78faa016c72da2",
    #     # "25ee19c27ca24d90847f9b3384392051",  #    "47865cb0914d48729a6bc64e1cedc59f",
    #     # "c436ec5d0916457db241cafcb0995ed6",  #    "4f3bef876b524a9aaae66a4d2b343d02"
    # ]
    # BE

    # run_ids = [
    #     # "02cc771dbc8c480d95bd24aa4e27d338",  #   "54056611228849538b06fe08485c1d58",
    #     # "aaa16e941ae943d9970aede667ff424e",  #    "d34aaeb17c3c4414b9c8eb8d16a7f7d8",
    #     # "314c1040fe9f41efbf4d8b8ba8c23245",  #    "04a396bce9664421aabed0dbb434810a"
    # ]

    data_sources = ["clinical_morphological_e1"]
    # E1 SE
    # run_ids = ["956cb00450dd4744a04b87231f9e65ba"]
    # E1 BE
    run_ids = ["48805eb9e8fe483888f1972d9d4aa50f"]
    # E2 SE
    # run_ids = ["da72c9be464742f19b82cb5e36e2b323"]
    # E2 BE
    # run_ids = ["87c456c0312f47f2bdff62588b35ac77"]

    # OFI
    # data_sources = ["clinical_ofi"]
    # SE
    # run_ids = [
    #     "4e6fd67e655445689d0a56d9b6c37436",
    #     # 'ff9ddf8dc0e14a37acb36ccc1a55e959',
    #     #    "f8907eeeaf2a427483c82a4115d58077"
    # ]
    # BE
    # run_ids = ["facc8e17bf39405d8ce9842bf330f3f4"]

    for i, datasource in enumerate(data_sources):
        run_experiment(
            dataset_name=datasource, experiment_mode="kfold", run_id=run_ids[i]
        )
    # search_space = {
    #     # No model params
    #     "data_source": "morphological_e1_ofi",
    #     "data_mode": "both",  # "both",
    #     "model_name": "xgboost",
    #     # Model params
    #     # No tunable params
    #     "objective": "binary:logistic",
    #     "eval_metric": "logloss",
    #     "early_stopping_rounds": None,
    #     "verbose": False,
    #     "booster": "gbtree",
    #     # Tunable params
    #     # No tuned params
    #     "max_leaves": 0,
    #     "max_delta_step": 0,
    #     "n_estimators": 180,
    #     "subsample": 1,
    #     "colsample_bylevel": 1,
    #     "scale_pos_weight": 3.47,
    #     # ---
    #     # Tuned params
    #     "learning_rate": tune.uniform(0.0001, 0.1),
    #     "max_depth": tune.qrandint(3, 18, 1),
    #     "gamma": tune.uniform(0.001, upper=4),
    #     "reg_lambda": tune.uniform(0, 1),
    #     "reg_alpha": tune.quniform(1, 40, 1),
    #     "colsample_bytree": tune.uniform(0.5, 1),
    #     "min_child_weight": tune.quniform(0, 10, 1),
    # }

    # if search_space["data_mode"] == "both":
    #     dataset = (
    #         ClinicalPlusOFIBE()
    #         # ClinicalPlusMorphologicalBE()
    #     )  # ClinicalPlusOFISE()  # PapilaTabularBothEyes()
    # else:
    #     dataset = (
    #         ClinicalPlusOFISE()
    #         # ClinicalPlusMorphologicalSE()
    #     )  # ClinicalPlusOFISE()  # PapilaTabularSingleEye()

    # model_ops = ModelOPS()

    # experiment = Experiment(model_ops=model_ops, dataset=dataset, config=search_space)  # type: ignore
    # experiment.run_kfold_search(
    #     max_num_epochs=1000,
    #     grace_period=100,
    #     reduction_factor=2,
    #     num_samples=10000,
    #     cpus_per_trial=1,
    #     mode=str(search_space["data_mode"]),
    #     datasource=str(search_space["data_source"]),
    # )
    #     config = {
    #         "model_name": "xgboost",
    #         "data_source": datasource,
    #         "data_mode": "single",
    #         "objective": "binary:logistic",
    #         "eval_metric": "logloss",
    #         "n_estimators": tune.choice([i for i in range(1000, 100000, 1000)]),
    #         # tune.choice([i for i in range(1, 20)]),
    #         "max_depth": tune.choice([i for i in range(3, 20)]),
    #         "max_leaves": 0,
    #         "learning_rate": tune.loguniform(1e-5, 1e-1),
    #         "booster": "gbtree",
    #         "gamma": tune.uniform(0.001, upper=1),
    #         # tune.choice([i for i in range(1, 1000)]),
    #         "min_child_weight": 0,
    #         # tune.choice([i for i in range(1, 10)]),
    #         "max_delta_step": 0,
    #         "subsample": tune.uniform(0.5, 1),
    #         "colsample_bytree": tune.uniform(0.5, 1),
    #         "colsample_bylevel": tune.uniform(0.5, 1),
    #         # tune.choice([i for i in range(1, 100)]),
    #         "reg_lambda": tune.uniform(0.001, upper=3),
    #         "reg_alpha": 0,  # tune.choice([i for i in range(1, 100)]),
    #         "scale_pos_weight": tune.choice([i for i in range(1, 15)]),
    #         # tune.choice([i for i in range(1, 1000)]),
    #         "early_stopping_rounds": 1000,
    #         "verbose": True,
    #     }

    # debug config

    # print(config)

    # config = {"model_name": "xgboost",
    #             "data_source": "clinical",

    #             "data_mode": "single",
    #             "objective": "binary:logistic",
    #             "eval_metric": ["auc", "logloss"],
    #             "n_estimators": 100,
    #             "max_depth": 1,
    #             "max_leaves": 0,
    #             "learning_rate": 0.0001,
    #             "booster": "gbtree",
    #             "gamma": 1,
    #             "min_child_weight": 1,
    #             "max_delta_step": 1,
    #             "subsample": 0.5,
    #             "colsample_bytree": 0.5,
    #             "colsample_bylevel": 0.5,
    #             "reg_lambda": 1,
    #             "reg_alpha": 1,
    #             "scale_pos_weight": 1,
    #             "early_stopping_rounds": 1,
    #             "verbose": true}
