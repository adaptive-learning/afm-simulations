import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold  # type: ignore
from tqdm import tqdm

from .models import AFM

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8.8s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

EXPORT_PATH = "fitted_params_tf.csv"
EXCLUDED_ITEMS = "excluded_items"

SCENARIO_DEF = tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
]
SCENARIO_FULL_DEF = tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]


def load_params(path: Path) -> pd.DataFrame:
    """
    Load ground truth parameters of scenarios

    :param path: Path containing scenarios data
    :return: A single dataframe with all scenarios
    """
    logger.info(f"Loading parameters from fitted scenarios in {path}")
    params = []
    _, scenarios, _ = next(os.walk(path))
    for scenario in tqdm(scenarios, desc="loading scenarios"):
        if "ignore" in scenario:
            continue
        Path(f"fig/{scenario}").mkdir(parents=True, exist_ok=True)
        base_folder = path / scenario
        base_folder_excluded_items = base_folder / EXCLUDED_ITEMS

        for fold in range(1, 6):
            df = load_fold(base_folder, fold, scenario)
            df["excluded_items"] = False
            params.append(df)

            if base_folder_excluded_items.exists():
                df_ex = load_fold(
                    base_folder, fold, scenario, fold_folder=base_folder_excluded_items
                )
                df_ex["excluded_items"] = True
                params.append(df_ex)

    return pd.concat(params)


def load_results(path: Path) -> pd.DataFrame:
    """
    Load fitted parameters of scenarios

    :param path: Path containing scenarios data
    :return: A single dataframe with all scenarios
    """
    logger.info(f"Loading parameters from fitted scenarios in {path}")
    results = []
    _, scenarios, _ = next(os.walk(path))
    for scenario in tqdm(scenarios, desc="loading scenarios"):
        if "ignore" in scenario:
            continue
        results_path = path / scenario / "results.csv"
        results.append(pd.read_csv(results_path))

    return pd.concat(results)


def load_fold(
    base_folder: Path, fold: int, scenario: str, fold_folder: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load ground truth parameter for a single fold of a scenario simulation

    :param base_folder: Path to a folder with simulated scenario
    :param fold: Number identifier of the fold to load
    :param scenario: Name of the scenario being loaded
    :param fold_folder: Optional subfolder containing the requested fold
    :return: Data Frame with all ground truth parameters used in a fold
    """
    if fold_folder is None:
        fold_folder = base_folder
    params_path = fold_folder / (f"fold_{fold}_" + EXPORT_PATH)
    df = pd.read_csv(params_path)
    df["scenario"] = scenario
    df["fold"] = fold

    df["num_answers_cheated"] = pd.read_csv(
        base_folder / f"fold_{fold}_cheating.csv"
    ).iloc[0, 0]

    log, true_alphas, true_betas, true_gammas, q_matrix, setting = load_scenario(
        base_folder
    )
    df["answers_cheated_percentage"] = (
        df["num_answers_cheated"] / (len(log) / 5) * 100
    ).round(1)

    names = (
        [f"alpha{i}" for i in range(true_alphas.shape[0])]
        + [f"beta{i}" for i in range(true_betas.shape[0])]
        + [f"gamma{i}" for i in range(true_gammas.shape[0])]
    )
    values = np.hstack((true_alphas, true_betas, true_gammas))
    true_params = pd.DataFrame({"name": names, "true_value": values})
    df = df.merge(true_params, on="name", how="outer")

    cheater = (
        log.groupby("student_id")["cheating"]
        .max()
        .sort_index()
        .rename("cheater")
        .astype(bool)
    )
    cheater = cheater.set_axis("alpha" + cheater.index.astype(str))
    assert len(cheater) == len(true_alphas)
    assert 0 <= max(cheater) <= 1
    df = df.merge(cheater, left_on="name", right_index=True, how="left")

    if "cheating_prevalence" in setting:
        df["cheating_prevalence"] = setting["cheating_prevalence"]

    if (
        "beta_distribution_kwargs" in setting
        and "values" in setting["beta_distribution_kwargs"]
    ):
        df["betas"] = str(setting["beta_distribution_kwargs"]["values"])
    if (
        "gamma_distribution_kwargs" in setting
        and "values" in setting["gamma_distribution_kwargs"]
    ):
        df["gammas"] = str(setting["gamma_distribution_kwargs"]["values"])

    return df


def load_fitted_scenarios(path: Path) -> dict[str, SCENARIO_FULL_DEF]:
    """Load all scenarios in a given directory"""
    logger.info(f"Loading setting from fitted scenarios in {path}")
    all_scenarios = dict()
    _, scenarios, _ = next(os.walk(path))
    for scenario in tqdm(scenarios, desc="loading scenarios"):
        if "ignore" in scenario:
            continue
        Path(f"fig/{scenario}").mkdir(parents=True, exist_ok=True)
        base_folder = path / scenario

        all_scenarios[scenario] = load_fitted_scenario(base_folder)

    return all_scenarios


def load_scenario(
    scenario_path: Path,
) -> SCENARIO_DEF:
    """Load simulation data into data frames and arrays."""

    log = pd.read_csv(scenario_path / "log.csv")
    params = pd.read_csv(scenario_path / "params.csv")
    alphas = get_parameters(params, "alpha")
    betas = get_parameters(params, "beta")
    gammas = get_parameters(params, "gamma")
    with open(scenario_path / "setting.json", "r") as f:
        setting = json.load(f)

    q_matrix = (
        pd.read_csv(scenario_path / "q_matrix.csv").set_index("item_id").to_numpy()
    )
    return log, alphas, betas, gammas, q_matrix, setting


def get_parameters(
    params: pd.DataFrame, param_name: Literal["alpha", "beta", "gamma"]
) -> np.ndarray:
    """Extract values of a single type of AFM parameter."""
    return params[params.name.str.contains(param_name)].value.values  # type: ignore


def load_fitted_scenario(
    scenario_path: Path,
) -> SCENARIO_FULL_DEF:
    """
    Load data about simulated scenario and AFM fitted to the simulated data

    :param scenario_path: Path to a folder containing data to load
    :return: tuple containing simulation data and description of fitted AFM
    """
    log, alphas, betas, gammas, q_matrix, setting = load_scenario(scenario_path)

    num_students = len(log["student_id"].unique())
    student_ids = log["student_id"].to_numpy()
    item_ids = log["item_id"].to_numpy()
    learning_opportunities: np.ndarray = AFM.compute_opportunities(
        student_ids, item_ids, q_matrix, num_students=num_students
    )
    learning_opportunities_df = pd.DataFrame(
        learning_opportunities,
        columns=[f"k{i}" for i in range(learning_opportunities.shape[1])],
    )

    predictions = pd.read_csv(scenario_path / "predictions.csv", index_col=0)

    return (
        log,
        alphas,
        betas,
        gammas,
        q_matrix,
        setting,
        learning_opportunities_df,
        predictions,
    )


def fit_all_in_path(
    path: Path,
    items_to_exclude: Optional[list[int]] = None,
    student_to_exclude: Optional[list[int]] = None,
) -> None:
    """Fit AFM to all scenarios in a directory"""
    _, scenarios, _ = next(os.walk(path))
    for scenario in tqdm(scenarios, desc="fitting scenarios"):
        if "ignore" in scenario:
            continue

        print("Fitting", scenario)
        fit_afm_on_simulated_data(
            path / scenario,
            items_to_exclude=items_to_exclude,
            student_to_exclude=student_to_exclude,
        )


def fit_afm_on_simulated_data(
    base_folder: Path,
    afm_export_path: str = EXPORT_PATH,
    items_to_exclude: Optional[list[int]] = None,
    student_to_exclude: Optional[list[int]] = None,
    q_matrix_override: Optional[np.ndarray] = None,
) -> None:
    """
    Fit AFM to a simulated data of a single scenario

    It internally splits simulated data into 5 folds such that all responses
    of a single student fall in the same fold. All results are saved as CSV
    files in the scenario folder.

    :param base_folder: Path to a folder with simulated data
    :param afm_export_path: Name of CSV file to serialize AFM model
    :param items_to_exclude: Optional list of item ids to exclude from fitting
    :param student_to_exclude: Optional list of student ids to exclude from
                               fitting
    :param q_matrix_override: Optional Q-matrix to use instead of scenario's
                              ground truth one
    """
    if items_to_exclude is None:
        items_to_exclude = list()
    if student_to_exclude is None:
        student_to_exclude = list()

    log, true_alphas, true_betas, true_gammas, q_matrix, setting = load_scenario(
        base_folder
    )
    if q_matrix_override is not None:
        q_matrix = q_matrix_override

    num_students = len(log["student_id"].unique())
    num_items = len(log["item_id"].unique())
    student_ids = log["student_id"].to_numpy()
    item_ids = log["item_id"].to_numpy()
    learning_opportunities = AFM.compute_opportunities(
        student_ids, item_ids, q_matrix, num_students=num_students
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_student_ids = (
        log["student_id"]  # type: ignore
        .drop_duplicates()[lambda x: ~x.isin(student_to_exclude)]
        .unique()
    )

    scenario = base_folder.parts[-1]
    results = []
    predictions = []

    fold = 1
    for _, student_indices in kf.split(all_student_ids):
        selected_student_ids = all_student_ids[student_indices]

        selected_log_indices = log["student_id"].isin(selected_student_ids) & (
            ~log["item_id"].isin(items_to_exclude)
        )
        fold_log = log[selected_log_indices]
        fold_num_observations = len(fold_log)
        fold_student_ids = fold_log["student_id"].to_numpy()
        fold_item_ids = fold_log["item_id"].to_numpy()
        fold_learning_opportunities = learning_opportunities[selected_log_indices]
        fold_correct = fold_log["solved"].to_numpy()

        pd.Series([fold_log["cheating"].sum()], name="num_answers_cheated").to_csv(
            base_folder / f"fold_{fold}_cheating.csv", index=False
        )

        if items_to_exclude:
            export_path = Path(
                base_folder / EXCLUDED_ITEMS / (f"fold_{fold}_" + afm_export_path)
            )
            export_path.parent.mkdir(parents=True, exist_ok=True)
            selected_log_indices.to_csv(
                base_folder / EXCLUDED_ITEMS / f"fold_{fold}_indices.csv", index=False
            )
        else:
            export_path = Path(base_folder / (f"fold_{fold}_" + afm_export_path))
            selected_log_indices.to_csv(
                base_folder / f"fold_{fold}_indices.csv", index=False
            )

        ll, rmse, y_pred, afm = fit_afm_on_simulated_data_fold(
            num_students=num_students,
            num_items=num_items,
            q_matrix=q_matrix,
            num_observations=fold_num_observations,
            student_ids=fold_student_ids,
            item_ids=fold_item_ids,
            learning_opportunities=fold_learning_opportunities,
            correct=fold_correct,
            export_path=export_path,
            init_alphas=true_alphas if "gt_alpha" in str(base_folder) else None,
            fit_alphas="gt_alpha" not in str(base_folder),
        )

        y_pred_df = pd.DataFrame(y_pred, columns=["fitted"], index=fold_log.index)
        y_pred_df.to_csv(base_folder / f"fold_{fold}_predictions.csv")
        y_pred_df["fold"] = fold

        y_pred_ground_truth = get_ground_truth_afm_predictions(
            num_students=num_students,
            num_items=num_items,
            q_matrix=q_matrix,
            num_observations=fold_num_observations,
            student_ids=fold_student_ids,
            item_ids=fold_item_ids,
            learning_opportunities=fold_learning_opportunities,
            init_alphas=true_alphas,
            init_betas=true_betas,
            init_gammas=true_gammas,
        )

        y_pred_df["ground_truth"] = pd.Series(y_pred_ground_truth, index=fold_log.index)

        predictions.append(y_pred_df)

        results.append([scenario, fold, ll, rmse])

        fold += 1

    pd.DataFrame(results, columns=["scenario", "fold", "ll", "rmse"]).to_csv(
        base_folder / "results.csv"
    )

    pd.concat(predictions).sort_index().join(log).to_csv(
        base_folder / "predictions.csv"
    )


def get_ground_truth_afm_predictions(
    num_students: int,
    num_items: int,
    q_matrix: np.ndarray,
    num_observations: int,
    student_ids: np.ndarray,
    item_ids: np.ndarray,
    learning_opportunities: np.ndarray,
    init_alphas: np.ndarray,
    init_betas: np.ndarray,
    init_gammas: np.ndarray,
) -> np.ndarray:
    """Obtain predictions of AFM with ground truth parameter values"""
    afm = AFM.get_model(
        num_students=num_students,
        num_items=num_items,
        q_matrix=q_matrix,
        num_observations=num_observations,
        init_alphas=init_alphas,
        init_betas=init_betas,
        init_gammas=init_gammas,
        fit_alphas=False,
        fit_betas=False,
        fit_gammas=False,
        alpha_penalty=0.1,
    )

    y_pred = afm.predict(
        {
            "student": student_ids,
            "item": item_ids,
            "learning_opportunities": learning_opportunities,
        },
    )

    return y_pred


def fit_afm_on_simulated_data_fold(
    num_students: int,
    num_items: int,
    q_matrix: np.ndarray,
    num_observations: int,
    student_ids: np.ndarray,
    item_ids: np.ndarray,
    learning_opportunities: np.ndarray,
    correct: np.ndarray,
    export_path: Optional[Path] = None,
    init_alphas: Optional[np.ndarray] = None,
    init_betas: Optional[np.ndarray] = None,
    init_gammas: Optional[np.ndarray] = None,
    fit_alphas: bool = True,
    fit_betas: bool = True,
    fit_gammas: bool = True,
) -> tuple[float, float, np.ndarray, AFM]:
    """
    Fit AFM to a simulated data of a single scenario and a single fold
    """
    afm = AFM.get_model(
        num_students=num_students,
        num_items=num_items,
        q_matrix=q_matrix,
        num_observations=num_observations,
        init_alphas=init_alphas,
        init_betas=init_betas,
        init_gammas=init_gammas,
        fit_alphas=fit_alphas,
        fit_betas=fit_betas,
        fit_gammas=fit_gammas,
        alpha_penalty=0.1,
    )

    afm.fit(
        {
            "student": student_ids,
            "item": item_ids,
            "learning_opportunities": learning_opportunities,
        },
        correct,
        batch_size=num_observations,
        epochs=3000,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50),
        ],
        verbose=0,
    )
    if export_path is not None:
        afm.save_variables(export_path)

    y_pred = afm.predict(
        {
            "student": student_ids,
            "item": item_ids,
            "learning_opportunities": learning_opportunities,
        },
    )

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    rmse = tf.keras.metrics.RootMeanSquaredError()  # type: ignore

    log_likelihood: float = bce(correct, y_pred).numpy()  # type: ignore
    rmse_value: float = rmse(correct, y_pred).numpy()  # type: ignore

    print(f"Results:\nLL:{log_likelihood}\nRMSE:{rmse_value}\n\n")

    return log_likelihood, rmse_value, y_pred, afm
