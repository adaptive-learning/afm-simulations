"""Reproduces Figure 4.3"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from scipy.special import expit  # type: ignore

from afm_simulations.afm.common import (
    fit_all_scenarios,
    load_fitted_scenario,
    load_fold,
)
from afm_simulations.simulations.common import run_simulation

matplotlib.use("agg")

sns.set()

LOG_PATH = "log.csv"
Q_MATRIX_PATH = "q_matrix.csv"
PARAMS_PATH = "params.csv"
EXPORT_PATH = "fitted_params_tf.csv"


def analyze_path(path: Path) -> None:
    scenario = path.name
    base_folder = path
    Path(f"fig/{scenario}").mkdir(parents=True, exist_ok=True)

    (
        log,
        true_alphas,
        true_betas,
        true_gammas,
        q_matrix,
        setting,
        learning_opportunities,
        predictions,
    ) = load_fitted_scenario(base_folder)

    df = load_fold(
        base_folder, 1, scenario
    )  # needed only for ground truth parameters that are all present in all folds

    plot_learning_curves(predictions, learning_opportunities, q_matrix, df, scenario)


def plot_learning_curves(
    predictions: pd.DataFrame,
    learning_opportunities: pd.DataFrame,
    q_matrix: np.ndarray,
    params_df: pd.DataFrame,
    scenario: str,
) -> None:
    print("Plotting learning curves for", scenario)
    item_knowledge_component_pairs = (
        pd.DataFrame(q_matrix, columns=[f"k{i}" for i in range(q_matrix.shape[1])])
        .reset_index()
        .rename(columns={"index": "item_id"})
        .melt(id_vars="item_id", var_name="KC", value_name="present")[lambda x: x["present"] == 1]
        .drop(columns="present")
    )

    # add opportunities to each attempt
    predictions_with_learning = pd.concat([predictions, learning_opportunities], axis=1)

    # set global identification whether a student is a cheater (i.e. has cheated at least once)
    is_cheater = predictions.groupby("student_id")["cheating"].max().rename("is_cheater")

    betas = params_df[params_df["name"].str.contains("beta")][["name", "value"]]
    betas["name"] = betas["name"].str.replace("beta", "k")
    betas.columns = ["KC", "beta"]  # type: ignore

    gammas = params_df[params_df["name"].str.contains("gamma")][["name", "value"]]
    gammas["name"] = gammas["name"].str.replace("gamma", "k")
    gammas.columns = ["KC", "gamma"]  # type: ignore

    qs = q_matrix[predictions_with_learning["item_id"]]  # Q-matrix row for a given item
    easiness_part = qs @ betas["beta"].to_numpy()  # q_i * Beta
    learning_part = np.einsum("ij,ij->i", learning_opportunities, gammas["gamma"].to_numpy() * qs)  # q_i * Gamma * t

    predictions_with_learning["fitted α=0"] = expit(easiness_part + learning_part)

    plot_data = pd.concat(
        [
            (
                predictions_with_learning.melt(
                    id_vars=list(predictions.columns) + ["fitted α=0"],
                    value_vars=learning_opportunities.columns,  # type: ignore
                    var_name="KC",
                    value_name="opportunity",
                )
                .melt(
                    id_vars=[
                        "fold",
                        "student_id",
                        "item_id",
                        "cheating",
                        "KC",
                        "opportunity",
                    ],
                    var_name="prediction_type",
                    value_name="probability",
                )
                .merge(is_cheater, left_on="student_id", right_index=True)
                .merge(
                    item_knowledge_component_pairs, on=["item_id", "KC"]
                )  # keep only rows where the predictions actually include the KC we are interested in
            ),
        ]
    ).reset_index(drop=True)

    prediction_labels = {
        "fitted": "fitted",
        "ground_truth": "ground truth",
        "solved": "observed",
        "fitted α=0": "fitted $\\alpha = 0$",
    }
    plot_data["prediction_type"] = plot_data["prediction_type"].apply(prediction_labels.get)

    is_cheater_labels = {
        0: "Not cheating",
        1: "Cheating",
    }
    plot_data["is_cheater"] = plot_data["is_cheater"].apply(is_cheater_labels.get)

    kc_labels = {
        "k0": "KC1",
        "k1": "KC2",
    }
    plot_data["KC"] = plot_data["KC"].apply(kc_labels.get)

    plot_data_copy = plot_data.copy()
    plot_data_copy["is_cheater"] = "All"
    plot_data = pd.concat([plot_data, plot_data_copy], ignore_index=True)

    g = sns.FacetGrid(
        plot_data,
        row="KC",
        col="is_cheater",
        col_order=["All", "Not cheating", "Cheating"],
        margin_titles=True,
    )
    g.map_dataframe(
        sns.lineplot,
        x="opportunity",
        y="probability",
        hue="prediction_type",
        hue_order=[
            "fitted",
            "fitted $\\alpha = 0$",
            "observed",
            "ground truth",
        ],
    )
    g.add_legend()
    g.set_titles(
        col_template="{col_name}",
        row_template="{row_name}",
    )
    g.set(xticks=[0, 2, 4, 6, 8])

    g.axes[0, 0].set_ylabel("Correct answer probability")

    g.axes[1, 1].set_xlabel("Practice opportunities")
    g.axes[1, 0].set_xlabel("")
    g.axes[1, 2].set_xlabel("")

    g.fig.subplots_adjust(top=0.9)
    image_name = f"fig/{scenario}/learning_curves.svg"
    g.savefig(image_name)
    print(f"Saved image to {image_name}")


if __name__ == "__main__":
    run_simulation(
        [
            "../scenarios/cheating_combined_thirds_simple.json",
        ]
    )

    relevant_scenarios = list(Path().glob("data/cheating_combined_thirds_50*"))

    fit_all_scenarios(relevant_scenarios)
    analyze_path(Path("data/cheating_combined_thirds_50"))
