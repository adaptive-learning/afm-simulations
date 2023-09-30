"""
Reproduces Figure 5.4 and 5.5
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

from afm_simulations.afm.common import (
    fit_all_scenarios,
    load_specific_params,
    load_specific_results,
)
from afm_simulations.simulations.common import run_simulation

matplotlib.use("agg")

sns.set()


def plot_beta_gamma(params: pd.DataFrame) -> None:
    relevant_params = (
        params[
            (params["name"].str.contains("beta")) | (params["name"].str.contains("gamma"))  # keep beta and gamma params
        ]
        .drop_duplicates(subset=["name", "scenario", "value"])
        .sort_values(["scenario", "name"])
        .assign(true_value=lambda df: df["true_value"].fillna(method="ffill"))
        .reset_index(drop=True)
    )

    relevant_params["parameter_type"] = relevant_params["name"].apply(lambda s: s[:-1])
    relevant_params["KC"] = relevant_params["name"].apply(lambda s: int(s[-1:]))

    relevant_params["lineplot_x"] = relevant_params["KC"].apply(lambda x: (x * 100) - 50)

    relevant_params["KC"] += 1
    relevant_params["parameter_type"] = (
        relevant_params["parameter_type"].str.replace("beta", "β").str.replace("gamma", "γ")
    )
    relevant_params["case"] = relevant_params["case"].str.replace("_", " ")

    g = sns.FacetGrid(
        relevant_params,
        col="base_scenario",
        row="parameter_type",
        legend_out=True,
        sharex=True,
        sharey="row",
    )

    g.map_dataframe(sns.lineplot, x="lineplot_x", y="true_value", color="k", linestyle=":")

    g.map_dataframe(
        sns.pointplot,
        x="KC",
        y="value",
        hue="case",
        hue_order=["well specified", "misspecified"],
        join=False,
        dodge=0.5,
        palette="tab10",
    )

    g.add_legend()

    g.set_axis_labels("KC", "Parameter value")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    image_name = "fig/item_ordering_bias_misspecified_beta_gamma.svg"
    plt.savefig(
        image_name,
        bbox_inches="tight",
        dpi=600,
    )
    print(f"Saved image to {image_name}")
    plt.clf()
    plt.close()


def plot_bars(results_df: pd.DataFrame, metric: str) -> None:
    ax = sns.barplot(data=results_df, x="scenario", y=metric, hue="case")
    if metric != "BIC":
        ax.legend_.remove()
    metric_min, metric_max = results_df[metric].min(), results_df[metric].max()
    ax.set_ylim(metric_min - 0.01 * metric_min, metric_max + 0.01 * metric_min)
    image_name = f"fig/results_misspecified_{metric}_comparison.svg"
    plt.savefig(image_name, bbox_inches="tight", dpi=600)
    print(f"Saved image to {image_name}")
    plt.clf()
    plt.close()


def analyze_beta_gamma(params_df: pd.DataFrame) -> None:
    params_df = add_scenario_metadata(params_df)

    plot_beta_gamma(params_df)


def analyze_metrics(results_df: pd.DataFrame) -> None:
    results_df = add_scenario_metadata(results_df)

    num_observation = 1000 * 20
    results_df["ll"] *= num_observation
    results_df["AIC"] = results_df.apply(
        lambda row: 2 * (1000 + (2 if row["case"] == "well_specified" else 4)) + 2 * row["ll"],
        axis=1,
    )
    results_df["BIC"] = results_df.apply(
        lambda row: np.log(num_observation) * (1000 + (2 if row["case"] == "well_specified" else 4)) + 2 * row["ll"],
        axis=1,
    )

    results_df["case"] = results_df["case"].str.replace("_", " ")
    results_df.drop(columns="scenario", inplace=True)
    results_df.rename(
        columns={"ll": "Negative log-likelihood", "rmse": "RMSE", "base_scenario": "scenario"}, inplace=True
    )

    for metric in ["Negative log-likelihood", "RMSE", "AIC", "BIC"]:
        plot_bars(results_df, metric)


def add_scenario_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df["base_scenario"] = (
        df["scenario"]
        .apply(lambda x: "_".join(x.split("_")[:4]))
        .str.replace("ordering_bias_", "")
        .str.replace("_", " ")
        .str.replace("in", "fixed")
    )
    df["case"] = df["scenario"].apply(lambda x: "_".join(x.split("_")[4:]))

    return df


def main() -> None:
    run_simulation(
        [
            "../scenarios/afm_item_ordering_misspecified.json",
        ]
    )

    relevant_scenarios = list(Path().glob("data/ordering_bias_*specified"))

    misspecified_q_matrix = np.array(
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    q_matrix_overrides = {
        "ordering_bias_in_order_misspecified": misspecified_q_matrix,
        "ordering_bias_random_order_misspecified": misspecified_q_matrix,
    }

    fit_all_scenarios(relevant_scenarios, q_matrix_overrides=q_matrix_overrides)

    results_df = load_specific_results(relevant_scenarios)
    analyze_metrics(results_df)

    params_df = load_specific_params(relevant_scenarios)
    analyze_beta_gamma(params_df)


if __name__ == "__main__":
    main()
