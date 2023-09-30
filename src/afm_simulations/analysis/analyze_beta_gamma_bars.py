"""Reproduces Figure 4.1"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore

from afm_simulations.afm.common import fit_all_scenarios, load_specific_params
from afm_simulations.simulations.common import run_simulation

matplotlib.use("agg")

sns.set()

LOG_PATH = "log.csv"
Q_MATRIX_PATH = "q_matrix.csv"
PARAMS_PATH = "params.csv"
EXPORT_PATH = "fitted_params_tf.csv"
EXCLUDED_ITEMS = "excluded_items"

SCENARIO_BASE_PROPER_NAMES = {
    "cheating_combined_thirds_gt_alpha": "⅓ overlap GT $\\alpha$ ",
    "cheating_combined_thirds": "⅓ overlap",
    "cheating_combined_0": "no overlap",
}


def plot_beta_gamma_bars(params: pd.DataFrame, tag=None) -> None:
    relevant_params = (
        params[
            (params["name"].str.contains("beta")) | (params["name"].str.contains("gamma"))  # keep beta and gamma params
        ]
        .drop_duplicates(subset=["name", "scenario", "value"])
        .reset_index(drop=True)
    )

    relevant_params["base_scenario_name"] = relevant_params["scenario"].apply(lambda s: s[:-3])
    relevant_params["cheating_prevalence"] = relevant_params["cheating_prevalence"].apply(lambda x: int(x * 100))
    relevant_params["lineplot_x"] = relevant_params["cheating_prevalence"].apply(lambda x: (x * 100) - 100)

    relevant_params["parameter_type"] = relevant_params["name"].apply(lambda s: s[:-1])

    relevant_params["KC"] = relevant_params["name"].apply(lambda s: s[-1:])

    relevant_params["base_scenario_name"] = relevant_params["base_scenario_name"].apply(SCENARIO_BASE_PROPER_NAMES.get)

    g = sns.FacetGrid(
        relevant_params,
        col="KC",
        row="parameter_type",
        row_order=["beta", "gamma"],
        legend_out=True,
        sharex=True,
        sharey="row",
        height=3,
        aspect=1,
    )
    g.map_dataframe(sns.lineplot, x="lineplot_x", y="true_value", color="k", linestyle=":", zorder=3)

    g.map_dataframe(
        sns.pointplot,
        x="cheating_prevalence",
        y="value",
        hue="base_scenario_name",
        hue_order=[v for _, v in SCENARIO_BASE_PROPER_NAMES.items()],
        join=False,
        dodge=0.5,
        palette="tab10",
    )

    g.add_legend()

    g.set_axis_labels("Cheater prevalence (%)", "Fitted parameter value")

    axes = g.axes.flatten()
    axes[0].set_title("$\\beta_1$")
    axes[1].set_title("$\\beta_2$")
    axes[2].set_title("$\\gamma_1$")
    axes[3].set_title("$\\gamma_2$")

    image_name = f"fig/cheating_beta_gamma_bars{'_' + tag if tag else ''}.svg"
    plt.savefig(image_name)
    plt.clf()
    print(f"Saved image to {image_name}")


def analyze_alpha_distributions(params_df: pd.DataFrame) -> None:
    for _, (betas, gammas) in list(params_df[["betas", "gammas"]].drop_duplicates().iterrows()):
        tag = f"{betas}-{gammas}"
        relevant_params_df = params_df[(params_df["betas"] == betas) & (params_df["gammas"] == gammas)]
        plot_beta_gamma_bars(
            relevant_params_df,
            tag=tag,
        )


if __name__ == "__main__":
    run_simulation(
        [
            "../scenarios/cheating_combined_0.json",
            "../scenarios/cheating_combined_thirds_simple.json",
        ]
    )

    relevant_scenarios = list(Path().glob("data/cheating_combined_thirds_[025g]*")) + list(
        Path().glob("data/cheating_combined_0*")
    )

    fit_all_scenarios(relevant_scenarios)
    params_df = load_specific_params(relevant_scenarios)
    analyze_alpha_distributions(params_df)
