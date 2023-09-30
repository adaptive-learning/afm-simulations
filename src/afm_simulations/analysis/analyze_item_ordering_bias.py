"""Reproduces Figure 5.3"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore

from afm_simulations.afm.common import fit_all_scenarios, load_specific_params
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

    relevant_params["lineplot_x"] = relevant_params["KC"] * 100 - 50

    relevant_params["KC"] += 1
    relevant_params["parameter_type"] = (
        relevant_params["parameter_type"].str.replace("beta", "β").str.replace("gamma", "γ")
    )
    relevant_params["case"] = relevant_params["case"].str.replace("_", " ").str.replace("^in", "fixed", regex=True)

    g = sns.FacetGrid(
        relevant_params,
        col="KC",
        row="parameter_type",
        legend_out=True,
        sharex=True,
        sharey="row",
        aspect=0.75,
    )

    g.map_dataframe(lambda color, data: plt.axhline(y=data["true_value"].iloc[0], color="k", linestyle=":", zorder=3))

    g.map_dataframe(
        sns.pointplot,
        x="KC",
        y="value",
        hue="case",
        hue_order=["fixed order", "random order"],
        join=False,
        dodge=0.5,
        palette="tab10",
    )

    g.add_legend()

    g.set_axis_labels("KC", "Parameter value")
    g.set_titles(template="${row_name}_{col_name}$")

    g.set_xlabels("")
    g.set_xticklabels([])

    sns.move_legend(g, "center right", bbox_to_anchor=(0.84, 0.62))

    image_name = "fig/item_ordering_bias_cumulative_beta_gamma.svg"
    plt.savefig(
        image_name,
        bbox_inches="tight",
        dpi=600,
    )
    print(f"Saved image to {image_name}")
    plt.clf()
    plt.close()


def analyze_beta_gamma(params_df: pd.DataFrame) -> None:
    params_df = add_scenario_metadata(params_df)

    plot_beta_gamma(params_df)


def add_scenario_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df["base_scenario"] = df["scenario"].apply(lambda x: "_".join(x.split("_")[2:]))
    df["case"] = df["scenario"].apply(lambda x: "_".join(x.split("_")[:2]))

    return df


def main() -> None:
    run_simulation(
        [
            "../scenarios/afm_item_ordering_bias.json",
        ]
    )

    relevant_scenarios = list(Path().glob("data/*_cumulative_KC*"))

    fit_all_scenarios(relevant_scenarios)
    params_df = load_specific_params(relevant_scenarios)
    analyze_beta_gamma(params_df)


if __name__ == "__main__":
    main()
