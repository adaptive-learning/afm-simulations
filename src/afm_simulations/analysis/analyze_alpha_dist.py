from itertools import repeat
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from afm_tf2 import common

matplotlib.use("agg")

sns.set()

LOG_PATH = "log.csv"
Q_MATRIX_PATH = "q_matrix.csv"
PARAMS_PATH = "params.csv"
EXPORT_PATH = "fitted_params_tf.csv"
EXCLUDED_ITEMS = "excluded_items"

SCENARIO_BASE_PROPER_NAMES = {
    "debug_cheating_combined_thirds_gt_alpha": "⅓ overlap GT $\\alpha$ ",
    "debug_cheating_combined_thirds": "⅓ overlap",
    "debug_cheating_combined_0": "no overlap",
}


def plot_hist(params, name, scenario=None, tag=None, separate_folds=False):
    relevant_params = (
        params[params["name"].str.contains(name)][lambda x: x["value"] != 0]
        .drop_duplicates(subset=["name", "scenario", "value"])
        .reset_index(drop=True)
    )

    # relevant_params = relevant_params[
    #     (~relevant_params["scenario"].str.contains("gt_alpha"))
    #     | (relevant_params["fold"] == 1)
    # ].reset_index(drop=True)

    relevant_params = relevant_params.reset_index(drop=True)

    relevant_params["base_scenario_name"] = relevant_params["scenario"].apply(
        lambda s: s[:-3]
    )

    relevant_params["base_scenario_name_plus_fold"] = (
        relevant_params["base_scenario_name"] + "_" + relevant_params["fold"].apply(str)
    )

    if separate_folds:
        row_faceting_col_name = "base_scenario_name_plus_fold"
    else:
        row_faceting_col_name = "base_scenario_name"

    relevant_params["base_scenario_name"] = relevant_params["base_scenario_name"].apply(
        SCENARIO_BASE_PROPER_NAMES.get
    )

    if "cheating_prevalence" in relevant_params.columns:
        relevant_params["cheating_prevalence"] = (
            relevant_params["cheating_prevalence"] * 100
        ).astype(int)

        relevant_params["cheater"] = relevant_params["cheater"].apply(
            lambda x: "Yes" if x else "No"
        )

        # g = sns.displot(
        #     data=relevant_params,
        #     x="value",
        #     hue="cheater",
        #     hue_order=["No", "Yes"],
        #     multiple="stack",
        #     kde=True,
        #     col="cheating_prevalence",
        #     row=row_faceting_col_name,
        #     kind="hist",
        #     bins=30,
        # )

        g = sns.FacetGrid(
            relevant_params,
            col="cheating_prevalence",
            row=row_faceting_col_name,
            # row_order=SCENARIO_BASE_PROPER_NAMES.items(),
            legend_out=True,
            sharex="col",
            sharey="none",
            margin_titles=True,
            # height=5,
            aspect=1.2,
            gridspec_kws={"wspace": 0.1, "hspace": 0.1},
        )

        g.map_dataframe(
            sns.histplot,
            x="value",
            hue="cheater",
            hue_order=["No", "Yes"],
            multiple="stack",
            kde=True,
            bins=10,
        )

        # The color cycles are going to all the same, doesn't matter which axes we use
        ax = g.axes[0][1]

        # Some how for a plot of 5 bars, there are 6 patches, what is the 6th one?
        boxes = [
            item
            for item in ax.get_children()
            if isinstance(item, matplotlib.patches.Rectangle)
        ][:-1]

        # There is no labels, need to define the labels
        legend_labels = ["No", "Yes"]

        # Create the legend patches
        legend_patches = [
            matplotlib.patches.Patch(color=C, label=L)
            for C, L in zip(
                [
                    boxes[-1].get_facecolor(),
                    boxes[0].get_facecolor(),
                ],
                legend_labels,
            )
        ]

        g.add_legend(
            legend_data={"No": legend_patches[0], "Yes": legend_patches[1]},
            title="Cheating",
        )
        g.set(yticks=[])

        for excluded_cheater_value, color, linestyle in zip(
            ["No", "Yes", 10], ["C1", "C0", "red"], repeat("--")
        ):
            # value 10 is a hack to include both cheaters and non-cheaters
            for base in g.row_names:
                for prevalence in g.col_names:
                    mean = relevant_params[
                        (relevant_params[row_faceting_col_name] == base)
                        & (relevant_params["cheating_prevalence"] == prevalence)
                        & (relevant_params["cheater"] != excluded_cheater_value)
                    ]["value"].mean()

                    g.axes_dict[(base, prevalence)].axvline(
                        x=mean, color=color, linestyle=linestyle, linewidth=2
                    )

        # g.set_axis_labels(f"Value of parameter {name}", f"Estimated density")
        g.set_axis_labels(f"$\\alpha$ value", "")
        g.set_yticklabels()
        g.set_titles(
            # template="{row_name} @ {col_name} % cheaters",
            col_template="{col_name} % cheaters",
            row_template="{row_name}",
        )
    else:
        sns.displot(
            data=relevant_params,
            x="value",
        )

    if scenario is not None:
        plt.savefig(f"fig/{scenario}/{name}_dist.svg")
    else:
        plt.savefig(
            f"fig/cheating_{name}_dist{'_' + tag if tag else ''}.svg",
            bbox_inches="tight",
        )
    # plt.show()
    plt.clf()


def analyze_alpha_distributions(params_df: pd.DataFrame) -> None:
    for _, (betas, gammas) in list(
        params_df[["betas", "gammas"]].drop_duplicates().iterrows()
    ):
        tag = f"{betas}-{gammas}"
        relevant_params_df = params_df[
            (params_df["betas"] == betas) & (params_df["gammas"] == gammas)
        ]
        plot_hist(
            relevant_params_df,
            "alpha",
            tag=tag,
        )


if __name__ == "__main__":
    # common.fit_all_in_path(Path("./data"))
    params_df = common.load_params(Path("./data"))
    analyze_alpha_distributions(params_df)
