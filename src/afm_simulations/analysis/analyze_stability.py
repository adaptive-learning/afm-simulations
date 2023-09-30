from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

from afm_simulations.afm.common import fit_all_scenarios, load_specific_params
from afm_simulations.simulations.common import run_simulation

matplotlib.use("agg")

sns.set()


def plot_convergence_2d(log: pd.DataFrame, base_scenario: str) -> None:
    beta_tex = r"\beta_"
    gamma_tex = r"\gamma_"
    true_values = log[["name", "true_value"]].drop_duplicates().sort_values("name")
    true_values_str = "; ".join(
        [
            f"${row['name'].replace('beta', beta_tex).replace('gamma', gamma_tex)}$={row['true_value']}"
            for _, row in true_values.iterrows()
        ]
    )

    for parameter_name in log["name"].unique():
        relevant_log = log[log["name"] == parameter_name]
        true_value = relevant_log["true_value"].iloc[0]
        pivot_means = relevant_log.pivot_table(
            index="num_students",
            columns="num_items",
            values="value",
            aggfunc="mean",
        )

        pivot_ci = relevant_log.pivot_table(
            index="num_students",
            columns="num_items",
            values="value",
            aggfunc=lambda a: tuple(sns.utils.ci(sns.algorithms.bootstrap(a), which=95)),
        )

        pivot_ci_size = pivot_ci.applymap(
            lambda ci: abs(ci[1] - ci[0]) if true_value == 0 else abs(ci[1] - ci[0]) / true_value * 100
        )

        labels = np.array(
            [
                [
                    f"{pivot_means.iloc[i, j]:.3f}\n({pivot_ci.iloc[i, j][0]:.3f}–{pivot_ci.iloc[i, j][1]:.3f})"
                    for j in range(pivot_means.shape[1])
                ]
                for i in range(pivot_means.shape[0])
            ]
        )

        ax = sns.heatmap(
            pivot_ci_size,
            vmin=0,
            vmax=0.1 if true_value == 0 else 100,
            annot=labels,
            annot_kws={"size": 7},
            fmt="",
            cmap="viridis",
            # cmap="vlag",
            cbar_kws={
                "label": f'{"Absolute" if true_value == 0 else "Relative"} 95% CI size'
                f'{"" if true_value == 0 else " [%]"}'
            },
        )
        ax.set_xlabel("Number of items")
        ax.set_ylabel("Number of students")

        ax.set_title(
            f"Estimates for parameter ${parameter_name.replace('beta', beta_tex).replace('gamma', gamma_tex)}$\n"
            f"True parameter values: {true_values_str}"
        )
        image_name = f"fig/{base_scenario}/{parameter_name}_stability_heat.svg"
        plt.savefig(image_name)
        print(f"Saved image to {image_name}")
        plt.clf()
        plt.close()


def scenario_stability(log: pd.DataFrame, base_scenario: str) -> None:
    log["num_students"] = log["scenario"].str.replace(r".*_(\d+)s_.*", r"\1", regex=True).apply(int)
    log["num_items"] = log["scenario"].str.replace(r".*s_(\d+)i$", r"\1", regex=True).apply(int)

    plot_convergence_2d(
        log[(log["name"].str.contains("beta")) | (log["name"].str.contains("gamma"))],
        base_scenario,
    )


def analyze_stability(params_df: pd.DataFrame) -> None:
    base_scenarios = {"_".join(scenario.split("_")[:-2]) for scenario in params_df["scenario"].unique()}

    for base_scenario in base_scenarios:
        Path(f"fig/{base_scenario}").mkdir(parents=True, exist_ok=True)
        scenarios = params_df["scenario"].drop_duplicates()[lambda x: x.str.contains(base_scenario)]  # type: ignore
        scenario_stability(params_df[params_df["scenario"].isin(scenarios)].copy(), base_scenario)


if __name__ == "__main__":
    package_root = Path(__file__).parent.parent
    run_simulation(
        [
            package_root / "scenarios/afm_stability_single_kc.json",
        ]
    )

    relevant_scenarios = list(Path().glob("data/afm_stability_single_KC*"))

    fit_all_scenarios(relevant_scenarios)
    params_df = load_specific_params(relevant_scenarios)
    analyze_stability(params_df)
