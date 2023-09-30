from pathlib import Path
from typing import Sequence, Union

from .scenario_reader import ScenarioReader
from .simulation import Simulation


def run_simulation(
    scenario_jsons: Sequence[Union[str, Path]],
    plot_results: bool = False,
    force_regenerate: bool = False,
) -> None:
    """
    Generate simulated log data

    :param scenario_jsons: A collection of string paths to JSON files with
                           scenarios to simulate
    :param plot_results: Plots aggregated simulations statistics when True
    :param force_regenerate: Regenerate simulation data even if they already exist
    """
    for scenario_json in scenario_jsons:
        reader = ScenarioReader(scenario_json)

        for scenario_name in reader.get_scenario_names():
            if Path(f"./data/{scenario_name}").exists() and not force_regenerate:
                print(f"Folder with data for scenario {scenario_name} already exists; skipping...")
                continue

            scenario_setting = reader.get_simulation_setting(scenario_name)

            simulation = Simulation(scenario_setting)
            simulation.run()

            if plot_results:
                simulation.plot_attempts_heatmap()
                simulation.plot_item_success_rates()

            simulation.export_afm_simulation()
