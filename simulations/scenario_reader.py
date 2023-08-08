import json
from typing import Dict, Any
from copy import deepcopy

import numpy as np

SETTING = Dict[str, Any]

"""
SETTINGS
"""

"""
Initial distributions
"""


def normal_distribution(n, mean, std):
    return np.random.normal(mean, std, n)


def uniform_distribution(n, min, max):
    return np.random.uniform(min, max, n)


def predefined(n, values):
    return np.array(values[:n], dtype=float)


DISTRIBUTIONS = {
    "normal": normal_distribution,
    "uniform": uniform_distribution,
    "predefined": predefined,
}

"""
Learning modes
"""

NO_LEARNING = {
    "state_updater_after_item_chain": [],
}

CONSTANT_LEARNING = {
    "state_updater_after_item_chain": ["constant_learning"],
}

OPPORTUNITIES_LEARNING = {
    "state_updater_after_item_chain": ["learning_opportunities"],
}

STEP_LEARNING = {
    "state_updater_after_item_chain": ["step_learning"],
    "learn_prob": 0.05,
    "max_gain": 6,
}

STEEP_LEARNING = {
    "state_updater_after_item_chain": ["steep_learning"],
    "learning_period": 10,
    "max_gain": 6,
}

"""
Student pass modes
"""

IN_ORDER_PASS = {
    "item_selector_chain": ["first_item"],
}

RANDOM_PASS = {
    "item_selector_chain": ["random_item"],
}

ADAPTATION_PASS = {
    "initializer_before_student_chain": ["scramble_and_sort"],
    "item_selector_chain": ["first_item"],
    "state_updater_after_student_chain": ["update_b_estimates"],
    "adaptation_after": 20,
}

FIRST_K_RANDOM_PASS = {
    "initializer_before_student_chain": ["scramble_and_sort"],
    "item_selector_chain": ["first_item"],
    "state_updater_after_student_chain": ["update_b_estimates_once"],
    "first_k": 20,
}

EPSILON_GREEDY_PASS = {
    "initializer_before_student_chain": ["scramble_and_sort", "epsilon"],
    "item_selector_chain": ["first_item"],
    "state_updater_after_student_chain": ["update_b_estimates_epsilon"],
    "epsilon": 0.05,
}

IN_ORDER_PASS_RANDOM_SKIP = {"item_selector_chain": ["first_item_random_skip"]}

"""
Other modifications
"""

ATTRITION = {
    "initializer_before_student_chain": ["set_remaining_time"],
    "item_solver_chain": ["attrition"],
    "practice_terminator_chain": ["attrition_termination"],
    "remaining_solve_time": lambda true_b: sum(true_b) * 0.6,
}

EARLY_STOPPING = {
    "initializer_before_student_chain": ["set_remaining_items"],
    "item_solver_chain": ["early_stopping"],
    "practice_terminator_chain": ["max_items_terminator"],
}

MASTERY = {
    "initializer_before_student_chain": ["initialize_streak"],
    "item_solver_chain": ["mastery"],
    "practice_terminator_chain": ["mastery_termination"],
}

CHEATING = {
    "initializer_before_student_chain": ["initialize_cheater"],
    "item_solver_chain": ["cheating"],
    "state_updater_after_item_chain": ["switch_cheater_state"],
    # "item_solver_chain": ["cheating_compromised_items"],
}

"""
Option dictionaries for easier configuration
"""

ORDERS = {
    "in order": IN_ORDER_PASS,
    "random order": RANDOM_PASS,
    "adaptation order": ADAPTATION_PASS,
    "first k random": FIRST_K_RANDOM_PASS,
    "epsilon greedy": EPSILON_GREEDY_PASS,
    "in order random skip": IN_ORDER_PASS_RANDOM_SKIP,
}

LEARNING_MODES = {
    "no": NO_LEARNING,
    "constant": CONSTANT_LEARNING,
    "step": STEP_LEARNING,
    "steep": STEEP_LEARNING,
    "opportunities": OPPORTUNITIES_LEARNING,
}


class ScenarioReader:
    general_keys = {
        "model",
        "order",
        "learning",
        "student_count",
        "item_count",
        "seed",
        "attrition",
        "mastery",
    }

    afm_settings_keys = {
        "alpha_distribution",
        "alpha_distribution_kwargs",
        "beta_distribution",
        "beta_distribution_kwargs",
        "gamma_distribution",
        "gamma_distribution_kwargs",
        "q_matrix",
    }

    mastery_keys = {"mastery_streak"}

    def __init__(self, scenarios_path: str):
        self.scenarios_path = scenarios_path
        self._check_json()
        self.scenarios = self._load_scenarios()

    def _check_json(self):
        config = json.load(open(self.scenarios_path))

        general = config.get("general")
        if general is None:
            raise ValueError("General setting is missing in scenarios file!")

        scenarios = config.get("scenarios")
        if scenarios is None or len(scenarios) == 0:
            raise ValueError("Scenarios description are missing in scenarios file!")

        for scenario_name in scenarios:
            complete_scenario = {**general, **scenarios[scenario_name]}
            all_keys = set(complete_scenario.keys())

            self._check_keys(scenario_name, self.general_keys, all_keys)
            if complete_scenario["model"] == "afm":
                self._check_keys(scenario_name, self.afm_settings_keys, all_keys)
            if complete_scenario["mastery"]:
                self._check_keys(scenario_name, self.mastery_keys, all_keys)

    @staticmethod
    def _check_keys(scenario_name, expected, actual):
        for key in expected:
            if key not in actual:
                raise ValueError(f"Scenario {scenario_name} is missing {key} setting.")

    def _load_scenarios(self) -> Dict[str, Any]:
        scenarios = dict()

        config = json.load(open(self.scenarios_path))
        general = config["general"]

        for scenario_name in config["scenarios"]:
            scenarios[scenario_name] = {**general, **config["scenarios"][scenario_name]}

        return scenarios

    def get_scenario_names(self):
        return list(self.scenarios.keys())

    def get_simulation_setting(self, scenario_name: str) -> SETTING:
        setting = self.scenarios[scenario_name]
        setting["scenario_name"] = scenario_name
        # item order
        setting.update(**ORDERS[setting["order"]])
        # student learning
        setting.update(**LEARNING_MODES[setting["learning"]])

        setting["practice_terminator_chain"] = ["available_item"]

        if setting["model"] == "afm":
            setting["item_solver_chain"] = ["afm"]
            for param in "alpha", "beta", "gamma":
                setting[f"{param}_distribution"] = DISTRIBUTIONS[
                    setting[f"{param}_distribution"]
                ]

        if setting.get("cheating", False):
            setting = self._modify_dict(setting, CHEATING)
        if setting.get("mastery", False):
            setting = self._modify_dict(setting, MASTERY)
        if setting.get("attrition", False):
            setting = self._modify_dict(setting, ATTRITION)
        return setting

    @staticmethod
    def _modify_dict(dict1: SETTING, dict2: SETTING) -> SETTING:
        modified = {**dict1}
        for key, value in dict2.items():
            if key not in modified:
                modified[key] = value
            else:
                assert type(value) == list, "Dictionaries contain contradicting keys"
                modified[key] = deepcopy(modified[key]) + value
        return modified
