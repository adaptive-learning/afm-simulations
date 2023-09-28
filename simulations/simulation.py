import json
import os
import random
from copy import deepcopy
from math import log
from typing import Any, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm

matplotlib.use("agg")

SETTING = dict[str, Any]


class Item:
    def __init__(self, id: int, concepts: list[int]) -> None:
        self.id = id
        self.concepts = concepts


class Student:
    def __init__(self, id: int, skill: float, opportunities: list[int]) -> None:
        self.id = id
        self.skill = skill
        self.opportunities = opportunities


class Simulation:
    def __init__(self, setting: SETTING) -> None:
        self.log_id: int = 0
        self.students: list[Student] = []
        self.items: list[Item] = []
        self.true_b = None
        self.means = None
        self.available_items: list[Item] = []
        self.active_student: Optional[Student] = None
        self.active_item: Optional[Item] = None
        self.item_order = None
        self.remaining_solve_time: Optional[float] = None
        self.solve_time: Optional[float] = None
        self.log: Optional[Union[pd.DataFrame, np.ndarray]] = None
        # self.order_log = None
        self.random_students_id: list[int] = []
        self.setting: SETTING = deepcopy(setting)
        self.alpha: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.gamma: Optional[np.ndarray] = None
        self.q_matrix: Optional[np.ndarray] = None
        self.solved: Optional[bool] = None
        self.streak: Optional[int] = None
        self.remaining_items: Optional[int] = None
        self.scenario_name: str = self.setting["scenario_name"]

    def _chain_executor(self, chain: Sequence[str]) -> None:
        """
        Execute all methods in the chain

        Input sequence must contain valid Simulation class method names. The
        methods are executed in the order they appear in the sequence.

        :param chain: Sequence of method names
        """
        for method_name in chain:
            getattr(self, method_name)()

    def _chain_executor_cumulative(self, chain: Sequence[str]) -> Any:
        """
        Execute all methods in the chain and accumulate results

        Input sequence must contain valid Simulation class method names. The
        methods are executed in the order they appear in the sequence. The
        result of each method is accumulated in `result` variable that is
        also available to each method as input.

        :param chain: Sequence of method names
        """
        result = []
        for method_name in chain:
            result = getattr(self, method_name)(result)
        return result

    def _set_seed(self) -> None:
        """Initialize random seeds"""
        random.seed(self.setting["seed"])
        np.random.seed(self.setting["seed"])

    def run(self):
        """Main simulation code"""
        # initialize
        self.initializer()
        for self.active_student in tqdm(
            self.students, desc="simulating students", leave=False
        ):
            # initialize state before student pass
            self.initializer_before_student()

            while self.practice_terminator():
                # select item
                if self.item_selector() is None:
                    continue

                # solve item
                self.item_solver()

                # log activity
                self.update_log()

                # update state after single response
                self.state_updater_after_item()

            # update state after student pass
            self.state_updater_after_student()

        self.log = pd.DataFrame(
            data=self.log,
            columns=["student_id", "item_id", "solved", "solve_time", "cheating"],
        ).dropna(subset=["student_id", "item_id"])
        self.log.student_id = self.log.student_id.astype(int)
        self.log.item_id = self.log.item_id.astype(int)

    def plot_attempts_heatmap(self):
        pivoted_log = self.log[["student_id", "item_id", "solved"]].pivot(
            index="student_id", columns="item_id"
        )
        palette = sns.color_palette(
            [sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]]
        )
        ax = sns.heatmap(data=pivoted_log, mask=pivoted_log.isnull(), cmap=palette)
        ax.set_facecolor("black")
        ax.set_title(f"Attempts heatmap -- {self.scenario_name}")
        plt.savefig(f"fig/{self.scenario_name}_heatmap.svg")
        plt.show()
        plt.clf()

    def plot_item_success_rates(self):
        ax = self.log.groupby("item_id")["solved"].mean().plot(kind="bar")
        ax.set_ylabel("Item success rate")
        ax.set_title(f"Item success rate -- {self.scenario_name}")
        plt.savefig(f"fig/{self.scenario_name}_item_success_rate.svg")
        plt.show()
        plt.clf()

    def update_log(self):
        # either one of solve time and solved have to be a valid simulation value
        should_log = self.solved is not None or self.solve_time is not None
        if should_log:
            self.log[self.log_id] = [
                self.active_student.id,
                self.active_item.id,
                self.solved,
                self.solve_time,
                getattr(self.active_student, "cheating", None),
            ]
            self.log_id += 1

    def export_q_matrix(self, path):
        columns = tuple("c{}".format(i) for i in range(self.q_matrix.shape[1]))
        q_matrix = pd.DataFrame(self.q_matrix, columns=columns)
        q_matrix = q_matrix.reset_index().rename(columns={"index": "item_id"})
        q_matrix.to_csv(path, index=False)

    def export_params(self, path):
        names = (
            ["alpha{}".format(i) for i in range(self.alpha.shape[0])]
            + ["beta{}".format(i) for i in range(self.beta.shape[0])]
            + ["gamma{}".format(i) for i in range(self.gamma.shape[0])]
        )
        values = np.hstack((self.alpha, self.beta, self.gamma))
        params = pd.DataFrame({"name": names, "value": values})
        params.to_csv(path, index=False)

    def export_afm_simulation(self):
        base_path = f"data/{self.scenario_name}/"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        self.log[["student_id", "item_id", "solved", "cheating"]].to_csv(
            base_path + "log.csv", index=False
        )
        self.export_q_matrix(base_path + "q_matrix.csv")
        self.export_params(base_path + "params.csv")
        with open(f"{base_path}/setting.json", "w") as f:
            json.dump(self.setting, f, ensure_ascii=False, cls=CustomJSONEncoder)

    """
    Initializer
    """

    def initializer(self):
        # Default stuff
        self._set_seed()

        student_count = self.setting["student_count"]
        item_count = self.setting["item_count"]
        self.q_matrix = np.array(self.setting["q_matrix"])
        concept_count = self.q_matrix.shape[1]

        alpha_distribution = self.setting["alpha_distribution"]
        beta_distribution = self.setting["beta_distribution"]
        gamma_distribution = self.setting["gamma_distribution"]

        self._set_seed()
        self.beta = beta_distribution(
            concept_count, **self.setting["beta_distribution_kwargs"]
        )
        self._set_seed()
        self.gamma = gamma_distribution(
            concept_count, **self.setting["gamma_distribution_kwargs"]
        )
        assert all(self.gamma >= 0)
        self._set_seed()
        self.alpha = alpha_distribution(
            student_count, **self.setting["alpha_distribution_kwargs"]
        )

        self.students = [
            Student(id, skill, np.full(concept_count, 0))
            for id, skill in enumerate(self.alpha)
        ]

        self.items = [
            Item(id, concept_mapping)
            for id, concept_mapping in enumerate(self.q_matrix)
        ]

        # for debug
        # pd.DataFrame(skills).plot(kind='hist', title='alpha')
        # plt.show()
        # pd.DataFrame(self.beta).plot(kind='hist', title='beta')
        # plt.show()
        # pd.DataFrame(self.gamma).plot(kind='hist', title='gamma')
        # plt.show()

        self.log = np.full((student_count * item_count, 5), fill_value=np.nan)
        self.log_id = 0
        self._set_seed()

        # other extensions for different scenarios
        self._chain_executor(self.setting.get("initializer_chain", ()))

    def create_fixed_scramble(self):
        self.item_order = [i for i in range(len(self.items))]
        random.shuffle(self.item_order)

    """
    Initializer before student
    """

    def initializer_before_student(self):
        self.available_items = self.items[:]
        self.item_order = 1
        self._chain_executor(self.setting.get("initializer_before_student_chain", ()))

    def scramble_and_sort(self):
        random.shuffle(self.available_items)
        self.available_items = sorted(
            self.available_items, key=lambda item: item.estimated_b
        )

    def epsilon(self):
        if random.random() < self.setting["epsilon"]:
            random.shuffle(self.available_items)
            self.random_students_id.append(self.active_student.id)

    def fixed_scramble(self):
        self.available_items.sort(key=lambda item: self.item_order[item.id])

    def set_remaining_time(self):
        self.remaining_solve_time = self.setting["remaining_solve_time"](
            self.true_b["true_b"]
        )

    def set_remaining_items(self):
        self.remaining_items = self.setting["max_items"]

    def initialize_streak(self):
        self.streak = 0

    def initialize_cheater(self):
        # is student cheating right now
        self.active_student.cheating = False
        # set if student will eventually cheat
        self.active_student.is_cheater = (
            random.random() < self.setting["cheating_prevalence"]
        )
        # for compromised items
        # if self.active_student.is_cheater:
        #     self.active_student.cheating = True

        # for switch
        # set when student starts cheating
        self.active_student.items_left_when_starts_cheating = random.randint(
            1, len(self.items)
        )

        # for switch and back only
        # # set when student stops cheating
        # self.active_student.items_left_when_stops_cheating = random.randint(
        #     0, self.active_student.items_left_when_starts_cheating - 1
        # )
        self.switch_cheater_state()

    """
    Practice terminator
    """

    def practice_terminator(self):
        return self._chain_executor_cumulative(
            self.setting.get("practice_terminator_chain", ())
        )

    def available_item(self, *args):
        return len(self.available_items) > 0

    def attrition_termination(self, result):
        return result and self.remaining_solve_time > 0

    def mastery_termination(self, result):
        return result and self.streak < self.setting["mastery_streak"]

    def max_items_terminator(self, result):
        return result and self.remaining_items > 0

    """
    Item selector
    """

    def item_selector(self):
        self.active_item = self._chain_executor_cumulative(
            self.setting.get("item_selector_chain", ())
        )
        return self.active_item

    def first_item(self, *args):
        return self.available_items.pop(0)

    def random_item(self, *args):
        return self.available_items.pop(
            random.randint(0, len(self.available_items) - 1)
        )

    def random_from_window(self, *args):
        window_size = min(self.setting["k"], len(self.available_items))
        return self.available_items.pop(random.randint(0, window_size - 1))

    def first_item_random_skip(self, *args):
        while (
            np.random.random() < self.setting["skip_probability"]
            and self.available_items
        ):
            self.available_items.pop(0)
        if not self.available_items:
            return None
        return self.available_items.pop(0)

    """
    Item solver
    """

    def item_solver(self):
        self._chain_executor(self.setting.get("item_solver_chain", ()))

    def log_time_model(self):
        student = self.active_student
        item = self.active_item

        self.solve_time = item.b + item.a * student.skill + np.random.normal(0, 1)
        student.solved_items += 1

    @staticmethod
    def _f(x):
        return 1 / (1 + np.exp(-x))
        # return x

    def afm(self):
        p = self.afm_prob()
        if "afm_noise_std" in self.setting:
            p += np.random.normal(0, self.setting["afm_noise_std"])
        self.solved = np.random.rand() <= p

    def afm_prob(self):
        student = self.active_student
        item = self.active_item
        p = self._f(
            self.active_student.skill
            + np.matmul(self.beta, item.concepts)
            + np.matmul(item.concepts, self.gamma * student.opportunities)
        )
        return p

    def cfm(self):
        p = self.cfm_prob()

        self.solved = np.random.rand() <= p

    def cfm_prob(self):
        student = self.active_student
        item = self.active_item
        # compute exponents
        xs = (
            self.active_student.skill + self.beta + (self.gamma * student.opportunities)
        )
        present_concepts = item.concepts != 0
        # take only concepts that are required to solve item
        xs = xs[present_concepts]
        # apply logistic function and multiply the probabilities
        p = np.prod(np.apply_along_axis(self._f, 0, xs))
        return p

    def afm_time(self):
        student = self.active_student
        item = self.active_item

        self.solve_time = (
            self.active_student.skill
            + np.matmul(self.beta, item.concepts)
            + np.matmul(item.concepts, self.gamma * np.log(student.opportunities + 1))
        )
        self.solved = True

    def attrition(self):
        self.remaining_solve_time -= self.solve_time
        if self.remaining_solve_time < 0:
            self.solved = None
            self.solve_time = None

    def mastery(self):
        self.streak = self.streak + 1 if self.solved else 0

    def early_stopping(self):
        self.remaining_items -= 1

    def cheating(self):
        if self.active_student.cheating:
            self.solved = True

    def cheating_compromised_items(self):
        if self.active_student.cheating and (
            self.active_item.id in self.setting["compromised_items"]
        ):
            self.solved = True

    """
    State updater after item
    """

    def state_updater_after_item(self):
        self.item_order += 1
        self.solved = None
        self.solve_time = None
        self._chain_executor(self.setting.get("state_updater_after_item_chain", ()))

    def learning_opportunities(self):
        self.active_student.opportunities += self.active_item.concepts

    def constant_learning(self):
        # skill from -3 to 3
        max_gain = self.setting["max_gain"]
        self.active_student.skill += max_gain / len(self.items)

    def step_learning(self):
        if (
            not hasattr(self.active_student, "has_learned")
            and random.random() < self.setting["learn_prob"]
        ):
            max_gain = self.setting["max_gain"]
            self.active_student.skill += max_gain
            self.active_student.has_learned = True

    def steep_learning(self):
        max_gain = self.setting["max_gain"]
        if self.active_student.skill < max_gain:
            base = self.setting["learning_period"]
            if self.active_student.solved_items < base:
                skill_increment = max_gain * log(
                    1 / self.active_student.solved_items + 1, base
                )
                self.active_student.skill += skill_increment

    def switch_cheater_state(self):
        if (
            self.active_student.is_cheater
            and self.active_student.items_left_when_starts_cheating
            >= len(self.available_items)
        ):
            self.active_student.cheating = True

        # for switch and back only
        # if (
        #     self.active_student.is_cheater
        #     and self.active_student.items_left_when_stops_cheating
        #     >= len(self.available_items)
        # ):
        #     self.active_student.cheating = False

    """
    State updater after student
    """

    def state_updater_after_student(self):
        self._chain_executor(self.setting.get("state_updater_after_student_chain", ()))

    def update_b_estimates(self):
        if (self.active_student.id + 1) % self.setting["adaptation_after"] == 0:
            self._update_b_estimates()

    def update_b_estimates_once(self):
        if (self.active_student.id + 1) == self.setting["first_k"]:
            self._update_b_estimates()

    def update_b_estimates_epsilon(self):
        if self.active_student.id in self.random_students_id:
            df = pd.DataFrame(
                data=self.log,
                columns=[
                    "student_id",
                    "item_id",
                    "item_order",
                    "student_skill",
                    "item_difficulty",
                    "item_true_difficulty",
                    "solve_time",
                ],
            )
            self._update_b_estimates(df[df.student_id.isin(self.random_students_id)])

    def _update_b_estimates(self, df=None):
        if df is None:
            df = pd.DataFrame(
                data=self.log,
                columns=[
                    "student_id",
                    "item_id",
                    "item_order",
                    "student_skill",
                    "item_difficulty",
                    "item_true_difficulty",
                    "solve_time",
                ],
            )
        for item, estimate in zip(
            self.items, df.groupby("item_id")["solve_time"].mean()[0:]
        ):
            item.estimated_b = estimate

    def get_true_difficulty_params(self):
        return self.true_b["true_b"].mean(), self.true_b["true_b"].std()

    def get_estimated_difficulty_params(self):
        estimated_difficulty_params = self.log.groupby("item_id")[
            "solve_time"
        ].describe()["mean"]
        return estimated_difficulty_params.mean(), estimated_difficulty_params.std()


def plot_mean_times(
    full_log, scenarios, title="Estimated item difficulties", colors=None, path=None
):
    assert scenarios
    if colors is not None:
        palette = [sns.color_palette("Paired", 12)[color] for color in colors]
    else:
        palette = None
    fig, ax = plt.subplots(figsize=(4.67, 3.3))
    ax = sns.lineplot(
        data=full_log[full_log.scenario.isin(scenarios)],
        x="item_id",
        y="solve_time",
        hue="scenario",
        palette=palette,
        ax=ax,
    )
    ax = sns.lineplot(
        data=full_log.groupby("item_id")["item_true_difficulty"].max(),
        ax=ax,
        label="true difficulty",
        color="Y",
    )
    ax.set_title(title)
    ax.set_xlabel("Item")
    ax.set_ylabel("Difficulty")

    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_mean_order_convergence(
    full_log,
    scenarios=[],
    window_size=10,
    path=None,
    labels=None,
    title="Item order convergence (Spearman corr. coef.)",
):
    assert scenarios
    ideal_order = list(
        full_log.groupby("item_id")["item_true_difficulty"].max().sort_values().index
    )

    correlations = [
        (
            i // window_size + 1,
            spearmanr(ideal_order[: len(row)], row)[0],
            labels[scenarios.index(scenario_name)] if labels else scenario_name,
        )
        for scenario_name in scenarios
        for i, row in enumerate(
            full_log[full_log.scenario == scenario_name]
            .groupby("student_id")
            .apply(lambda g: list(g.item_id))
        )
    ]
    df = pd.DataFrame(correlations, columns=["i", "correlation", "scenario"])

    fig, ax = plt.subplots(figsize=(4.67, 3.3))
    ax = sns.lineplot(data=df, x="i", y="correlation", hue="scenario", ci=None, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("{}s of students".format(window_size))
    ax.set_ylabel("Mean correlation")
    ax.set_ylim(-0.25, 1)

    if path:
        plt.savefig(path)
    else:
        plt.show()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            json.JSONEncoder.default(self, obj)
        except TypeError:
            return repr(obj)
