import json
import os
import random
from copy import deepcopy
from math import log
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from scipy.stats import spearmanr  # type: ignore
from tqdm import tqdm

matplotlib.use("agg")

SETTING = dict[str, Any]


class Item:
    def __init__(self, item_id: int, concepts: np.ndarray) -> None:
        self.id = item_id
        self.concepts = concepts
        self.estimated_b: float = 0
        self.a: float = 0
        self.b: float = 0


class Student:
    def __init__(
        self, student_id: int, skill: float, opportunities: np.ndarray
    ) -> None:
        self.id = student_id
        self.skill = skill
        self.opportunities = opportunities
        self.cheating = False
        self.is_cheater = False
        self.items_left_when_starts_cheating = 0
        self.solved_items = 0
        self.has_learned = False


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
        self.item_order: Any = None
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
        result: Any = []
        for method_name in chain:
            result = getattr(self, method_name)(result)
        return result

    def _set_seed(self) -> None:
        """Initialize random seeds"""
        random.seed(self.setting["seed"])
        np.random.seed(self.setting["seed"])

    def run(self) -> None:
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

    def plot_attempts_heatmap(self) -> None:
        """
        Plot answer correctness as heatmap (students X items)
        """
        assert isinstance(self.log, pd.DataFrame)
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

    def plot_item_success_rates(self) -> None:
        """
        Plot item success rates as bar plot
        """
        assert isinstance(self.log, pd.DataFrame)
        ax = self.log.groupby("item_id")["solved"].mean().plot(kind="bar")
        ax.set_ylabel("Item success rate")
        ax.set_title(f"Item success rate -- {self.scenario_name}")
        plt.savefig(f"fig/{self.scenario_name}_item_success_rate.svg")
        plt.show()
        plt.clf()

    def update_log(self) -> None:
        """
        Update log after simulated student answer
        """
        assert self.log is not None
        assert self.active_student is not None
        assert self.active_item is not None
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

    def export_q_matrix(self, path: Union[str, Path]) -> None:
        """
        Save Q-matrix used in simulation as CSV file

        :param path: Path to a file that will contain stored Q-matrix
        """
        assert self.q_matrix is not None
        columns = tuple("c{}".format(i) for i in range(self.q_matrix.shape[1]))
        q_matrix = pd.DataFrame(self.q_matrix, columns=columns)
        q_matrix = q_matrix.reset_index().rename(columns={"index": "item_id"})
        q_matrix.to_csv(path, index=False)

    def export_params(self, path: Union[str, Path]) -> None:
        """
        Export AFM parameters as CSV file

        :param path: Path to a file that will contain stored AFM parameters
        """
        assert self.alpha is not None
        assert self.beta is not None
        assert self.gamma is not None
        names = (
            ["alpha{}".format(i) for i in range(self.alpha.shape[0])]
            + ["beta{}".format(i) for i in range(self.beta.shape[0])]
            + ["gamma{}".format(i) for i in range(self.gamma.shape[0])]
        )
        values = np.hstack((self.alpha, self.beta, self.gamma))
        params = pd.DataFrame({"name": names, "value": values})
        params.to_csv(path, index=False)

    def export_afm_simulation(self) -> None:
        """
        Export result of simulation with AFM model

        This exports simulated log file as well as Q-matrix, AFM parameters,
        and simulation settings.
        """
        assert isinstance(self.log, pd.DataFrame)
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

    def initializer(self) -> None:
        """
        Initialize simulation state at the beginning
        """
        # Default stuff
        self._set_seed()

        student_count: int = self.setting["student_count"]
        item_count: int = self.setting["item_count"]
        self.q_matrix = np.array(self.setting["q_matrix"])
        concept_count: int = self.q_matrix.shape[1]

        alpha_distribution: Callable[..., np.ndarray] = self.setting[
            "alpha_distribution"
        ]
        beta_distribution: Callable[..., np.ndarray] = self.setting["beta_distribution"]
        gamma_distribution: Callable[..., np.ndarray] = self.setting[
            "gamma_distribution"
        ]

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
            Student(student_id, skill, np.full(concept_count, 0))
            for student_id, skill in enumerate(self.alpha)
        ]

        self.items = [
            Item(item_id, concept_mapping)
            for item_id, concept_mapping in enumerate(self.q_matrix)
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

    def create_fixed_scramble(self) -> None:
        """
        Randomly shuffle item ordering

        It is intended to simulate student solving items in the same order,
        but the order is random.
        """
        self.item_order = [i for i in range(len(self.items))]
        random.shuffle(self.item_order)

    """
    Initializer before student
    """

    def initializer_before_student(self) -> None:
        """
        Initialize simulation state before new students starts solving
        """
        self.available_items = self.items[:]
        self.item_order = 1  # type: ignore
        self._chain_executor(self.setting.get("initializer_before_student_chain", ()))

    def scramble_and_sort(self) -> None:
        """
        Order items base on their estimated difficulty

        There is a random shuffle step to break ties (especially in
        the beginning where all estimates are the same).
        """
        random.shuffle(self.available_items)
        self.available_items = sorted(
            self.available_items, key=lambda item: item.estimated_b
        )

    def epsilon(self) -> None:
        """
        Randomly reorder items for the active student with some probability
        """
        assert self.active_student is not None
        if random.random() < self.setting["epsilon"]:
            random.shuffle(self.available_items)
            self.random_students_id.append(self.active_student.id)

    def fixed_scramble(self) -> None:
        """
        Order items based on predefined fix item order
        """
        assert self.available_items is not None
        assert self.item_order is not None
        self.available_items.sort(key=lambda item: self.item_order[item.id])

    def set_remaining_time(self) -> None:
        """
        Update remaining student solving time
        """
        self.remaining_solve_time = self.setting["remaining_solve_time"](
            self.true_b["true_b"]  # type: ignore
        )

    def set_remaining_items(self) -> None:
        """
        Initialize number of item student can solve
        """
        self.remaining_items = self.setting["max_items"]

    def initialize_streak(self) -> None:
        """
        Initialize streak counter

        A streak count is the number of consecutively successfully solved items.
        """
        self.streak = 0

    def initialize_cheater(self) -> None:
        """
        Randomly assign student to a cheating group
        """
        assert self.active_student is not None
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
        """
        Determine if current student should stop solving items

        There can be multiple reasons to stop solving from no unsolved items
        left to expended solving time budget.
        """
        return self._chain_executor_cumulative(
            self.setting.get("practice_terminator_chain", ())
        )

    def available_item(self, *args: Any) -> bool:
        """
        Check if the student should stop due to no unsolved items left

        :return: True if there are any unsolved items left and False otherwise
        """
        return len(self.available_items) > 0

    def attrition_termination(self, result: bool) -> bool:
        """
        Check if the student should stop due to no remaining solving time left

        :param result: Results of previous checks
        :return: True if there is any remaining solving time left and previous
                 conditions are also satisfied and False otherwise
        """
        assert self.remaining_solve_time is not None
        return result and self.remaining_solve_time > 0

    def mastery_termination(self, result: bool) -> bool:
        """
        Check if the student should stop due to achieving mastery

        :param result: Results of previous checks
        :return: True if there is any remaining solving tim e left and
                 previous conditions are also satisfied and False otherwise
        """
        return result and self.streak < self.setting["mastery_streak"]

    def max_items_terminator(self, result: bool) -> bool:
        """
        Check if the student should stop due to solving predefined number of items

        :param result: Results of previous checks
        :return: True if student can still solve some items and previous
                 conditions are also satisfied and False otherwise
        """
        assert self.remaining_items is not None
        return result and self.remaining_items > 0

    """
    Item selector
    """

    def item_selector(self) -> Item:
        """
        Select the next item for student to solve
        """
        self.active_item = self._chain_executor_cumulative(
            self.setting.get("item_selector_chain", ())
        )
        assert self.active_item is not None
        return self.active_item

    def first_item(self, *args: Any) -> Item:
        assert self.available_items is not None
        return self.available_items.pop(0)

    def random_item(self, *args: Any) -> Item:
        assert self.available_items is not None
        return self.available_items.pop(
            random.randint(0, len(self.available_items) - 1)
        )

    def random_from_window(self, *args: Any) -> Item:
        assert self.available_items is not None
        window_size = min(self.setting["k"], len(self.available_items))
        return self.available_items.pop(random.randint(0, window_size - 1))

    def first_item_random_skip(self, *args: Any) -> Optional[Item]:
        while (
            np.random.rand() < self.setting["skip_probability"] and self.available_items
        ):
            self.available_items.pop(0)
        if not self.available_items:
            return None
        return self.available_items.pop(0)

    """
    Item solver
    """

    def item_solver(self) -> None:
        """Simulate a student solving an item"""
        self._chain_executor(self.setting.get("item_solver_chain", ()))

    def log_time_model(self) -> None:
        """
        Simulate student answer using Log time model

        The student is assumed to always solve the item and the response
        time is modeled using a variation of IRT model.
        """
        assert self.active_student is not None
        assert self.active_item is not None
        student = self.active_student
        item = self.active_item

        self.solve_time = item.b + item.a * student.skill + np.random.normal(0, 1)  # type: ignore
        student.solved_items += 1

    @staticmethod
    def _f(x: Any) -> Any:
        """Logistic sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def afm(self) -> None:
        """Simulate student answer using AFM"""
        p = self.afm_prob()
        if "afm_noise_std" in self.setting:
            p += np.random.normal(0, self.setting["afm_noise_std"])  # type: ignore
        self.solved = np.random.rand() <= p  # type: ignore

    def afm_prob(self) -> float:
        """Compute correct answer probability using AFM formula"""
        assert self.active_student is not None
        assert self.active_item is not None
        assert self.beta is not None
        assert self.gamma is not None
        student = self.active_student
        item = self.active_item
        p = self._f(
            self.active_student.skill
            + np.matmul(self.beta, item.concepts)
            + np.matmul(item.concepts, self.gamma * student.opportunities)
        )
        return p

    def cfm(self) -> None:
        """
        Simulate student answer using CFM

        CFM is a variation of AFM where the concepts are modeled as
        conjunctive instead of additive.
        """
        p = self.cfm_prob()

        self.solved = np.random.rand() <= p  # type: ignore

    def cfm_prob(self) -> float:
        """Compute correct answer probability using CFM formula"""
        assert self.active_student is not None
        assert self.active_item is not None
        assert self.beta is not None
        assert self.gamma is not None
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

    def afm_time(self) -> None:
        """Simulate student response time using AFM-inspired formula"""
        assert self.active_student is not None
        assert self.active_item is not None
        assert self.beta is not None
        assert self.gamma is not None
        student = self.active_student
        item = self.active_item

        self.solve_time = (
            self.active_student.skill
            + np.matmul(self.beta, item.concepts)
            + np.matmul(item.concepts, self.gamma * np.log(student.opportunities + 1))
        )
        self.solved = True

    def attrition(self) -> None:
        """Modify the state if a student has stopped solving due to attrition"""
        assert self.remaining_solve_time is not None
        assert self.solve_time is not None
        self.remaining_solve_time -= self.solve_time
        if self.remaining_solve_time < 0:
            self.solved = None
            self.solve_time = None

    def mastery(self) -> None:
        """Update streak counter"""
        assert self.streak is not None
        assert self.solved is not None
        self.streak = self.streak + 1 if self.solved else 0

    def early_stopping(self) -> None:
        """Reduce counter of items a student will solve"""
        assert self.remaining_items is not None
        self.remaining_items -= 1

    def cheating(self) -> None:
        """Modify the state if a student was cheating"""
        assert self.active_student is not None
        if self.active_student.cheating:
            self.solved = True

    def cheating_compromised_items(self) -> None:
        """Modify the state if a student was cheating"""
        assert self.active_student is not None
        assert self.active_item is not None
        if self.active_student.cheating and (
            self.active_item.id in self.setting["compromised_items"]
        ):
            self.solved = True

    """
    State updater after item
    """

    def state_updater_after_item(self) -> None:
        """Update simulation state after a simulated student response"""
        self.item_order += 1
        self.solved = None
        self.solve_time = None
        self._chain_executor(self.setting.get("state_updater_after_item_chain", ()))

    def learning_opportunities(self) -> None:
        """Update learning opportunities counter"""
        assert self.active_student is not None
        assert self.active_item is not None
        self.active_student.opportunities += self.active_item.concepts

    def constant_learning(self) -> None:
        """Update student skill"""
        assert self.active_student is not None
        # skill from -3 to 3
        max_gain = self.setting["max_gain"]
        self.active_student.skill += max_gain / len(self.items)

    def step_learning(self) -> None:
        """Update student skill"""
        assert self.active_student is not None
        if (not self.active_student.has_learned) and random.random() < self.setting[
            "learn_prob"
        ]:
            max_gain = self.setting["max_gain"]
            self.active_student.skill += max_gain
            self.active_student.has_learned = True

    def steep_learning(self) -> None:
        """Update student skill"""
        assert self.active_student is not None
        max_gain = self.setting["max_gain"]
        if self.active_student.skill < max_gain:
            base = self.setting["learning_period"]
            if self.active_student.solved_items < base:
                skill_increment = max_gain * log(
                    1 / self.active_student.solved_items + 1, base
                )
                self.active_student.skill += skill_increment

    def switch_cheater_state(self) -> None:
        """Update student cheating state"""
        assert self.active_student is not None
        assert self.available_items is not None
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

    def state_updater_after_student(self) -> None:
        """Update simulation state after a student has responded to all items"""
        self._chain_executor(self.setting.get("state_updater_after_student_chain", ()))

    def update_b_estimates(self) -> None:
        """Update item difficulty estimate"""
        assert self.active_student is not None
        if (self.active_student.id + 1) % self.setting["adaptation_after"] == 0:
            self._update_b_estimates()

    def update_b_estimates_once(self) -> None:
        """Update item difficulty estimate"""
        assert self.active_student is not None
        if (self.active_student.id + 1) == self.setting["first_k"]:
            self._update_b_estimates()

    def update_b_estimates_epsilon(self) -> None:
        """Update item difficulty estimate"""
        assert self.active_student is not None
        assert self.random_students_id is not None
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

    def _update_b_estimates(self, df: Optional[pd.DataFrame] = None) -> None:
        """Update item difficulty estimate"""
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

    def get_true_difficulty_params(self) -> tuple[float, float]:
        return self.true_b["true_b"].mean(), self.true_b["true_b"].std()  # type: ignore

    def get_estimated_difficulty_params(self) -> tuple[float, float]:
        assert isinstance(self.log, pd.DataFrame)
        estimated_difficulty_params = self.log.groupby("item_id")[
            "solve_time"
        ].describe()["mean"]
        return estimated_difficulty_params.mean(), estimated_difficulty_params.std()


def plot_mean_times(
    full_log: pd.DataFrame,
    scenarios: Sequence[str],
    title: str = "Estimated item difficulties",
    colors: Optional[list[int]] = None,
    path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot estimated and true item difficulties as line plots"""
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
