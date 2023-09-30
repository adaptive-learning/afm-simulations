from hashlib import sha256
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import Regularizer, l2
from tqdm import tqdm


class AFM(Model):
    def __init__(
        self,
        num_students: int,
        num_items: int,
        q_matrix: np.ndarray,
        num_knowledge_components: Optional[int] = None,
        max_learning_opportunities: Optional[np.ndarray] = None,
        init_alphas: Optional[np.ndarray] = None,
        init_betas: Optional[np.ndarray] = None,
        init_gammas: Optional[np.ndarray] = None,
        fit_alphas: bool = True,
        fit_betas: bool = True,
        fit_gammas: bool = True,
        alpha_regularizer: Regularizer = l2(l2=0.01),
    ) -> None:
        super().__init__()
        self.num_students = num_students
        self.num_items = num_items
        self.num_knowledge_components = (
            num_knowledge_components if num_knowledge_components is not None else q_matrix.shape[1]
        )
        self.max_learning_opportunities = (
            max_learning_opportunities if max_learning_opportunities is not None else q_matrix.sum(axis=0)
        )
        assert q_matrix.shape == (
            self.num_items,
            self.num_knowledge_components,
        ), (
            f"The shape of Q matrix {q_matrix.shape} does not match the number of items {num_items} "
            f"times number of knowledge components {num_knowledge_components}"
        )

        self.q_matrix = tf.constant(value=q_matrix, name="q_matrix", dtype=tf.float32)

        self.alphas = tf.Variable(
            initial_value=init_alphas if init_alphas is not None else np.zeros(self.num_students),  # type: ignore
            name="alphas",
            dtype=tf.float32,
            trainable=fit_alphas,
        )
        self.betas = tf.Variable(
            initial_value=init_betas  # type: ignore
            if init_betas is not None
            else np.zeros(self.num_knowledge_components),
            name="betas",
            dtype=tf.float32,
            trainable=fit_betas,
        )
        self.gammas = tf.Variable(
            initial_value=init_gammas  # type: ignore
            if init_gammas is not None
            else np.zeros(self.num_knowledge_components),
            name="gammas",
            dtype=tf.float32,
            trainable=fit_gammas,
        )
        self.alpha_regularizer = alpha_regularizer

    def call(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Compute predictions using AFM model

        `inputs` has to contain keys `student` with student ids, `item` with
        item ids, and `learning_opportunities` with learning opportunities of
        a student at each point.

        :param inputs: Dictionary with model inputs
        :return: Tensor with correct answer probability predictions
        """
        assert "student" in inputs
        assert "item" in inputs
        assert "learning_opportunities" in inputs

        student_ids, item_ids, learning_opportunities = (
            inputs["student"],
            inputs["item"],
            tf.cast(inputs["learning_opportunities"], tf.float32),
        )

        # gather vector with \alpha_i based on student id i in input
        student_proficiencies = tf.gather(self.alphas, student_ids, name="student_proficiencies")
        # gather matrix q_jk consisting of lines q_j from Q-matrix
        qs = tf.gather(self.q_matrix, item_ids, name="qs")

        # compute \sum_{k=1}^K \beta_k * q_{jk}
        tasks_easiness = tf.reduce_sum(qs * self.betas, axis=1, name="tasks_easiness")

        # learning opportunities normalization described in paper
        # T. Effenberger, R. Pelanek, and J. Cechak.
        # Exploration of the robustness and generalizability of the additive factors model.
        gammas_prime = self.gammas * self.max_learning_opportunities
        normalized_learning_opportunities = learning_opportunities / self.max_learning_opportunities

        # compute \sum_{k=1}^K \gamma_k q_{jk} t_{ik}
        learning_effects = tf.reduce_sum(normalized_learning_opportunities * gammas_prime * qs, axis=1)

        # compute logits z_{ij}
        logits = tf.add_n([student_proficiencies + tasks_easiness + learning_effects], name="logits")

        # convert logits to probability values
        correct_answer_probabilities = tf.math.sigmoid(logits)

        # parameter visualization in Tensorboard
        tf.summary.histogram("alphas", self.alphas)
        tf.summary.histogram("betas", self.betas)
        tf.summary.histogram("gammas", self.gammas)

        # regularization of alphas
        alpha_penalization_term = self.alpha_regularizer(self.alphas) / self.num_students
        self.add_loss(alpha_penalization_term)

        return correct_answer_probabilities

    def check_inputs(
        self,
        student_ids: np.ndarray,
        item_ids: np.ndarray,
        learning_opportunities: np.ndarray,
    ) -> None:
        """
        Assert basic input data constrains

        :param student_ids: Vector with student ids
        :param item_ids: Vector with item ids
        :param learning_opportunities: Matrix with student learning opportunities
        """
        assert (
            0 <= student_ids.min() and student_ids.max() < self.num_students
        ), "Students ids must be between 0 and NUM_STUDENTS - 1"
        assert 0 <= item_ids.min() and item_ids.max() < self.num_items, "Items ids must be between 0 and NUM_ITEMS - 1"
        assert student_ids.shape[0] == item_ids.shape[0], "Student and item ids shape mismatch"
        assert (
            student_ids.shape[0] == learning_opportunities.shape[0]
        ), "Student ids and learning opportunities shape mismatch"
        assert (
            learning_opportunities.shape[1] == self.num_knowledge_components
        ), "Learning opportunities should be a matrix NUM_OBSERVATIONS x NUM_KC"

    @staticmethod
    def compute_opportunities(
        student_ids: np.ndarray,
        item_ids: np.ndarray,
        q_matrix: np.ndarray,
        num_students: Optional[int] = None,
        cache_dir: str = "./cache",
    ) -> np.ndarray:
        """
        Precompute learning opportunities from log data

        This method assumes that:
        1) It has complete log data history up to some point in time.
        2) No other practicing is happening outside the received data.
        3) Input ids are order as they occurred from oldest to the most recent

        The computed opportunities are cached to save computational cost for
        future use.

        :param student_ids: Vector of student ids from the log
        :param item_ids: Vector of item ids from the log
        :param q_matrix: Complete binary Q-matrix
        :param num_students: Number of students for which to compute learning
                             opportunities. If not provided, length of
                             `student_ids` is used.
        :param cache_dir: Path for storing precomputed learning opportunities.
                          Nothing is stored when empty string is supplied.
        :return: Learning opportunities matrix
        """
        if cache_dir:
            # compute hash of inputs
            hasher = sha256()
            hasher.update(student_ids.data)
            hasher.update(item_ids.data)
            hasher.update(np.ascontiguousarray(q_matrix).data)
            checksum = hasher.hexdigest()

            # check for exiting cached file
            cache_file_path = Path(cache_dir) / f"{checksum}.npy"
            if cache_file_path.exists():
                print("Found cached opportunities files!")
                opportunities = np.load(cache_file_path)
                assert isinstance(opportunities, np.ndarray)
                return opportunities

        # initialize variables
        assert student_ids.shape[0] == item_ids.shape[0]
        num_observations = student_ids.shape[0]
        num_knowledge_components = q_matrix.shape[1]
        opportunities = np.zeros(
            shape=(num_observations, num_knowledge_components),
        )
        if num_students is None:
            num_students = student_ids.max() + 1
        student_opportunities = np.zeros(shape=(num_students, num_knowledge_components))

        # compute learning opportunities by accumulating Q-matrix lines for each student and student-item interaction
        for i in tqdm(range(num_observations)):
            student_id = student_ids[i]
            item_id = item_ids[i]
            # save opportunities when student have seen the item
            opportunities[i, :] = student_opportunities[student_id, :]

            # change opportunities for later
            student_opportunities[student_id, :] += q_matrix[item_id]

        if cache_dir:
            path = Path(cache_dir)
            path.mkdir(parents=True, exist_ok=True)
            checksum = sha256(np.array([student_ids, item_ids]).data).hexdigest()
            np.save(path / f"{checksum}.npy", opportunities)
        return opportunities

    @staticmethod
    def get_model(
        num_students: int,
        num_items: int,
        num_observations: int,
        q_matrix: np.ndarray,
        num_knowledge_components: Optional[int] = None,
        max_learning_opportunities: Optional[np.ndarray] = None,
        init_alphas: Optional[np.ndarray] = None,
        init_betas: Optional[np.ndarray] = None,
        init_gammas: Optional[np.ndarray] = None,
        fit_alphas: bool = True,
        fit_betas: bool = True,
        fit_gammas: bool = True,
        alpha_penalty: float = 0.01,
    ) -> "AFM":
        """
        Factory method for initialized AFM model

        :param num_students: Number of students to model (i.e., keep track of)
        :param num_items: Number of items
        :param num_observations: Number of observed student-item interactions
        :param q_matrix: Q-matrix
        :param num_knowledge_components: Number of KCs. Derived from shape of
                                         Q-matrix if not provided.
        :param max_learning_opportunities: Maximal number of achievable
                                           learning opportunities. Derived
                                           from Q-matrix content if not
                                           provided.
        :param init_alphas: Initial value of alpha parameters. Zeros are used
                            by default.
        :param init_betas: Initial value of beta parameters. Zeros are used by
                           default.
        :param init_gammas: Initial value of gamma parameters. Zeros are used
                            by default.
        :param fit_alphas: Flag whether to estimate alpha parameters. If False
                           alpha parameters keep their initial values.
        :param fit_betas: Flag whether to estimate beta parameters. If False
                          beta parameters keep their initial values.
        :param fit_gammas: Flag whether to estimate gamma parameters. If False
                           gamma parameters keep their initial values.
        :param alpha_penalty: Alpha parameter regularization coefficient
        :return: An instance of ready to use AFM class
        """
        afm = AFM(
            num_students=num_students,
            num_items=num_items,
            q_matrix=q_matrix,
            num_knowledge_components=num_knowledge_components,
            max_learning_opportunities=max_learning_opportunities,
            init_alphas=init_alphas,
            init_betas=init_betas,
            init_gammas=init_gammas,
            fit_alphas=fit_alphas,
            fit_betas=fit_betas,
            fit_gammas=fit_gammas,
            alpha_regularizer=l2(l2=alpha_penalty),
        )

        # experimentally discovered suitable hyperparameter values
        initial_learning_rate = 1e-2
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=20 * num_observations,
            decay_rate=0.96,
            staircase=True,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, name="binary_crossentropy_loss")

        afm.compile(
            optimizer=optimizer,
            loss=loss_object,
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),  # type: ignore
            ],
        )

        return afm

    def save_variables(self, path: Union[str, Path]) -> None:
        """
        Export AFM model parameters to CSV file

        :param path: Path to store exported values
        """
        names = (
            [f"alpha{i}" for i in range(len(self.alphas.numpy()))]
            + [f"beta{i}" for i in range(len(self.betas.numpy()))]
            + [f"gamma{i}" for i in range(len(self.gammas.numpy()))]
        )
        values = np.hstack((self.alphas.numpy(), self.betas.numpy(), self.gammas.numpy()))
        params = pd.DataFrame({"name": names, "value": values})
        params.to_csv(path, index=False)
