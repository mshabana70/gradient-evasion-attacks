import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class GradientEvasionAttack():

    def __init__(self, classifier: BaseEstimator, step_size: float = 0.1, max_iter: int = 100, mimicry_param: float = 0.0, norm='l2'):
        """
        Initialize the Gradient Evasion Attack.

        Parameters:
        - classifier: The target classifier to attack (must have a decision_function method).
        - step_size: The step size for each iteration of gradient descent.
        - max_iter: The maximum number of iterations to perform.
        - mimicry_param: The parameter controlling the influence of mimicry in the attack.
        - norm: The norm to use for gradient normalization ('l1', 'l2', or 'inf').
        """
        self.classifier = classifier
        self.step_size = step_size
        self.max_iter = max_iter
        self.mimicry_param = mimicry_param
        self.norm = norm