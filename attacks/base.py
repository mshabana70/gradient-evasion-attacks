"""
Base Attack Abstract Class

Provides standardized interface for all adversarial attacks

Author: Mahmoud Shabana
Date: 2025-10
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import warnings

@dataclass
class AttackConfig:
    """Configuration for attack parameters."""
    eps: float = 0.3
    eps_iter: float = 0.01
    nb_iter: int = 40
    norm: str = 'linf' # 'linf', 'l2', 'l1'
    targeted: bool = False
    clip_min: float = 0.0
    clip_max: float = 1.0
    verbose: bool = False

class BaseAttack(ABC):
    """
    Abstract base class for adversarial attacks.

    This class provides a standardized interfact and common utilities for implementing
    adversarial attacks. All attacks should inherit from this class and implement the 
    perturb() method.

    Attributes:
        model (nn.Module): Target model to attack
        config (AttackConfig): Attack configuration parameters
        device (torch.device): Device for computation ('cpu' or 'cuda')
        attack_name (str): Name of the attack for logging

    Paper References:
        - Goodfellow et al., 2014: "Explaining and Harnessing Adversarial Examples"
        - Madry et al., 2017: "Towards Deep Learning Models Resistant to Adversarial Attacks"
    """

    def __init__(self, model: nn.Module, config: Optional[AttackConfig] = None, device: Optional[torch.device] = None):
        """
        Initialize attack.

        Args:
            model: PyTorch model to attack
            config: Attack config parameters
            device: Computation device (CPU/GPU)
        """
        self.model = model
        self.config = config or AttackConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # attack metadata for logging
        self.attack_name = self.__class__.__name__
        self.stats = {
            "queries": 0,
            "success_rate": 0.0,
            "avg_distortion": 0.0
        }

    @abstractmethod
    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial pertubations.

        Args:
            x: Input samples (batch_size, C, H, W)
            y: True labels or target labels for targeted attacks

        Returns:
            Adversarial examples
        """
        pass