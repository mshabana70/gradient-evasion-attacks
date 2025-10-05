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

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Make the class callable."""
        return self.perturb(x, y)
    
    def _get_loss_fn(self) -> Callable:
        """
        Get the appropriate loss function based on attack config.

        Returns:
            Loss function for attack
        """
        if self.config.targeted:
            def loss_fn(outputs, labels):
                return -nn.CrossEntropyLoss()(outputs, labels)
        else:
            def loss_fn(outputs, labels):
                return nn.CrossEntropyLoss()(outputs, labels)
        return loss_fn
    
    def _clip_pertubation(self, perturbation: torch.Tensor, norm: str) -> torch.Tensor:
        """
        Clip perturbation to be within the specified norm ball.

        Args:
            perturbation: Raw Perturbation
            norm: Type of norm constraint ('linf', 'l2', 'l1')
        
        Returns:
            Clipped perturbation
        """
        
    
    def _project_to_ball(self, x: torch.Tensor, x_adv: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Project adversarial example to epsilon ball around original samples.
        
        Args:
            x_adv: Adversarial examples
            x: Original samples
            eps: Epsilon radius

        Returns:
            Projected adversarial examples
        """

        if self.config.norm == 'linf':
            x_adv = torch.min(x + eps, torch.max(x - eps, x_adv)) # L_inf(x_adv - x) = max_i |x_adv_i - x_i | <= eps 
        elif self.config.norm == 'l2': # L_2(x_adv - x) = sqrt(sum_i (x_adv_i - x_i)^2) <= eps
            delta = x_adv - x
            delta_flat = delta.view(delta.size(0), -1)
            norm = torch.norm(delta_flat, p=2, dim=1)
            mask = norm > eps # finds all adv examples outside the L2 ball
            if mask.any():
                delta_flat[mask] = delta_flat[mask] * (eps / norm[mask]).view(-1, 1) # projection step
                delta = delta_flat.view_as(delta)
                x_adv = x + delta
        
        # ensure valid pixel range
        x_adv = torch.clamp(x_adv, self.config.clip_min, self.config.clip_max)
        return x_adv
