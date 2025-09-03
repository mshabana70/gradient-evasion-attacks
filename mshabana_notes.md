# Evasion Attacks Paper Notes

[link to paper](https://arxiv.org/pdf/1708.06131)

Some protection mechanisms that learning system designers should use:

1. Finding potential vulnerabilities of learning *before* they are exploited by the adversary
2. Investigating the impact of the corresponding attacks (i.e., evaluating classifier security)
3. Devising appropriate countermeasures if an attack is found to significantly degrade the classifier's performance

The proposed evasion strategy is a gradient-based approach inspired by Golland's discriminative directions technique. This makes the approach applicable to any classifier with a differentiable discriminant function.

## Optimal evasion at test time

The classification algorithm is $f: \bm{\mathcal{X}} \mapsto \bm{\mathcal{Y}}$, which basically means a function $f$ maps from input space $\bm{\mathcal{X}}$ to output space $\bm{\mathcal{Y}}$. This is the proposed classifier in the paper.

The classifier $f$ output space $\bm{\mathcal{Y}}$ equals $\{-1, +1\}$, meaning there are only two possible class labels the output of the classifier can have: -1 and +1 (likely "spam" or "not spam").

The training data $\bm{\mathcal{D}} = \{x_{i}, y_{i}\}_{i=1}^{n}$ is a dataset with n samples. Each sample has features $x_{i}$ and a true label $y_{i}$. The data comes from some underlying distribution $p(X, Y$)$

In this paper, classifier output is defined as $f(x) = y^{c}$ to differentiate from the true label $y$. This $y^{c}$ output is given from a thresholding function $g : \bm{\mathcal{X}} \mapsto \mathbb{R}$, meaning the classifier function's output is computed in two stages: the first stage is a continuous function that produces a real number (something like a confidence score) by the *continuous discriminant function* $g(x)$. The is then handed off to the second stage to be converted to a discrete value by the threshold. In the case of this paper, the threshold is $(g(x) < 0) \to (f(x) = -1)$ and $f(x) = +1$ otherwise. So $g(x)$ does NOT return the $y^{c}$ label but rather a confidence score, and $f(x)$ applies a threshold to get the predicted label $y^{c}$.

### Adversary model

Assumptions are made about the adversary's motives, knowledge of the target, and modification capability to the underlying target's data distribution.

The goal of the adversary should be to maximize some utility function, so in the evasion use-case, it would be manipulating a single sample that should be misclassified.

An attacker can simply find a sample $\bold{x}$ such that $g(\bold{x}) < -\epsilon$ for any $\epsilon > 0$. This means the attack sample only needs to just cross the decision boundary of the classifier for misclassification. But this can be easily defended against by adjusting the decision threshold to be more robust in classification.

So, to combat this defense, it would make more sense for the attacker to create a sample that misclassifies with a high confidence; a sample that minimizes the discriminant function we discussed earlier, $g(x)$ (as long as the set of rules and limitations are realistic for adversarial pertubations).

Adversary knowledge can vary and generally includes the following:
- the training set or part of it
- the feature representation of each sample; i.e., how *real* objects such as emails, network packets are mapped into the classifier's feature space
- the type of a learning algorithm and the form of it's decision function
- the (trained) classifier model; e.g., weights of a linear classifier
- feedback from the classifier; e.g., classifier labels for samples chosen by the adversary

Adversary capabilities are limited to modification of test data in evasion scenarios. Under this restriction, variations in attacker's power may include:
- mods to the input data (limited or not)
- mods to the feature vectors (limited or not)
- independent mods to specific feaures (semantics of the input data may dictate that certain features are interdependent)

This paper defines two attack scenarios, **Perfect Knowledge (PK)** and **Limited Knowledge (LK)**:

- **Perfect Knowledge (PK)**: The goal of the adversary is to minimize $g(x)$ and they have knowledge of the feature space, type of classifier, and the trained model. The attacker can also transform attack points in the test data but this will be variable and is defined by $d_{max}$, which is the maximum distance a transformed attack point can be from the original attack sample. For example, the attack will be bounded by a certain number of words they can change in a spam email to evade a spam classifier. The distance measure will be defined as $d : \bm{\mathcal{X}} \times \bm{\mathcal{X}} \mapsto \mathbb{R}^{+}$ and is application specific.
- **Limited Knowledge (LK)**: The goal of the adversary is the same, minimize $g(x)$ under the same constraints that each transformed attack point must remain within a maximum distance of $d_{max}$ from the corresponding original attack sample. However the attacker will not know the learned classifier $f$ or its training data $\bm{\mathcal{D}}$, consequently not be able to compute $g(x)$. 