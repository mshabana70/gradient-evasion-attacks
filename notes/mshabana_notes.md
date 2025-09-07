# Evasion Attacks Paper Notes

<blockquote>WARNING: these are my unfiltered thoughts and understanding of the paper as I read through it and break it down to a fundamental understanding. I am not responsible for the brain pain these notes might bring you...</blockquote>

<br>

[link to paper](https://arxiv.org/pdf/1708.06131)

Some protection mechanisms that learning system designers should use:

1. Finding potential vulnerabilities of learning *before* they are exploited by the adversary
2. Investigating the impact of the corresponding attacks (i.e., evaluating classifier security)
3. Devising appropriate countermeasures if an attack is found to significantly degrade the classifier's performance

The proposed evasion strategy is a gradient-based approach inspired by Golland's discriminative directions technique. This makes the approach applicable to any classifier with a differentiable discriminant function.

## Optimal evasion at test time

The classification algorithm is $f: \mathcal{X} \mapsto \mathcal{Y}$, which basically means a function $f$ maps from input space $\mathcal{X}$ to output space $\mathcal{Y}$. This is the proposed classifier in the paper.

The classifier $f$ output space $\mathcal{Y}$ equals $\{-1, +1\}$, meaning there are only two possible class labels the output of the classifier can have: -1 and +1 (likely "spam" or "not spam").

The training data $`\mathcal{D} = \{x_{i}, y_{i}\}_{i=1}^{n}`$ is a dataset with n samples. Each sample has features $x_{i}$ and a true label $y_{i}$. The data comes from some underlying distribution $`p(\mathbf{X}, Y)`$

In this paper, classifier output is defined as $f(x) = y^{c}$ to differentiate from the true label $y$. This $y^{c}$ output is given from a thresholding function $g : \mathcal{X} \mapsto \mathbb{R}$, meaning the classifier function's output is computed in two stages: the first stage is a continuous function that produces a real number (something like a confidence score) by the *continuous discriminant function* $g(x)$. The is then handed off to the second stage to be converted to a discrete value by the threshold. In the case of this paper, the threshold is $(g(x) < 0) \to (f(x) = -1)$ and $f(x) = +1$ otherwise. So $g(x)$ does NOT return the $y^{c}$ label but rather a confidence score, and $f(x)$ applies a threshold to get the predicted label $y^{c}$.

### Adversary model

Assumptions are made about the adversary's motives, knowledge of the target, and modification capability to the underlying target's data distribution.

The goal of the adversary should be to maximize some utility function, so in the evasion use-case, it would be manipulating a single sample that should be misclassified.

An attacker can simply find a sample $`\mathbf{x}`$ such that $`g(\mathbf{x}) < -\epsilon`$ for any $\epsilon > 0$. This means the attack sample only needs to just cross the decision boundary of the classifier for misclassification. But this can be easily defended against by adjusting the decision threshold to be more robust in classification.

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

#### Attack Scenarios

This paper defines two attack scenarios, **Perfect Knowledge (PK)** and **Limited Knowledge (LK)**:

- **Perfect Knowledge (PK)**: The goal of the adversary is to minimize $g(x)$ and they have knowledge of the feature space, type of classifier, and the trained model. The attacker can also transform attack points in the test data but this will be variable and is defined by $d_{max}$, which is the maximum distance a transformed attack point can be from the original attack sample. For example, the attack will be bounded by a certain number of words they can change in a spam email to evade a spam classifier. The distance measure will be defined as $d : \mathcal{X} \times \mathcal{X} \mapsto \mathbb{R}^{+}$ and is application specific.
- **Limited Knowledge (LK)**: The goal of the adversary is the same, minimize $g(x)$ under the same constraints that each transformed attack point must remain within a maximum distance of $d_{max}$ from the corresponding original attack sample. However the attacker will not know the learned classifier $f$ or its training data $\mathcal{D}$, consequently not be able to compute $g(x)$. The adversary will instead have a surrogate dataset $`\mathcal{D'} = \{\hat{x}_{i}, \hat{y}_{i}\}_{i=1}^{n_{q}}`$ of $n_{q}$ samples from the same underlying distribution $p(\mathbf{X}, Y)$ that $\mathcal{D}$ was drawn from. The adversary approximates $`g(\mathbf{x})`$ as $`\hat{g}(\mathbf{x})`$, where $`\hat{g}(\mathbf{x})`$ is the discriminant function of the surrogate classifier $\hat{f}$ learnt on $`\mathbf{\mathcal{D'}}`$ An adversary will want $\hat{f}$ to closely approximate the targeted classifier $f$. So it stands to reason that instead of using the true class labels $\hat{y_{i}}$ to train $\hat{f}$, the adversary can query $f$ with sample from $\mathcal{D'}$ and train $\hat{f}$ using the labels $`\hat{y}_{i}^{c} = f(\hat{\mathbf{x}}_{i})`$ for each $`\mathbf(x)_{i}`$, resulting in an $\hat{f}$ that closely mimics $f$ on the surrogate data.

#### Attack Strategy

Moving on to breaking down the attack strategy from the Biggio paper. Based on the attack scenarios we covered earlier, we can define our attack strategy in a formal way. For any target malicious sample $x^{0}$ (this is the target point that the attacker wants to reach in the feature space), the best attack strategy would be finding a sample $x^{*}$ to minimize $g(\cdot)$ or it's estimate $\hat{g}(\cdot)$. This is minimization is bounded of course by it's distance from $x^{0}$, which essentially means there is a hard limit to how much the adversary can *change* or *modify* a sample to be classified as the target point. This is all detailed by the following optimization problem:

$$\textbf{x}^{*} = \text{arg}\min_{x} \space \hat{g}(\textbf{x}) \space \space \text{s.t} \space\space d(\text{x}, \textbf{x}^{0}) \leq d_{max}$$

The paper details a problem simply minimizing the discrimnant function using gradient descent. I need to revisit this section to better understand the problem and why it is a problem in this context, as well as the significance of their solution...

Alright, its the next day and we're back! Let's try to tackle this section again...

So the authors are describing a problem with using gradient descent to find effective adversarial examples, $x^{0}$. This is the problems that they detail (spoken in laymen's terms):

1. **Local Optima Trapping**: So in the landscape where we want to apply optimization (in order to find adversarial examples), it is generally non-convex. What "non-convex" means is a space with many dips or minimums, with one being the true global minimum. Gradient descent will tend towards a minimum, but not always the global minimum. SO, your gradient descent optimization can get stuck at a local minimum and not the global, leading to an adversarial example that may not be optimal at all.
2. **Decision Boundary Issues**: This is a bit trickier to understand but gradient descent paths in the optimization algorithm can also get stuck at the descision boundary of a model. Stuff like following the descision boundary line and never actually crossing it from "malicious" to "benign" sample, defeating the purpose of our adversarial example. The GD paths could also lead to regions in the feature space that are "unrealistic", which means a combination of features that aren't really possible. In the case of emails, think of a combination of features like "spam words", "personal greeting", "known sender", and "sender reputation" being all true or highly likely. That wouldn't really happen (or atleast not with high probability, where $p(\textbf{x}) \approx 0)$). And even if they did exist, these feature combinations would have very low density in the legitimate training data. There is also the case where GD can lead to a point on the feature space that doesn't really look like a legitimate sample or data point.  
3. **Flat Gradient Regions**: This refers to regions in the feature space that are flat and cause the GD algorithm to level out and get stuck from finding any minimum. 

To combat all these challenges, Biggio et al. introduce an extra component to the optimization algorithm called *mimicry*. Let's dive into what actually makes up that component:

To avoid entering regions of the feature space that don't have legitimate and realistic points, why not just steer the optimization algorithm to target the regions of the feature space with higher density or cluster of legitimate samples? The authors do exactly this by using a **kernel density estimation (KDE)** to estimate where legitimate samples are densely populated in the overall feature space. Wiki says the KDE is defined as the following:

$$\hat{f}_{h}(x) = \frac{1}{nh}\sum_{i=1}^{n}k(\frac{x - x_{i}}{h})$$

- $\hat{f}_{h}(x)$ is the unknown density we want to estimate about the feature space at a given point $x$.
- $n$ is the number of data points/samples.
- $h$ is a smoothing parameter (sometimes called *bandwidth*), mainly to keep the KDE from being too volatile in its estimates at a given point $x$
- $k$ is the actual kernel, which is usually a non-negative function. Think of functions like the Guassian, Uniform, Cosine, etc. What this function actually does is it places a small "bump" at a given, legitimate point x, which is later averaged and if a given point x is near a lot of these bumps, it'll have a high density value, otherwise low density.

So the authors use the KDE to estimate density, but how do you tell the KDE what density to estimate? The author's goal is to find legitimate sample that look like the real data. And if we think back to what the data actually looks like, we remember that the data are these pairs that look like $\{x, y^{c}\}$ where $y^{c}$ is the true label of the sample that can either be +1 or -1. So, since we want to find densities in the feature space that are legitimate $x$ samples, we can define this as probability densities. Specifically, $p(\textbf{x}|y^{c} = -1)$, which computes the probability of a sample x given its label is -1. We now can point our optimization path towards density estimates where the probability of a sample $x$ being -1 is very high, ensuring this density region will have legitimate samples to attack with high confidence. 

Another paramter is introduced into this new optimization, which is $\lambda$. This is a "weight" that basically controls how much of the KDE effects the optimization search. Whenever $\lambda > 0$ then the optimization algo will prioritize searching high density regions of legitimate sample $x$, otherwise it will just follow traditional gradient descent throughout the feature space to find a sample $x$.

The final modified optimization problem becomes:

$$\text{arg}\min_{x} F(\textbf{x}) = \hat{g}(\textbf{x}) - \frac{\lambda}{n}\sum_{i|y_{i}^{c}=-1}k(\frac{\textbf{x} - \textbf{x}_{i}}{h}) \space\space \text{s.t} \space\space d(\textbf{x},\textbf{x}^{0}) < d_{max}$$

where $n$ is the number of benign samples $(y^{c} = -1)$ available to the adversary. This computes $g(\textbf{x})$. It is also important that we highlight the minus sign in front of the KDE component. The reason for this is because when minimizing $F(\textbf{x})$, we will maximize the KDE of the above equation, "pulling" our optimization path towards high-density legitimate regions. The optimization problem now becomes a two objective approach where you aim to "fool the classifier" AND "look like real legitimate data". Pretty elegant once you know what's going on lol.

THOUGHT: I think this paper opts to use the gradient descent algorithm in their attack strategy, but states that quadratic techniques like Newton's Method, BFGS and L-BFGS can be an approach. This could be a potential spin-off point worth experimenting with in the future once everything is implemented.


### Gradient descent attacks

Something mentioned in this section is a future work avenue: 

<blockquote>
However, note that if $g$ is non-differentiable or insufficiently smooth, one may still use the mimicry / KDE term of Eq. (2) as a search heuristic. This investigation is left to future work. 
</blockquote><br>

The authors provide an algorithm to solve the optimization problem posed by the previous section's equation. Let's break down this algorithm:

**Input:** $\textbf{x}^{0}$, the initial attack point; $t$, the step size; $\lambda$, the trade-off parameter; $\epsilon > 0$ a small constant. <br>
**Output:** $`\textbf{x}^{*}`$, the final attack point. <br>
1: $`m \leftarrow 0`$ <br>
2: **repeat:** <br>
3: &emsp; $`m \leftarrow m + 1`$ <br>
4: &emsp; Set $\nabla F(\textbf{x}^{m-1})$ to a unit vector aligned with $\nabla g(\textbf{x}^{m-1}) - \lambda \nabla p(\textbf{x}^{m-1}|y^{c} = -1)$. <br>
5: &emsp; $\textbf{x}^{m} \leftarrow \textbf{x}^{m-1} - t\nabla F(\textbf{x}^{m-1})$<br>
6: &emsp; **if** $d(\textbf{x}^{m}, \textbf{x}^{0}) > d_{max}$ **then** <br>
7: &emsp;&emsp; Project $\textbf{x}^{m}$ onto the boundary of the feasible region. <br>
8: &emsp; **end if** <br>
9: **until** $F(\textbf{x}^{m}) - F(\textbf{x}^{m -1}) < \epsilon$ <br>
10: **return:** $\textbf{x}^{*} = \textbf{m}^{m}$

Let's walk through this algorithm line by line and understand how we conduct the gradient-based attack and actually solved the predefined optimization problem:

Understanding the inputs:
- $\textbf{x}^{0}$: The starting point (your original malicious sample)
- $t$: The step size or how big of steps you take in each iteration of the algorithm.
- $\lambda$: Trade-off parameter - how much to care about mimicry vs evasion
- $\epsilon$: Convergence threshold (when to stop)

The first three steps (1-3) are just basic steps detailing iteration. The authors use $m$ as a counter variable that increments by 1.

Step 4 is really the heart of the algorithm. It's computing the gradient of the objective function $F(\textbf{x}) = \hat{g}(\textbf{x}) - \lambda \nabla p(\textbf{x}|y^{c} = -1)$:
- $\nabla g(\textbf{x}^{m-1})$: This is the gradient pointing towards "fool the classifier more".
- $\lambda \nabla p(\textbf{x}^{m-1}|y^{c} = -1)$: Gradient pointing towards "higher density legitimate regions"
- The minus sign in this step subtracts $\lambda$ times the density gradient so that we're actually moving towards higher density.
- We set this gradient of the objective function to a unit vector so that we normalize the direction. This allows for step size $t$ to control distance consistently. "break this down more"

Step 5 is standard gradient steps. simply moving in the computed direction

Steps 6, 7 and 8 are enforcing the constraint we discussed earlier: $d(\textbf{x}^{m}, \textbf{x}^{0}) > d_{max}$. This means the adversarial example can't be too different from the original. If our optimization computation does step outside of this feasible region, project back onto the boundary.

Finally, step 9 does a convergence check. Stop the algorithm once the improvements to the objective function $F(\textbf{x}^{m-1})$ are tiny.

