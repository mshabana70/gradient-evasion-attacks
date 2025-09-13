As an aspiring PhD student in AI, it's crucial to understand the mathematical foundations of various learning algorithms, especially when you're diving into areas like adversarial machine learning where the properties of these functions become paramount. Let's break down the discriminant function for a neural network, specifically a multi-layer perceptron (MLP) with a single hidden layer and a sigmoidal activation function, as described in the sources
.
At its core, a classifier, like an SVM or a neural network, assigns samples (represented as feature vectors) to predefined classes
. This assignment is typically done by thresholding a continuous discriminant function, often denoted as $g(\mathbf{x})$, where $\mathbf{x}$ is the input feature vector. For binary classification, we might say $f(\mathbf{x}) = -1$ if $g(\mathbf{x}) < 0$, and $+1$ otherwise
.
Now, for a neural network, let's consider the architecture described in the sources: a multi-layer perceptron with a single hidden layer of $m$ neurons and a sigmoidal activation function
. The structure of this discriminant function $g(\mathbf{x})$ can be decomposed step-by-step:
1. Output Layer Activation: The final output of the network, which forms the discriminant function $g(\mathbf{x})$, is often derived from a sigmoid function applied to a linear combination of the hidden layer's outputs. The source defines it as: $g(\mathbf{x}) = (1 + e^{-h(\mathbf{x})})^{-1}$
 Here, $h(\mathbf{x})$ is the weighted sum of the hidden layer's outputs, plus a bias. The function $(1 + e^{-z})^{-1}$ is the logistic sigmoid function, which squashes the output to a range between 0 and 1, suitable for probabilistic interpretations (e.g., as a posterior estimate $p(y_c=-1|\mathbf{x})$)
.
2. Hidden Layer Summation: The term $h(\mathbf{x})$ is itself a linear combination of the outputs of the $m$ hidden neurons: $h(\mathbf{x}) = \sum_{k=1}^{m} w_k \delta_k(\mathbf{x}) + b$
 Here, $w_k$ are the weights connecting the $k$-th hidden neuron to the output layer, and $b$ is the bias for the output layer.
3. Hidden Neuron Activation: Each hidden neuron $k$ also uses a sigmoidal activation function, taking a linear combination of the input features: $\delta_k(\mathbf{x}) = (1 + e^{-h_k(\mathbf{x})})^{-1}$
 This means $\delta_k(\mathbf{x})$ is the output of the $k$-th hidden neuron.
4. Input-to-Hidden Layer Summation: The input to each hidden neuron $k$, denoted $h_k(\mathbf{x})$, is a linear combination of the input features $\mathbf{x} = [x_1, \dots, x_d]$: $h_k(\mathbf{x}) = \sum_{j=1}^{d} v_{kj} x_j + b_k$
 Here, $v_{kj}$ are the weights connecting the $j$-th input feature to the $k$-th hidden neuron, and $b_k$ is the bias for the $k$-th hidden neuron.
Understanding the Gradient
In adversarial machine learning, particularly with evasion attacks, understanding the gradient of this discriminant function is crucial because gradient-based approaches are often used to find adversarial examples
. The gradient $\nabla g(\mathbf{x})$ tells us the direction of the steepest ascent of $g(\mathbf{x})$. To evade detection, an adversary wants to minimize $g(\mathbf{x})$ (or its estimate $\hat{g}(\mathbf{x})$), moving towards a region classified as legitimate, often by taking steps in the direction opposite to the gradient
.
For the neural network discriminant function defined above, the $i$-th component of the gradient $\nabla g(\mathbf{x})$ with respect to the $i$-th input feature $x_i$ is derived using the chain rule
:
$\frac{\partial g}{\partial x_i} = \frac{\partial g}{\partial h} \sum_{k=1}^{m} \left( \frac{\partial h}{\partial \delta_k} \frac{\partial \delta_k}{\partial h_k} \frac{\partial h_k}{\partial x_i} \right)$
Let's break down each term:
• $\frac{\partial g}{\partial h}$: This is the derivative of the output sigmoid with respect to its input $h(\mathbf{x})$. For $g(z) = (1+e^{-z})^{-1}$, its derivative is $g(z)(1-g(z))$. So, $\frac{\partial g}{\partial h} = g(\mathbf{x})(1-g(\mathbf{x}))$
.
• $\frac{\partial h}{\partial \delta_k}$: This is the derivative of $h(\mathbf{x})$ with respect to the output of the $k$-th hidden neuron, $\delta_k(\mathbf{x})$. From $h(\mathbf{x}) = \sum_{j=1}^{m} w_j \delta_j(\mathbf{x}) + b$, we get $\frac{\partial h}{\partial \delta_k} = w_k$
.
• $\frac{\partial \delta_k}{\partial h_k}$: Similar to the output layer, this is the derivative of the hidden neuron's sigmoid output $\delta_k(\mathbf{x})$ with respect to its input $h_k(\mathbf{x})$. So, $\frac{\partial \delta_k}{\partial h_k} = \delta_k(\mathbf{x})(1-\delta_k(\mathbf{x}))$
.
• $\frac{\partial h_k}{\partial x_i}$: This is the derivative of the $k$-th hidden neuron's input sum with respect to the $i$-th input feature. From $h_k(\mathbf{x}) = \sum_{j=1}^{d} v_{kj} x_j + b_k$, we get $\frac{\partial h_k}{\partial x_i} = v_{ki}$
.
Combining these terms, the $i$-th component of the gradient is:
$\frac{\partial g}{\partial x_i} = g(\mathbf{x})(1-g(\mathbf{x})) \sum_{k=1}^{m} \left( w_k \delta_k(\mathbf{x})(1-\delta_k(\mathbf{x}))v_{ki} \right)$
This formula allows for computing the gradient of the discriminant function with respect to each input feature, which is essential for gradient-descent-based evasion attacks against neural networks
. The sources note that neural networks appear "much more robust" against the proposed evasion attack compared to SVMs when only $g(\mathbf{x})$ is minimized, because their decision function might have "flat regions" where the gradient is close to zero, hindering gradient descent. This highlights why understanding the gradient's behavior across the feature space is so important in adversarial settings.