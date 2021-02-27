import numpy as np
import utils
import typing
np.random.seed(1)

def sigmaoid(z: np.ndarray):
    return 1/(1+np.exp(-z))

def d_sigmaoid(z: np.ndarray):
    return sigmaoid(z)*(1-sigmaoid(z))


def imp_sigmaoid(z: np.ndarray):
    return 1.7159 * np.tanh((2/3)*z)

def d_imp_sigmaoid(z: np.ndarray):
    return 1.7159 * (2/3) * (1- np.tanh((2/3)*z) ** 2)

def softmax(z: np.ndarray):
    exps = np.exp(z)
    softmax = exps/exps.sum(axis=1, keepdims=1)
    return softmax

def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    m = 33.55 # mean of X_train
    s = 78.87 # standard deviation of X_train
    
    X = (X-m)/s # normalize images
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1) # adding bias column

    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """

    #loss = -(targets*np.log(outputs) + (1-targets)*np.log(1-outputs))
    #loss = np.mean(loss)

    loss = targets * np.log(outputs)
    loss = -np.sum(loss, axis=1, keepdims=1)
    loss = np.mean(loss)

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return loss


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)

            # w = np.zeros(w_shape)
            if use_improved_weight_init:
                std = 1/np.sqrt(w_shape[0])
                w = np.random.normal(0, std, w_shape)
            else:
                w = np.random.uniform(-1,1, w_shape) # init with random weights

            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

        self.layer_output = [] #the z output from each hidden layer

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """

        if self.use_improved_sigmoid:
            sigmaoid_func = imp_sigmaoid
        else:
            sigmaoid_func = sigmaoid

        self.layer_output = []

        for i in range(len(self.neurons_per_layer)):

            try: _input = sigmaoid_func(self.layer_output[i-1])
            except IndexError: _input = X

            z = np.dot(_input, self.ws[i])

            if i != len(self.neurons_per_layer)-1:
                self.layer_output.append(z)
                
            else: #output layer
                return softmax(z)

        ##output layer 1
        #self.zj = np.dot(X, self.ws[0])
        #self.A = sigmaoid_func(self.zj)

        ##ouput layer 2
        #self.zk = np.dot(self.A, self.ws[1])
        #Y = softmax(self.zk)

        #return Y

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        N = X.shape[0]

        if self.use_improved_sigmoid:
            d_sigmaoid_func = d_imp_sigmaoid
            sigmaoid_func = imp_sigmaoid
        else:
            d_sigmaoid_func = d_sigmaoid
            sigmaoid_func = sigmaoid

        # calculate first gradient
        delta = -(targets-outputs) 
        A = sigmaoid_func(self.layer_output[-1])
        self.grads[-1] = A.T.dot(delta)/N 


        # calculate gradients for other hidden layers
        for i, z in enumerate(reversed(self.layer_output), start=1):
            d_f = d_sigmaoid_func(z)
            
            try: _input = sigmaoid_func(self.layer_output[-i-1])
            except IndexError: _input = X

            delta = d_f * delta.dot(self.ws[-i].T)
            self.grads[-i-1] = _input.T.dot(delta)/N

        ## batch size
        #N = X.shape[0]

        #self.A = sigmaoid(self.layer_output[0])
        #self.zj = self.layer_output[0]

        ## calculate dC/dw_kj
        #d_k = -(targets-outputs) #delta_k
        #grad_k = self.A.T.dot(d_k)/N
        #self.grads[1] = grad_k
        ##self.grads[1] = np.mean(grad_k, axis=0, keepdims=1).transpose()

        ## calculate dC/dw_ji
        #if self.use_improved_sigmoid:
        #    d_f = d_imp_sigmaoid(self.zj)
        #else:
        #    d_f = d_sigmaoid(self.zj)
        
        #d_j = d_f * d_k.dot(self.ws[1].T) #delta_j
        #grad_j = X.T.dot(d_j)/N
        #self.grads[0] = grad_j
        ##self.grads[0] = np.mean(grad_j, axis=0, keepdims=1).transpose()

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """

    out = np.zeros((Y.shape[0], num_classes))
    out[np.arange(Y.shape[0]), Y.T] = 1

    return out

def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
