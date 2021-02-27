

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

        self.layer_ouput = []
        self.layer_input.append(X)

        for i in range(len(self.neurons_per_layer)):

            z = np.dot(self.layer_input[i], self.ws[i])

            if i != len(self.neurons_per_layer)-1:
                self.layer_ouput.append(z)
                input_ = sigmaoid_func(z)
                self.layer_input.append(input_)
                
            else: #output layer
                return softmax(z)

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
        # batch size
        N = X.shape[0]

        if self.use_improved_sigmoid:
            d_sigmaoid_func = d_imp_sigmaoid
            sigmaoid_func = imp_sigmaoid
        else:
            d_sigmaoid_func = d_sigmaoid
            sigmaoid_func = sigmaoid

        # calculate first gradient
        delta = -(targets-outputs) 
        A = sigmaoid_func(self.layer_ouput[-1])
        self.grads[-1] = A.T.dot(delta)/N 


        # calculate gradients for other hidden layers
        for i, z in enumerate(reversed(self.layer_ouput), start=1):
            d_f = d_sigmaoid_func(z)

            delta = d_f * delta.dot(self.ws[-i].T)
            self.grads[-i-1] = self.layer_input[-i-1].T.dot(delta)/N


        ## calculate dC/dw_kj