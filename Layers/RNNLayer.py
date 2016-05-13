__author__ = 'vapaspen'
__name__ = 'RNNLayer'

import numpy as np

class ANN_Layer(object):
    """
    ANN/RNN use is_recurrent to switch between the two.
    """

    def __init__(self, X_count, H_count, is_recurrent=True, MU=0.01, learning_rate=0.01):
        rng = np.random.RandomState(1234)

        self.Layer = {}

        self.Layer["X_count"] = X_count
        self.Layer["H_count"] = H_count
        self.Layer["is_recurrent"] = is_recurrent
        self.Layer["learning_rate"] = learning_rate

        self.Layer["Last_input"] = np.zeros((self.Layer["X_count"],), dtype=float)

        self.Layer["last_deltal_update"] = None
        self.Layer["MU"] = MU

        self.Layer["nodes"] = {}


        self.Layer["nodes"]["neurons"] = np.zeros((self.Layer["H_count"],), dtype=float)
        self.Layer["nodes"]["neurons_delta"] = np.zeros_like(self.Layer["nodes"]["neurons"])

        self.Layer["nodes"]["bias"] = np.zeros((self.Layer["H_count"],), dtype=float)
        self.Layer["nodes"]["bias_delta"] = np.zeros_like(self.Layer["nodes"]["bias"])
        self.Layer["nodes"]["bias_delta_last"] = np.zeros_like(self.Layer["nodes"]["bias"])

        self.Layer["nodes"]["input_weights"] = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (self.Layer["X_count"] + self.Layer["H_count"])),
                    high=np.sqrt(6. / (self.Layer["X_count"] + self.Layer["H_count"])),
                    size=(self.Layer["X_count"], self.Layer["H_count"])
                ),
                dtype=float
        )
        self.Layer["nodes"]["input_weights_delta"] = np.zeros_like(self.Layer["nodes"]["input_weights"])
        self.Layer["nodes"]["input_weights_delta_last"] = np.zeros_like(self.Layer["nodes"]["input_weights"])

        if self.Layer["is_recurrent"]:
            self.Layer["nodes"]["hidden_weights"] = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (self.Layer["H_count"] + self.Layer["H_count"])),
                    high=np.sqrt(6. / (self.Layer["H_count"] + self.Layer["H_count"])),
                    size=(self.Layer["H_count"], self.Layer["H_count"])
                ),
                dtype=float
        )
            self.Layer["nodes"]["hidden_weights_delta"] = np.zeros_like(self.Layer["nodes"]["hidden_weights"])
            self.Layer["nodes"]["hidden_weights_delta_last"] = np.zeros_like(self.Layer["nodes"]["hidden_weights"])

            self.Layer["nodes"]["last_neurons"] = np.zeros((self.Layer["H_count"],), dtype=float)

    def feed_foward(self, input_nodes):
        """
        The basic feed forward Process for this layer.

        :param input_nodes:
            One dimensional Numpy array with the same length as the configured X size

        :return:
            Activated One dimensional numpy array

        :raises:
            Exception when input length doesnt match input settings.
        """

        if not len(input_nodes) == self.Layer["X_count"]:
            raise Exception("Input given not the same shape as Layer settings.")

        self.Layer["Last_input"] = input_nodes
        if self.Layer["is_recurrent"]:
            self.Layer["nodes"]["last_neurons"] = self.Layer["nodes"]["neurons"]
            self.Layer["nodes"]["neurons"] = np.tanh(np.dot(input_nodes, self.Layer["nodes"]["input_weights"]) + np.dot(self.Layer["nodes"]["neurons"], self.Layer["nodes"]["hidden_weights"]) + self.Layer["nodes"]["bias"])
        else:
            self.Layer["nodes"]["neurons"] = np.tanh(np.dot(input_nodes, self.Layer["nodes"]["input_weights"]) + self.Layer["nodes"]["bias"])
        return self.Layer["nodes"]["neurons"]

    def reset_bias(self):
        """
        Function to reset all of the Gradients for this layer
        :return: void
        """
        self.Layer["nodes"]["neurons_delta"] = np.zeros_like(self.Layer["nodes"]["neurons"])
        self.Layer["nodes"]["bias_delta"] = np.zeros_like(self.Layer["nodes"]["bias"])
        self.Layer["nodes"]["bias_delta_last"] = self.Layer["nodes"]["bias_delta"]

        self.Layer["nodes"]["input_weights_delta"] = np.zeros_like(self.Layer["nodes"]["input_weights"])
        self.Layer["nodes"]["input_weights_delta_last"] =self.Layer["nodes"]["input_weights_delta"]

        if self.Layer["is_recurrent"]:
            self.Layer["nodes"]["hidden_weights_delta"] = np.zeros_like(self.Layer["nodes"]["hidden_weights"])
            self.Layer["nodes"]["hidden_weights_delta_last"] = self.Layer["nodes"]["hidden_weights_delta"]


    def get_gradients(self, test_error, neurons=None):

        if not neurons is None:
            if not len(neurons) == len(self.Layer["nodes"]["neurons"]):
                raise ValueError("Length of given neuron array, " + str(len(neurons)) + " does not match the Length in this object: " + str(len(self.Layer["nodes"]["neurons"])) + ".")
            self.Layer["nodes"]["neurons"] = neurons

        self.reset_bias()

        gradients = {}

        self.Layer["nodes"]["neurons_delta"] = test_error
        self.Layer["nodes"]["bias_delta"] += (1 - self.Layer["nodes"]["neurons"] * self.Layer["nodes"]["neurons"]) * test_error
        self.Layer["nodes"]["input_weights_delta"] += self.Layer["nodes"]["bias_delta"] * np.array([self.Layer["Last_input"]]).T

        gradients["bias_delta"] = self.Layer["nodes"]["bias_delta"]
        gradients["input_weights_delta"] = self.Layer["nodes"]["input_weights_delta"]

        if self.Layer["is_recurrent"]:
            self.Layer["nodes"]["hidden_weights_delta"] += self.Layer["nodes"]["bias_delta"] * np.array([self.Layer["nodes"]["last_neurons"]]).T
            gradients["hidden_weights_delta"] = self.Layer["nodes"]["hidden_weights_delta"]


        gradients["input"] = np.dot(self.Layer["nodes"]["bias_delta"], self.Layer["nodes"]["input_weights"].T)

        return gradients



    def update_param(self, gradients=None):

        if not gradients == None:
            self.Layer["nodes"]["bias_delta"] = gradients["bias_delta"]
            self.Layer["nodes"]["input_weights_delta"] = gradients["input_weights_delta"]

            if self.Layer["is_recurrent"]:
                self.Layer["nodes"]["hidden_weights_delta"] = gradients["hidden_weights_delta"]


        if self.Layer["is_recurrent"]:
            for param, delta_param, delta_last in zip([self.Layer["nodes"]["bias"], self.Layer["nodes"]["input_weights"], self.Layer["nodes"]["hidden_weights"]],
                                          [self.Layer["nodes"]["bias_delta"], self.Layer["nodes"]["input_weights_delta"], self.Layer["nodes"]["hidden_weights_delta"]],
                                           [self.Layer["nodes"]["bias_delta_last"], self.Layer["nodes"]["input_weights_delta_last"], self.Layer["nodes"]["hidden_weights_delta_last"]]):
                np.clip(delta_param, -10, 10, out=delta_param)
                param -= (self.Layer["learning_rate"] * delta_param) + (delta_last * self.Layer["MU"])

        else:
            for param, delta_param, delta_last in zip([self.Layer["nodes"]["bias"], self.Layer["nodes"]["input_weights"]],
                                          [self.Layer["nodes"]["bias_delta"], self.Layer["nodes"]["input_weights_delta"]],
                                           [self.Layer["nodes"]["bias_delta_last"], self.Layer["nodes"]["input_weights_delta_last"]]):
                np.clip(delta_param, -10, 10, out=delta_param)
                param -= (self.Layer["learning_rate"] * delta_param) + (delta_last * self.Layer["MU"])

    def backpropagate(self, test_error):
        """
        :param test_error:
            One dimensional Numpy array that represents the Gradients of this layer Post activation.
        :return:
            One dimensional Numpy array that represents the Gradients of this layer Post activation of the Next Layer
        """
        gradients = self.get_gradients(test_error)

        self.update_param()
        """
        self.reset_bias()
        self.Layer["nodes"]["neurons_delta"] = test_error
        self.Layer["nodes"]["bias_delta"] += (1 - self.Layer["nodes"]["neurons"] * self.Layer["nodes"]["neurons"]) * test_error
        self.Layer["nodes"]["input_weights_delta"] += self.Layer["nodes"]["bias_delta"] * np.array([self.Layer["Last_input"]]).T


        if not self.Layer["is_recurrent"]:

            #Update all Layer Parameters at once.
            for param, delta_param, delta_last in zip([self.Layer["nodes"]["bias"], self.Layer["nodes"]["input_weights"]],
                                          [self.Layer["nodes"]["bias_delta"], self.Layer["nodes"]["input_weights_delta"]],
                                           [self.Layer["nodes"]["bias_delta_last"], self.Layer["nodes"]["input_weights_delta_last"]]):
                np.clip(delta_param, -10, 10, out=delta_param)
                param -= (self.Layer["learning_rate"] * delta_param) + (delta_last * self.Layer["MU"])

        else:
            self.Layer["nodes"]["hidden_weights_delta"] += self.Layer["nodes"]["bias_delta"] * np.array([self.Layer["nodes"]["last_neurons"]]).T

            #Update all Layer Parameters at once.
            for param, delta_param, delta_last in zip([self.Layer["nodes"]["bias"], self.Layer["nodes"]["input_weights"], self.Layer["nodes"]["hidden_weights"]],
                                          [self.Layer["nodes"]["bias_delta"], self.Layer["nodes"]["input_weights_delta"], self.Layer["nodes"]["hidden_weights_delta"]],
                                           [self.Layer["nodes"]["bias_delta_last"], self.Layer["nodes"]["input_weights_delta_last"], self.Layer["nodes"]["hidden_weights_delta_last"]]):
                np.clip(delta_param, -10, 10, out=delta_param)



                param -= (self.Layer["learning_rate"] * delta_param) + (delta_last * self.Layer["MU"])


        return np.dot(self.Layer["nodes"]["bias_delta"], self.Layer["nodes"]["input_weights"].T)
        """
        return gradients["input"]
