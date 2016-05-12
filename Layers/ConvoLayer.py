__author__ = 'vapaspen'
__name__ = 'ConvoLayer'


import numpy as np
import math

class Convo_Layer:
    """
    Standard Convolution layer.
    """

    def __init__(self, visual_space_definition, depth_of_field, step_size, MU=0.01):
        """ Standard Convolution layer that contains all weights, and node values.

        :param visual_space_definition:
            Max index for each dimension in the visual space. Taken as an array for each Max Index
        :param depth_of_field:
            Number of Nodes viewed on each step. Taken as an array of with the Max index for that dimension in position.
        :param step_size:
            Number of nodes skip after after each step. Taken as an array of with the Max index for that dimension in position.


        :return:
        """
        self.Layer = {}

        self.Layer["H_count"] = self.check_parm_and_get_H(visual_space_definition, depth_of_field, step_size)

        self.Layer["visual_space_definition"] = visual_space_definition
        self.Layer["depth_of_field"] = depth_of_field
        self.Layer["step_size"] = step_size

        self.Layer["Last_input"] = np.zeros_like((self.Layer["visual_space_definition"],), dtype=float)

        self.Layer["last_deltal_update"] = None
        self.Layer["MU"] = MU

        self.Layer["nodes"] = {}


        self.Layer["nodes"]["neurons"] = np.zeros((tuple(self.Layer["H_count"])), dtype=float)
        self.Layer["nodes"]["neurons_delta"] = np.zeros_like(self.Layer["nodes"]["neurons"])

        self.Layer["nodes"]["bias"] = np.zeros(tuple(self.Layer["H_count"]), dtype=float)
        self.Layer["nodes"]["bias_delta"] = np.zeros_like(self.Layer["nodes"]["bias"])
        self.Layer["nodes"]["bias_delta_last"] = np.zeros_like(self.Layer["nodes"]["bias"])

        self.Layer["nodes"]["input_weights"] = np.random.random((self.Layer["depth_of_field"]))*0.1
        self.Layer["nodes"]["input_weights_delta"] = np.zeros_like(self.Layer["nodes"]["input_weights"])
        self.Layer["nodes"]["input_weights_delta_last"] = np.zeros_like(self.Layer["nodes"]["input_weights"])


    def check_parm_and_get_H(self, visual_space_definition, depth_of_field, step_size):
        if not len(visual_space_definition) == len(depth_of_field) or not len(depth_of_field) == len(step_size):
            raise Exception("Arguments do not have the same dimensions. visual_space_definition: " + str(len(visual_space_definition)) + ", depth_of_field: " + str(len(depth_of_field)) + ", step_size: " + str(len(step_size)))

        H = np.zeros_like(visual_space_definition)
        for i in range(len(visual_space_definition)):
            H_remainder = ((visual_space_definition[i] - depth_of_field[i]) % step_size[i])
            if not H_remainder == 0:
                raise Exception("The visual field of will fall off the end.")
            H[i] = math.floor((visual_space_definition[i] - (depth_of_field[i] - step_size[i])) / step_size[i])
        return H[::-1]

    def feed_foward(self, input_layer):
        """

        :param input_layer:
        :return:
        """



        count = np.zeros_like(self.Layer["H_count"])
        depth_start = len(self.Layer["nodes"]["neurons"]) - 1

        def loop_on_data(depth):

            for i in range(len(self.Layer["nodes"]["neurons"][depth])):
                count[depth] = i
                if depth > 0:
                    new_depth = depth - 1
                    loop_on_data(new_depth)
                else:
                    slice_definitions = ([slice(count[step_dimension], count[step_dimension] + self.Layer["depth_of_field"][step_dimension], None) for step_dimension in range(len(self.Layer["depth_of_field"]))])
                    print(input_layer[tuple(slice_definitions)])

        loop_on_data(depth_start)

