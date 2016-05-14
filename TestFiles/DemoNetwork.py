__author__ = 'vapaspen'
__name__ = 'DemoNetwork'
import numpy as np
from scipy import stats
from Layers.RNNLayer import ANN_Layer as ANN

def bulk_assign(left_dict, right_dict):
    """ Support function for combining complex dictionary's that may be in an empty initial state.

    :param left_dict: Dict to assign values too. Can be passed in as None valued Dict

    :param right_dict: Dict with values to combine or add to the left. Must not be None

    :return: returns the assign or sum'ed dict's
    """
    if left_dict is None:
        return right_dict

    results ={}
    for left_key, right_key in zip(left_dict, right_dict):
        results[left_key] = left_dict[left_key] + right_dict[right_key]


    return results

class FileClassifier(object):

    def __init__(self, hyparams):
        """ The main Class for the String Classifier. Initializes the network structure by the given hyparams. The network structure is fixed but individual elements of the network can be configured

        :param hyparams: this is a Dict type item that contains all of networks controllable structure items.
           hyparams = {"x_len": Neuron count for first layer. Most be the same as the Max input string size.
                "feature_h": Neuron Count
                "feature_y": Neuron Count
                "context_h": Neuron Count
                "context_y": Neuron Count
                "analysis_h1": Neuron Count
                "analysis_h2": Neuron Count
                "class_count": Neuron Count
                "confidence_threshold": 0.1
                "learning_rate":
                "MU":
               }

        :return:
        """

        self.hyperparams = hyparams

        self.loss = 0

        self.Main_Feature_Layer_In = ANN(self.hyperparams["x_len"], self.hyperparams["feature_h"], is_recurrent=False, learning_rate=self.hyperparams["learning_rate"], MU=self.hyperparams["MU"])
        self.Main_Feature_Layer_Out = ANN(self.hyperparams["feature_h"], self.hyperparams["feature_y"],is_recurrent=False, learning_rate=self.hyperparams["learning_rate"], MU=self.hyperparams["MU"])

        self.Context_Layer_In = ANN(self.hyperparams["feature_y"], self.hyperparams["context_h"], learning_rate=self.hyperparams["learning_rate"], MU=self.hyperparams["MU"])
        self.Context_Layer_Out = ANN(self.hyperparams["context_h"],self.hyperparams["context_y"], learning_rate=self.hyperparams["learning_rate"], MU=self.hyperparams["MU"])

        self.analysis_data = np.zeros((self.hyperparams["context_y"] + self.hyperparams["feature_y"],), dtype=float)
        self.Analysis_Layer_In = ANN(len(self.analysis_data), self.hyperparams["analysis_h1"], is_recurrent=False, learning_rate=self.hyperparams["learning_rate"], MU=self.hyperparams["MU"])
        self.Analysis_Layer_Hidden = ANN(self.hyperparams["analysis_h1"], self.hyperparams["analysis_h2"], is_recurrent=False, learning_rate=self.hyperparams["learning_rate"], MU=self.hyperparams["MU"])
        self.Analysis_Layer_Out = ANN(self.hyperparams["analysis_h2"], self.hyperparams["class_count"], is_recurrent=False, learning_rate=self.hyperparams["learning_rate"], MU=self.hyperparams["MU"])

    def string_to_float(self, input_as_string):
        if type(input_as_string) is not str:
            raise TypeError("Input of " + str(type(input_as_string)) + " is not the required type of str.")

        if(len(input_as_string) > self.hyperparams["x_len"]):
            raise ValueError("Input String of " + str(len(input_as_string)) + " bigger then given Max Input Size: " + str(self.hyperparams["x_len"]))

        data_as_string = np.array(input_as_string, 'c').view(np.uint8)
        padding = self.hyperparams["x_len"] - len(data_as_string)
        data_as_string = np.lib.pad(data_as_string, (0, padding), 'constant', constant_values=(0, 0))
        data_as_string = data_as_string.astype(float)

        return data_as_string

    def feed_foward_context(self, input_as_string):

        input_as_float = self.string_to_float(input_as_string)

        MFLI = self.Main_Feature_Layer_In.feed_foward(input_as_float)
        MFLI = stats.threshold(MFLI, threshmin=np.median(MFLI), newval=0)
        MFLO = self.Main_Feature_Layer_Out.feed_foward(MFLI)
        MFLO = stats.threshold(MFLO, threshmin=np.median(MFLO), newval=0)

        CLI = self.Context_Layer_In.feed_foward(MFLO)
        CLO = self.Context_Layer_Out.feed_foward(CLI)

        feed_foward_result_state = {
            "Main_Feature_Layer_In": MFLI,
            "Main_Feature_Layer_Out": MFLO,
            "Context_Layer_In": CLI,
            "Context_Layer_Out": CLO
        }
        return [CLO, feed_foward_result_state]

    def feed_foward_analysis(self, input_as_string, context_layer_results):

        input_as_float = self.string_to_float(input_as_string)

        MFLI = self.Main_Feature_Layer_In.feed_foward(input_as_float)
        MFLI = stats.threshold(MFLI, threshmin=np.median(MFLI), newval=0)
        MFLO = self.Main_Feature_Layer_Out.feed_foward(MFLI)
        MFLO = stats.threshold(MFLO, threshmin=np.median(MFLO), newval=0)

        self.analysis_data = np.concatenate((context_layer_results, MFLO), axis=0)

        ALI = self.Analysis_Layer_In.feed_foward(self.analysis_data)
        ALH = self.Analysis_Layer_Hidden.feed_foward(ALI)
        ALO = self.Analysis_Layer_Out.feed_foward(ALH)

        feed_foward_result_state = {
            "Main_Feature_Layer_In": MFLI,
            "Main_Feature_Layer_Out": MFLO,
            "analysis_data": self.analysis_data,
            "Analysis_Layer_In": ALI,
            "Analysis_Layer_Hidden": ALH,
            "Analysis_Layer_Out": ALO
    }

        return [ALO, feed_foward_result_state]

    def feed_foward(self, item_package_data):
        results = {}

        results["context"] = []
        context_FF = None
        for string_item in item_package_data["context"]:
            context_FF = self.feed_foward_context(string_item)
            results["context"].append(context_FF[1])

        results["analysis"] = []
        results["result"] = []
        results["confidence"] = []
        for string_item in item_package_data["analysis"]:
            analysis_FF = self.feed_foward_analysis(string_item, context_FF[0])
            results["result"].append(analysis_FF[0])
            results["analysis"].append(analysis_FF[1])


        for class_prob in results["result"]:
            for value in class_prob:
                if value > self.hyperparams["confidence_threshold"]:
                     results["confidence"].append((1 - value))
                else:
                    results["confidence"].append(value)

        return results

    def backpropagate(self, test_error, feed_foward_results):
        updates = {
            "Main_Feature_Layer_In": None,
            "Main_Feature_Layer_Out": None,
            "Context_Layer_In": None,
            "Context_Layer_Out": None,
            "Analysis_Layer_In": None,
            "Analysis_Layer_Hidden": None,
            "Analysis_Layer_Out": None
        }


        for i in range(len(feed_foward_results["analysis"])):

            updates["Analysis_Layer_Out"] = bulk_assign(
                updates["Analysis_Layer_Out"],
                self.Analysis_Layer_Out.get_gradients(
                    test_error[i],
                    neurons=feed_foward_results["analysis"][i]["Analysis_Layer_Out"]
                )
            )


            updates["Analysis_Layer_Hidden"] = bulk_assign(
                updates["Analysis_Layer_Hidden"],
                self.Analysis_Layer_Hidden.get_gradients(
                    updates["Analysis_Layer_Out"]["input"],
                    neurons=feed_foward_results["analysis"][i]["Analysis_Layer_Hidden"]
                )
            )

            updates["Analysis_Layer_In"] = bulk_assign(
                updates["Analysis_Layer_In"],
                self.Analysis_Layer_In.get_gradients(
                    updates["Analysis_Layer_Hidden"]["input"],
                    neurons=feed_foward_results["analysis"][i]["Analysis_Layer_In"]
                )
            )



            updates["Main_Feature_Layer_Out"] = bulk_assign(
                updates["Main_Feature_Layer_Out"],
                self.Main_Feature_Layer_Out.get_gradients(
                    (updates["Analysis_Layer_In"]["input"][self.hyperparams["context_y"]:]),
                    neurons=feed_foward_results["analysis"][i]["Main_Feature_Layer_Out"]
                )
            )

            updates["Main_Feature_Layer_In"] = bulk_assign(
                updates["Main_Feature_Layer_In"],
                self.Main_Feature_Layer_In.get_gradients(
                    updates["Main_Feature_Layer_Out"]["input"],
                    neurons=feed_foward_results["analysis"][i]["Main_Feature_Layer_In"]
                )
            )


            for j in range(max(0, min(25, len(feed_foward_results["context"])))):

                updates["Context_Layer_Out"] = bulk_assign(
                    updates["Context_Layer_Out"],
                    self.Context_Layer_Out.get_gradients(
                        updates["Analysis_Layer_In"]["input"][:self.hyperparams["context_y"]],
                        neurons=feed_foward_results["context"][j]["Context_Layer_Out"]
                    )
                )

                updates["Context_Layer_In"] = bulk_assign(
                    updates["Context_Layer_In"],
                    self.Context_Layer_In.get_gradients(
                        updates["Context_Layer_Out"]["input"],
                        neurons=feed_foward_results["context"][j]["Context_Layer_In"]
                    )
                )


                updates["Main_Feature_Layer_Out"] = bulk_assign(
                updates["Main_Feature_Layer_Out"],
                self.Main_Feature_Layer_Out.get_gradients(
                    (updates["Context_Layer_In"]["input"]),
                    neurons=feed_foward_results["context"][j]["Main_Feature_Layer_Out"]
                    )
                )

                updates["Main_Feature_Layer_In"] = bulk_assign(
                    updates["Main_Feature_Layer_In"],
                    self.Main_Feature_Layer_In.get_gradients(
                        updates["Main_Feature_Layer_Out"]["input"],
                        neurons=feed_foward_results["context"][j]["Main_Feature_Layer_In"]
                    )
                )

            self.Main_Feature_Layer_In.update_param(gradients=updates["Main_Feature_Layer_In"])
            self.Main_Feature_Layer_Out.update_param(gradients=updates["Main_Feature_Layer_Out"])
            self.Context_Layer_In.update_param(gradients=updates["Context_Layer_In"])
            self.Context_Layer_Out.update_param(gradients=updates["Context_Layer_Out"])
            self.Analysis_Layer_In.update_param(gradients=updates["Analysis_Layer_In"])
            self.Analysis_Layer_Hidden.update_param(gradients=updates["Analysis_Layer_Hidden"])
            self.Analysis_Layer_Out.update_param(gradients=updates["Analysis_Layer_Out"])

    def train_on_data(self, data, targets):

        if not type(targets) == type(np.array):
            targets = np.array(targets)


        sample = self.feed_foward(data)

        error = -(targets - sample["result"])

        self.loss = error

        self.backpropagate(error, sample)

        return sample