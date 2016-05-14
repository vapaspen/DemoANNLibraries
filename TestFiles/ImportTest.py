from Layers.RNNLayer import ANN_Layer as RNN
from scipy import signal
from scipy import ndimage
from Layers.ConvoLayer import Convo_Layer as Convo
import numpy as np
from DemoNetwork import FileClassifier
import SampleData as data

parm = {"x_len": 100,
        "feature_h": 100,
        "feature_y": 100,
        "context_h": 2,
        "context_y": 2,
        "analysis_h1": 200,
        "analysis_h2": 25,
        "class_count": 2,
        "confidence_threshold": 0.1,
        "learning_rate":0.1,
        "MU":0.001
       }
MAX_ITER = 100

FC = FileClassifier(parm)
sum_mean_loss = 0
for i in range(MAX_ITER):
    sample_path = np.random.choice(len(data.Train_Data), len(data.Train_Data))

    epoch_loss = 0
    for item in sample_path:

        sample = data.Train_Data[str(item)]
        train_result = FC.train_on_data(sample, sample["train"])

        epoch_loss += np.mean(np.abs(FC.loss))

    sum_mean_loss = epoch_loss/len(sample_path)
    if i % 10 == 0:
        print(str(i) + ": Loss: " + str(sum_mean_loss) +" Current Raw: " + str(train_result["result"]))

for item in data.Test_Data:

    sample = data.Test_Data[item]
    result = FC.feed_foward(sample)
    found_class = 'Part of the same Game'
    count = 0
    print(result["result"])
    '''
    for j in result["result"]:
        if j[0] < .1:
            found_class = 'For a Different Game'
        print(found_class + " Confidence: " + str(result["confidence"]))
        print("Item: " + str(data.Test_Data[item]["analysis"][count]) + " Raw: " + str(j))
        print()
        count += 1
    '''