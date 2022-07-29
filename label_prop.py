import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics.cluster import normalized_mutual_info_score
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('mnist_test.csv')
real = data.pop('label') 
real = real.to_numpy().transpose()
data = data / 255
label_prop_model = LabelPropagation()
rng = np.random.RandomState(42)

def semi_supervised_test(percent): 
    random_unlabeled_points = rng.rand(len(real)) < 1 - percent / 100
    labels = np.copy(real)
    labels[random_unlabeled_points] = -1
    res = label_prop_model.fit(data, labels).transduction_
    score = normalized_mutual_info_score(res.astype(int), np.array(real),average_method='arithmetic')
    #print(score * 100) #print the score because we want to know the real value
    return score * 100 



if __name__ == '__main__':
    
    percents = range(10 , 110 , 10) 
    results = []
    for percent in percents: 
        results.append(semi_supervised_test(percent))

    plt.plot(percents, results)
  
    # naming the x axis
    plt.xlabel('percent of labeled data')
    # naming the y axis
    plt.ylabel('score')
  
    # giving a title to my graph
    plt.title('graph of score with respect to the percents of labeled data')
  
    # function to show the plot
    plt.show()
    







