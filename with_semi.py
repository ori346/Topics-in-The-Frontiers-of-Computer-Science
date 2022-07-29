import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from dpmmpython.dpmmwrapper import DPMMPython
import json
from sklearn.semi_supervised import LabelPropagation

dim = 128
pca = PCA(dim)
rng = np.random.RandomState(42)
label_prop_model = LabelPropagation()

#using standart mnist
data = pd.read_csv('mnist_test.csv')
labels = data.pop('label') 
labels = labels.to_numpy().transpose()

#normlize the data
data = data.to_numpy()

#using 80% of the data as trainig set
x_train = data[:8000]
y_train = labels[:8000]
x_test = data[8000:]
y_test = labels[8000:]


f = open('result.json')
j = json.load(f) 
result = j['res']

result = np.array(result)

def relabel_data(labeled):
    #implimting the majority rule 
    vote = [0] * 10
    tr = []
    for i in range(10): 
        clu = np.where(result == i)[0]
        for j in clu: 
            if labeled[j] != -1:
                vote[labeled[j]] += 1
            else: 
                tr.append(j)  
    
        mi = 0 
        mv = 0 
        for k in range(10):
            if mv < vote[k]:
                mi = k
                mv = vote[k]
        labeled[tr] = mi 
        vote = [0] * 10
        tr = []
    
    
    return labeled

pca_data = pca.fit_transform(x_train)

def semi_supervised_test(percent): 
    random_unlabeled_points = rng.rand(len(y_train)) < 1 - percent / 100
    labeled = np.copy(y_train)
    labeled[random_unlabeled_points] = -1

    #use the majority rule
    relabeled_result = relabel_data(labeled)

    #use LabelPropagation
    #relabeled_result = label_prop_model.fit(pca_data, labeled).transduction_ 

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(x_train, relabeled_result, epochs=3)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_acc * 100)
    return val_acc * 100


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