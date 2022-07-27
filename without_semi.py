import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

pca = PCA(128)
kmeans = KMeans(n_clusters=10, random_state=0)
rng = np.random.RandomState(42)

#using standart mnist
data = pd.read_csv('mnist_test.csv')
labels = data.pop('label') 
labels = labels.to_numpy().transpose()
#normlize the data
data = data.to_numpy()
data = data / 255

def test(N):
    total = 80 * N   
    x_train = data[:total]
    y_train = labels[:total]
    x_test = data[8000:]
    y_test = labels[8000:]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    return val_acc * 100

if __name__ == '__main__':
    
    percents = range(10 , 110 , 10) 
    results = []
    for percent in percents: 
        results.append(test(percent))

    plt.plot(percents, results)
  
    # naming the x axis
    plt.xlabel('percent of labeled data')
    # naming the y axis
    plt.ylabel('score')
  
    # giving a title to my graph
    plt.title('graph of score with respect to the percents of labeled data')
  
    # function to show the plot
    plt.show()
