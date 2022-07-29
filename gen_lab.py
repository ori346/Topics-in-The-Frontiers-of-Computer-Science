import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from dpmmpython.dpmmwrapper import DPMMPython
from sklearn.preprocessing import normalize
import codecs, json

#clustring the data and save it into result.json


pca = PCA(128)
data = pd.read_csv('mnist_test.csv')
labels = data.pop('label') 
labels = labels.to_numpy().transpose()
data = data.to_numpy()

x_train = data[:8000]
y_train = labels[:8000]

data_pca = pca.fit_transform(x_train)
data_pca = normalize(data_pca)

labels_res,_,results= DPMMPython.fit(data_pca.transpose() , 10.0 ,verbose = True)

x = labels_res.tolist()
x = list(map(lambda x : x - 1 , x))
b = {'res' : x}

json.dump(b, codecs.open('result.json', 'w', encoding='utf-8'))


