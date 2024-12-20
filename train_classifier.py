import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
#The data and labels are then converted into NumPy arrays for easy manipulation and model training.

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#data (input features) and labels (output targets) are divided into training and testing sets.
#The data is shuffled before splitting to ensure randomness.

model = RandomForestClassifier()
#Random Forest is an ensemble learning method that constructs multiple decision trees and combines their predictions to improve accuracy and reduce overfitting

model.fit(x_train, y_train)#trains the model based on the data

y_predict = model.predict(x_test)#now we need to predict or evaluate the test data

score = accuracy_score(y_predict, y_test)#here we are going to compare the accuracy rate blw the obtained and expected(y_test) results

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
