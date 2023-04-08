import pickle
import numpy as np


def prediction(Age, Hepatomegaly, Ascites, Sex, Cholestrol):
    pickle_in = open('C:/Users/ASUS/Desktop/GSA/ensemble.pkl', 'rb')
    classifier = pickle.load(pickle_in)
    input_data = np.array([[Age, Hepatomegaly, Ascites, Sex, Cholestrol]])
    return classifier.predict(input_data)

   

#print(prediction('Y', 'Y', 'F', 261))
#print(type(prediction(20, 0, 1, 1, 261)))