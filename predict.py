import pickle
import numpy as np


def prediction(Age, Hepatomegaly, Ascites, Sex, Cholestrol):
    pickle_in = open('Lr_model.pkl', 'rb')
    classifier = pickle.load(pickle_in)
    input_data = np.array([[Age, Hepatomegaly, Ascites, Sex, Cholestrol]])
    with open('ans.txt','w') as f:
        print(classifier.predict(input_data))
        f.write(str(classifier.predict(input_data)[0]))
    # classifier.predict(input_data)

   

#prediction('Y', 'Y', 'F', 261)
#prediction(20, 0, 1, 1, 261)