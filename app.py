import streamlit as st
import predict

def welcome():
	return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs

def main():

    st.title('Cirrhosis Prediction')
    st.write('Please enter the patient details below.')
    
    Age = st.slider('Age', min_value=0, max_value=100, value=50)
    
    Hepatomegaly = 0 if st.selectbox('Hepatomegaly:', ['Yes', 'No'])=='No' else 1
    
    Ascites = 0 if st.selectbox('Ascites:', ['Yes', 'No'])=='No' else 1
    
    Sex = 1 if st.selectbox('Sex:', ['Male', 'Female'])=='Male' else 0

    Cholestrol = st.slider('Cholestrol (U/L)', min_value=0, max_value=500, value=250)
    
    result =""
    
    if st.button("Predict"):
        predict.prediction(Age, Hepatomegaly, Ascites, Sex, Cholestrol)
        with open('ans.txt', 'r') as f:
            result=f.read()
        
        
    st.success('The output is {}'.format(result))
	
if __name__=='__main__':
	main()
