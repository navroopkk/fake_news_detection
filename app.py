import streamlit as st
import pickle
import numpy as np
model_p=pickle.load(open('model.pkl','rb'))

def predict_news(title,text,subject,date):
    prediction= model_p.predict([text])
    print(prediction)
    if prediction == 'fake':
        pred= False
    else:
        pred= True

    return pred

def main():
    st.title("Fake News Detection")
    html_temp= """
    <h2>Fake News Detection</h2>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    title= st.text_input("title","type here")
    text = st.text_input("text", "type here")
    subject = st.text_input("subject", "type here")
    date = st.text_input("date", "type here")

    safe_html = """
        <h2> The news article input is Valid </h2>
    """
    danger_html = """
            <h2> The news article input is Fake</h2>
    """
    if st.button("predict"):
        output=predict_news(title,text,subject,date)
        #st.success('the given news is',format(output))

        if output== True:
            st.markdown(safe_html,unsafe_allow_html=True)
        else:
            st.markdown(danger_html, unsafe_allow_html=True)

if __name__=='__main__':
    main()

