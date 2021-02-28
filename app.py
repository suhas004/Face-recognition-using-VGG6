# -*- coding: utf-8 -*-
"""

Created on Sat Oct 31 10:08:43 2020

@author: suhas
"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image,ImageOps



@st.cache(allow_output_mutation=True)
def get_model():
    from tensorflow.keras.models import load_model
    model=load_model('model.h5')
    return model



def predict(image):

    model = get_model()
    size = (224, 224)
    image = ImageOps.fit(image, size)
    image = np.asarray(image)
    image_reshape = image[np.newaxis, ...]
    result = model.predict(image_reshape)
    return result



def main():
    html_temp = """
    <div style="background-color:cyan;padding:10px">
    <h2 style="color:black;text-align:center;">Face Recognition using Transfer Learning</h2>
    </div>
           """

    st.markdown(html_temp, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])

    if uploaded_file is not None:

        image= Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        result=predict(image)

        if st.button("Predict"):

            st.write("Classifying...")

            st.write(result)

            if result[0][0] >=0.45:
                prediction = 'Dhoni'
                st.success(prediction)

            elif result[0][1] > 0.45:
                prediction = 'kohli'
                st.success(prediction)



            if result[0][2] >0.45:
                prediction = 'sachin'
                st.success(prediction)




if __name__ == '__main__':
    main()