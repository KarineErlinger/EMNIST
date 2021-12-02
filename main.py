from numpy.core.defchararray import title
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.utils import *
import tensorflow as tf
from keras.models import *
from PIL import Image
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Reconnaissance de lettres",
    page_icon=":pencil:",
)

hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("Reconnaissance de lettres")


def predict(image):
    #dictionnaire pour traduire les labels en lettres correspondantes
    dictionnaire = dict({0:"?", 1:"a", 2:"b", 3:"c", 4:"d", 5:"e", 6:"f", 7:"g", 8:"h", 9:"i", 10:"j", 11:"k",
                    12:"l", 13:"m", 14:"n", 15:"o", 16:"p", 17:"q", 18:"r", 19:"s", 20:"t", 21:"u", 22:"v",
                    23:"w", 24:"x", 25:"y", 26:"z"})

    #processing de l'image
    image = Image.fromarray((image[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    image = image.convert('L')
    image = (tf.keras.utils.img_to_array(image)/255)
    image = image.reshape(1,28,28,1)
    x_2 = tf.convert_to_tensor(image)

    #appel au modèle
    model = load_model('fat_model_corrected.hdf5')

    #prédiction des labels
    prediction = model.predict(image)
    prediction = np.around(prediction, 2)
    prediction = pd.DataFrame(prediction)
    prediction.rename(columns=dictionnaire, inplace=True)
    prediction = prediction.T.sort_values(by=0, ascending=False)
    prediction = prediction.rename(columns={0:'Pourcentage'})
    # st.write(prediction)
    return prediction


# réglage possible de la taille du pinceau dans le menu latéral
stroke_width = st.sidebar.slider("Stroke width: ", 10, 50, 20)
# le canva
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#fff",
    background_color="#000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

result = st.button("Predict")

if canvas_result.image_data is not None and result:
    # st.image(canvas_result.image_data)
    img = canvas_result.image_data
    outputs = predict(img)

    # Converting index to equivalent letter
    ind_max = outputs.idxmax().values[0] # Index of the max element

    #Barre de progrès un peu classe pour faire patienter le chaland
    progress_bar = st.progress(0)
    for i in range(20):
        progress_bar.progress(i + 1)
        time.sleep(0.01)

    #Message rabaissant
    if ind_max == "?":
        st.markdown("<h1 style='text-align: center; color: red;'>Merci d'apprendre à dessiner.</h1>", unsafe_allow_html=True)

    else:
    
        #Sortie du label seul
        st.markdown("<h3 style = 'text-align: center;'>Prediction : {}<h3>".format(
            ind_max), unsafe_allow_html=True)

        st.dataframe(outputs.head(10)) #affichage dataframe
        st.balloons() #affichage ballons dégueux


