import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
### NLTK
import nltk # Procesamiento del lenguaje natural
nltk.download('averaged_perceptron_tagger') # Etiquetar las palabras
nltk.download('vader_lexicon') # Analisis de sentimiento
nltk.download('wordnet') # Categorizacion de las palabras
nltk.download('stopwords') # Stopwords
from nltk.corpus import wordnet
from nltk import pos_tag # Clasificacion de palabras
from nltk.corpus import stopwords # Eliminar palabras vacias
from nltk.tokenize import WhitespaceTokenizer # Tokenizar
from nltk.stem import WordNetLemmatizer # Lematizar
from nltk.stem.wordnet import WordNetLemmatizer # Lematizar
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Analisis de sentimiento
import re # Para reemplazar las contracciones con las palabras completas
from textblob import TextBlob

    
def app():
    
    c1, c2, c3 = st.beta_columns((1, 2, 1))

    with c1:
        st.write(" ")
        
    with c2:    
        st.markdown("<h1 style='text-align: center; color: white; font-size:1cm;'> ◽ Análisis de sentimiento para una review ◽ </h1>", unsafe_allow_html=True)
        st.markdown(" ")
        st.markdown(" ")
        
                
        # Se carga el modelo 
        modelo_sentimientos = 'model_procesamiento_lenguaje.pkl'

        model=''

        if model == '':
            with open(modelo_sentimientos, 'rb') as file:
                model = pickle.load(file)
                
        # Función de supresión de contracciones 
        def decontracted(frase):
            # Específicas
            frase = re.sub(r"won\'t", "will not", frase)
            frase = re.sub(r"can\'t", "can not", frase)
            # Generales
            frase = re.sub(r"n\'t", " not", frase)
            frase = re.sub(r"\'re", " are", frase)
            frase = re.sub(r"\'s", " is", frase)
            frase = re.sub(r"\'d", " would", frase)
            frase = re.sub(r"\'ll", " will", frase)
            frase = re.sub(r"\'t", " not", frase)
            frase = re.sub(r"\'ve", " have", frase)
            frase = re.sub(r"\'m", " am", frase)
            return frase    
        
                
        # Se crea la caja donde meter el comentario para su posterior análisis. Se deja con un ejemplo 
        ejemplo = "Stayed with parents, wife twin toddlers in two triple rooms. The hotel is easy to reach and the rooms were well placed well furnished. The best feature was extremely friendly helpful staff, particularly Ms. Annalucia Ms. Anna who were always ready to listen help out with big smiles. The breakfasts were very good, with good spread and the guests were made welcome to sit and eat at leisure (more important when you are with toddlers!) Would surely go back to Venice would happily stay again at Russo Palace. I would come back."
        
        input_review = st.text_area("Introduce un comentario o review en inglés (hasta 10.000 caracteres permitidos):", value = ejemplo, max_chars=10000, height=330)
        
        # Se crea un dataframe con el texto introducido
        df = pd.DataFrame()
        lista = [input_review]
        df['input_review'] = lista
        
        # Con Vader, del módulo nltk. Este módulo añade un score positivo, negativo, neutro y una integración de todas las anteriores
        analizador = SentimentIntensityAnalyzer()
        
        df["input_review"] = df['input_review'].apply(lambda x: decontracted(x))
        
        df["sentimiento"] = df['input_review'].apply(lambda x: analizador.polarity_scores(x))
        df = pd.concat([df.drop(['sentimiento'], axis=1), df['sentimiento'].apply(pd.Series)], axis=1)
        
        # Con TextBlob
        df['polaridad'] = df['input_review'].apply(lambda x: TextBlob(x).sentiment.polarity) 
        df['subjetividad'] = df['input_review'].apply(lambda x: TextBlob(x).sentiment.subjectivity) 
        
        # Características del texto
        df_caract = df.iloc[: , 1:]
        
        with st.beta_expander("Click aquí para mostrar las características de la review introducida: ⬇️"):
            st.write("Las características de la review en cuanto a negatividad, neutralidad, positividad, polaridad y subjetividad son las siguientes:")
            st.write(df_caract)
            st.write(" ")
            st.write(" ")
            
        x_in = df_caract # Con el conjunto de datos    
        
        # Análisis de sentimiento
        if st.button("Analizar sentimiento 🏁:"): 
        
            sentimiento = model.predict(x_in)
            
            if sentimiento == 1: 
                sentimiento = 'Positivo'
                
            elif sentimiento == 0: 
                sentimiento = 'Negativo'
                
            if sentimiento == 'Positivo':
                st.success('El sentimiento de esta review es: {} '.format(sentimiento)) 
                st.balloons()
                
            elif sentimiento == 'Negativo':
                st.error('El sentimiento de esta review es: {} '.format(sentimiento))
                
                
        # Ejemplo review negativa:
            """Walls extremely thin, you can hear everything. Excessive hoovering every morning outside bedroom. There. Is a bar, but no one to tend it."""
            
    with c3:
        st.write(" ")    
    