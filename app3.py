import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import io
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
### NLTK
import nltk # Procesamiento del lenguaje natural
nltk.download('averaged_perceptron_tagger') # Etiquetar las palabras
nltk.download('vader_lexicon') # Analisis de sentimiento
nltk.download('wordnet') # Categorizacion de las palabras
from nltk.corpus import wordnet
from nltk import pos_tag # Clasificacion de palabras
from nltk.corpus import stopwords # Eliminar palabras vacias
from nltk.tokenize import WhitespaceTokenizer # Tokenizar
from nltk.stem import WordNetLemmatizer # Lematizar
from nltk.stem.wordnet import WordNetLemmatizer # Lematizar
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Analisis de sentimiento
import string
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

    
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
                
                
        # Se crea la caja donde meter el comentario para su posterior análisis. Se deja con un ejemplo 
        ejemplo = "Stayed with parents, wife twin toddlers in two triple rooms. The hotel is easy to reach and the rooms were well placed well furnished. The best feature was extremely friendly helpful staff, particularly Ms. Annalucia Ms. Anna who were always ready to listen help out with big smiles. The breakfasts were very good, with good spread and the guests were made welcome to sit and eat at leisure (more important when you are with toddlers!) Would surely go back to Venice would happily stay again at Russo Palace. I would come back."
        
        input_review = st.text_area("Introduce un comentario o review en inglés (hasta 10.000 caracteres permitidos):", value = ejemplo, max_chars=10000, height=330)
        
        # Funciones para procesar el comentario 
        # Etiquetado de nombres, verbos, adjetivos o adverbios
        def get_wordnet_pos(pos_tag):
            if pos_tag.startswith('J'):
                return wordnet.ADJ
            elif pos_tag.startswith('V'):
                return wordnet.VERB
            elif pos_tag.startswith('N'):
                return wordnet.NOUN
            elif pos_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
            
        # Función para limpiar el texto
        def limpiar_texto(texto):
            # Poner el texto en minúsculas
            texto = texto.lower()
            # Tokenizar el texto y quitar los signos de puntuación
            texto = [word.strip(string.punctuation) for word in texto.split(" ")]
            # Quitar las palabras que contengan números
            texto = [word for word in texto if not any(c.isdigit() for c in word)]
            # Quitar las stop words
            stop = stopwords.words('english')
            texto = [x for x in texto if x not in stop]
            # Quitar los tokens vacíos
            texto = [t for t in texto if len(t) > 0]
            # Pos tags
            pos_tags = pos_tag(texto)
            # Lematizar el texto
            texto = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
            # Quitar las palabras con sólo una letra
            texto = [t for t in texto if len(t) > 1]
            # Unir todo
            texto = " ".join(texto)
            return(texto)
        
        # Se crea un dataframe con el texto introducido
        df = pd.DataFrame()
        lista = [input_review]
        df['input_review'] = lista
        
        # Limpieza de texto 
        df['review_limp'] = df['input_review'].apply(lambda x: limpiar_texto(x))
        
        # Con Vader, del módulo nltk. Este módulo añade un score positivo, negativo, neutro y una integración de todas las anteriores
        analizador = SentimentIntensityAnalyzer()
        
        df["sentimiento"] = df['review_limp'].apply(lambda x: analizador.polarity_scores(x))
        df = pd.concat([df.drop(['sentimiento'], axis=1), df['sentimiento'].apply(pd.Series)], axis=1)
        
        # Con TextBlob
        df['polaridad'] = df['review_limp'].apply(lambda x: TextBlob(x).sentiment.polarity) 
        df['subjetividad'] = df['review_limp'].apply(lambda x: TextBlob(x).sentiment.subjectivity) 
        
        # Características del texto
        df_caract = df.iloc[: , 2:]
        
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
    