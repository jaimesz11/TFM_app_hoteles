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
nltk.download('stopwords') # Stopwords
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
from wordcloud import WordCloud
st.set_option('deprecation.showPyplotGlobalUse', False) # Quitar cuadro de warning
st.set_option('deprecation.showfileUploaderEncoding', False) # Quitar cuadro de warning

def app():
    # Columnas
    c1, c2, c3 = st.beta_columns((1, 2, 1))

    with c1:
        st.write(" ")
        
    with c2:    
        st.markdown("<h1 style='text-align: center; color: white; font-size:1cm;'> ◽ Análisis de sentimiento para múltiples reviews ◽ </h1>", unsafe_allow_html=True)
        st.markdown(" ")
        st.markdown(" ")
        
                
        # Se carga el modelo 
        modelo_sentimientos = 'model_procesamiento_lenguaje.pkl'

        model=''

        if model == '':
            with open(modelo_sentimientos, 'rb') as file:
                model = pickle.load(file)
                
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
        
            # Función para dibujar la nube de palabras
        def show_wordcloud(data, title = None):
            wordcloud = WordCloud(
                background_color = 'black',
                max_words = 200,
                max_font_size = 40, 
                scale = 3,
                random_state = 42,
                contour_width=0
            ).generate(str(data))

            fig = plt.figure(1, figsize = (50, 50))
            plt.axis('off')
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig)
        
        # La columna de las reviews debe llamarse 'review'
        html_columna_nec = """
        <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
            Los datos a introducir deben tener, al menos, la columna <strong>'review'</strong>. Si esta columna no se encuentra, la aplicación dará error:        
        """
        st.markdown(html_columna_nec, unsafe_allow_html=True)
        
        with st.beta_expander("Campos/nombre de columnas de los datos a introducir ⬇️"):    
            st.write("La estructura de los datos a introducir debe ser la siguiente (campos):")
            df_estructura = pd.DataFrame(columns=['review'])
            st.markdown(df_estructura.columns.to_list())
        
        # Carga de archivo csv o xlsx
        carga = st.file_uploader(
            label='Introduce tu archivo CSV o Excel (200MB máximo):',
            type = ['csv', 'xlsx']
        )
        
        # Lectura de los datos
        global df
        if carga is not None:
            print(carga)
            
            # Al leer los archivos, que coja sólo la coumna review
            try: 
                df = pd.read_csv(carga, usecols = ['review'])
                
            except Exception as e:
                df = pd.read_excel(carga, usecols = ['review'])
                
            # Se convierte la columna review a string
            df = df.astype(str)
            copia_df = df.copy() # Se copia el dataframe para posteriormente pegar las columnas con el análisis de sentimiento (positivo o negativo)
            
            # Se imprime la nube de palabras con la función cargada anteriormente
            st.write("La nube de palabras, que destaca los conceptos principales de las reviews, es la siguiente:")
            show_wordcloud(df["review"])
            
            
            # Análisis de sentimiento    
            st.subheader('Análisis de sentimiento 🏁:')
            st.write(" ")
            
            # Limpieza del texto de las reviews 
            df['review_limp'] = df['review'].apply(lambda x: limpiar_texto(x))
            
            # Con Vader, del módulo nltk. Este módulo añade un score positivo, negativo, neutro y una integración de todas las anteriores
            analizador = SentimentIntensityAnalyzer()
            
            df["sentimiento"] = df['review_limp'].apply(lambda x: analizador.polarity_scores(x))
            df = pd.concat([df.drop(['sentimiento'], axis=1), df['sentimiento'].apply(pd.Series)], axis=1)
            
            # Con TextBlob
            df['polaridad'] = df['review_limp'].apply(lambda x: TextBlob(x).sentiment.polarity) 
            df['subjetividad'] = df['review_limp'].apply(lambda x: TextBlob(x).sentiment.subjectivity) 
            
            # Características del texto
            df_caract = df.iloc[: , 2:]
            
            # Análisis de sentimiento
            sentimiento = model.predict(df_caract)
            
            # Dataframe de análisis de sentimiento
            df_sentimientos = pd.DataFrame({'sentimiento': sentimiento})
            
            # Negativa si es 0 y positiva si es 1
            df_sentimientos['sentimiento'] = df_sentimientos['sentimiento'].map({0 : 'Negativo', 1 : 'Positivo'})  
            
            # Añadir la tabla de predicciones a la tabla original de las reviews 
            copia_df['sentimiento'] = df_sentimientos['sentimiento']
            
            st.write("Primeras reviews con análisis de sentimiento:")
            st.write(copia_df.head())
            
            # Para el gráfico:
            rev_positiva = df_sentimientos[df_sentimientos['sentimiento'] == 'Positivo']
            rev_positiva_conteo = rev_positiva.shape[0]
            rev_negativa = df_sentimientos[df_sentimientos['sentimiento'] == 'Negativo']
            rev_negativa_conteo = rev_negativa.shape[0]
            labels = ['Reviews positivas', 'Reviews negativas']
            colors = ['darkolivegreen', 'firebrick']
            sizes = [rev_positiva_conteo, rev_negativa_conteo]
            
            st.write("Distribución de reviews positivas y negativas del archivo introducido:")
            # Gráfico Pie Chart para ver distribuciones de sentimiento de reviews
            fig = go.Figure(data = ([go.Pie(labels = labels, values = sizes, pull=[0, 0.1])]))
            fig.update_traces(textinfo='value', textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=1)))
            st.plotly_chart(fig)
            
            # Descarga Excel/CSV
            st.subheader('Selecciona formato del archivo de descarga con el sentimiento de las reviews:')
            opciones = st.radio('', ('EXCEL (.xlsx)','CSV (.csv)'))
            
            # Para csv
            if opciones == 'CSV (.csv)':
                df_descarga = pd.DataFrame(copia_df)
                df_descarga
                csv = df_descarga.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode() 
                link = f'<a href="data:file/csv;base64,{b64}" download="analisis_sentimiento.csv">Descargar archivo csv</a>'
                st.markdown(link, unsafe_allow_html=True)
                
            # Para excel
            if opciones == 'EXCEL (.xlsx)':
                towrite = io.BytesIO()
                downloaded_file = copia_df.to_excel(towrite, encoding='utf-8', index=False, header=True)
                towrite.seek(0)  
                b64 = base64.b64encode(towrite.read()).decode()  
                link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="analisis_sentimiento.xlsx">Descargar archivo excel</a>'
                st.markdown(link, unsafe_allow_html=True)
                

    with c3: 
        st.write(" ")
            