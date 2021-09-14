import streamlit as st
from PIL import Image

def app():
    # Para las columnas
    c1, c2, c3 = st.beta_columns((1, 2, 1))
    
    with c1:
        st.write(" ")
    
    with c2:
        st.markdown("<h1 style='text-align: center; color: white; font-size:1cm;'> ◽ Introducción a la aplicación ◽</h1>", unsafe_allow_html=True)
        st.header("1️⃣ Utilidad de la aplicación")
        
        #with st.beta_expander("Click para mostrar/ocultar el texto ⬇️"):
            
        html_utilidad = """
        <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
            <p>Conocer el porcentaje de ocupación de un hotel es uno de los datos más importantes a prever para los hoteleros. 
            Es por ello que surgió la idea de crear una aplicación a través de la cual se puede obtener la probabilidad de cancelación de las reservas.
            El resultado de dicha predicción ayudará a los hoteleros a estimar las habitaciones disponibles para vender, 
            las provisiones necesarias para el departamento de alimentos y bebidas, el número de camareros de piso necesarios, 
            el establecimiento de precios de cada habitación en base a la demanda, etc. En conclusión, para los profesionales de la
            industria hotelera puede ser de mucha utilidad conocer la probabilidad de cancelación de las reservas.</p>
            <p>Además, en la propia aplicación también se ha creado una interfaz en la que los hoteleros pueden introducir, mediante un archivo, 
            todas las opiniones de su/s hotel/es y, como resultado, la aplicación analiza y devuelve el sentimiento de estas opiniones. Esto sirve, entre otras cosas, para   
            proporcionar al experto hotelero qué sentimiento de sus clientes sobre su estancia predomina y los conceptos más relevantes introducidos 
            en esas opiniones.</p>
            </h2>
        </div>
        """
        st.markdown(html_utilidad, unsafe_allow_html=True)

        st.header("2️⃣ Estructura de la aplicación")
        
        with st.beta_expander("Click para mostrar/ocultar el texto ⬇️"):

            html_estructura = """
            <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
                La aplicación consta de cinco partes:  
                <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">  
                - <strong>Introducción a la app</strong>: se explica la utilidad de la aplicación, se proporciona acceso a los dos
                modelos entrenados y a sus respectivos datos de entrenamiento (probabilidad de cancelación y análisis de sentimiento) y se enuncia
                distintas métricas y datos interesantes sobre los modelos previamente comentados.</div>
                <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
                - <strong>Predicción de cancelación de reserva individual</strong>: consiste en una interfaz en la que se permite introducir datos correspondientes a una reserva
                y extraer en la propia aplicación la probabilidad de que esa reserva sea cancelada. El sentido de añadir esta pestaña es conocer los datos a introducir por el usuario
                para hallar la probabilidad de cancelación de reservas y, adicionalmente, proporcionar una breve noción de cómo funcionaría el modelo.
                <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
                - <strong>Predicción de cancelación de reservas múltiples</strong>: permite al usuario introducir un archivo .csv o .xlsx con los datos de diferentes reservas (200MB máximo). 
                Tras ello, la aplicación devuelve un análisis gráfico y el archivo (en el formato que el usuario prefiera) con una columna adicional en la que se indica la probabilidad de cancelación de cada una de 
                las reservas.
                <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
                - <strong>Análisis de sentimiento para una review</strong>: consiste en una interfaz en la que se permite al usuario introducir una opinión y extraer medidas
                de sentimiento, así como el propio sentimiento (positivo o negativo). La utilidad de esta interfaz es proporcionar al usuario una breve noción de como funcionaría
                el modelo de análisis de sentimiento. 
                <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
                - <strong>Análisis de sentimiento para múltiples reviews</strong>: permite al usuario introducir un archivo .csv o .xlsx con las distintas opiniones que los clientes
                hayan escrito sobre su estancia en el/los hotel/es. La aplicación crea una "nube de palabras", que extrae los principales conceptos de las opiniones en general y
                analiza el sentimiento de estas opiniones. Posteriormente, se crea un gráfico en el que se contempla la distribución de las opiniones positivas y negativas para 
                que el experto hotelero tenga información sobre cuál es el sentimiento predominante en cuanto a las opiniones de sus clientes. Además, se proporciona la posibilidad
                de descarga de un archivo en formato .xlsx o .csv del archivo con las opiniones y su correspondiente sentimiento en una columna adicional.
            </div>
            """    
            st.markdown(html_estructura, unsafe_allow_html=True)
        
        st.header("3️⃣ Modelo aplicado para predecir las cancelaciones")
        
        image_mapa = Image.open('mapa.png')
        image_imp_variables = Image.open('import-variables-cancelacion.png')
        
        
        with st.beta_expander("Click para mostrar/ocultar el texto ⬇️"):
            
            html_modelo_pred_canc = """
            <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
                <p>Los datos que se han utilizado para entrenar el modelo se encuentran disponibles en el siguiente link (click en la imagen) de la plataforma Kaggle: 
                <a href="https://www.kaggle.com/jessemostipak/hotel-booking-demand"><img src="https://www.analyticsvidhya.com/wp-content/uploads/2015/06/kaggle-logo-transparent-300-300x136.png" 
                width=60" height="25"></a></p>
                <p>El modelo se ha desarrollado en un jupyter notebook, el cual, para el interés del usuario de esta aplicación, se ha
                colgado en Google Colab. Se puede encontrar en el siguiente enlace:
                <a href="https://colab.research.google.com/drive/12M9c3olwU1GdEDmyg9umSnZc3J2YN1EI?usp=sharing"><img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" 
                width=60" height="25"></a></p>
                <p>La precisión del modelo entrenado es de un <strong>98.29%</strong>. Está formado de <strong>21 variables</strong>, cuyos datos son generales para todos los hoteles. 
                Es por ello que este modelo es genérico para distintos hoteles. Es importante mencionar que este modelo se podría mejorar si se entrena para los datos de 
                un hotel en concreto, puesto que particularizaría el modelo para las reservas de ese hotel.</p>
                <p>Los hoteles que más han aportado en el entrenamiento del modelo se encuentran sobre todo en Europa, en especial Portugal. En el Colab hay un mapa interactivo
                donde se puede visualizar en qué países se alojan de forma mayoritaria los huéspedes recogidos en el conjunto de datos. El mapa es el siguiente:</p> 
                """
                
            html_modelo_pred_canc2 ="""    
                <p> En la pestaña <strong>"Predicción de cancelación de reserva individual"</strong> se pueden observar (e introducir) valores para las distintas variables que el modelo
                usa para la predicción. Esta predicción vendrá en forma de probabilidad, concretamente será la probabilidad en tanto por ciento de que esa reserva sea cancelada. En el modelo
                de Colab, las predicciones son (1) si la reserva se cancela y (0) si no se cancela. Sin embargo, en la aplicación se ha decidido sacar la probabilidad de que 
                las reservas sean canceladas, dejando así en manos del usuario (profesional hotelero) la decisión, según la probabilidad, de si finalmente considerar si
                esa reserva va a ser cancelada o no para tomar las decisiones que considere oportunas.</p>
                <p>Mediante la importancia de variables se puede interpretar el modelo, observando qué variables son las que juegan un papel más importante en cuanto a que la 
                reserva sea cancelada o no. A continuación se plasma la importancia de las variables, así como su interpretación según el modelo:</p>
                """
            html_modelo_pred_canc3 = """    
                <p>- <strong>Lead_time o número de días transcurridos entre la fecha de reserva y la fecha de llegada</strong>: Es la que mayor correlación tiene con que
                una reserva se cancele. Cuanto más son los días transcurridos entre fecha de reserva y de llegada, más probabilidad existirá de que cancele la reserva.</p>
                <p>- <strong>Total_of_special_requests o peticiones especiales del cliente</strong>: Cuantas más peticiones extra sean realizadas por el cliente, mayor probabilidad
                de cancelación existe.</p>
                <p>- <strong>Required_car_parking_spaces o plazas de aparcamiento requeridas por el cliente</strong>: Cuantas más plazas de parking pida el cliente, mayor probabilidad
                de que cancele la reserva. Esta variable está correlacionada también con lead_time, por lo que que esta variable esté correlacionada positivamente con la cancelación de
                la reserva puede ser ocasionado por cambios en planificación del viaje e imprevistos.</p>
                <p>- <strong>Booking_changes o número de cambios en la reserva</strong>: Al realizar más cambios en la reserva, mayor probabilidad tiene en cancelar la reserva de hotel.
                El cliente tiene cambios en sus necesidades, y, al igual que modifica la reserva, también a su vez busca otras opciones, las cuales podrían interesarle más.</p>
                <p>- <strong>Previous_cancellations o número de cancelaciones previas realizadas por el cliente anteriores a la reserva actual</strong>: A mayor número de cancelaciones
                previas en la reserva actual, mayor probabilidad de que esa reserva se acabe cancelando definitivamente.</p>
                <p>- <strong>Is_repeated_guest o si es un cliente que repite su reserva</strong>: Si es un cliente que repite reserva, es más probable que cancele debido a que puede buscar
                a su vez nuevos destinos que le puedan convencer.</p>
                <p><strong>Para más información acceder a: </strong><a href="https://colab.research.google.com/drive/12M9c3olwU1GdEDmyg9umSnZc3J2YN1EI?usp=sharing"><img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" 
                width=60" height="25"></a></p>
                </div>
                """    

            st.markdown(html_modelo_pred_canc, unsafe_allow_html=True)    
            st.image(image_mapa, caption='Países más utilizados por los clientes del conjunto de datos')
            st.markdown(html_modelo_pred_canc2, unsafe_allow_html=True)
            st.image(image_imp_variables, caption='Importancia de variables en el conjunto de datos')            
            st.markdown(html_modelo_pred_canc3, unsafe_allow_html=True)
        
        st.header("4️⃣ Modelo aplicado para el análisis de sentimiento")
        
        image_curvaroc = Image.open('curva-roc.png')
        
        with st.beta_expander("Click para mostrar/ocultar el texto ⬇️"):
            
            html_modelo_analisis_sentimiento = """
            <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
                <p>Los datos que se han utilizado para entrenar el modelo se encuentran disponibles en el siguiente link (click en la imagen) de la plataforma Kaggle: 
                <a href="https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe"><img src="https://www.analyticsvidhya.com/wp-content/uploads/2015/06/kaggle-logo-transparent-300-300x136.png" 
                width=60" height="25"></a></p>
                <p>El modelo se ha desarrollado en un jupyter notebook, el cual, para el interés del usuario de esta aplicación, se ha
                colgado en Google Colab. Se puede encontrar en el siguiente enlace:
                <a href="https://colab.research.google.com/drive/1EROJC0Rid1f-RZRmiO3QqsIYwa3sQ24g?usp=sharing"><img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" 
                width=60" height="25"></a></p>
                <p>El modelo de análisis de sentimiento tiene un área bajo la curva ROC de <strong>87%</strong>, tal y como se puede observar a continuación:</p>
                """
            html_modelo_analisis_sentimiento2 = """
                <p>El dataset con el que se ha realizado el entrenamiento consta de
                un total de 493.457 opiniones positivas (95.68%) y un total de 22.281 opiniones negativas (4.32%). Es por este motivo que el modelo se ha balanceado para entrenarlo.</p>
                <p>La matriz de confusión tiene los siguientes valores: <strong>3.483 verdaderos negativos, 898 falsos negativos, 75.878 verdaderos positivos y 22.890 falsos positivos.</strong>
                La precisión general del modelo es del <strong>77%</strong>.</p>
                <p><strong>Para más información acceder a: </strong><a href="https://colab.research.google.com/drive/1EROJC0Rid1f-RZRmiO3QqsIYwa3sQ24g?usp=sharing"><img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" 
                width=60" height="25"></a></p>
            </div>
            """   
            
            st.markdown(html_modelo_analisis_sentimiento, unsafe_allow_html=True)
            st.image(image_curvaroc, caption='Área bajo la curva ROC para modelo de análisis de sentimiento')
            st.markdown(html_modelo_analisis_sentimiento2, unsafe_allow_html=True)
    
    with c3:
        st.write()
    
        