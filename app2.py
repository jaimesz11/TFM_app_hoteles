import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def app():
    
    c1, c2, c3 = st.beta_columns((1, 2, 1))

    with c1:
        st.write(" ")
    
    with c2:
        # Título
        st.markdown("<h1 style='text-align: center; color: white; font-size:1cm;'> ◽ Predicción de cancelación de reservas múltiples ◽ </h1>", unsafe_allow_html=True)
        st.write(" ")
        st.write(" ")
        
        html_columnas_modelo = """
        <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
            Los datos a introducir deben tener, al menos, los siguientes campos con alguno de los valores que se introducen en la anterior pestaña de la presente aplicación.
            Si estas columnas no son encontradas en el archivo introducido, no se podrán realizar las predicciones y dará error:        
        """
        st.markdown(html_columnas_modelo, unsafe_allow_html=True)
        
        with st.beta_expander("Campos/nombre de columnas de los datos a introducir ⬇️"):    
            st.write("La estructura de los datos a introducir debe ser la siguiente (nombres de columnas):")
            df_estructura = pd.DataFrame(columns=['hotel', 'meal', 'reserved_room_type', 'customer_type', 'year', 'month',
                                                'day', 'lead_time', 'deposit_type', 'arrival_date_week_number',
                                                'arrival_date_day_of_month', 'stays_in_weekend_nights',
                                                'stays_in_week_nights', 'is_repeated_guest', 'previous_cancellations',
                                                'previous_bookings_not_canceled', 'adr', 'required_car_parking_spaces',
                                                'total_of_special_requests', 'Total_Guests', 'booking_changes'])
            st.markdown(df_estructura.columns.to_list())
            st.write('**Ir a la página anterior ("Predicción de cancelación de reserva individual") si se tiene alguna duda sobre la estructura de las variables.**')
        
        # Función de predicciones
        def model_prediction(x_in, model):
            preds = model.predict_proba(x_in)[:, 1]
            return preds
        
        
        # Se carga el modelo 
        modelo_cancelaciones = 'model_prediccion_cancelaciones.pkl'
        model=''

        if model == '':
            with open(modelo_cancelaciones, 'rb') as file:
                model = pickle.load(file)
        
        # Configuración
        st.set_option('deprecation.showfileUploaderEncoding', False)
        
        # Carga de archivo 
        carga = st.file_uploader(
            label='Introduce tu archivo CSV o Excel (200MB máximo):',
            type = ['csv', 'xlsx']
        )
        
        # Lectura de los datos
        global df
        if carga is not None:
            print(carga)
            
            # Al leer los archivos, que coja sólo las columnas que nos interesan para las predicciones (por si hay añadidas en el archivo más variables)
            try: 
                df = pd.read_csv(carga, usecols = ['hotel', 'meal', 'reserved_room_type', 'customer_type', 'year', 'month',
                                                'day', 'lead_time', 'deposit_type', 'arrival_date_week_number',
                                                'arrival_date_day_of_month', 'stays_in_weekend_nights',
                                                'stays_in_week_nights', 'is_repeated_guest', 'previous_cancellations',
                                                'previous_bookings_not_canceled', 'adr', 'required_car_parking_spaces',
                                                'total_of_special_requests', 'Total_Guests', 'booking_changes'])
            except Exception as e:
                df = pd.read_excel(carga, usecols = ['hotel', 'meal', 'reserved_room_type', 'customer_type', 'year', 'month',
                                                'day', 'lead_time', 'deposit_type', 'arrival_date_week_number',
                                                'arrival_date_day_of_month', 'stays_in_weekend_nights',
                                                'stays_in_week_nights', 'is_repeated_guest', 'previous_cancellations',
                                                'previous_bookings_not_canceled', 'adr', 'required_car_parking_spaces',
                                                'total_of_special_requests', 'Total_Guests', 'booking_changes'])
        
        # Visualización      
        if carga is not None:
            try: 
                st.write("Las primeras filas del archivo introducido son las siguientes:")
                st.write(df.head(5))
                
            except Exception as e:
                print(e) 
        
            df2 = df.copy() # Copio el dataframe introducido con el archivo para cambiarle las variables en el nuevo dataframe
            
            # Conversión a numéricas de los tipo texto 
            # Hotel
            df2['hotel'] = df2['hotel'].map({'Resort Hotel' : 0, 'City Hotel' : 1})

            # Meal
            df2['meal'] = df2['meal'].map({'BB: Bed & Breakfast' : 0, 'FB: Full Board (desayuno, comida y cena)' : 1, 
                                        'HB: Half Board (Desayuno y otra comida, normalmente cena)': 2, 'SC': 3, 'Sin definir': 4})
            
            # Reserved_room_type
            df2['reserved_room_type'] = df2['reserved_room_type'].map({'A' : 1, 'B': 8, 'C': 0, 'D': 2, 'E': 3, 'G': 4, 'H': 6, 'L': 7, 'F': 5})
            
            # Customer_type
            df2['customer_type'] = df2['customer_type'].map({'Transitorios' : 0, 'Con contrato asignado' : 1, 
                                        'Transitorios asociados a otros': 2, 'Grupos': 3})            

            # Deposit_type
            df2['deposit_type'] = df2['deposit_type'].map({'Sin depósito' : 0, 'Reembolsable' : 1, 'No reembolsable': 2})    
            
            # Is_repeated_guest
            df2['is_repeated_guest'] = df2['is_repeated_guest'].map({'No' : 0, 'Sí' : 1})  
            
            
            # Normalización de variables lead_time, arrival_date_week_number, arrival_date_day_of_month, adr
            df2['lead_time'] = np.log(df2['lead_time'] + 1)
            df2['arrival_date_week_number'] = np.log(df2['arrival_date_week_number'] + 1)
            df2['arrival_date_day_of_month'] = np.log(df2['arrival_date_day_of_month'] + 1)
            df2['adr'] = np.log(df2['adr'] + 1)
            # Se sustituyen los NaN con la media de Adr
            df2['adr'] = df2['adr'].fillna(value = df2['adr'].mean())
            
            
            # Gráfico de distribución de precio según el tipo de hotel
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write('Gráfico de distribución de precio medio (adr) según el tipo de habitación reservada (reserved_room_type):')
            
            city_hotel = df[df['hotel'] == 'City Hotel']
            resort_hotel = df[df['hotel'] == 'Resort Hotel']
            
            
            figs = go.Figure()
            figs.add_trace(go.Box(
                y=city_hotel['adr'],
                x=city_hotel['reserved_room_type'],
                name='City Hotel',
                marker_color='#d8b70d'
            ))
            
            figs.add_trace(go.Box(
                y=resort_hotel['adr'],
                x=resort_hotel['reserved_room_type'],
                name='Resort Hotel',
                marker_color='#3d5b59'
            ))
            
            figs.update_layout(
            xaxis=dict(title='normalized moisture', zeroline=False),
            boxmode='group'
            )
            
            st.write(figs)

        # Predicciones
        if carga is not None:
            # Realización de predicciones
            x_in = df2
            predictions = model_prediction(x_in, model)
            
            # Resultado de las predicciones
            resultado_predicciones = pd.DataFrame({'Probabilidad cancelación (%)':predictions*100})
            
            # Insertarlo en la tabla original
            df["Probabilidad_cancelación_%"] = predictions*100
            
            # Predicción de cancelación y análisis de distribución de valores   
            st.subheader('Predicción de cancelación de reservas y análisis de distribución 🏁:')
            st.write(" ")
                
            st.write("Primeros registros con predicciones (en la última variable):")
            
            st.write(df.head())
            
            # Gráfico de distribución de las predicciones
            html_concepto = """
                <h2 style="line-height:150%; text-align:justify; font-size: 0.4cm;">
                Si supera el 70% de probabilidad de cancelación, será muy probable que esa reserva sea cancelada. Es por ello que, para añadir más información, 
                se ha construido un gráfico que separa las predicciones cuyos resultados son probabilidades de cancelación mayores y menores de 70%, con el objetivo
                de obtener rápidamente información acerca de cómo se distribuyen las predicciones de cancelación.
                </h2>
            </div>
            """   
                
            st.write(html_concepto, unsafe_allow_html=True)
            st.write("Distribución de probabilidades de la predicción (distinguiendo entre si supera el 70% o no):")
            
            # Para el gráfico:
            si_cancela = df[df['Probabilidad_cancelación_%'] >= 70]
            si_cancela_conteo = si_cancela.shape[0]
            no_cancela = df[df['Probabilidad_cancelación_%'] < 70]
            no_cancela_conteo = no_cancela.shape[0]
            labels = ['<70% (no cancela)', '>70% (sí cancela)']
            colors = ['darkolivegreen', 'firebrick']
            sizes = [no_cancela_conteo, si_cancela_conteo]
        
            fig = go.Figure(data = ([go.Pie(labels = labels, values = sizes, pull=[0, 0.1])]))
            fig.update_traces(textinfo='value', textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=1)))

            st.plotly_chart(fig)
            
            # Descarga Excel/CSV
            st.subheader('Selecciona formato del archivo de descarga con las predicciones:')
            opciones = st.radio('', ('EXCEL (.xlsx)','CSV (.csv)'))
            
            # Para csv
            if opciones == 'CSV (.csv)':
                df_descarga = pd.DataFrame(df)
                df_descarga
                csv = df_descarga.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode() 
                link = f'<a href="data:file/csv;base64,{b64}" download="predicciones.csv">Descargar archivo csv</a>'
                st.markdown(link, unsafe_allow_html=True)
            # Para excel
            elif opciones == 'EXCEL (.xlsx)':
                towrite = io.BytesIO()
                downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
                towrite.seek(0)  
                b64 = base64.b64encode(towrite.read()).decode()  
                link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="predicciones.xlsx">Descargar archivo excel</a>'
                st.markdown(link, unsafe_allow_html=True)
            


    with c3:
        st.write(" ")
            
        






