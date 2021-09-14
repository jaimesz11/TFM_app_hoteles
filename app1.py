import streamlit as st
import numpy as np
import pickle
from sklearn import svm
import streamlit as st
import altair as alt
import plotly.express as px
np.set_printoptions(suppress=True) # Para quitar la notación científica en los arrays


def app():

    # Path del modelo preentrenado
    modelo_cancelaciones = 'model_prediccion_cancelaciones.pkl'

    # Se recibe el modelo y se devuelve la predicción (en probabilidad)
    def model_prediction(x_in, model):
        x = np.asarray(x_in).reshape(1,-1)
        preds = model.predict_proba(x)[:, 1]
        return preds

    model=''

    # Se carga el modelo
    if model == '':
        with open(modelo_cancelaciones, 'rb') as file:
            model = pickle.load(file)


# Lectura de datos
    # Columnas
    
    c1, c2, c3 = st.beta_columns((1, 2, 1))

    with c1:
        st.write(" ")
        
    with c2:
        # Título
        st.markdown("<h1 style='text-align: center; color: white; font-size:1cm;'> ◽ Predicción de cancelación de reserva individual ◽ </h1>", unsafe_allow_html=True)
        st.markdown(" ")
        st.markdown(" ")
        
        # Introducir variables
        Hotel = st.radio("1. Tipo de hotel (hotel):", ["Resort Hotel", "City Hotel"]) 
        Meal = st.selectbox("2. Tipo de régimen (meal):", ["BB: Bed & Breakfast", "FB: Full Board (desayuno, comida y cena)", 
                                                    "HB: Half Board (Desayuno y otra comida, normalmente cena)", "SC", "Sin definir"]) 
                            #(BB: 0, FB: 1, HB: 2, SC: 3, Sin definir: 4):")
        Reserved_room_type = st.selectbox("3. Tipo de habitación (reserved_room_type):", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L']) 
                            #C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6, 'L': 7, 'B': 8:")
        Customer_type = st.selectbox("4. Tipo de cliente (customer_type):", ["Transitorios", "Con contrato asignado", "Transitorios asociados a otros", "Grupos"])
                            #(Transitorios: 0, Con contrato asignado: 1, Transitorios asociados a otros: 2, Grupos: 3):")
        Year = st.slider("5. Año en el que se produce la reserva (year):", min_value = 2014, max_value = 2050)
        Month = st.slider("6. Mes en el que se produce la reserva (month):", min_value = 1, max_value = 12)
        Day = st.slider("7. Día en el que se produce la reserva (day):", min_value = 1, max_value = 31)
        Lead_time = st.text_input("8. Número de días transcurridos entre la fecha de reserva y la fecha de llegada (lead_time):")
        Deposit_type = st.radio("9. Tipo de depósito/pago realizado por el cliente (deposit_type):", ["Sin depósito", "Reembolsable", "No reembolsable"])
                            #(No deposit: 0, Refundable: 1, Non Refund: 2")
        Arrival_date_week_number = st.slider("10. Número de la semana del año en la que llegan los clientes (arrival_date_week_number):", min_value = 1, max_value = 52)
        Arrival_date_day_of_month = st.slider("11. Día del mes en el que llegan los clientes (arrival_date_day_of_month):", min_value = 1, max_value = 31)
        Stays_in_weekend_nights = st.text_input("12. Número de noches (sábados o domingos) que reservan los clientes (stays_in_weekend_nights):")
        Stays_in_week_nights = st.text_input("13. Número de semanas (de lunes a viernes) que reservan los clientes (stays_in_week_nights):")
        Is_repeated_guest = st.radio("14. ¿Es un cliente que ya ha hecho reservas anteriormente? (is_repeated_guest)", ["Sí", "No"])
                            #(0: no, 1: sí):")
        Previous_cancellations = st.text_input("15. Número de cancelaciones previas realizadas por el cliente anteriores a la reserva actual (previous_cancellations):")
        Previous_bookings_not_canceled = st.text_input("16. Número de reservas previas no canceladas por el cliente anteriores a la reserva actual (previous_bookings_not_canceled):")
        Adr = st.text_input("17. Promedio de la tarifa diaria (adr):")
        Required_car_parking_spaces = st.text_input("18. Número de plazas de parking requeridas por el cliente (required_car_parking_spaces):")
        Total_of_special_requests = st.slider("19. Número de peticiones especiales (por ejemplo: planta alta) solicitadas por el cliente (total_of_special_requests):", min_value = 0, max_value = 30)
        Total_Guests = st.slider("20. Número total de personas por habitación (Total_Guests):", min_value = 0, max_value = 10)
        Booking_changes = st.slider("21. Número de cambios/modificaciones realizadas en la reserva hasta el check-in o cancelación (booking_changes):", min_value = 0, max_value = 20)
    
        # Conversión de valores a formato float
        
        # Hotel
        if Hotel == "Resort Hotel":
            Hotel = 0
        elif Hotel == "City Hotel": 
            Hotel = 1
            
        # Meal
        if Meal == "BB: Bed & Breakfast": 
            Meal = 0
        elif Meal == "FB: Full Board (desayuno, comida y cena)":
            Meal = 1
        elif Meal == "HB: Half Board (Desayuno y otra comida, normalmente cena)":
            Meal = 2
        elif Meal == "SC":
            Meal = 3
        elif Meal == "Sin definir":
            Meal == 4
            
        # Reserved_room_type 
        if Reserved_room_type == 'A':
            Reserved_room_type = 1
        elif Reserved_room_type == 'B':
            Reserved_room_type = 8
        elif Reserved_room_type == 'C':
            Reserved_room_type = 0
        elif Reserved_room_type == 'D':
            Reserved_room_type = 2
        elif Reserved_room_type == 'E':
            Reserved_room_type = 3
        elif Reserved_room_type == 'G':
            Reserved_room_type = 4
        elif Reserved_room_type == 'H':
            Reserved_room_type = 6
        elif Reserved_room_type == 'L':
            Reserved_room_type = 7
        elif Reserved_room_type == 'F':
            Reserved_room_type = 5
            
        # Customer_type
        if Customer_type == 'Transitorios':
            Customer_type = 0
        elif Customer_type == "Con contrato asignado":
            Customer_type = 1
        elif Customer_type == "Transitorios asociados a otros":
            Customer_type = 2
        elif Customer_type == 'Grupos':
            Customer_type = 3
                
        
        # Deposit_type
        if Deposit_type == "Sin depósito":
            Deposit_type = 0
        elif Deposit_type == 'Reembolsable':
            Deposit_type = 1
        elif Deposit_type == 'No reembolsable':
            Deposit_type = 2    
            
        # Is_repeated_guest
        if Is_repeated_guest == "No":
            Is_repeated_guest = 0
        elif Is_repeated_guest == 'Sí':
            Is_repeated_guest = 1       
            
        # Espacios
        st.write(" ")
        st.write(" ")
        
        


        # El botón predicción se usa para iniciar el procesamiento
        if st.button("Predicción 🏁:"): 
            
            Lead_time = np.float_(Lead_time)
            Arrival_date_week_number = np.float_(Arrival_date_week_number)
            Arrival_date_day_of_month = np.float_(Arrival_date_day_of_month)
            Adr = np.float_(Adr)         
        
        
        # Normalización de variables lead_time, arrival_date_week_number, arrival_date_day_of_month, adr
            Lead_time = np.log(Lead_time + 1)
            Arrival_date_week_number = np.log(Arrival_date_week_number + 1)
            Arrival_date_day_of_month = np.log(Arrival_date_day_of_month + 1)
            Adr = np.log(Adr + 1)
            
            x_in = [np.float_(Hotel),
                    np.float_(Meal),
                    np.float_(Reserved_room_type),
                    np.float_(Customer_type),
                    np.float_(Year),
                    np.float_(Month),
                    np.float_(Day),
                    np.float_(Lead_time),
                    np.float_(Deposit_type),
                    np.float_(Arrival_date_week_number),
                    np.float_(Arrival_date_day_of_month),
                    np.float_(Stays_in_weekend_nights),
                    np.float_(Stays_in_week_nights),
                    np.float_(Is_repeated_guest),
                    np.float_(Previous_cancellations),
                    np.float_(Previous_bookings_not_canceled),
                    np.float_(Adr),
                    np.float_(Required_car_parking_spaces),
                    np.float_(Total_of_special_requests),
                    np.float_(Total_Guests),
                    np.float_(Booking_changes)]
            
            predictS = model_prediction(x_in, model)
            
            if predictS[0] < 0.5:
                st.success('La probabilidad de cancelación de reserva es de: {:.3f}%'.format(predictS[0]*100))
                st.write("Hay bajas probabilidades de que se cancele esta reserva 👍👌")
            
            if predictS[0] > 0.5:
                st.error('La probabilidad de cancelación de reserva es de: {:.3f}%'.format(predictS[0]*100))
                st.write("Hay altas probabilidades de que esta reserva sea cancelada 👎")
            
            #st.success('La probabilidad de cancelación de reserva es de: {:.5f}%'.format(predictS[0]*100))
            
    with c3:
        st.write(" ")
        