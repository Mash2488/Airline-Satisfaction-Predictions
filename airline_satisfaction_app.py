import streamlit as st
import pandas as pd
import joblib

# Load the trained model
rf = joblib.load("random_forest_model.pkl")

st.header("Enter Passenger Information")

# Categorical inputs
gender = st.selectbox("Gender", ["Male", "Female"])
class_type = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])

# Demographics & flight details
age = st.slider("Age", 7, 85, 30)
flight_distance = st.slider("Flight Distance", 30, 5000, 1000)

# Delay inputs
departure_delay = st.number_input("Departure Delay (minutes)", min_value=0, max_value=2000, value=0)
arrival_delay = st.number_input("Arrival Delay (minutes)", min_value=0, max_value=2000, value=0)

# Service ratings (0 to 5 scale)
wifi = st.slider("Inflight Wifi Service", 0, 5, 3)
convenience = st.slider("Departure/Arrival Time Convenient", 0, 5, 3)
booking = st.slider("Ease of Online Booking", 0, 5, 3)
gate_location = st.slider("Gate Location", 0, 5, 3)
food = st.slider("Food and Drink", 0, 5, 3)
online_boarding = st.slider("Online Boarding", 0, 5, 3)
seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
onboard = st.slider("On-board Service", 0, 5, 3)
legroom = st.slider("Leg Room Service", 0, 5, 3)
baggage = st.slider("Baggage Handling", 0, 5, 3)
checkin = st.slider("Check-in Service", 0, 5, 3)
inflight_service = st.slider("Inflight Service", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)


# Feature engineering
is_business_class = 1 if class_type == "Business" else 0
is_business_travel = 1 if travel_type == "Business travel" else 0
is_loyal_customer = 1 if customer_type == "Loyal Customer" else 0
gender_binary = 1 if gender == "Male" else 0

# Create final input DataFrame
input_df = pd.DataFrame([{
    'gender': gender_binary,
    'age': age,
    'flight_distance': flight_distance,
    'inflight_wifi_service': wifi,
    'departure/arrival_time_convenient': convenience,
    'ease_of_online_booking': booking,
    'gate_location': gate_location,
    'food_and_drink': food,
    'online_boarding': online_boarding,
    'seat_comfort': seat_comfort,
    'inflight_entertainment': entertainment,
    'on-board_service': onboard,
    'leg_room_service': legroom,
    'baggage_handling': baggage,
    'checkin_service': checkin,
    'inflight_service': inflight_service,
    'cleanliness': cleanliness,
    'departure_delay_in_minutes': departure_delay,
    'arrival_delay_in_minutes': arrival_delay,
    'is_business_class': is_business_class,
    'is_business_travel': is_business_travel,
    'is_loyal_customer': is_loyal_customer
}])



if st.button("Predict Satisfaction"):
    prediction = rf.predict(input_df)[0]
    prob = rf.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"Passenger is likely **Satisfied** ({prob:.2%} confidence).")
    else:
        st.error(f"Passenger is likely **Neutral or Dissatisfied** ({prob:.2%} confidence).")



