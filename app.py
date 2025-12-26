import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("car_mpg_model.pkl", "rb"))

st.title("🚗 MPG Predictor")

# ================= INPUTS =================
year = st.number_input("Year", 2000, 2025, 2018)
price = st.number_input("Price", 0, 2000000, 10000)
mileage = st.number_input("Mileage", 0, 300000, 40000)
tax = st.number_input("Tax", 0, 1000, 150)
engineSize = st.number_input("Engine Size", 0.5, 6.0, 1.5)

model_name = st.selectbox(
    "Car Model",
    [
        'B-MAX','C-MAX','EcoSport','Edge','Escort','Fiesta','Focus',
        'Fusion','Galaxy','Grand C-MAX','Grand Tourneo Connect','KA',
        'Ka+','Kuga','Mondeo','Mustang','Puma','Ranger','S-MAX',
        'Streetka','Tourneo Connect','Tourneo Custom','Transit Tourneo'
    ]
)

fuel = st.selectbox("Fuel Type", ["Petrol","Diesel","Electric","Hybrid","Other"])
trans = st.selectbox("Transmission", ["Automatic","Manual","Semi-Auto"])

# ================= BASE DATA =================
input_data = {
    "year": year,
    "price": price,
    "mileage": mileage,
    "tax": tax,
    "engineSize": engineSize
}

# ================= TRAINED COLUMNS =================
trained_cols = [
 'year','price','mileage','tax','engineSize',
 'model_ B-MAX','model_ C-MAX','model_ EcoSport','model_ Edge',
 'model_ Escort','model_ Fiesta','model_ Focus','model_ Fusion',
 'model_ Galaxy','model_ Grand C-MAX','model_ Grand Tourneo Connect',
 'model_ KA','model_ Ka+','model_ Kuga','model_ Mondeo',
 'model_ Mustang','model_ Puma','model_ Ranger','model_ S-MAX',
 'model_ Streetka','model_ Tourneo Connect','model_ Tourneo Custom',
 'model_ Transit Tourneo','model_Focus',
 'fuelType_Diesel','fuelType_Electric','fuelType_Hybrid',
 'fuelType_Other','fuelType_Petrol',
 'transmission_Automatic','transmission_Manual','transmission_Semi-Auto'
]

# ================= ONE HOT =================
input_data[f"model_{model_name}"] = 1
input_data[f"fuelType_{fuel}"] = 1
input_data[f"transmission_{trans}"] = 1

input_df = pd.DataFrame([input_data])

# ================= ALIGN COLUMNS (MOST IMPORTANT) =================
for col in trained_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[trained_cols]

# ================= PREDICT =================
if st.button("Predict MPG"):
    pred = model.predict(input_df)[0]
    kmpl=pred*1.609/3.785
    st.success(f"✅ Predicted KMPL: {kmpl:.2f}")
    st.success(f"✅ Predicted MPG: {pred:.2f}")