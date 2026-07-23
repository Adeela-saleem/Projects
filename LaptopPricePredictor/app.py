import streamlit as st
import pandas as pd
import joblib

# ---------- Trained pipeline load karo ----------
# pipeline.pkl ke andar encoding + scaling + log1p/expm1 + model sab kuch already hai
pipeline = joblib.load('pipeline.pkl')

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.title("💻 Laptop Price Predictor")
st.caption(
    "⚠️ Ye prediction ek purane (historic) laptop-price dataset par trained model se aa rahi hai. "
    "Current/live market price is se different ho sakti hai — sirf ek andaza (estimate) samjhein."
)

st.header("Laptop Ki Specs Daaliye")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", ['ASUS', 'Lenovo', 'acer', 'Avita', 'HP', 'DELL', 'MSI', 'APPLE'])
    processor_brand = st.selectbox("Processor Ka Brand", ['Intel', 'AMD', 'M1'])
    processor_name = st.selectbox(
        "Processor Ka Naam",
        ['Core i3', 'Core i5', 'Celeron Dual', 'Ryzen 5', 'Core i7',
         'Core i9', 'M1', 'Pentium Quad', 'Ryzen 3', 'Ryzen 7', 'Ryzen 9']
    )
    processor_gnrtn = st.number_input("Processor Ki Generation (jaise 10, 11, 12)", min_value=1, max_value=13, value=10)
    ram_gb = st.selectbox("RAM (GB mein)", [4, 8, 16, 32])
    ram_type = st.selectbox("RAM Ka Type", ['DDR4', 'LPDDR4', 'LPDDR4X', 'DDR5', 'DDR3', 'LPDDR3'])
    ssd = st.selectbox("SSD (GB mein)", [0, 128, 256, 512, 1024])
    hdd = st.selectbox("HDD (GB mein)", [0, 512, 1024, 2048])

with col2:
    os_ = st.selectbox("Operating System", ['Windows', 'DOS', 'Mac'])
    os_bit = st.selectbox("OS Bit", [32, 64])
    graphic_card_gb = st.selectbox("Graphic Card (GB mein)", [0, 2, 4, 6, 8])
    weight = st.selectbox("Weight Ki Category", ['Casual', 'ThinNlight', 'Gaming'])
    warranty = st.selectbox("Warranty", ['No warranty', '1 year', '2 years', '3 years'])
    touchscreen = st.selectbox("Touchscreen Hai?", ['No', 'Yes'])
    msoffice = st.selectbox("MS Office Included Hai?", ['No', 'Yes'])
    rating = st.slider("Rating (stars mein)", 1, 5, 3)

st.info(
    "Note: 'Number of Ratings' aur 'Number of Reviews' ek naye/abhi tak release na hue laptop ke liye "
    "available nahi hoti, isliye yahan 0 rakh rahe hain."
)

if st.button("Price Predict Karo"):
    # Raw input — koi manual encoding/scaling nahi karni, pipeline khud sab handle kar leta hai
    input_df = pd.DataFrame([{
        'brand': brand,
        'processor_brand': processor_brand,
        'processor_name': processor_name,
        'processor_gnrtn': processor_gnrtn,
        'ram_gb': ram_gb,
        'ram_type': ram_type,
        'ssd': ssd,
        'hdd': hdd,
        'os': os_,
        'os_bit': os_bit,
        'graphic_card_gb': graphic_card_gb,
        'weight': weight,
        'warranty': warranty,
        'Touchscreen': 1 if touchscreen == 'Yes' else 0,
        'msoffice': 1 if msoffice == 'Yes' else 0,
        'rating': rating,
        'Number of Ratings': 0,
        'Number of Reviews': 0,
    }])

    # pipeline.predict() seedha real ₹ value deta hai (log1p/expm1 andar hi handle ho jata hai)
    prediction = pipeline.predict(input_df)[0]

    st.success(f"Andazan Price: ₹{prediction:,.0f}")
    st.caption("⚠️ Ye sirf historic data ke basis par estimate hai, real-time market price nahi.")