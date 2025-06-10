import streamlit as st
import joblib
import re
import base64


# Load components
model = joblib.load("xgb_model2.pkl")
vectorizer = joblib.load("vectorizer2.pkl")
label_encoder = joblib.load("label_encoder2.pkl")

def clean_text(text):
    return re.sub(r"http\S+|@\w+|[^a-zA-Z\s]", "", text.lower())

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img = get_base64_image("Mask.webp")


st.markdown(
    f"""
    <style>
    body {{
        background-image: url("data:image/webp;base64,{bg_img}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}

    .stApp {{
        background-color: rgba(255, 255, 255, 0.85); 
        padding: 2rem;
        border-radius: 20px;
    }}

    input[type="text"] {{
    background-color: #121212	 !important;  /* box background color */
    color: #FF7F7F	 !important;             /* text color inside box */
    border: 1px solid #444 !important;
    border-radius: 10px;
    padding: 0.5rem;
        }}

    .css-10trblm, .stTextInput > label, .stButton {{
        color: #333333;
        font-size: 18px;
    }}

    .stSuccess {{
        font-weight: bold;
        color: green;
    }}
    .footer {{
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        font-size: 12px;
        color: #0B3D91;
    }}


    </style>
    <div class="footer">
        Developed by <b>Ankuj Saha</b><br>
        Data Analyst and Machine Learning Developer<br>
        © 2025 Ankuj. All rights reserved.

    </div>
    """,
    unsafe_allow_html=True
)

st.title("Emotion Classifier")

user_input = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    emotion = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Emotion: {emotion}")