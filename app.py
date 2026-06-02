import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Medicine Assistant",
    page_icon="💊",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.block-container {
    padding-top: 1rem;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
}

h1, h2, h3 {
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

import gdown

url = "https://drive.google.com/file/d/XXXXX/view?usp=sharing"

gdown.download(url, "pill_model.h5", quiet=False)

model = tf.keras.models.load_model("pill_model.h5")

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = [
    "multivitamin",
    "paracetamol",
    "vitamin_c"
]

# --------------------------------------------------
# MEDICINE DATABASE
# --------------------------------------------------

medicine_info = {

    "paracetamol": {
        "use": "Used for fever and pain relief",
        "dosage": "500 mg after meals",
        "warning": "Avoid overdose",
        "side_effects": "Nausea, stomach upset",
        "storage": "Store below 25°C",
        "manufacturer": "Sun Pharma"
    },

    "vitamin_c": {
        "use": "Supports immunity",
        "dosage": "500 mg daily",
        "warning": "Take after food",
        "side_effects": "Acidity, stomach irritation",
        "storage": "Store in a cool dry place",
        "manufacturer": "Cipla"
    },

    "multivitamin": {
        "use": "Nutritional supplement",
        "dosage": "1 tablet daily",
        "warning": "Do not overdose",
        "side_effects": "Mild nausea",
        "storage": "Store below 30°C",
        "manufacturer": "Abbott"
    }
}

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("💊 Smart Medicine Assistant")

st.sidebar.success("Healthcare AI Solution")

st.sidebar.markdown("""
### Features

✅ Medicine Detection

✅ Confidence Score

✅ Dosage Guidance

✅ Side Effects

✅ Storage Instructions

✅ Safety Warning
""")

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div style='padding:30px;
background:linear-gradient(90deg,#4facfe,#00f2fe);
border-radius:15px;
text-align:center;
color:white;'>

<h1>💊 Smart Medicine Assistant</h1>

<h3>
AI-Powered Medicine Recognition & Guidance System
</h3>

<p>
Fast • Intelligent • Reliable
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Medicines Supported", "3")

with col2:
    st.metric("AI Framework", "TensorFlow")

with col3:
    st.metric("Project Type", "Healthcare AI")

st.markdown("---")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Medicine Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Medicine",
            width=400
        )

    img = image.resize((224, 224))

    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    medicine = class_names[predicted_class]

    confidence = np.max(prediction) * 100

    with col2:

        st.success(
            f"✅ Detected Medicine: {medicine}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.subheader("Confidence Meter")

        st.progress(int(confidence))

        if confidence > 80:
            st.success("High Confidence Prediction")
        elif confidence > 60:
            st.warning("Moderate Confidence Prediction")
        else:
            st.error("Low Confidence Prediction")

    st.markdown("---")

    d1, d2 = st.columns(2)

    with d1:

        st.info(
            f"💡 Use\n\n{medicine_info[medicine]['use']}"
        )

        st.success(
            f"💊 Dosage\n\n{medicine_info[medicine]['dosage']}"
        )

        st.warning(
            f"⚠️ Warning\n\n{medicine_info[medicine]['warning']}"
        )

    with d2:

        st.error(
            f"🚨 Side Effects\n\n{medicine_info[medicine]['side_effects']}"
        )

        st.info(
            f"📦 Storage\n\n{medicine_info[medicine]['storage']}"
        )

        st.success(
            f"🏭 Manufacturer\n\n{medicine_info[medicine]['manufacturer']}"
        )

    st.markdown("---")

    st.warning("""
🚨 Emergency Guidance

• Consult a doctor immediately if overdose occurs.

• Stop usage if severe allergic reactions appear.

• Seek emergency help for breathing difficulties.

• Follow medical advice before changing dosage.
""")

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

with st.expander("ℹ️ About Project"):

    st.write("""
This project uses Artificial Intelligence,
Machine Learning, and Computer Vision
to identify medicines from images and
provide guidance regarding dosage,
usage, storage, and safety instructions.
""")

# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.markdown("---")

st.subheader("🔄 How It Works")

w1, w2, w3, w4 = st.columns(4)

with w1:
    st.info("1️⃣ Upload Image")

with w2:
    st.info("2️⃣ AI Analysis")

with w3:
    st.info("3️⃣ Medicine Detection")

with w4:
    st.info("4️⃣ Guidance Display")

# --------------------------------------------------
# FUTURE SCOPE
# --------------------------------------------------

st.markdown("---")

st.subheader("🚀 Future Scope")

st.write("✅ OCR Medicine Text Recognition")
st.write("✅ Expiry Date Detection")
st.write("✅ Voice Assistant")
st.write("✅ Nearby Pharmacy Finder")
st.write("✅ Drug Interaction Checker")

# --------------------------------------------------
# TEAM SECTION
# --------------------------------------------------

st.markdown("---")

st.subheader("👩‍💻 Team")

st.write("• Gayatri Joshi")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown("""
### 🏆 Developed By

**Gayatri Joshi**

💊 Smart Medicine Assistant

🚀 Healthcare AI Hackathon Project 2026
""")
