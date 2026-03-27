import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# CONFIGURACIÓN
# =========================
IMG_HEIGHT = 180
IMG_WIDTH = 180

# ⚠️ Ajusta esto si cambiaste clases
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

MODEL_PATH = "modelo_flores.keras"

# =========================
# CARGA DEL MODELO
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# FUNCIONES
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    probabilities = tf.nn.softmax(prediction).numpy()[0]

    predicted_index = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index]

    return predicted_class, confidence, probabilities

# =========================
# INTERFAZ
# =========================
st.set_page_config(page_title="Clasificador de Flores", layout="centered")

st.title("🌸 Clasificador de Flores")
st.write("Carga una imagen o usa la cámara para identificar el tipo de flor.")

# Mostrar clases disponibles
st.subheader("📂 Clases que reconoce el modelo:")
st.write(", ".join(CLASS_NAMES))

# =========================
# INPUTS
# =========================
uploaded_file = st.file_uploader("📁 Subir imagen", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Tomar foto")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)

elif camera_image is not None:
    image = Image.open(camera_image)

# =========================
# PREDICCIÓN
# =========================
if image is not None:
    st.image(image, caption="Imagen seleccionada", use_column_width=True)

    if st.button("🔍 Predecir"):
        predicted_class, confidence, probabilities = predict(image)

        st.subheader("🌼 Resultado:")
        st.success(f"**{predicted_class.upper()}** ({confidence*100:.2f}%)")

        # =========================
        # PROBABILIDADES
        # =========================
        st.subheader("📊 Probabilidades por clase:")

        prob_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(prob_dict)

        # Resaltado adicional
        st.info(f"La clase más probable es: **{predicted_class}**")