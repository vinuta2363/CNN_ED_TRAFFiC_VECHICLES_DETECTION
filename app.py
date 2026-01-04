import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Traffic Vehicle Clustering", layout="centered")
st.title("🚦 Traffic Vehicle Clustering (Pretrained CNN + KMeans)")

IMG_SIZE = 128

# -------------------------------
# LOAD SAVED MODELS
# -------------------------------
@st.cache_resource
def load_models():
    encoder = load_model("encoder.keras")
    kmeans = joblib.load("kmeans.pkl")
    return encoder, kmeans

encoder, kmeans = load_models()

st.success("Models loaded successfully")

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a traffic image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load & preprocess image (NO cv2)
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # FEATURE EXTRACTION
    # -------------------------------
    features = encoder.predict(img_array, verbose=0)
    features = features.reshape(1, -1)

    # -------------------------------
    # CLUSTER PREDICTION
    # -------------------------------
    cluster_id = int(kmeans.predict(features)[0])

    st.markdown("### 🧠 Prediction Result")
    st.success(f"Image belongs to **Cluster {cluster_id}**")

    st.info(
        "Clusters are learned automatically using CNN Autoencoder + KMeans. "
        "Cluster IDs can later be mapped to vehicle types (Car / Truck / Bus)."
    )

else:
    st.warning("Please upload an image to classify.")
