import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans

# -------------------------------
# SETTINGS
# -------------------------------
IMG_SIZE = 128
CLUSTERS = 5   # car, bus, bike, pedestrian, traffic sign

st.set_page_config(page_title="Traffic Image Clustering", layout="wide")
st.title("🚦 Traffic Image Clustering using CNN + KMeans")

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload traffic images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) >= CLUSTERS:

    images = []
    file_names = []

    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        file_names.append(file.name)

    images = np.array(images) / 255.0
    st.success(f"Loaded {len(images)} images")

    # -------------------------------
    # CNN AUTOENCODER
    # -------------------------------
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    encoded = Conv2D(128, (3,3), activation='relu', padding='same')(x)

    # Decoder
    x = UpSampling2D((2,2))(encoded)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------
    with st.spinner("Training CNN Autoencoder..."):
        autoencoder.fit(images, images, epochs=5, batch_size=32, verbose=0)

    st.success("Autoencoder training completed")

    # -------------------------------
    # FEATURE EXTRACTION
    # -------------------------------
    encoder = Model(input_img, encoded)
    features = encoder.predict(images, verbose=0)
    features = features.reshape(features.shape[0], -1)

    # -------------------------------
    # K-MEANS CLUSTERING
    # -------------------------------
    kmeans = KMeans(n_clusters=CLUSTERS, n_init=10)
    labels = kmeans.fit_predict(features)

    st.success("Clustering completed")

    # -------------------------------
    # DISPLAY CLUSTERS
    # -------------------------------
    st.subheader("Clustered Images")

    for c in range(CLUSTERS):
        st.markdown(f"### Cluster {c}")
        idx = np.where(labels == c)[0]

        cols = st.columns(5)
        for i, img_idx in enumerate(idx[:10]):
            with cols[i % 5]:
                st.image(images[img_idx], use_container_width=True)

    # -------------------------------
    # SAVE RESULTS
    # -------------------------------
    results_text = ""
    for i in range(len(file_names)):
        results_text += f"{file_names[i]} -> Cluster {labels[i]}\n"

    st.download_button(
        "Download Cluster Results",
        results_text,
        file_name="aton_clusters.txt"
    )

else:
    st.info("Upload at least 5 images to start clustering")
