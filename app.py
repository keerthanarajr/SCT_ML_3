import streamlit as st
import joblib
import cv2
import numpy as np
from skimage.feature import hog

model = joblib.load("model.pkl")
IMG_SIZE = 64

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    layout="wide"
)

st.title("🐱🐶 Cat vs Dog Classifier")

st.write("Upload an image and get prediction with confidence scores.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )


# ---------------- UI LOGIC ----------------
if uploaded_file is not None:

    col1, col2 = st.columns(2)

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(img_rgb, width=300)

    features = extract_features(img)

    proba = model.predict_proba([features])[0]

    cat_prob = float(proba[0])
    dog_prob = float(proba[1])

    with col2:
        st.subheader("🧠 Prediction")

        if cat_prob > dog_prob:
            st.success("🐱 Cat Detected")
        else:
            st.success("🐶 Dog Detected")

        st.write("### Confidence Scores")

        st.progress(cat_prob)
        st.text(f"Cat: {cat_prob*100:.2f}%")

        st.progress(dog_prob)
        st.text(f"Dog: {dog_prob*100:.2f}%")

    with st.expander("ℹ️ How this works"):
        st.write("""
        - Image is converted to grayscale  
        - HOG features are extracted  
        - SVM model predicts Cat or Dog  
        - Probabilities show confidence level  
        """)