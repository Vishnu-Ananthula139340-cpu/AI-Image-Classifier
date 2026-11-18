
import streamlit as st
import torch
from torchvision import models
from PIL import Image

# ==========================================
# 1. PAGE SETUP
# ==========================================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="📷",
    layout="centered"
)

# ==========================================
# 2. LOAD PRETRAINED AI MODEL
# ==========================================

@st.cache_resource
def load_model():

    # Load pretrained MobileNetV2 model
    weights = models.MobileNet_V2_Weights.DEFAULT

    model = models.mobilenet_v2(weights=weights)

    # Set model to evaluation mode
    model.eval()

    # Get ImageNet category names
    categories = weights.meta["categories"]

    # Get the correct preprocessing steps
    preprocess = weights.transforms()

    return model, categories, preprocess


# Load model
model, categories, preprocess = load_model()


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def process_image(image):

    # Convert image into RGB format
    image = image.convert("RGB")

    # Apply preprocessing
    input_tensor = preprocess(image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


# ==========================================
# 4. WEBSITE TITLE AND DESCRIPTION
# ==========================================

st.title("📷 AI Image Classifier")

st.write(
    "Upload an image and let Artificial Intelligence "
    "identify the objects in your picture!"
)

st.info(
    "This application uses MobileNetV2 "
    "trained on the ImageNet dataset."
)


# ==========================================
# 5. IMAGE UPLOAD SECTION
# ==========================================

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 6. IMAGE CLASSIFICATION
# ==========================================

if uploaded_file is not None:

    try:

        # Open uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Display uploaded image
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.write("### 🔍 Classifying your image...")

        # Prepare image
        input_batch = process_image(image)

        # Perform AI prediction
        with torch.inference_mode():
            output = model(input_batch)

        # Calculate probabilities
        probabilities = torch.nn.functional.softmax(
            output[0],
            dim=0
        )

        # Get top 3 predictions
        top3_prob, top3_catid = torch.topk(
            probabilities,
            3
        )

        # ======================================
        # 7. DISPLAY PREDICTION RESULTS
        # ======================================

        st.subheader("🏆 Top 3 Predictions")

        for i in range(3):

            # Get object name
            class_name = categories[
                top3_catid[i].item()
            ]

            # Calculate confidence percentage
            confidence = top3_prob[i].item() * 100

            # Display prediction
            st.write(
                f"**{i + 1}. {class_name.capitalize()}**"
            )

            st.progress(confidence / 100)

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

            st.divider()

        st.success("Image classification completed!")

    except Exception as e:

        st.error(f"Something went wrong: {e}")


# ==========================================
# 8. FOOTER
# ==========================================

st.caption(
    "AI Image Classifier | Powered by PyTorch "
    "and MobileNetV2"
)
