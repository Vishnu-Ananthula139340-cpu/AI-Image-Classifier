
import streamlit as st
import torch
from torchvision import models
from PIL import Image
import io
import csv

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VisionAI - Image Classifier",
    page_icon="📸",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #101525;
}
h1, h2, h3 {
    color: #ffffff;
}
div[data-testid="stMetric"] {
    background: #202b43;
    padding: 15px;
    border-radius: 12px;
}
div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #9333ea);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
    min-height: 45px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD RESNET50 AI MODEL
# ==========================================

@st.cache_resource
def load_model():

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)

    model.eval()

    categories = weights.meta["categories"]

    preprocess = weights.transforms()

    return model, categories, preprocess


st.title("📸 VisionAI")
st.subheader("Advanced AI Image Classification")

st.write(
    "Upload multiple photographs and let AI "
    "identify their contents."
)

st.info(
    "Powered by PyTorch and ResNet50 "
    "with pretrained ImageNet weights."
)


# ==========================================
# 3. LOAD MODEL
# ==========================================

with st.spinner("Loading AI model..."):

    model, categories, preprocess = load_model()

st.success("AI model is ready!")


# ==========================================
# 4. MULTIPLE IMAGE UPLOAD
# ==========================================

st.header("📂 Upload Your Images")

uploaded_files = st.file_uploader(
    "Choose multiple images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

MAX_IMAGES = 10

if uploaded_files:

    if len(uploaded_files) > MAX_IMAGES:
        st.warning(
            f"Only the first {MAX_IMAGES} images "
            "will be processed in this batch."
        )

    selected_files = uploaded_files[:MAX_IMAGES]

    st.write(
        f"**{len(selected_files)} images selected**"
    )

    # ======================================
    # 5. IMAGE PREVIEW
    # ======================================

    st.subheader("Image Gallery")

    preview_columns = st.columns(3)

    for i, uploaded_file in enumerate(selected_files):

        try:

            image = Image.open(
                io.BytesIO(uploaded_file.getvalue())
            ).convert("RGB")

            with preview_columns[i % 3]:

                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

        except Exception:
            st.warning(
                f"Could not open {uploaded_file.name}"
            )


    # ======================================
    # 6. CLASSIFY ALL IMAGES
    # ======================================

    if st.button(
        "🚀 CLASSIFY ALL IMAGES",
        use_container_width=True
    ):

        valid_images = []

        results = []

        for uploaded_file in selected_files:

            try:

                image = Image.open(
                    io.BytesIO(uploaded_file.getvalue())
                ).convert("RGB")

                tensor = preprocess(image)

                valid_images.append({
                    "name": uploaded_file.name,
                    "image": image,
                    "tensor": tensor
                })

            except Exception as error:

                st.error(
                    f"Error reading {uploaded_file.name}: {error}"
                )


        if valid_images:

            progress = st.progress(0)

            status = st.empty()

            # Process in small batches
            batch_size = 2

            total = len(valid_images)

            for start in range(0, total, batch_size):

                batch = valid_images[
                    start:start + batch_size
                ]

                input_batch = torch.stack([
                    item["tensor"] for item in batch
                ])

                with torch.inference_mode():

                    output = model(input_batch)

                    probabilities = torch.nn.functional.softmax(
                        output,
                        dim=1
                    )

                    top_prob, top_ids = torch.topk(
                        probabilities,
                        k=3,
                        dim=1
                    )

                for index, item in enumerate(batch):

                    predictions = []

                    for rank in range(3):

                        class_id = top_ids[
                            index, rank
                        ].item()

                        confidence = top_prob[
                            index, rank
                        ].item() * 100

                        predictions.append({
                            "class": categories[class_id],
                            "confidence": confidence
                        })

                    results.append({
                        "name": item["name"],
                        "image": item["image"],
                        "predictions": predictions
                    })

                completed = min(start + len(batch), total)

                progress.progress(completed / total)

                status.write(
                    f"Classified {completed} of {total} images"
                )


            # ==================================
            # 7. DISPLAY RESULTS
            # ==================================

            st.success(
                "All selected images have been processed!"
            )

            st.header("🏆 Classification Results")

            st.metric(
                "Images classified",
                len(results)
            )

            for number, result in enumerate(results, 1):

                with st.expander(
                    f"Image {number}: {result['name']}",
                    expanded=True
                ):

                    left, right = st.columns([1, 1])

                    with left:

                        st.image(
                            result["image"],
                            use_container_width=True
                        )

                    with right:

                        st.write("### Top 3 Predictions")

                        for rank, prediction in enumerate(
                            result["predictions"], 1
                        ):

                            confidence = prediction["confidence"]

                            st.write(
                                f"**{rank}. {prediction['class']}**"
                            )

                            st.progress(
                                min(confidence / 100, 1.0)
                            )

                            st.caption(
                                f"Model score: {confidence:.2f}%"
                            )

            # ==================================
            # 8. DOWNLOAD CSV REPORT
            # ==================================

            csv_buffer = io.StringIO()

            writer = csv.writer(csv_buffer)

            writer.writerow([
                "Image Name",
                "Rank",
                "Predicted Class",
                "Model Score (%)"
            ])

            for result in results:

                for rank, prediction in enumerate(
                    result["predictions"], 1
                ):

                    writer.writerow([
                        result["name"],
                        rank,
                        prediction["class"],
                        f"{prediction['confidence']:.2f}"
                    ])

            st.download_button(
                label="📥 Download All Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="visionai_results.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "VisionAI | Powered by PyTorch and ResNet50"
)

st.caption(
    "Predictions are model estimates, not guaranteed "
    "identifications. Confidence scores are not "
    "measured accuracy."
)
