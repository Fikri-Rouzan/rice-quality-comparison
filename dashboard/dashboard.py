import os
import time

# Suppress compiler log spam from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image

# Streamlit page configuration
st.set_page_config(page_title="Ricelytics MultiNet", page_icon="🌾", layout="wide")

# Global constant definitions
LABELS = ["whole", "chalky", "broken", "discolored"]
TARGET_SIZE = (224, 224)

# Model architecture file registry
MODEL_REGISTRY = {
    "MobileNetV2": {
        "file": "mobilenet.keras",
        "params": "2.5M",
        "badge": "⚡ Ultra-Fast / Recommended for Edge Deployment",
        "description": "Optimal for edge computing and real-time inspection with minimal memory footprint.",
    },
    "ResNet50": {
        "file": "resnet50.keras",
        "params": "24.1M",
        "badge": "🔬 Deep Residual Architecture",
        "description": "Deep feature extraction utilizing residual skip-connections.",
    },
    "EfficientNetB0": {
        "file": "efficientnet.keras",
        "params": "4.3M",
        "badge": "📐 Compound Scaling Architecture",
        "description": "Balanced scalability across network depth, width, and image resolution.",
    },
}


# Cached model loading function based on selected architecture
@st.cache_resource
def load_deep_learning_model(selected_model_name):
    filename = MODEL_REGISTRY[selected_model_name]["file"]

    # Search possible model directory paths
    possible_paths = [
        os.path.join("models", filename),
        filename,
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path:
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            return model, model_path
        except Exception as e:
            st.sidebar.error(f"Failed to load {selected_model_name}: {e}")
            return None, None
    else:
        st.sidebar.error(f"File `{filename}` was not found in the `models/` folder.")
        return None, None


def enhance_image(image):
    # Reduce sensor noise using Gaussian Blur
    return cv2.GaussianBlur(image, (3, 3), 0)


# Sidebar: Controls and model selection
with st.sidebar:
    if os.path.exists("image/icon.png"):
        st.image("image/icon.png", width=120)
    else:
        st.title("🌾 Ricelytics MultiNet")

    st.write("")
    st.subheader("⚙️ Architecture Configuration")
    st.write("")

    selected_model_name = st.selectbox(
        "Select CNN Model:",
        options=list(MODEL_REGISTRY.keys()),
        index=0,
        help="Select a model to perform rice grain classification.",
    )

    # Load selected model
    model, loaded_path = load_deep_learning_model(selected_model_name)
    model_info = MODEL_REGISTRY[selected_model_name]

    if model is not None:
        st.write("")
        st.success(f"✅ **{selected_model_name}** active")
        st.caption(f"📁 Path: `{loaded_path}`")
        st.caption(f"📊 Parameters: `~{model_info['params']}`")
        st.info(f"{model_info['badge']}\n\n*{model_info['description']}*")
    else:
        st.write("")
        st.warning(
            "⚠️ Model is not ready. Ensure the `.keras` file exists in the folder."
        )

    st.markdown("---")
    st.caption(
        "This dashboard is designed to detect, digitally segment, "
        "and classify rice grain quality into 4 commodity categories "
        "using Convolutional Neural Network (CNN) architectures."
    )

# Main header
st.title("Ricelytics MultiNet: Comparative CNN System for Rice Quality Assessment")
st.markdown(
    f"Automated inspection system powered by digital image segmentation and **{selected_model_name}** architecture."
)
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Quality Inspection", "📖 Kernel Type Guide"])

# Tab 1: Rice Quality Inspection
with tab1:
    with st.container(border=True):
        st.subheader("Image Input Settings")
        st.markdown("Select an image capture method or upload rice grain images below:")

        input_method = st.radio(
            "Image Input Method:",
            ("Upload Image File", "Live Camera"),
            horizontal=True,
        )

        st.warning(
            "💡 Recommendation: Place rice grains against a dark/high-contrast background with the "
            "camera positioned perpendicular (top-down 90° view) for maximum visual accuracy."
        )

    uploaded_file = None
    if input_method == "Upload Image File":
        uploaded_file = st.file_uploader(
            "Upload rice grain image (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
        )
    else:
        uploaded_file = st.camera_input(
            "Position the rice grain directly in the center of the camera frame"
        )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        with st.spinner(
            f"Running segmentation and classification using {selected_model_name}..."
        ):
            enhanced_img = enhance_image(img_bgr)

            # HSV color space segmentation
            hsv_image = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
            v_channel = hsv_image[:, :, 2]

            max_v_main = np.max(v_channel) if np.max(v_channel) > 0 else 1
            dynamic_thresh_main = max(int(max_v_main * 0.35), 50)

            _, binary_mask = cv2.threshold(
                v_channel, dynamic_thresh_main, 255, cv2.THRESH_BINARY
            )

            # Find all independent rice grain contours in the image
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

        st.write("")
        st.subheader("🖼️ Comparative Analysis and Detection Results")

        # Initialize counters and grain tracking variables
        grain_counts = {
            "whole": 0,
            "chalky": 0,
            "broken": 0,
            "discolored": 0,
        }
        detected_any_rice = False
        inference_latencies = []

        if model is not None:
            if contours:
                largest_contour_main = max(contours, key=cv2.contourArea)
                max_grain_area = cv2.contourArea(largest_contour_main)

                clean_full_mask = np.zeros_like(binary_mask)
                valid_grains_data = []

                # Filter rice morphological contours
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 100 or area < (max_grain_area * 0.10):
                        continue

                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    solidity_score = float(area) / hull_area if hull_area > 0 else 0
                    if solidity_score < 0.75:
                        continue

                    x, y, w, h = cv2.boundingRect(c)
                    aspect_ratio_score = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                    if aspect_ratio_score < 1.10:
                        continue

                    # Object successfully passed as a valid rice grain
                    detected_any_rice = True
                    cv2.drawContours(
                        clean_full_mask, [c], -1, 255, thickness=cv2.FILLED
                    )
                    valid_grains_data.append((x, y, w, h))

                # Segmented clean image
                segmented_clean_bgr = cv2.bitwise_and(
                    img_bgr, img_bgr, mask=clean_full_mask
                )
                img_rgb_annotated = cv2.cvtColor(segmented_clean_bgr, cv2.COLOR_BGR2RGB)
                segmented_full_rgb = img_rgb_annotated.copy()

                # Infer each grain object using 1:1 square padding
                for x, y, w, h in valid_grains_data:
                    grain_crop = segmented_clean_bgr[y : y + h, x : x + w]

                    max_side = max(w, h)
                    grain_square = np.zeros((max_side, max_side, 3), dtype=np.uint8)
                    df_x = (max_side - w) // 2
                    df_y = (max_side - h) // 2
                    grain_square[df_y : df_y + h, df_x : df_x + w] = grain_crop

                    # Resize square canvas proportionally to model input dimensions
                    grain_resized = cv2.resize(
                        grain_square, TARGET_SIZE, interpolation=cv2.INTER_AREA
                    )
                    grain_input_rgb = cv2.cvtColor(grain_resized, cv2.COLOR_BGR2RGB)

                    # Pixel normalization
                    normalized_input = grain_input_rgb / 255.0
                    input_batch = np.expand_dims(normalized_input, axis=0)

                    # Prediction & latency recording
                    start_t = time.perf_counter()
                    predictions = model(input_batch, training=False).numpy()[0]
                    end_t = time.perf_counter()
                    inference_latencies.append((end_t - start_t) * 1000)

                    predicted_class_idx = np.argmax(predictions)
                    predicted_label = LABELS[predicted_class_idx]
                    confidence_score = predictions[predicted_class_idx] * 100

                    # Update grain quality count dictionary
                    grain_counts[predicted_label] += 1

                    # Bounding box color mapping by category
                    color_map = {
                        "whole": (0, 255, 0),
                        "chalky": (255, 165, 0),
                        "broken": (255, 0, 0),
                        "discolored": (255, 255, 0),
                    }
                    box_color = color_map[predicted_label]

                    # Dynamically scale label font size
                    dynamic_font_scale = max(0.35, img_bgr.shape[1] / 3200.0)
                    dynamic_thickness = max(1, int(img_bgr.shape[1] / 2200.0))

                    # Bounding box on black background image
                    cv2.rectangle(
                        img_rgb_annotated,
                        (x, y),
                        (x + w, y + h),
                        box_color,
                        dynamic_thickness + 1,
                    )
                    label_text = f"{predicted_label.upper()} ({confidence_score:.0f}%)"
                    cv2.putText(
                        img_rgb_annotated,
                        label_text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        dynamic_font_scale,
                        box_color,
                        dynamic_thickness,
                    )

            # Dashboard column visualization
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    pil_image,
                    caption="Original Input Image",
                    width="stretch",
                )
            with col2:
                if detected_any_rice:
                    st.image(
                        img_rgb_annotated,
                        caption=f"Segmentation & Classification Result ({selected_model_name})",
                        width="stretch",
                    )
                else:
                    st.image(
                        segmented_full_rgb,
                        caption="Segmentation Result (No Valid Rice Grains Detected)",
                        width="stretch",
                    )

            # Cumulative quantitative statistics
            if detected_any_rice:
                st.markdown("---")
                st.subheader("📊 Rice Commodity Quantitative Analysis Results")

                # Compute total grain count across all categories
                total_grains = sum(grain_counts.values())
                avg_latency = (
                    np.mean(inference_latencies) if inference_latencies else 0.0
                )

                st.markdown(
                    f"🎯 **TOTAL GRAINS DETECTED:** `{total_grains}` | "
                    f"⏱️ **AVERAGE LATENCY:** `{avg_latency:.2f} ms/grain` | "
                    f"🧠 **MODEL:** `{selected_model_name}`"
                )

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        label="🌾 TOTAL WHOLE",
                        value=grain_counts["whole"],
                    )
                with m2:
                    st.metric(
                        label="⚪ TOTAL CHALKY",
                        value=grain_counts["chalky"],
                    )
                with m3:
                    st.metric(
                        label="❌ TOTAL BROKEN",
                        value=grain_counts["broken"],
                    )
                with m4:
                    st.metric(
                        label="🍂 TOTAL DISCOLORED",
                        value=grain_counts["discolored"],
                    )

                # Accumulated grain count bar chart per label
                st.write("")
                st.markdown("#### Cumulative Rice Grain Distribution Chart:")

                categories = [lbl.capitalize() for lbl in LABELS]
                total_counts = [grain_counts[lbl] for lbl in LABELS]

                fig = px.bar(
                    x=categories,
                    y=total_counts,
                    labels={
                        "x": "Rice Quality Category",
                        "y": "Grain Count",
                    },
                    color=categories,
                    color_discrete_sequence=px.colors.qualitative.Pastel1,
                    text=total_counts,
                )
                fig.update_traces(textposition="auto")
                fig.update_layout(
                    xaxis=dict(tickangle=-45, title_font=dict(size=12)),
                    yaxis=dict(title_font=dict(size=12)),
                    showlegend=False,
                    height=380,
                    margin=dict(l=40, r=40, t=20, b=60),
                    template="plotly_white",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.error(
                    "🚨 **Validation Failed:** No rice grain objects met the geometric morphology criteria of the system."
                )
        else:
            st.error(
                f"Classification process halted because `{model_info['file']}` is not available in the `models/` directory."
            )
    else:
        st.info(
            "Please upload a rice image or activate the camera module to begin the classification process."
        )

# Tab 2: Rice Kernel Type Guide
with tab2:
    st.subheader("Rice Quality Standard Guide")
    st.write(
        "Explanation of rice quality classification parameters based on physical kernel characteristics:"
    )

    with st.expander("🌾 Whole Kernel"):
        st.markdown("""
        - **Characteristics**: Rice grains that are fully intact or exhibit minor breakage not exceeding **1/10** of the average length of a normal grain.
        - **Indicators**: Distinct symmetrical elongated morphology with predominant translucency.
        """)

    with st.expander("⚪ Chalky Grain"):
        st.markdown("""
        - **Characteristics**: Rice grains with an opaque, chalky, or milky white area covering **1/2** or more of the total grain body.
        - **Indicators**: Uneven amylose density caused by incomplete endosperm filling during the grain ripening stage.
        """)

    with st.expander("❌ Broken Kernel"):
        st.markdown("""
        - **Characteristics**: Rice grains with distinct physical breakage, ranging between **2/10** and **8/10** of the average length of an unbroken grain.
        - **Indicators**: Missing head or tail portions resulting from suboptimal milling or high kernel brittleness.
        """)

    with st.expander("🍂 Discolored Grain"):
        st.markdown("""
        - **Characteristics**: Rice grains showing macro surface discoloration across partial or whole areas, turning yellow, brownish-yellow, or developing dark spots.
        - **Indicators**: Damage due to microbial activity (fungi), poor storage humidity, or excess heat prior to drying.
        """)

# Footer
st.markdown("---")
st.caption("© 2026 Ricelytics MultiNet")
