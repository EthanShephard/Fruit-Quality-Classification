import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from model3 import load_model as load_price_model_fn, predict_price

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Fruit Pricing System",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.header-title {
    font-size: 40px;
    font-weight: bold;
    color: white;
}
.subtext {
    color: #aaaaaa;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/home/dhruv/Documents/projects/Fruit-Quality-Classification/runs/apple_detector6/weights/best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
AVERAGE_APPLE_WEIGHT = 180

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_price_model():
    return load_price_model_fn()

yolo_model = load_yolo_model()
price_model = load_price_model()

# ===============================
# FUNCTIONS
# ===============================
def get_detections(results):
    boxes = results[0].boxes
    if boxes is None:
        return []
    return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()

def estimate_weight(count):
    total_weight_g = count * AVERAGE_APPLE_WEIGHT
    total_weight_kg = total_weight_g / 1000
    return total_weight_g, total_weight_kg

# ===============================
# HEADER
# ===============================
st.markdown('<div class="header-title">🍎 AI Fruit Pricing System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Computer Vision + ML based pricing engine</div>', unsafe_allow_html=True)

st.divider()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Controls")

market_rate = st.sidebar.slider("Market Rate (₹/kg)", 80, 200, 120)
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","png","jpeg"])

st.sidebar.markdown("---")
st.sidebar.info("Adjust parameters to simulate market conditions.")

# ===============================
# MAIN TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["📸 Detection", "📊 Insights", "⚙️ Debug"])

if uploaded_file:

    with st.spinner("Processing image..."):

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_np is None:
            st.error("Invalid image file.")
            st.stop()

        # YOLO
        results = yolo_model.predict(
            img_np,
            conf=confidence_threshold,
            iou=IOU_THRESHOLD,
            imgsz=640,
            verbose=False
        )

        boxes, confs = get_detections(results)
        apple_count = len(boxes)

        weight_g, weight_kg = estimate_weight(apple_count)

        price = predict_price(price_model, apple_count, weight_kg, market_rate)

        avg_conf = np.mean(confs) if len(confs) > 0 else 0

        # Draw boxes
        annotated = img_np.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ===============================
    # TAB 1: DETECTION
    # ===============================
    with tab1:

        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.image(annotated, caption="Detection Output", use_container_width=True)

        with col2:
            st.subheader("Results")

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            c1.metric("🍎 Apples", apple_count)
            c2.metric("⚖️ Weight (kg)", f"{weight_kg:.2f}")
            c3.metric("📈 Market Rate", f"₹{market_rate}")
            c4.metric("💰 Price", f"₹{price:.2f}")

            st.divider()

            if apple_count == 0:
                st.warning("No apples detected.")
            else:
                st.success("Detection successful.")

    # ===============================
    # TAB 2: INSIGHTS
    # ===============================
    with tab2:

        st.subheader("Analysis")

        st.write("### Pricing Logic")
        st.write("""
        - Price is predicted using ML model (XGBoost)
        - Factors:
            - Number of apples
            - Estimated weight
            - Market rate
        """)

        st.write("### System Behavior")
        st.write(f"""
        - Apples detected: **{apple_count}**
        - Avg confidence: **{avg_conf:.2f}**
        - Weight estimated: **{weight_kg:.2f} kg**
        """)

        st.write("### Observations")
        if apple_count > 10:
            st.info("Large batch detected → higher pricing impact")
        elif apple_count > 0:
            st.info("Small batch detected")
        else:
            st.warning("No valid objects found")

    # ===============================
    # TAB 3: DEBUG
    # ===============================
    with tab3:

        st.subheader("Debug Info")

        st.write("Detection boxes:")
        st.write(boxes)

        st.write("Confidence scores:")
        st.write(confs)

        st.write("Raw prediction output:")
        st.write(results)

else:
    st.info("Upload an image to start.")

# ===============================
# FOOTER
# ===============================
st.divider()
st.caption("Built with Streamlit | YOLOv8 | XGBoost")