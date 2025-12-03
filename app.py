import streamlit as st
import requests
from PIL import Image
import io
import json
import time
import zipfile

API_URL = "http://127.0.0.1:8000/detect"

st.set_page_config(page_title="YOLOv8 Detection App", layout="centered")

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        color-scheme: light !important;
        background: #eef4ff !important;
    }

    * {
        color: #1e1e1e !important;
        font-weight: 500;
    }

    .hero-header, .hero-header * {
        color: white !important;
    }

    [data-testid="stFileUploader"], 
    [data-testid="stFileUploader"] * {
        color: #1e1e1e !important;
        background: #f5f7ff !important;
    }

    [data-testid="stExpander"] label {
        color: #1e1e1e !important;
    }

    .dark-btn > button {
        background: #0d1117 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
    }
    .dark-btn > button * {
        color: white !important;
    }

    .run-btn > button {
        background: #d6336c !important;
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
    }

</style>
<style>
    @media (prefers-color-scheme: light) {
        button, button * {
            background: #1e3c72 !important; 
            color: white !important;
            border: none !important;
        }
    }

    @media (prefers-color-scheme: dark) {
        button, button * {
            background: #1e3c72 !important;
            color: #ffffff !important;
            border: none !important;
        }
    }

    button:hover {
        opacity: 0.9 !important;
    }

</style>

""", unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.annotated_bytes = None
    st.session_state.json_str = None
    st.session_state.inference_ms = None
    st.session_state.fps = None
    st.session_state.image_size = None
    st.session_state.model_name = "YOLOv8n"
    st.session_state.params = "3.2M (approx)"

def reset_app():
    st.markdown("<script>window.scrollTo(0,0);</script>", unsafe_allow_html=True)
    st.session_state.clear()
    import time
    st.session_state.uploader_key = f"uploader_{time.time()}"
    st.rerun()


st.markdown("""
    <div class="hero-header" style="
        width: 100%;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background: linear-gradient(90deg, #4b79a1, #283e51);
        margin-top: 15px;
        margin-bottom: 30px;
    ">
        <h1 style="margin-bottom: 10px; font-size: 38px;">YOLOv8 Object Detection</h1>
        <p style="font-size: 18px; opacity: 0.9;">
            AI-driven object detection for images — fast, accurate, and production-ready⚡
        </p>
    </div>
""", unsafe_allow_html=True)


st.markdown("""
<h3 style="color:#1e3c72; font-size:26px; margin-bottom:4px;">📁 Upload Image</h3>
<p style="color:#555; margin-top:0; font-size:15px;">
    Upload an image and get instant object detection.
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key=st.session_state.get("uploader_key", "uploader_1")
)


def show_detection_output():

    result = st.session_state.result
    detections = result["detections"]
    annotated_bytes = st.session_state.annotated_bytes
    json_str = st.session_state.json_str

    st.markdown("### Annotated Output Image")
    st.image(annotated_bytes, width='stretch')

    st.markdown("### Downloads")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "⬇ Annotated Image",
            data=annotated_bytes,
            file_name=result["annotated_image"],
            mime="image/jpeg",
            key="dl1"
        )
        st.markdown("<style>#dl1{}</style>", unsafe_allow_html=True)

    with c2:
        st.download_button(
            "⬇ JSON Output",
            data=json_str,
            file_name=result["json_file"],
            mime="application/json",
            key="dl2"
        )
        st.markdown("<style>#dl2{}</style>", unsafe_allow_html=True)

    with c3:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            zipf.writestr(result["json_file"], json_str)
            zipf.writestr(result["annotated_image"], annotated_bytes)
        zip_data = zip_buffer.getvalue()

        st.download_button(
            "⬇ All Outputs (ZIP)",
            data=zip_data,
            file_name="yolo_detection_bundle.zip",
            mime="application/zip",
            key="dl3"
        )
        st.markdown("<style>#dl3{}</style>", unsafe_allow_html=True)


    with st.expander("Show Detection JSON"):
        st.json(detections)


    col_sum, col_conf = st.columns([1, 1])

    with col_sum:
        st.markdown("<h3>Object Summary</h3>", unsafe_allow_html=True)
        from collections import Counter
        labels = [d["class_name"] for d in detections]
        counts = Counter(labels)

        pill_html = "<div style='margin-bottom: 12px;'>"
        for cls, cnt in counts.items():
            pill_html += (
                "<span style=\"padding:6px 14px;margin-right:8px;margin-bottom:6px;"
                "display:inline-block;background-color:#e4ecfb;color:#1e3c72;"
                "font-weight:600;border-radius:20px;border:1px solid #c7d7f5;\">"
                f"{cls}: {cnt}</span>"
            )
        pill_html += "</div>"
        st.markdown(pill_html, unsafe_allow_html=True)


    with col_conf:
        st.markdown("<h3>Confidence Scores</h3>", unsafe_allow_html=True)
        for det in detections:
            st.write(f"**{det['class_name']}** — {det['confidence']*100:.1f}%")
            st.progress(det["confidence"])


    st.markdown("### Performance Metrics")
    m1, m2 = st.columns(2)

    with m1:
        st.markdown(f"""
            <div style="padding:15px;border-radius:12px;border:1px solid #ddd;background:white;">
                <h4>Inference Time</h4>
                <p><b>{st.session_state.inference_ms} ms</b></p>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
            <div style="padding:15px;border-radius:12px;border:1px solid #ddd;background:white;">
                <h4>FPS</h4>
                <p><b>{st.session_state.fps}</b></p>
            </div>
        """, unsafe_allow_html=True)


    st.markdown("### Image & Model Metadata")
    st.markdown(f"""
        <div style="padding:15px;border-radius:12px;border:1px solid #ddd;background:white;">
            <p><b>Image Size:</b> {st.session_state.image_size}</p>
            <p><b>Model:</b> {st.session_state.model_name}</p>
            <p><b>Model Parameters:</b> {st.session_state.params}</p>
            <p><b>YOLO Input Size:</b> 640 × 640</p>
        </div>
    """, unsafe_allow_html=True)


    center = st.columns([1, 2, 1])[1]
    with center:
        if st.button("🔄 Detect New Image", use_container_width=True):
            reset_app()


if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width='stretch')

    col1, col2, col3 = st.columns([1, 0.5, 1.1])

    with col2:
        run_clicked = st.button("Run Detection", key="runbtn")

    if run_clicked:

        progress = st.progress(0)
        start_time = time.time()

        with st.spinner("Running YOLO inference..."):
            for i in range(0, 70, 10):
                time.sleep(0.07)
                progress.progress(i)

            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(API_URL, files=files)

        progress.progress(100)
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            st.session_state.result = result

            inference_ms = round((end_time - start_time) * 1000, 2)
            st.session_state.inference_ms = inference_ms
            st.session_state.fps = round(1000 / inference_ms, 2)

            pil_img = Image.open(uploaded_file)
            st.session_state.image_size = f"{pil_img.width} × {pil_img.height}"

            annotated_url = f"http://127.0.0.1:8000/outputs/{result['annotated_image']}"
            img_response = requests.get(annotated_url)

            st.session_state.annotated_bytes = img_response.content
            st.session_state.json_str = json.dumps(result["detections"], indent=4)

            st.success("Detection Completed!")

            show_detection_output()

    elif st.session_state.result is not None:
        show_detection_output()

else:
    st.session_state.result = None
    st.session_state.annotated_bytes = None
    st.session_state.json_str = None


st.markdown("""
<hr>
<p style='text-align:center; color:#aaa;'>
Built with ❤️ by Harini Mohan using YOLOv8, FastAPI & Streamlit
</p>
""", unsafe_allow_html=True)
