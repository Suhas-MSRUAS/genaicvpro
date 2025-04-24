import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import json
import tempfile

# Load the trained model
model = YOLO("D:/genaicvpro/runs/detect/train2/weights/best.pt")

st.set_page_config(page_title="Pen Box Layout Interpreter", layout="wide")
st.title("🖊️ Pen Box Layout Interpretation")

# Layout: Two columns
left_col, right_col = st.columns(2)

# --- LEFT COLUMN: Image Upload and Prediction ---
with left_col:
    st.header("Upload Pen Box Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Submit for Layout Analysis"):
            with st.spinner("Running prediction..."):
                # Save temp image
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, uploaded_file.name)
                image.save(image_path)

                # Run prediction
                results = model(image_path)[0]
                rendered_image = results.plot()
                st.image(rendered_image, caption="Prediction", use_container_width=True)

                # Get image size
                img_width, img_height = image.size

                # Extract layout info for structured JSON
                layout_json = {}
                centers = {}
                dimensions = {}

                for box in results.boxes:
                    cls = int(box.cls.cpu().item())
                    conf = round(float(box.conf.cpu().item()), 2)
                    label = model.names[cls]
                    x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().tolist())

                    x_center = round((x1 + x2) / 2 / img_width, 2)
                    y_center = round((y1 + y2) / 2 / img_height, 2)
                    width = round(x2 - x1)
                    height = round(y2 - y1)

                    centers[label] = (x_center, y_center)
                    dimensions[label] = [width, height]

                    layout_json[label] = {
                        "dimensions": [width, height]
                    }

                    if label == "insert":
                        layout_json[label]["type"] = "pen_holder"
                        layout_json[label]["material"] = "black velvet"

                # Compute relative position of pen with respect to insert
                if "pen" in centers and "insert" in centers:
                    pen_x, pen_y = centers["pen"]
                    insert_x, insert_y = centers["insert"]

                    dx = pen_x - insert_x
                    dy = pen_y - insert_y

                    if abs(dx) < 0.1:
                        position = "centre"
                    elif dx < 0:
                        position = "left corner"
                    else:
                        position = "right corner"

                    if "pen" in layout_json:
                        layout_json["pen"]["relative_position"] = position
                    

                # Pass layout data to right column
                st.session_state["layout_json"] = layout_json

# --- RIGHT COLUMN: JSON Layout Output ---
with right_col:
    st.markdown("<h3 style='text-align: center;'>📐 Extracted Layout (JSON)</h3>", unsafe_allow_html=True)
    if "layout_json" in st.session_state:
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.json(st.session_state["layout_json"])
        st.markdown("</div>", unsafe_allow_html=True)

        json_bytes = json.dumps(st.session_state["layout_json"], indent=2).encode('utf-8')
        st.download_button(
            label="⬇️ Download JSON",
            data=json_bytes,
            file_name="layout.json",
            mime="application/json"
        )
    else:
        st.markdown("<div style='text-align: center;'>Upload an image and click submit to see the layout structure.</div>", unsafe_allow_html=True)
