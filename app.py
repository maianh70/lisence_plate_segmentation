from io import BytesIO
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from helper import mask_extracting, cropping_plate

@st.cache_resource
def load_model():
    return YOLO("best.pt")

max_files = 7
st.title("License Plate Detection")
st.write("Upload an image and get the plate!!!")

up_loaded_files = st.file_uploader(
    "Upload images", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
    )

if len(up_loaded_files) > max_files:
    st.warning(f"We only process {max_files} at a time")

if not up_loaded_files:
    st.warning(f"Please upload your images of plates")


for i, file in enumerate(up_loaded_files):
    # Read the uploaded file to images:
    model = load_model()
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) 
    # 1 as color (BGR), excluding alpha, 0 for gray scale, 1  loads the image as-is (including alpha if present)
    im_bgr_to_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #st.image(im_bgr_to_rgb, caption="Original Image", use_column_width=True)

    with st.spinner("Cutting...."):
        mask, resized_im, cont, conf = mask_extracting(im_bgr_to_rgb, model)
        plate = None
        if mask is not None and conf >= 0.01:
            plate = cropping_plate(resized_im, mask, cont)

        st.image(resized_im, caption=f"The image_{i} has a plate with {round(conf.item(), 2) * 100}% certainty.", use_container_width=True)
    
        if plate is not None:
            st.image(plate, caption=f"The plate_{i}", use_container_width=True)
            
            img_for_pil = Image.fromarray(plate)
            buf = BytesIO()
            img_for_pil.save(buf, format="PNG")
            result_in_byte = buf.getvalue()
        
            st.download_button(
                label="Download Plate",
                data=result_in_byte,
                file_name=f"plate{i}.png",
                mime="image/png",
                key=f"download_plate_{i}"
            )
        else:
            st.write(f"Can not detect any plates in image_{i}")



