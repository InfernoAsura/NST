import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import io
import os

image_dir = "images"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
options = ["-- Select an image --"] + image_files

@st.cache_resource
def load_model():
    model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
    return model

def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    return img[tf.newaxis, :]

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.cast(tensor[0], tf.uint8)
    return Image.fromarray(tensor.numpy())


st.title("Neural Style Transfer")
style_col, content_col = st.columns(2)
with style_col:
    st.subheader("Style Image")
    style_image = st.file_uploader("Upload the style image", type=['jpg', 'png', 'jpeg'])
    selected_file = st.selectbox("Choose an image:", options)
    if selected_file != "-- Select an image --" or style_image:
        if selected_file != "-- Select an image --":   
            style = Image.open(os.path.join(image_dir, selected_file))
            st.image(style, caption=selected_file, use_container_width=True)
        else:
            style = Image.open(style_image)
            st.image(style, caption="Style", use_container_width = True)

with content_col:
    st.subheader("Content Image")
    content_image1 = st.camera_input("Take a photo")
    content_image2 = st.file_uploader("Upload the content image", type=['jpg', 'png', 'jpeg'])
    if content_image1 is not None:
        content_image = content_image1
    else:
        content_image = content_image2
    if content_image:
        content = Image.open(content_image)
        st.image(content, caption="Content", use_container_width = True)


process = st.button("Process")

if process:
    if (style_image == None and selected_file == "-- Select an image --") or content_image == None:
        st.error("Please upload both the style as well as the content image!!")
    else:
        model = load_model()
        with st.spinner("Processing..."):
            style_tensor = preprocess_image(style)
            content_tensor = preprocess_image(content)
            stylised_tensor = model(content_tensor, style_tensor)[0]
            stylised_image = tensor_to_image(stylised_tensor)
        st.subheader("🎨 Stylized Image")
        st.image(stylised_image, use_container_width=True)

        buf = io.BytesIO()
        stylised_image.save(buf, format="PNG")
        st.download_button("Download Image", buf.getvalue(), "stylised.png", "image/png")
