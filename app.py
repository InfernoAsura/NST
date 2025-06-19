import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from PIL import Image
import numpy as np
import io



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
    style_image = st.file_uploader("Upload the style image", type=['jpg', 'png'])
    if style_image:
        style = Image.open(style_image)
        st.image(style, caption="Style", use_container_width = True)

with content_col:
    st.subheader("Content Image")
    content_image = st.file_uploader("Upload the content image", type=['jpg', 'png'])
    if content_image:
        content = Image.open(content_image)
        st.image(content, caption="Content", use_container_width = True)


process = st.button("Process")

if process:
    if style_image == None or content_image == None:
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