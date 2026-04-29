import streamlit as st
from PIL import Image
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import your backend
from src.tryon.tryon import try_on

st.set_page_config(page_title="Virtual Try-On", layout="wide")

st.title("🧥 Virtual Try-On")

st.write("Upload your photo and a garment to see how it looks on you.")

# ---------------- INPUT ----------------
person_file = st.file_uploader("Upload Person Image", type=["jpg", "png"])
garment_file = st.file_uploader("Upload Garment Image", type=["jpg", "png"])

# ---------------- PROCESS ----------------
if st.button("Try On"):

    if person_file is None or garment_file is None:
        st.warning("Please upload both images")
    else:
        # Save uploaded files temporarily
        os.makedirs("temp", exist_ok=True)

        person_path = os.path.join("temp", "person.png")
        garment_path = os.path.join("temp", "garment.png")

        with open(person_path, "wb") as f:
            f.write(person_file.read())

        with open(garment_path, "wb") as f:
            f.write(garment_file.read())

        # Call your backend function
        output_path = try_on(person_path, garment_path)

        # ---------------- OUTPUT ----------------
        st.subheader("Result")
        st.image(output_path, caption="Virtual Try-On Result")