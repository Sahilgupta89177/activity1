'''Name: Rucha patil
Roll No: 210
'''

import streamlit as st
import cv2
import numpy as np


def apply_translation(image, tx, ty):
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return translated_image


def apply_scaling(image, sx, sy):
    scaled_image = cv2.resize(image, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
    return scaled_image


def apply_rotation(image, angle):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image


def apply_reflection(image, axis):
    if axis == 'horizontal':
        reflected_image = cv2.flip(image, 1)
    elif axis == 'vertical':
        reflected_image = cv2.flip(image, 0)
    else:
        raise ValueError("Invalid reflection axis. Use 'horizontal' or 'vertical'.")
    return reflected_image


def apply_shearing(image, shear_factor):
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    sheared_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return sheared_image


def main():
    st.title("Affine Image Transformation App")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        input_image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        st.image(input_image, caption="Original Image", use_column_width=True)

        # Transformation options
        st.subheader("Choose Transformation")
        transform_option = st.selectbox("Select a transformation:",
                                        ["Translation", "Scaling", "Rotation", "Horizontal Reflection",
                                         "Vertical Reflection", "Shearing"])

        if transform_option == "Translation":
            tx = st.slider("X-Translation", -100, 100, 0)
            ty = st.slider("Y-Translation", -100, 100, 0)
            transformed_image = apply_translation(input_image, tx, ty)
        elif transform_option == "Scaling":
            sx = st.slider("X-Scaling", 0.1, 3.0, 1.0)
            sy = st.slider("Y-Scaling", 0.1, 3.0, 1.0)
            transformed_image = apply_scaling(input_image, sx, sy)
        elif transform_option == "Rotation":
            angle = st.slider("Rotation Angle (degrees)", -180, 180, 0)
            transformed_image = apply_rotation(input_image, angle)
        elif transform_option == "Horizontal Reflection":
            transformed_image = apply_reflection(input_image, 'horizontal')
        elif transform_option == "Vertical Reflection":
            transformed_image = apply_reflection(input_image, 'vertical')
        elif transform_option == "Shearing":
            shear_factor = st.slider("Shearing Factor", -1.0, 1.0, 0.0)
            transformed_image = apply_shearing(input_image, shear_factor)

        st.image(transformed_image, caption="Transformed Image", use_column_width=True)


if __name__ == '__main__':
    main()
