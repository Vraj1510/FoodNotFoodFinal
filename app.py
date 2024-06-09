import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from streamlit_extras.stylable_container import stylable_container
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face Hub
# Replace 'your-username/your-model-id' with your actual model ID
model_path = hf_hub_download(repo_id='rudrashah/is-this-food', filename='food_or_not_v2.keras')

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)
# Check the expected input shape
st.set_page_config(layout="wide")
input_shape = model.input_shape

def preprocess_image(image):
    # Convert image to RGB if it's RGBA
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Resize image to match model input size
    image = image.resize((224, 224))  # Resize to 224x224 pixels
    # Convert image to numpy array
    image = np.asarray(image)
    # Expand dimensions to match model input shape
    image = np.expand_dims(image, axis=0)
    return image

# Function to make prediction
def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(processed_image)
    return prediction

# Main function to run the app
def main():
    # Inject custom CSS
  
    with stylable_container(
        key="cat_container",
        css_styles=[
            """
            {
                padding: 0.5em;
                border-radius: 1em;
                position: fixed;
            }
            """,
            """
            .stMarkdown {
                padding-right: 1.5em;
                margin-top: -95px;
                position: fixed;
            }
            """,
        ],
    ):
        # Title of the app
        st.markdown("<div style='text-align: start; font-weight:0px; margin-left:-20px; font-size:50px; margin-top:30px;'>Food Classifier</div>", unsafe_allow_html=True)

        # Create columns with specific widths
        col1, col2 = st.columns([2, 3])

        with col1:
            # Upload image
            with stylable_container(
                key="upload_container",
                css_styles=[
                    """
                    {
                        margin-top:30px;
                        margin-bottom:100px;
                        margin-left:-20px;
                    }
                    """,
                    """
                    .stMarkdown {
                        padding-right: 1.5em;
                    }
                    """,
                ],
            ):
             uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                    # Predict whether the image is food or not
                    image = Image.open(uploaded_image)
                    prediction = predict(image)
                    with stylable_container(
                      key="upload_container",
                      css_styles=[
                          """
                          {
                              margin-bottom:10px;
                          }
                          """,
                          """
                          .stMarkdown {
                              padding-right: 1.5em;
                          }
                          """,
                      ],
                  ):
                     if prediction[0][0] < 0.5:
                        st.success("This is a food image.")
                     else:
                        st.error("This is not a food image.")

        with col2:
            if uploaded_image is not None:
                # Display the uploaded image
                image = Image.open(uploaded_image)
                with stylable_container(
                    key="image_container",
                    css_styles=[
                        """
                        {
                            margin-top: 0px;
                            padding: 0.5em;
                            border-radius: 1em;
                            margin-top: -45px;
                            padding-bottom: 40px;
                            margin-left:20px;
                            height:400px;
                          
                        }
                        """,
                    ],
                ):
                 st.image(image, caption="", width=770, use_column_width=None)

# Run the app
if __name__ == "__main__":
    main()