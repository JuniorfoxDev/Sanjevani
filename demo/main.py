import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model import ImprovedTinyVGGModel
from utils import load_and_preprocess_image, predict_image

# Dictionary to store treatment information for each disease
treatment_info = {
    'AMD': "Age-related macular degeneration (AMD) treatment may include injections, laser therapy, or photodynamic therapy.",
    'Cataract': "Cataract treatment typically involves surgery to remove the cloudy lens and replace it with an artificial one.",
    'Glaucoma': "Treatment for glaucoma usually involves prescription eye drops, oral medications, laser therapy, or surgery.",
    'Myopia': "Myopia can be managed with prescription glasses, contact lenses, or refractive surgery.",
    'Non-eye': "The detected condition does not seem to be an eye disease. Consult a healthcare professional for further evaluation.",
    'Normal': "No eye disease detected. Continue regular eye check-ups for preventive care."
}

def main():
    st.title("Sanjevani")
    st.markdown('Retinal Disease Detection')

    # Setting device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Trained Model
    MODEL_SAVE_PATH = "models/MultipleEyeDiseaseDetectModel.pth"
    model_info = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'))

    # Instantiate Model
    model = ImprovedTinyVGGModel(
        input_shape=3,
        hidden_units=48,
        output_shape=6).to(device)

    # Define paths
    data_path = Path("demo/test_images/")

    # Image upload section
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image
        custom_image_path = data_path / uploaded_file.name
        with open(custom_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and preprocess the image
        custom_image_transformed = load_and_preprocess_image(custom_image_path)

        # Load the model
        model.load_state_dict(model_info)
        model.eval()

        # Predict the label for the image
        class_names = np.array(['AMD', 'Cataract', 'Glaucoma', 'Myopia', 'Non-eye', 'Normal'])
        predicted_label, image_pred_probs = predict_image(model,
                                                          custom_image_transformed,
                                                          class_names)

        # Prediction result section
        st.markdown(
            f'<h3 style="color: green;">Prediction Result</h3>', 
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 3])

        # Display prediction label and confidence rate on the left column
        col1.write(f"Predicted eye disease: **{predicted_label[0]}**")

        # Display the treatment information based on the predicted disease
        if predicted_label[0] in treatment_info:
            col1.write(f"**Treatment:** {treatment_info[predicted_label[0]]}")
        else:
            col1.write("**Treatment information not available.**")

        # Display the uploaded image on the right column
        with col2:
            image = Image.open(custom_image_path)
            st.image(image, caption='Uploaded Image', use_column_width=True)

if __name__ == "__main__":
    main()
