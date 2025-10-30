import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Coin Value Classifier",
    page_icon="🪙",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    }
    .prediction-value {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained ResNet50 model"""
    # Create model architecture
    model = models.resnet50(pretrained=False)
    
    # Modify first layer for grayscale input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final layer for regression (single output)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1)
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load('model.pth', map_location='cpu'))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'model.pth' not found. Please ensure the model file is in the same directory as this script.")
        return None

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize and pad to square (224x224)
    size = 224
    w, h = image.size
    scale = size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Create square with white background
    square = Image.new('L', (size, size), 255)
    
    # Paste in center
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    square.paste(resized, (x, y))
    
    # Normalize (ImageNet grayscale normalization)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])
    ])
    
    tensor = transform(square)
    return tensor.unsqueeze(0)  # Add batch dimension

def predict(model, image_tensor):
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        output = model(image_tensor)
        predicted_value = output.item()
        
        # Round to nearest 5
        rounded_value = round(predicted_value / 5) * 5
        
        # Clamp between 0 and 100
        rounded_value = max(0, min(100, rounded_value))
        
        return predicted_value, int(rounded_value)

# Main app
def main():
    # Header
    st.title("Coin Classifier")
    st.markdown("### Upload a coin image to predict its value (0p - 100p)")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a clear image of a coin"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Show image info
            st.caption(f"Size: {image.size[0]}×{image.size[1]} pixels")
            st.caption(f"Mode: {image.mode}")
        
        with col2:
            st.subheader("🔍 Preprocessed Image")
            # Show preprocessed image
            preprocessed = image.convert('L')
            st.image(preprocessed, use_container_width=True, channels="GRAY")
            st.caption("Converted to grayscale")
        
        # Predict button
        st.markdown("---")
        if st.button("🎯 Classify Coin"):
            with st.spinner("Analyzing image..."):
                # Preprocess
                image_tensor = preprocess_image(image)
                
                # Predict
                raw_prediction, rounded_prediction = predict(model, image_tensor)
                
                # Display results
                st.success("✅ Classification Complete!")
                
                # Prediction box
                st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Predicted Value</h3>
                        <div class="prediction-value">{rounded_prediction}p</div>
                        <p style="color: #666; margin-top: 10px;">
                            Raw prediction: {raw_prediction:.2f}p
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence indicator
                st.markdown("### 📊 Prediction Details")
                
                # Create a visual representation
                possible_values = list(range(0, 105, 5))
                st.write(f"**Possible values:** {', '.join([f'{v}p' for v in possible_values])}")
                
                # Show which class was predicted
                class_index = possible_values.index(rounded_prediction)
                st.progress((class_index + 1) / len(possible_values))
                st.caption(f"Class {class_index + 1} of {len(possible_values)}")
    
    else:
        # Instructions when no image is uploaded
        st.info("👆 Please upload a coin image to get started")
        
        # Example section
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload** a clear image of a coin using the file uploader above
            2. **Review** the original and preprocessed images
            3. **Click** the 'Classify Coin' button
            4. **View** the predicted coin value
            
            **Supported coin values:** 0p, 5p, 10p, 15p, 20p, 25p, 30p, 35p, 40p, 45p, 50p, 55p, 60p, 65p, 70p, 75p, 80p, 85p, 90p, 95p, 100p
            
            **Tips for best results:**
            - Use clear, well-lit images
            - Ensure the coin is the main subject
            - Avoid blurry or heavily shadowed images
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Powered by ResNet50 & PyTorch</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
