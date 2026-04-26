import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import cv2
import numpy as np
import hashlib

def overlay_heatmap_with_contour(heatmap, original_image):

    import cv2
    import numpy as np

    # resize heatmap
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)

    # colored heatmap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # overlay heatmap
    overlay = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)

    # threshold to detect tumor
    _, thresh = cv2.threshold(heatmap_uint8, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contour
    cv2.drawContours(overlay, contours, -1, (0,255,0), 3)

    return overlay
def is_valid_mri_image(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    if mean_intensity < 10 or mean_intensity > 245:
        return False, "Image brightness is not suitable."

    if std_intensity < 15:
        return False, "Image contrast is too low."

    return True, "Valid MRI image."
def gradcam_to_mask(heatmap, original_image):

    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Stronger threshold to avoid whole-brain activation
    _, mask = cv2.threshold(heatmap, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask)

    if len(contours) > 0:
        h, w = mask.shape
        image_area = h * w

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 150:
                continue

            if area > image_area * 0.18:
                continue

            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            break

    return clean_mask
# AI Model Imports
import google.generativeai as gen_ai

# Brain Tumor Classification Imports
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from vnet_model import VNet

import streamlit.components.v1 as components

# Load environment variables
load_dotenv()
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

# Configure Streamlit page settings
st.set_page_config(
    page_title="TumorScanAI.com",
    page_icon=":brain:",
    layout="centered",
)


# Load the Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
chat_model = gen_ai.GenerativeModel('gemini-2.5-flash')



# Set up the brain tumor classification model (EfficientNet)
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'best_efficientnet_brain_tumor_classification.pth'

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

if not os.path.exists(model_path):
    st.error(f"Classifier model file not found: {model_path}")
    st.stop()

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Load VNet model for tumor segmentation
vnet_model = VNet().to(device)

if not os.path.exists("vnet_trained.pth"):
    st.error("Segmentation model file not found: vnet_trained.pth")
    st.stop()

vnet_model.load_state_dict(torch.load("vnet_trained.pth", map_location=device))
vnet_model.eval()


class_names = ['Glioma','Meningioma','No Tumor','Pituitary']    

def generate_gradcam(model, image):

    gradients = None
    activations = None

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    # Last convolution layer of EfficientNet
    target_layer = model.features[-1][0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.eval()

    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0).to(device)

    output = model(image)
    class_idx = output.argmax()

    model.zero_grad()
    output[0, class_idx].backward()

    # Safety check
    if gradients is None or activations is None:
        return None

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.relu(heatmap)

    heatmap /= torch.max(heatmap)

    forward_handle.remove()
    backward_handle.remove()

    return heatmap.detach().cpu().numpy()

# Function to predict the class of an uploaded image
def predict_image(image_file, model):

    image = image_file.convert("RGB")

    image = image_transforms(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return class_names[predicted.item()], confidence.item() * 100
def segment_tumor_vnet(image):

    img = cv2.resize(np.array(image), (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # normalize
    img = img.astype(np.float32) / 255.0

    # standardization
    img = (img - img.mean()) / (img.std() + 1e-8)

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    img = torch.tensor(img, dtype=torch.float32).to(device)

    with torch.no_grad():
        mask = vnet_model(img)
        mask = torch.sigmoid(mask)

    mask = mask.squeeze().cpu().numpy()

    # Stronger threshold for cleaner tumor region
    mask = (mask > 0.55).astype(np.uint8) * 255

    # Strong cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
def calculate_vnet_tumor_size(mask, original_image):

    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    _, brain_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    brain_pixels = np.sum(brain_mask > 0)
    tumor_pixels = np.sum(mask > 0)

    if brain_pixels == 0:
        return 0

    tumor_percentage = (tumor_pixels / brain_pixels) * 100

    return tumor_percentage
# Function to assess emergency level and suggest treatment or vitamins
def assess_tumor_emergency(predicted_class):
    emergency_info = {
    
  "Pituitary": {
    "emergency_level": "Medium",
    "action": "Visit a doctor to discuss treatment options. Pituitary tumors can affect hormone levels and require medical attention.",
    "origin": "Arise from the pituitary gland, a small gland at the base of the brain responsible for hormone production.",
    "description": """
        The pituitary gland is a small, pea-sized gland located at the base of the brain. 
        It is responsible for producing and releasing hormones that regulate a wide range 
        of bodily functions, including growth, metabolism, and reproduction.
        
        Pituitary tumors are growths that develop in the pituitary gland. They can be benign
         (non-cancerous) or malignant (cancerous). 
        Benign pituitary tumors are more common than malignant pituitary tumors.

        Pituitary tumors can cause a variety of symptoms, depending on their size 
        and location. Common symptoms include:
        - Headaches
        - Vision problems
        - Double vision
        - Blurred vision
        - Loss of peripheral vision
        - Nausea and vomiting
        - Fatigue
        - Weight gain
        - Increased thirst
        - Frequent urination
        - Menstrual irregularities
        - Infertility
        - Erectile dysfunction
        
        Pituitary tumors are diagnosed with a combination of physical 
        examination, blood tests, and imaging tests. Blood tests can
         be used to measure the levels of hormones produced by the pituitary 
         gland. Imaging tests, such as MRI and CT scans, can be used to 
         visualize the pituitary gland and to identify any tumors.

        Treatment for pituitary tumors depends on the size, location, and
         type of tumor. Treatment options include:
        - Surgery
        - Radiation therapy
        - Medication
        - Observation

        Surgery is the primary treatment for pituitary tumors. The goal of 
        surgery is to remove the tumor while preserving the function of the
         pituitary gland. Surgery is typically performed through the nose
          or through a small incision in the forehead.

        Radiation Therapy: Radiation therapy uses high-energy radiation to
         kill tumor cells. It can be used before surgery to shrink the tumor
          or after surgery to kill any remaining tumor cells.

        Medication: Medication can be used to treat pituitary tumors that
         are not amenable to surgery or radiation therapy. Medications can
          be used to lower hormone levels, shrink the tumor, or relieve symptoms.

        Observation: Observation is an option for patients with small,
         slow-growing pituitary tumors that are not causing any symptoms.
          Patients who are observed will have regular blood tests and imaging 
          tests to monitor the tumor.

        Prognosis: The prognosis for pituitary tumors depends on the size, location,
         and type of tumor, as well as the patient's age and overall health.
          The five-year survival rate for patients with pituitary tumors is about 95%.
    """
  },
  "Glioma": {
    "emergency_level": "High",
    "action": "Seek immediate medical consultation and treatment options, as gliomas are often aggressive and require prompt attention.",
    "origin": "Gliomas arise from glial cells, which support and protect neurons in the brain and spinal cord.",
    "description": """
        Gliomas are tumors that arise from the glial cells in the brain or spinal cord. They are the most common type of brain tumor, accounting for about 80% of all brain tumor cases. Gliomas can be classified into different grades based on their aggressiveness, ranging from grade I (least aggressive) to grade IV (most aggressive).

        Symptoms of gliomas can vary depending on the location and size of the tumor.
         Common symptoms include:
        - Headaches
        - Seizures
        - Nausea and vomiting
        - Vision or hearing problems
        - Weakness or numbness
        - Memory problems
        - Personality changes

        Gliomas are diagnosed through physical examinations, imaging tests like MRI 
        or CT scans, and biopsy to confirm the diagnosis and determine the grade of
         the tumor.

        Treatment for gliomas includes:
        - Surgery: Removing as much of the tumor as possible while preserving 
        healthy tissue.
        - Radiation therapy: High-energy radiation to kill tumor cells.
        - Chemotherapy: Drugs to kill tumor cells or stop them from growing.

        Prognosis varies depending on the tumor's grade. Lower-grade gliomas 
        have a better prognosis with treatment, but higher-grade gliomas,
        particularly grade IV gliomas (glioblastomas), have a poor 
        prognosis with a survival rate lower than 5% for five years.

        Gliomas require close monitoring and aggressive treatment, especially
        for high-grade types, to manage symptoms and improve the patient's 
        quality of life.
    """
  },
  "Meningioma": {
    "emergency_level": "Low",
    "action": "Consult a doctor for routine follow-ups and management options, as most meningiomas are benign and treatable.",
    "origin": "Meningiomas arise from the meninges, the membranes that cover the brain and spinal cord.",
    "description": """
        Meningiomas are typically benign tumors that form in the meninges, which are the protective layers surrounding the brain and spinal cord. They are the most common type of primary brain tumor, accounting for about 30% of all brain tumors. Meningiomas can grow slowly and often don't cause symptoms right away.

        Symptoms of meningiomas can vary based on their size and location.
         Common symptoms include:
        - Headaches
        - Seizures
        - Vision problems
        - Hearing issues
        - Weakness or numbness
        - Balance problems

        Diagnosis of meningiomas is usually through imaging tests like MRI 
        and CT scans, followed by biopsy to confirm the nature of the tumor.

        Treatment options for meningiomas include:
        - Surgery: Removing the tumor, especially if it is causing symptoms.
        - Radiation therapy: Used for inoperable or recurrent meningiomas to 
          shrink the tumor.
        - Observation: If the tumor is small and not causing significant
         symptoms, doctors may opt for regular monitoring rather than immediate
          intervention.

        Most meningiomas are benign and can be successfully treated with surgery.
         The prognosis is excellent for patients who undergo surgical treatment,
          with many achieving full recovery. However, for malignant meningiomas,
           the prognosis depends on the extent of tumor removal and other factors
            like age and overall health.
    """
  },
  "No Tumor": {
    "emergency_level": "None",
    "action": "No medical action required. Regular check-ups are recommended for maintaining health and well-being.",
    "origin": "No tumor detected in the body or brain. Healthy tissue.",
    "description": """
        No tumor means that there are no abnormal growths or masses present
        in the body or brain. This is the ideal scenario for an individual’s 
        health, indicating that there are no cancerous or benign growths that
         could cause harm or disrupt normal bodily functions.

        In this scenario, individuals typically experience no symptoms related
         to tumors, such as headaches, seizures, vision issues, or pain. This 
         is the result of healthy and normal functioning tissues.

        Regular health check-ups, including physical exams and imaging tests 
        (such as MRIs or CT scans when necessary), are always recommended to 
        ensure continued good health and early detection of any potential health
         concerns. Early detection of abnormal growths or tumors can be critical
          for successful treatment.

        For individuals with no tumors, the overall prognosis is excellent, as no
         medical conditions related to tumors are present.
    """
  }
    }
    
    # Check if the predicted class is in the dictionary
    if predicted_class in emergency_info:
        return emergency_info[predicted_class]
    else:
        return {
            "emergency_level": "None",
            "action": "No medical action required. Regular check-ups are recommended for maintaining health and well-being.",
            "origin": "No tumor detected in the body or brain. Healthy tissue.",
            "description": """
        No tumor means that there are no abnormal growths or masses present
        in the body or brain. This is the ideal scenario for an individual’s 
        health, indicating that there are no cancerous or benign growths that
         could cause harm or disrupt normal bodily functions.

        In this scenario, individuals typically experience no symptoms related
         to tumors, such as headaches, seizures, vision issues, or pain. This 
         is the result of healthy and normal functioning tissues.

        Regular health check-ups, including physical exams and imaging tests 
        (such as MRIs or CT scans when necessary), are always recommended to 
        ensure continued good health and early detection of any potential health
         concerns. Early detection of abnormal growths or tumors can be critical
          for successful treatment.

        For individuals with no tumors, the overall prognosis is excellent, as no
         medical conditions related to tumors are present.
    """
        }

if "page" not in st.session_state:
    st.session_state.page = "main"  # Default page

def navigate_to(page):
    st.session_state.page = page


#main page
import streamlit as st

def main_page():
    # Set the page background and text color
    st.markdown("""
        <style>
        body {
            background-color: white;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>MEDICAL AI PLATFORM</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Advanced brain tumor classification and medical chatbot assistance powered by artificial intelligence.</h3>", unsafe_allow_html=True)

    # Button and box styling
    button_style = """
        <style>
        .stButton>button {
            width: 100%;
            height: 80px;
            font-size: 24px;
            border: 2px solid black;
            background-color: white;
            color: black;
        }
        .stButton>button:hover {
            background-color: black;
            color: white;
        }
        
        /* Styling for boxes */
        .stTextInput, .stNumberInput, .stTextArea, .stSelectbox, .stCheckbox {
            border: 2px solid black;
            background-color: white;
            color: black;
        }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Sign-In"):
            navigate_to("signin")

    with col2:
        if st.button("Register"):
            navigate_to("register")


import streamlit as st
import json
import os

# Register Page
def register_page():
    st.title("Register")

    # Input fields for username, password, and confirm password
    username = st.text_input("Enter a username:")
    password = st.text_input("Enter a password:", type="password")
    confirm_password = st.text_input("Confirm your password:", type="password")

    # Create two columns for the buttons
    col1, col2 = st.columns([1, 1])

    # Register button
    with col1:
        if st.button("Register"):
            # Validate inputs
            if username == "" or password == "" or confirm_password == "":
                st.error("All fields are required. Please fill in all the fields.")
            elif password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                # Save user credentials to JSON file
                if os.path.exists("users.json"):
                    with open("users.json", "r") as f:
                        users = json.load(f)
                else:
                    users = {}

                if username in users:
                    st.error("Username already exists. Please choose a different username.")
                else:
                    users[username] = hash_password(password)
                    with open("users.json", "w") as f:
                        json.dump(users, f)
                    st.success("Registration successful! You can now sign in.")
                    navigate_to("signin")

    # Sign In button (right-aligned)
    with col2:
        if st.button("Sign In"):
            navigate_to("signin")

# Sign In Page
def signin_page():
    st.title("Sign In")
    
    # Input fields for username and password
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    # Create two columns for the buttons
    col1, col2 = st.columns([1, 1])

    # Sign-In button (left-aligned)
    with col1:
        if st.button("Sign In"):
            # Validate inputs
            if username == "" or password == "":
                st.error("Both username and password are required.")
            else:
                if os.path.exists("users.json"):
                    with open("users.json", "r") as f:
                        users = json.load(f)

                    if username in users and verify_password(password, users[username]):
                        st.success("Successfully signed in!")
                        navigate_to("dashboard")
                    else:
                        st.error("Invalid credentials. Please try again.")
                else:
                    st.error("No users registered yet. Please register first.")

    # Sign-Up button (right-aligned)
    with col2:
        if st.button("Sign Up"):
            navigate_to("register")



#dashboard
def dashboard_page():
    # Set the background color to white and text color to black
    st.markdown("""
        <style>
        body {
            background-color: white;
            text_color: black;
        }
        .stButton>button {
            width: 100%;
            height: 80px;
            font-size: 24px;
            border: 2px solid black;
            background-color: white;
            color: black;
        }
        .stButton>button:hover {
            background-color: black;
            color: white;
        }
        .logout-button {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 10;
            background-color: white;
            border: 2px solid black;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header and description
    st.markdown("<h1 style='text-align: center; color: black;'>MEDICAL AI PLATFORM</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>Advanced brain tumor classification and medical chatbot assistance powered by artificial intelligence.</h3>", unsafe_allow_html=True)

    # Create three columns for the main buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    # Buttons for each section
    with col1:
        with st.container():
            st.button("Symptom Analyzer", on_click=go_to_symptom_analyzer)

    with col2:
        with st.container():
            st.button("Brain Tumor Analyzer", on_click=go_to_brain_tumor_analyzer)

    with col3:
        with st.container():
            st.button("BrainyBot", on_click=go_to_chatbot)



# Placeholder function to simulate navigating to the Symptom Analyzer page
def go_to_symptom_analyzer():
    st.session_state.page = "symptom_analyzer"

# Placeholder function to simulate navigating to the Brain Tumor Analyzer page
def go_to_brain_tumor_analyzer():
    st.session_state.page = "brain_tumor_analyzer"

# Function to initiate the chatbot page
def go_to_chatbot():
    st.session_state.page = "chatbot"

# Symptom Analyzer Page (Loading HTML file)
def symptom_analyzer_page():
    st.markdown("<h1 style='text-align: center;'>Symptom Analyser</h1>", unsafe_allow_html=True)

    # Load and display the HTML file using components
    html_file_path = 'index.html'  # Make sure the path is correct
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except UnicodeDecodeError:
        st.error("Error reading the HTML file. Please check the file encoding.")
        return

    # Display HTML content in Streamlit
    components.html(html_content, height=600, scrolling=True)

    # Back to main page
    if st.button("Go Back"):
        st.session_state.page = "dashboard"


def brain_tumor_analyzer_page():

    st.title("Brain Tumor Analyzer")
    st.markdown("Upload an MRI image to analyze a potential brain tumor.")

    uploaded_file = st.file_uploader("Upload MRI image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        original_image = np.array(image)

        valid, message = is_valid_mri_image(original_image)
        if not valid:
            st.error(f"Invalid or low-quality MRI image: {message}")
            return
        # --- Brain region cropping ---
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            brain_crop = original_image[y:y+h, x:x+w]
        else:
            brain_crop = original_image

        # convert cropped image back to PIL
        image = Image.fromarray(brain_crop)
        original_image = brain_crop
        # -----------------------------
        st.image(image, caption="Uploaded MRI Image", width=300)

        if st.button("Classify Tumor"):

            tumor_percentage = 0

            predicted_class, confidence = predict_image(image, model)

            st.success(f"Predicted Tumor Type: {predicted_class}")
            st.info(f"Prediction Confidence: {confidence:.2f}%")

            if confidence < 60:
                st.warning("Low confidence prediction. Please upload a clearer MRI image.")

            # ---------- GradCAM ----------
            heatmap = generate_gradcam(model, image)

            if heatmap is None:
                st.warning("GradCAM could not be generated.")
                st.image(original_image, caption="MRI Image", width=300)
                tumor_percentage = 0

            else:
                combined_image = overlay_heatmap_with_contour(heatmap, original_image)

                # =========================================================
                # ✅ GLIOMA → GradCAM + Segmentation
                # =========================================================
                if predicted_class == "Glioma":

                    seg_mask = segment_tumor_vnet(image)
                    seg_mask = cv2.resize(seg_mask, (original_image.shape[1], original_image.shape[0]))

                    kernel = np.ones((5, 5), np.uint8)
                    seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel)
                    seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel)

                    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    clean_mask = np.zeros_like(seg_mask)

                    if len(contours) > 0:
                        contours = sorted(contours, key=cv2.contourArea, reverse=True)
                        cv2.drawContours(clean_mask, [contours[0]], -1, 255, -1)

                    overlay_seg = original_image.copy()
                    overlay_seg[clean_mask > 0] = [255, 0, 0]

                    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay_seg, contours, -1, (0, 255, 0), 2)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(combined_image, caption="GradCAM Detection", width=300)

                    with col2:
                        st.image(overlay_seg, caption="Glioma Segmentation", width=300)

                    tumor_percentage = calculate_vnet_tumor_size(clean_mask, original_image)

                # =========================================================
                # ✅ MENINGIOMA / PITUITARY → ONLY GradCAM
                # =========================================================
                elif predicted_class in ["Meningioma", "Pituitary"]:

                    st.info(f"{predicted_class} detected. Showing tumor localization using GradCAM.")

                    st.image(combined_image, caption="Tumor Localization (GradCAM)", width=300)

                    clean_mask = gradcam_to_mask(heatmap, original_image)
                    tumor_percentage = calculate_vnet_tumor_size(clean_mask, original_image)

                # =========================================================
                # ✅ NO TUMOR
                # =========================================================
                else:
                    st.success("No Tumor Detected")
                    st.image(original_image, caption="MRI Image", width=300)
                    tumor_percentage = 0

            # Tumor severity classification
            
            if predicted_class == "No Tumor":
                severity = "No tumor detected"

            elif tumor_percentage == 0:
                severity = "Tumor not clearly detected"

            elif tumor_percentage < 3:
                severity = "Low"

            elif tumor_percentage < 7:
                severity = "Medium"

            elif tumor_percentage < 12:
                severity = "High"

            else:
                severity = "Critical"

            st.markdown("### Tumor Analysis")
            st.info(f"Tumor Coverage in Brain: {tumor_percentage:.2f}%")
            st.warning(f"Tumor Severity Level: {severity}")


            # =========================================================
            # ✅ EMERGENCY = SAME AS SEVERITY
            # =========================================================
            assessment = assess_tumor_emergency(predicted_class)

            if predicted_class == "No Tumor":
                emergency_override = "None"
            else:
                emergency_override = severity

            st.markdown("## 🧾 AI Medical Report")

            st.write("Tumor Type:", predicted_class)
            st.write("Prediction Confidence:", f"{confidence:.2f}%")
            st.write("Emergency Level:", emergency_override)

            if tumor_percentage == 0:
                st.write("Estimated Tumor Coverage: Segmentation could not detect tumor region")
            else:
                st.write("Estimated Tumor Coverage:", f"{tumor_percentage:.2f}%")

            st.write("Recommended Action:", assessment["action"])
            st.write("Tumor Origin:", assessment["origin"])
            st.write("Description:", assessment["description"])

            if tumor_percentage == 0:
                tumor_coverage_text = "Segmentation could not detect tumor region"
            else:
                tumor_coverage_text = f"{tumor_percentage:.2f}%"

            report_text = f"""
AI MEDICAL REPORT
-------------------------
Tumor Type: {predicted_class}
Prediction Confidence: {confidence:.2f}%
Emergency Level: {emergency_override}
Estimated Tumor Coverage: {tumor_coverage_text}

Recommended Action: {assessment["action"]}
Tumor Origin: {assessment["origin"]}

Description:
{assessment["description"]}
"""

            st.download_button(
                label="Download AI Medical Report",
                data=report_text,
                file_name="ai_medical_report.txt",
                mime="text/plain"
            )

    if st.button("Go Back"):
        st.session_state.page = "dashboard"
def chatbot_section():
    st.header("💬 Chat with Our Brainy Bot")

    # ---------------- SESSION SETUP ----------------
    if "hospital_chat_session" not in st.session_state:
        gen_ai.configure(api_key=GOOGLE_API_KEY)
        hospital_model = gen_ai.GenerativeModel("gemini-2.5-flash")
        st.session_state.hospital_chat_session = hospital_model.start_chat(history=[])

    if "general_chat_session" not in st.session_state:
        gen_ai.configure(api_key=GOOGLE_API_KEY)
        general_model = gen_ai.GenerativeModel("gemini-2.5-flash")
        st.session_state.general_chat_session = general_model.start_chat(history=[])

    if "hospital_chat_history" not in st.session_state:
        st.session_state.hospital_chat_history = []

    if "general_chat_history" not in st.session_state:
        st.session_state.general_chat_history = []

    # ---------------- MODE SELECTOR ----------------
    mode = st.radio(
        "Choose Mode:",
        ["🏥 Hospital Guidance", "💬 General Smart Chatbot"],
        horizontal=True
    )

    # =========================================================
    # MODE 1: HOSPITAL GUIDANCE
    # =========================================================
    if mode == "🏥 Hospital Guidance":
        st.subheader("🏥 Find Hospitals by Specialist / Area / City")
        st.caption("Examples: neurologist hyderabad, wakad pune, cardiologist mumbai, hospitals in vizag")

        for role, msg in st.session_state.hospital_chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        hospital_prompt = st.chat_input(
            "Enter area, city, specialist or hospital need...",
            key="hospital_input"
        )

        if hospital_prompt:
            st.session_state.hospital_chat_history.append(("user", hospital_prompt))
            with st.chat_message("user"):
                st.markdown(hospital_prompt)

            hospital_system_prompt = f"""
You are a hospital finder assistant for India.

User input: {hospital_prompt}

Understand the user naturally:

1. If user types only a place like:
   - warangal
   - wakad pune
   - hyderabad
   - gachibowli
   then treat it as:
   "Show hospitals in that location"

2. If user types specialist + place like:
   - cardiologist hyderabad
   - neurologist pune
   - skin doctor vizag
   then treat it as:
   "Show hospitals suitable for that specialty in that location"

IMPORTANT RESPONSE FORMAT:
You MUST give the answer in this exact structure for every hospital:

Hospital Name: <hospital name>
Description: <1 line why this hospital is useful>
Address: <detailed address with area, locality, city, state as much as possible>

RULES:
- Give 5 to 8 hospitals if possible.
- Always include address.
- Address should be as detailed as possible.
- Prefer branch name, road name, area, locality, city, state.
- Do NOT give generic answer first.
- Start directly with hospitals.
- Keep it clean and practical.
- If exact street number is not known, give best-known detailed branch/locality.
- No paragraph introduction.
- No explanation before the hospital list.

At the end write exactly:
Please call the hospital before visiting to confirm doctor availability.
"""

            response = st.session_state.hospital_chat_session.send_message(hospital_system_prompt)

            hospital_reply = response.text

            formatted_reply = hospital_reply
            formatted_reply = formatted_reply.replace("Hospital Name:", "\n### 🏥 **Hospital Name:** ")
            formatted_reply = formatted_reply.replace("Description:", "\n- **Description:** ")
            formatted_reply = formatted_reply.replace("Address:", "\n- **Address:** ")

            with st.chat_message("assistant"):
                st.markdown(formatted_reply)

            st.session_state.hospital_chat_history.append(("assistant", formatted_reply))

    # =========================================================
    # MODE 2: GENERAL SMART CHATBOT
    # =========================================================
    elif mode == "💬 General Smart Chatbot":
        st.subheader("💬 Ask Anything")

        for role, msg in st.session_state.general_chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        general_prompt = st.chat_input(
            "Ask anything... (Example: what is brain tumor, tell me a joke, explain cloud computing)",
            key="general_input"
        )

        if general_prompt:
            st.session_state.general_chat_history.append(("user", general_prompt))
            with st.chat_message("user"):
                st.markdown(general_prompt)

            general_system_prompt = f"""
You are Brainy Bot, a smart, friendly, and helpful AI assistant inside a Brain Tumor Analysis platform.

User asked: {general_prompt}

Rules:
1. Answer all types of questions naturally.
2. The user may ask:
   - medical questions
   - brain tumor questions
   - health awareness questions
   - educational questions
   - coding / technical questions
   - random normal questions
3. If it is a medical question:
   - answer in simple English
   - do not diagnose with certainty
   - suggest consulting a doctor if serious
4. If it is a normal question:
   - answer like a smart chatbot
5. Keep the tone helpful, simple, and natural.
6. Avoid robotic wording.
"""

            response = st.session_state.general_chat_session.send_message(general_system_prompt)

            with st.chat_message("assistant"):
                st.markdown(response.text)

            st.session_state.general_chat_history.append(("assistant", response.text))

    # ---------------- RESET BUTTONS ----------------
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Reset Hospital Chat"):
            if "hospital_chat_session" in st.session_state:
                del st.session_state["hospital_chat_session"]
            if "hospital_chat_history" in st.session_state:
                del st.session_state["hospital_chat_history"]
            st.rerun()

    with col2:
        if st.button("🔄 Reset General Chat"):
            if "general_chat_session" in st.session_state:
                del st.session_state["general_chat_session"]
            if "general_chat_history" in st.session_state:
                del st.session_state["general_chat_history"]
            st.rerun()

    # ---------------- GO BACK ----------------
    if st.button("Go Back"):
        st.session_state.page = "dashboard"

# Initialize page state
# App logic for navigation
if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "signin":
    signin_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
elif st.session_state.page == "symptom_analyzer":
    symptom_analyzer_page()
elif st.session_state.page == "brain_tumor_analyzer":
    brain_tumor_analyzer_page()
elif st.session_state.page == "chatbot":
    chatbot_section()