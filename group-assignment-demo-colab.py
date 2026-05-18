# --- 1. INSTALLATION ---
# !pip install -q -U google-genai gradio nest-asyncio
# !pip install --upgrade gradio

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr
from google.colab import userdata
from google import genai
from google.genai import types # Required for multimodal parts
import nest_asyncio
import io

# Initialize environment
gr.close_all()
nest_asyncio.apply()

# --- 2. GPU & MODEL SETUP ---
device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    from tensorflow.keras import mixed_precision
    try:
        mixed_precision.set_global_policy('mixed_float16')
        print("GPU Mixed Precision Enabled.")
    except: pass

MODEL_PATH = '/content/best_gender_model.keras'
IMG_SIZE = (320, 320)

if os.path.exists(MODEL_PATH):
    with tf.device('/GPU:0' if device_name else '/CPU:0'):
        model = tf.keras.models.load_model(MODEL_PATH)
    INPUT_NAME = model.input_names[0] if hasattr(model, 'input_names') else model.inputs[0].name.split(':')[0].split('/')[-1]
    print(f"Model loaded. Input: {INPUT_NAME}")
else:
    print("ERROR: Upload 'best_gender_model.keras' to Colab.")

# --- 3. PREPROCESSING & HEATMAP ---
def medical_preprocessing(img_np):
    img = img_np.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img.shape) == 2 or img.shape[2] == 1:
        img_gray = img if len(img.shape) == 2 else img[:, :, 0]
        img_clahe = clahe.apply(img_gray)
        img_final = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        img_final = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img_final.astype(np.float32) / 255.0

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    global INPUT_NAME
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        inputs_dict = {INPUT_NAME: img_array}
        last_conv_layer_output, preds = grad_model(inputs_dict, training=False)
        if isinstance(preds, (list, tuple)): preds = preds[0]
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = last_conv_layer_output[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy().astype(np.float32)

# --- 4. MULTIMODAL AGENT LOGIC ---
def analyze_xray(img):
    global INPUT_NAME
    if img is None: return "Error", "0%", None, "No image provided."

    try:
        # A. Neural Network Prediction
        img_resized = img.resize(IMG_SIZE)
        img_np = np.array(img_resized)
        processed_img = medical_preprocessing(img_np)
        img_array = np.expand_dims(processed_img, axis=0)

        with tf.device('/GPU:0' if device_name else '/CPU:0'):
            inputs_dict = {INPUT_NAME: img_array}
            raw_output = model(inputs_dict, training=False)
            if isinstance(raw_output, (list, tuple)): raw_output = raw_output[0]
            preds = raw_output.numpy()

        p_fem = float(preds[0][0])
        gen = "Female" if p_fem > 0.5 else "Male"
        conf = p_fem if p_fem > 0.5 else 1.0 - p_fem

        # B. Grad-CAM Visualization
        h_raw = make_gradcam_heatmap(img_array, model, 'relu')
        h_res = cv2.resize(h_raw, (IMG_SIZE[1], IMG_SIZE[0]))
        h_uint = np.uint8(255 * h_res)
        jet = cv2.applyColorMap(h_uint, cv2.COLORMAP_JET)
        # We use a pure heatmap and a superimposed version
        super_img = cv2.addWeighted(img_np, 0.6, jet, 0.4, 0)
        heatmap_pil = Image.fromarray(jet)
        final_viz = Image.fromarray(super_img)

        # C. Multi-modal Gemini Agent
        summary = ""
        try:
            client = genai.Client(api_key=userdata.get('GEMINI_API_KEY'))

            # Prepare images for Gemini (Original + Superimposed)
            # We pass these as PIL images directly in the contents list

            agent_prompt = f"""
            ROLE: Senior Radiologist Consultant.
            TASK: Evaluate the Deep Learning model's prediction for this Chest X-ray.

            AI PREDICTION: {gen}
            CONFIDENCE: {conf:.2%}

            INSTRUCTIONS:
            1. Look at the 'Original Image' to assess anatomical structure (clavicle width, rib cage, breast tissue shadows).
            2. Look at the 'AI Focus Map' (Grad-CAM) to see which pixels triggered the decision.
            3. Confirm if the AI is looking at valid biological markers or just artifacts (like text labels or medical leads).
            4. Provide a professional reasoning for the identified gender.
            5. Keep the report concise and clinical.
            """

            # Multi-modal call (Text + 2 Images)
            response = client.models.generate_content(
                model='gemini-flash-latest', # Or 'gemini-1.5-flash' depending on your regional access
                contents=[
                    agent_prompt,
                    img,           # Original PIL Image
                    final_viz      # Superimposed Heatmap PIL Image
                ]
            )
            summary = response.text
        except Exception as e:
            summary = f"**Agent Analysis Offline:** {str(e)}"

        return gen, f"{conf:.1%}", final_viz, summary

    except Exception as e:
        return "Error", "0%", None, f"Analysis failed: {str(e)}"

# --- 5. UI INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Multimodal Clinical X-Ray Diagnostics")
    gr.Markdown("Combining DenseNet121 Computer Vision with Gemini Flash Multimodal Reasoning.")

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.Image(type="pil", label="Upload Frontal X-Ray")
            btn = gr.Button("Execute Clinical Analysis", variant="primary")
        with gr.Column(scale=1):
            with gr.Row():
                out_gen = gr.Textbox(label="Identified Biological Gender")
                out_conf = gr.Textbox(label="AI Confidence")
            out_heat = gr.Image(label="Anatomical Feature Saliency (Grad-CAM)")

    gr.Markdown("### 🩺 Senior Radiologist Consultant Report")
    out_summary = gr.Markdown()

    btn.click(analyze_xray, input_file, [out_gen, out_conf, out_heat, out_summary])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)