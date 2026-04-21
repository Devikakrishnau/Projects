# =========================================================
# PYTHON PATH FIX
# =========================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# Imports
# =========================================================
import gradio as gr
import torch
import torch.nn as nn
from torchvision import models
from datetime import datetime

from backend.agents.orchestrator import Orchestrator
from frontend.report_generator import generate_pdf

# =========================================================
# Path Setup
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "backend" / "trained_models" / "tumor_model.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}\nRun: python backend/train_model.py"
    )

# =========================================================
# Load Model
# =========================================================
print("Loading model for frontend...")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
model.eval()

orch = Orchestrator(model)

print("Model loaded successfully")

# =========================================================
# GLOBAL CACHE
# =========================================================
analysis_cache = {}

# =========================================================
# Helper formatters
# =========================================================
def format_conf(c):
    return f"{c * 100:.1f} %"

def risk_label(r):
    if r > 0.7:
        return "High"
    if r > 0.4:
        return "Medium"
    return "Low"

# =========================================================
# MAIN AI ANALYSIS
# =========================================================
def analyze(image, clinical_text):

    if image is None:
        return ("No image", "", "", "", None)

    temp_path = BASE_DIR / "temp_ui.jpg"
    image.save(temp_path)

    result = orch.run_pipeline(str(temp_path), clinical_text)

    analysis_cache["data"] = result
    analysis_cache["gradcam"] = result["gradcam_image"]
    analysis_cache["original"] = str(temp_path)

    return (
        result["prediction"].upper(),
        format_conf(result["confidence"]),
        risk_label(result["risk_score"]),
        f'{result["tumor_size_cm"]:.2f} cm',
        result["gradcam_image"]
    )

# =========================================================
# BUILD REPORT
# =========================================================
def build_report(name, age, gender, blood_group, doctor):

    if "data" not in analysis_cache:
        return None

    patient = {
        "name": name,
        "age": age,
        "gender": gender,
        "blood_group": blood_group,
        "doctor": doctor,
        "date": datetime.now().strftime("%d-%m-%Y")
    }

    analysis = analysis_cache["data"]

    img_path = analysis_cache["original"]

    pdf_path = generate_pdf(patient, analysis, img_path)

    return pdf_path

# =========================================================
# UI
# =========================================================
with gr.Blocks(theme=gr.themes.Soft()) as ui:

    gr.Markdown("# Agentic AI Tumor Detection System")

    with gr.Tabs():

        # ================= PAGE 1 =================
        with gr.Tab("AI Analysis"):

            with gr.Row():

                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload MRI Scan")

                    clinical_input = gr.Textbox(
                        label="Clinical Notes",
                        placeholder="e.g. seizure, headache, blurred vision"
                    )

                    run_btn = gr.Button("Run Agentic Analysis")

                with gr.Column():
                    pred_box = gr.Textbox(label="Predicted Tumor Type")
                    conf_box = gr.Textbox(label="Model Confidence")
                    risk_box = gr.Textbox(label="Clinical Risk Level")
                    size_box = gr.Textbox(label="Estimated Tumor Size (cm)")

            heatmap_img = gr.Image(label="Grad-CAM Tumor Localization")

            run_btn.click(
                analyze,
                inputs=[image_input, clinical_input],
                outputs=[pred_box, conf_box, risk_box, size_box, heatmap_img]
            )

        # ================= PAGE 2 =================
        with gr.Tab("Patient Report"):

            gr.Markdown("## Enter Patient Details")

            name = gr.Textbox(label="Patient Name")

            age = gr.Textbox(label="Age")

            gender = gr.Dropdown(
                ["Male", "Female", "Other"],
                label="Gender"
            )

            blood_group = gr.Dropdown(
                ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                label="Blood Group"
            )

            doctor = gr.Textbox(label="Doctor Name")

            generate_btn = gr.Button("Generate Radiology Report PDF")

            pdf_output = gr.File(label="Download Report")

            generate_btn.click(
                build_report,
                inputs=[name, age, gender, blood_group, doctor],
                outputs=pdf_output
            )

# =========================================================
# Launch
# =========================================================
if __name__ == "__main__":
    ui.launch(server_name="127.0.0.1", server_port=7860)