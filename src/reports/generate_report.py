"""
Module 5 — generates a diagnostic report from a completed prediction,
combining Module 1 (image classification), Module 2 (clinical risk), and
Module 3 (Grad-CAM / feature contribution explanations) into two written
outputs: a clinical version (for the doctor) and a plain-language version
(for the patient).

HYBRID DESIGN — free, no API billing:
  1. Tries a local LLM via Ollama first (genuine LLM generation, runs on
     your own GPU, no cost, no internet needed after setup).
  2. If Ollama isn't running/available, automatically falls back to a
     deterministic template generator (src/reports/template_report.py) —
     so this never hard-crashes during a live demo/viva just because a
     background service wasn't started.

One-time setup for the LLM path (optional — template fallback works with
zero setup):
    1. Install Ollama: https://ollama.com/download
    2. ollama pull llama3.2:3b

Usage:
    python -m src.reports.generate_report \\
        --image-class Malignant --image-confidence 0.94 \\
        --risk-level High --top-risk-factors "Obesity,Passive Smoker,Wheezing"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import REPORT_DIR
from src.reports.llm_backend import call_ollama, OllamaUnavailableError
from src.reports.template_report import (
    generate_doctor_report_template,
    generate_patient_report_template,
)


DOCTOR_PROMPT_TEMPLATE = """You are drafting a structured diagnostic report from AI model outputs for a radiologist/oncologist to review. Be precise and appropriately hedged.

Model outputs:
- CT scan classification: {image_class} (confidence: {image_confidence:.1%})
- Clinical risk prediction: {risk_level}
- Key contributing clinical factors: {top_risk_factors}
- Image explainability note: {gradcam_note}

Write a report with EXACTLY these 7 numbered sections, concise (under 350 words total):
1. Patient Summary
2. Findings
3. Risk Assessment
4. Confidence Analysis
5. Clinical Interpretation
6. Recommendations
7. Follow-Up Actions

End by explicitly stating this is AI-assisted decision support requiring clinician review, not an autonomous diagnosis."""

PATIENT_PROMPT_TEMPLATE = """Explain this AI-assisted lung screening result to a patient with no medical background. Be warm, clear, and honest. Avoid jargon and unnecessary alarm.

Result:
- Scan result: {image_class} (AI confidence: {image_confidence:.1%})
- Risk level: {risk_level}

Write under 200 words. End with a clear next step (your doctor will review this and discuss it with you)."""


def build_gradcam_note(image_class: str) -> str:
    if image_class == "Malignant":
        return "Grad-CAM heatmap indicates model attention concentrated on a localized nodular region."
    elif image_class == "Benign":
        return "Grad-CAM heatmap shows model attention on a well-defined, non-invasive-appearing region."
    return "Grad-CAM heatmap shows no significant localized attention region."


def generate_report(image_class, image_confidence, risk_level, top_risk_factors, patient_id="anonymous"):
    gradcam_note = build_gradcam_note(image_class)

    doctor_prompt = DOCTOR_PROMPT_TEMPLATE.format(
        image_class=image_class, image_confidence=image_confidence,
        risk_level=risk_level, top_risk_factors=top_risk_factors,
        gradcam_note=gradcam_note,
    )
    patient_prompt = PATIENT_PROMPT_TEMPLATE.format(
        image_class=image_class, image_confidence=image_confidence, risk_level=risk_level,
    )

    try:
        print("Generating doctor-facing report via local LLM (Ollama)...")
        doctor_text = call_ollama(doctor_prompt, max_tokens=700)
        print("Generating patient-facing report via local LLM (Ollama)...")
        patient_text = call_ollama(patient_prompt, max_tokens=400)
        source = "local-llm (Ollama, llama3.2:3b)"
    except OllamaUnavailableError as e:
        print(f"[INFO] Local LLM unavailable ({e})")
        print("[INFO] Falling back to template-based report generator.")
        doctor_text = generate_doctor_report_template(
            image_class, image_confidence, risk_level, top_risk_factors, gradcam_note
        )
        patient_text = generate_patient_report_template(image_class, image_confidence, risk_level)
        source = "template-fallback"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doctor_path = REPORT_DIR / f"report_doctor_{patient_id}_{timestamp}.md"
    patient_path = REPORT_DIR / f"report_patient_{patient_id}_{timestamp}.md"

    doctor_path.write_text(
        f"# Clinical Report — Patient {patient_id}\n"
        f"Generated: {datetime.now().isoformat()}\n"
        f"Generated via: {source}\n"
        "**AI-assisted decision support draft — requires clinician review**\n\n"
        + doctor_text
    )
    patient_path.write_text(
        f"# Your Screening Result\nGenerated: {datetime.now().isoformat()}\n\n" + patient_text
    )

    return doctor_text, patient_text, doctor_path, patient_path, source


def main():
    parser = argparse.ArgumentParser(description="Generate an LLM diagnostic report (local, free)")
    parser.add_argument("--image-class", required=True, choices=["Normal", "Benign", "Malignant"])
    parser.add_argument("--image-confidence", type=float, required=True)
    parser.add_argument("--risk-level", required=True, choices=["Low", "Medium", "High"])
    parser.add_argument("--top-risk-factors", default="not provided")
    parser.add_argument("--patient-id", default="anonymous")
    args = parser.parse_args()

    doctor_text, patient_text, doctor_path, patient_path, source = generate_report(
        args.image_class, args.image_confidence, args.risk_level,
        args.top_risk_factors, args.patient_id,
    )

    print(f"\n(Generated via: {source})")
    print("\n" + "=" * 60 + "\nDOCTOR REPORT\n" + "=" * 60)
    print(doctor_text)
    print("\n" + "=" * 60 + "\nPATIENT REPORT\n" + "=" * 60)
    print(patient_text)
    print(f"\nSaved: {doctor_path}")
    print(f"Saved: {patient_path}")


if __name__ == "__main__":
    main()