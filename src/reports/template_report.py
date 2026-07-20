"""
Template-based fallback report generator for Module 5.

Zero external dependencies, zero network calls, always works. Used
automatically by generate_report.py whenever the local LLM (Ollama) is
unavailable, so report generation — and a live demo/viva — never hard-fails
just because a background service wasn't running.
"""

FINDING_TEXT = {
    "Normal": "no findings suggestive of malignancy were identified on the CT scan",
    "Benign": "a non-malignant-appearing lesion was identified on the CT scan",
    "Malignant": "findings on the CT scan are suspicious for malignancy",
}

RECOMMENDATION_TEXT = {
    "Low": "routine follow-up per standard screening guidelines is recommended",
    "Medium": "closer monitoring and follow-up imaging within 3-6 months is recommended",
    "High": "prompt referral for further diagnostic workup (e.g. biopsy, PET-CT) is recommended",
}

PATIENT_FINDING_TEXT = {
    "Normal": "Your scan did not show anything concerning.",
    "Benign": "Your scan showed a spot that appears non-cancerous.",
    "Malignant": "Your scan showed something that needs a closer look.",
}


def confidence_band(confidence: float) -> str:
    if confidence >= 0.9:
        return "high"
    elif confidence >= 0.7:
        return "moderate"
    return "low"


def generate_doctor_report_template(image_class, image_confidence, risk_level, top_risk_factors, gradcam_note):
    return f"""1. Patient Summary
AI-assisted CT screening and clinical risk assessment completed.

2. Findings
{FINDING_TEXT[image_class]}. Model confidence: {image_confidence:.1%}. {gradcam_note}

3. Risk Assessment
Clinical risk model output: {risk_level} risk. Contributing factors: {top_risk_factors}.

4. Confidence Analysis
Image classification confidence of {image_confidence:.1%} is {confidence_band(image_confidence)} \
and should be weighed alongside clinical judgment and any discordant signals.

5. Clinical Interpretation
Combined image and clinical signals are consistent with {image_class} imaging findings \
at {risk_level} clinical risk.

6. Recommendations
{RECOMMENDATION_TEXT[risk_level]}.

7. Follow-Up Actions
Clinician review of the full case is required before any result is communicated to the \
patient. This output is AI-assisted decision support only, not an autonomous diagnosis."""


def generate_patient_report_template(image_class, image_confidence, risk_level):
    return f"""{PATIENT_FINDING_TEXT[image_class]}

Based on the health information you provided, your risk level was assessed as {risk_level.lower()}.

This result was generated with the help of an AI tool. Your doctor will review everything \
carefully and talk to you about what it means and what happens next. If anything is unclear, \
please don't hesitate to ask your doctor directly."""