from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import os


# ------------------------------------------------
# CLINICAL INTERPRETATION
# ------------------------------------------------
def clinical_interpretation(prediction, risk, size_cm):

    findings = []

    if "meningioma" in prediction.lower():
        findings += [
            "Well-defined extra-axial lesion noted.",
            "Imaging features suggest benign meningioma."
        ]

    elif "glioma" in prediction.lower():
        findings += [
            "Intra-axial infiltrative lesion identified.",
            "Findings suspicious for primary glial tumor."
        ]

    elif "pituitary" in prediction.lower():
        findings += [
            "Sellar/suprasellar mass observed.",
            "Features suggest pituitary adenoma."
        ]

    else:
        findings.append("No definite intracranial mass lesion detected.")

    findings.append(f"Estimated tumor size: {size_cm:.2f} cm.")
    findings.append(f"Clinical risk category: {risk}")

    # Impression
    if risk == "High":
        impression = "Large tumor with significant clinical risk requiring urgent medical evaluation."
    elif risk == "Medium":
        impression = "Moderate tumor presence requiring specialist consultation."
    else:
        impression = "Small lesion with low immediate clinical concern."

    # Suggestions
    suggestions = [
        "Correlate findings with clinical symptoms",
        "Recommend contrast-enhanced MRI for detailed evaluation",
        "Consult neurosurgeon or neurologist for treatment planning"
    ]

    return findings, impression, suggestions


# ------------------------------------------------
# PDF GENERATION
# ------------------------------------------------
def generate_pdf(patient, analysis, img_path):

    file_path = "Radiology_Report.pdf"

    doc = SimpleDocTemplate(
        file_path,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=20
    )

    story = []

    # -------- Styles --------
    section_style = ParagraphStyle(
        "section",
        fontSize=11,
        textColor=colors.HexColor("#0B5394")
    )

    normal_style = ParagraphStyle(
        "normal",
        fontSize=9
    )

    # -------- HEADER --------
    header = Table([["AI NEURO IMAGING CENTER"]], colWidths=[7.5 * inch])

    header.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#0B5394")),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 16),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8)
    ]))

    story.append(header)
    story.append(Spacer(1,6))

    story.append(Paragraph("MRI BRAIN RADIOLOGY REPORT", section_style))
    story.append(Spacer(1,10))

    # -------- Doctor --------
    story.append(
        Paragraph(
            f"<b>Doctor:</b> {patient['doctor']}",
            normal_style
        )
    )

    story.append(Spacer(1,6))

    # -------- PATIENT DETAILS --------
    story.append(Paragraph("PATIENT DETAILS", section_style))
    story.append(Spacer(1,6))

    info = [
        ["Patient Name", patient["name"]],
        ["Age", patient["age"]],
        ["Blood Group", patient["blood_group"]],
        ["Gender", patient["gender"]],
        ["Date", datetime.now().strftime("%d-%m-%Y")]
    ]

    table = Table(info, colWidths=[2.2*inch, 4.5*inch])

    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#D9E1F2")),
        ('GRID',(0,0),(-1,-1),0.4,colors.grey)
    ]))

    story.append(table)
    story.append(Spacer(1,10))

    # -------- AI MEASUREMENTS --------
    story.append(Paragraph("AI MEASUREMENTS", section_style))

    story.append(
        Paragraph(
            f"<b>Tumor Type:</b> {analysis['prediction']}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Model Confidence:</b> {analysis['confidence']*100:.2f}%",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Risk Category:</b> {analysis['risk_level']}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Estimated Tumor Size:</b> {analysis['tumor_size_cm']:.2f} cm",
            normal_style
        )
    )

    story.append(Spacer(1,8))

    # -------- MRI IMAGE --------
    if os.path.exists(img_path):

        story.append(Paragraph("Tumor Localization", section_style))

        img = Image(
            img_path,
            width=2.4 * inch,
            height=2.4 * inch
        )

        img.hAlign = "CENTER"

        story.append(img)
        story.append(Spacer(1,6))

    # -------- FINDINGS --------
    findings, impression, suggestions = clinical_interpretation(
        analysis["prediction"],
        analysis["risk_level"],
        analysis["tumor_size_cm"]
    )

    story.append(Paragraph("KEY FINDINGS", section_style))

    for f in findings:
        story.append(Paragraph("• " + f, normal_style))

    story.append(Spacer(1,6))

    story.append(Paragraph("IMPRESSION", section_style))
    story.append(Paragraph(impression, normal_style))

    story.append(Spacer(1,6))

    # -------- SUGGESTIONS --------
    story.append(Paragraph("SUGGESTIONS", section_style))

    for s in suggestions:
        story.append(Paragraph("• " + s, normal_style))

    doc.build(story)

    return file_path