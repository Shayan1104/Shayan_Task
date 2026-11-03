import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def convert_to_python_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    else:
        try:
            return obj.item()
        except:
            return obj

def generate_json_report(profile, file_path):

    profile = convert_to_python_types(profile)

    report = {
        "generated_at": str(datetime.now()),
        "profile_summary": profile
    }

    with open(file_path, "w") as f:
        json.dump(report, f, indent=4)
def generate_pdf_report(profile, file_path):

    c = canvas.Canvas(file_path, pagesize=letter)

    y = 750

    c.drawString(50, y, "DATA CLEANING REPORT")
    y -= 40

    for key, value in profile.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    c.save()       