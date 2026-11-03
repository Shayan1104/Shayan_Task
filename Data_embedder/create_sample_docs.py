import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_sample_pdf(filepath):
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter

    # Page 1
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, height - 72, "NETSOL Technologies - Global Employee Handbook")
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 90, "Document ID: NT-HB-2026 | Version: 4.2 | Classification: Internal")
    c.line(72, height - 100, width - 72, height - 100)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 130, "1. Executive Overview & Core Mission")
    c.setFont("Helvetica", 11)
    text_p1 = (
        "NETSOL Technologies is a worldwide leader in innovative software solutions for asset finance and leasing. "
        "Our flagship platform, NFS Ascent, provides end-to-end automation for credit evaluation, loan servicing, "
        "and digital asset lifecycle management across Tier-1 financial institutions worldwide. "
        "Our core mission is to empower global financial enterprises with autonomous intelligence, "
        "seamless scalability, and bank-grade data security."
    )
    y = height - 160
    for line in [text_p1[:80], text_p1[80:160], text_p1[160:240], text_p1[240:]]:
        c.drawString(72, y, line.strip())
        y -= 18

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y - 20, "2. Core Principles & Engineering Standards")
    y -= 50
    principles = [
        "Customer Centricity: Deliver unmatched reliability and high availability for our clients.",
        "Engineering Rigor: Implement automated unit, integration, and security testing in all codebases.",
        "Continuous AI Innovation: Integrate vector search, retrieval-augmented generation, and automated workflows."
    ]
    for p in principles:
        c.drawString(85, y, f"• {p}")
        y -= 22

    c.setFont("Helvetica-Oblique", 9)
    c.drawRightString(width - 72, 36, "Page 1 of 3")
    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "NETSOL Technologies - Global Employee Handbook")
    c.line(72, height - 85, width - 72, height - 85)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 120, "3. Work Environment & Hybrid Office Policy")
    c.setFont("Helvetica", 11)
    y = height - 150
    policy = (
        "NETSOL operates on a flexible hybrid work model. Team members are expected to collaborate on-site "
        "on Tuesdays and Thursdays, with flexible remote work permitted on Mondays, Wednesdays, and Fridays. "
        "Core collaboration hours are between 10:00 AM and 4:00 PM local time. "
        "All engineers are provided with dual-monitor ergonomic setups, high-performance laptops, and cloud sandbox credits."
    )
    for line in [policy[:80], policy[80:160], policy[160:240], policy[240:]]:
        c.drawString(72, y, line.strip())
        y -= 18

    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, "4. Information Security & Compliance")
    y -= 30
    sec_points = [
        "Data Encryption: All customer data must be encrypted at rest (AES-256) and in transit (TLS 1.3).",
        "Credential Hygiene: Mandatory multi-factor authentication (MFA) on all developer repositories.",
        "Incident Protocol: Report any security anomalies to secops@netsoltech.com within 30 minutes."
    ]
    for sp in sec_points:
        c.drawString(85, y, f"• {sp}")
        y -= 22

    c.setFont("Helvetica-Oblique", 9)
    c.drawRightString(width - 72, 36, "Page 2 of 3")
    c.showPage()

    # Page 3
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "NETSOL Technologies - Global Employee Handbook")
    c.line(72, height - 85, width - 72, height - 85)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 120, "5. Benefits, Paid Time Off & Healthcare")
    c.setFont("Helvetica", 11)
    y = height - 150
    benefits = [
        "Annual Leave: 24 paid vacation days per calendar year, accrued monthly.",
        "Health Coverage: Comprehensive medical, dental, and vision insurance with 100% company-paid premiums for employees.",
        "Learning Stipend: $2,500 annual personal budget for AI conferences, certifications, and technical books.",
        "Wellness Allowance: $75 monthly reimbursement for gym memberships, fitness trackers, or ergonomic equipment.",
        "Parental Leave: 16 weeks of fully paid parental leave for primary caregivers, 8 weeks for secondary caregivers."
    ]
    for b in benefits:
        c.drawString(85, y, f"• {b[:85]}")
        y -= 18
        if len(b) > 85:
            c.drawString(100, y, b[85:])
            y -= 18
        y -= 8

    c.setFont("Helvetica-Oblique", 9)
    c.drawRightString(width - 72, 36, "Page 3 of 3")
    c.save()
    print(f"Created 3-page sample PDF: {filepath}")

def create_sample_files():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_docs"))
    os.makedirs(out_dir, exist_ok=True)

    # 1. 3-Page PDF
    pdf_path = os.path.join(out_dir, "netsol_handbook.pdf")
    create_sample_pdf(pdf_path)

    # 2. Markdown File
    md_content = """# AI Vector Architecture & RAG Pipeline Specification

## 1. Overview
This technical specification outlines the end-to-end vector embedding and retrieval architecture for the NIAI Task Data Embedder.

## 2. Ingestion & Document Parsing
The system accepts raw documents in various formats:
- Adobe Portable Document Format (.pdf) with page-level text extraction
- Plain text (.txt), Markdown (.md), CSV, and JSON data
- Microsoft Word documents (.docx)
- Web URLs parsed via WebBaseLoader / BeautifulSoup
- Recursive folder trees and compressed ZIP archives

## 3. Semantic Chunking Strategy
Documents are partitioned using RecursiveCharacterChunker:
- Target Chunk Size: 800 characters
- Chunk Overlap: 100 characters
- Separator Hierarchy: `["\\n\\n", "\\n", ". ", "? ", "! ", " "]`
- Metadata: Each chunk preserves the source filename, exact page number (1-indexed), chunk index, and estimated token counts.

## 4. Vector Embedding Models
Vector representations are computed using:
- Local CPU Mode: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- Optional Cloud Mode: OpenAI `text-embedding-3-small` (1536-d) or Google Gemini `text-embedding-004` (768-d).
- Distance Metric: Cosine similarity via normalized dot products.
"""
    with open(os.path.join(out_dir, "ai_architecture.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    # 3. Financial TXT File
    txt_content = """NETSOL TECHNOLOGIES Q3 FISCAL PERFORMANCE REPORT
Date: March 31, 2026
Prepared By: Global Finance Operations

Key Performance Highlights:
- Total Quarterly Revenue reached $16.4 million, representing an 18% year-over-year increase.
- Cloud SaaS Subscription Revenue grew by 42% to $6.8 million.
- Operating Income stood at $2.9 million, compared to $1.8 million in the prior year period.
- NFS Ascent enterprise implementations were completed across 4 major clients in North America and Asia-Pacific.
- Research and Development expenditure was $3.2 million, focused on autonomous AI agents and generative intelligence.
- Cash and cash equivalents at quarter-end totaled $28.5 million with zero long-term debt.
"""
    with open(os.path.join(out_dir, "q3_financial_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt_content)

    print("Sample documents created in sample_docs/ folder.")

if __name__ == "__main__":
    create_sample_files()
