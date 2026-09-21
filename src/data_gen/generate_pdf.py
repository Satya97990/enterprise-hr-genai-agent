import os
from fpdf import FPDF

# Configuration
OUTPUT_DIR = "data/raw/pdf"

# Unstructured corporate policies
POLICIES = {
    "Leave_and_Attendance_Policy_Bengaluru": """
1. Purpose
This document outlines the leave policies for employees based in the Bengaluru regional offices.

2. Privilege Leave (PL)
Employees are entitled to 21 Privilege Leaves per calendar year. PLs are accrued proportionally at the end of each month. 

3. Sick Leave (SL)
Employees are entitled to 12 Sick Leaves per annum. Unused SLs cannot be encashed and lapse at the end of the calendar year.

4. Casual Leave (CL)
7 Casual Leaves are granted annually to attend to unforeseen personal matters.
    """,
    "IT_Asset_Requisition_Policy": """
1. Overview
This policy governs the assignment, use, and return of corporate IT assets across all departments.

2. Standard Allocation
Associate Analysts and Analysts will be assigned a standard Windows laptop. Managers and Directors are eligible for premium laptops or macOS devices based on departmental approval.

3. Return of Assets
All IT assets must be returned to the Internal IT department on or before the last working day during the separation process. Failure to do so will result in a deduction from the Full & Final (FnF) settlement.
    """,
    "Compensation_and_Gratuity_Policy": """
1. Base Salary and Provident Fund
The standard Provident Fund (PF) deduction is calculated at 12% of the basic salary tier.

2. Gratuity Eligibility
As per the Payment of Gratuity Act, employees are eligible for gratuity payouts only after completing 5 continuous years (1825 days) of active service with the organization.

3. Verification
Employment status and exact dates of joining are strictly maintained in the centralized HR JSON records.
    """
}

class PolicyPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Internal Corporate Policy Document - Confidential", align="C", ln=1)
        self.ln(5)

def generate_pdfs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for title, content in POLICIES.items():
        pdf = PolicyPDF()
        pdf.add_page()
        
        # Add Policy Title
        pdf.set_font("Arial", "B", 16)
        formatted_title = title.replace("_", " ")
        pdf.cell(0, 10, formatted_title, ln=1)
        pdf.ln(5)
        
        # Add Policy Content
        pdf.set_font("Arial", "", 11)
        # Using multi_cell to handle line breaks and unstructured paragraph text
        pdf.multi_cell(0, 6, content.strip())
        
        filepath = os.path.join(OUTPUT_DIR, f"{title}.pdf")
        pdf.output(filepath)
        
    print(f"Successfully generated {len(POLICIES)} unstructured PDF policies in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_pdfs()