import json
import random
import os
from datetime import datetime, timedelta

# Configuration
NUM_RECORDS = 100
OUTPUT_DIR = "data/raw/json"

# Categorical data for randomization
DEPARTMENTS = ["Risk Advisory", "Consulting", "Tax", "Audit", "Internal IT"]
DESIGNATIONS = ["Associate Analyst", "Analyst", "Senior Analyst", "Manager", "Director"]
LOCATIONS = ["Bengaluru - Outer Ring Road", "Bengaluru - Whitefield", "Mumbai", "Gurugram"]

def generate_employee_record(emp_id):
    """Generates a single synthetic employee dictionary."""
    joining_date = datetime.now() - timedelta(days=random.randint(100, 1500))
    designation = random.choice(DESIGNATIONS)
    
    # Base salary tiers based on designation to maintain data consistency
    salary_bands = {
        "Associate Analyst": (600000, 800000),
        "Analyst": (800000, 1200000),
        "Senior Analyst": (1200000, 1800000),
        "Manager": (1800000, 3000000),
        "Director": (3000000, 5000000)
    }
    
    base_salary = random.randint(*salary_bands[designation])
    
    record = {
        "employee_id": f"EMP{str(emp_id).zfill(5)}",
        "first_name": f"Employee_{emp_id}_First",
        "last_name": f"Employee_{emp_id}_Last",
        "professional_details": {
            "designation": designation,
            "department": random.choice(DEPARTMENTS),
            "location": random.choice(LOCATIONS),
            "date_of_joining": joining_date.strftime("%Y-%m-%d"),
            "employment_status": "Active"
        },
        "leave_balances": {
            "privilege_leave": random.randint(0, 21), # Standard PL capping
            "sick_leave": random.randint(0, 12),      # Standard SL capping
            "casual_leave": random.randint(0, 7)      # Standard CL capping
        },
        "compensation": {
            "base_salary_inr": base_salary,
            "provident_fund_tier": "Standard 12%",
            "gratuity_eligible": (datetime.now() - joining_date).days >= 1825 # 5 years
        },
        "assets_assigned": [
            {"asset_type": "Laptop", "asset_tag": f"LT-{random.randint(1000, 9999)}"}
        ]
    }
    return record

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i in range(1, NUM_RECORDS + 1):
        record = generate_employee_record(i)
        filepath = os.path.join(OUTPUT_DIR, f"{record['employee_id']}.json")
        
        with open(filepath, 'w') as f:
            json.dump(record, f, indent=4)
            
    print(f"Successfully generated {NUM_RECORDS} employee JSON records in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()