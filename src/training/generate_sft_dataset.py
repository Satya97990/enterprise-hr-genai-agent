import json
import os
import random

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_train.jsonl")

def generate_sft_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # The strict system prompt we want the model to internalize
    system_prompt = (
        "You are a strict, professional HR assistant for an enterprise company. "
        "Use the provided context to answer the employee's question accurately. "
        "Do not hallucinate. Do not state that you are an AI."
    )
    
    # Templates for our perfect training examples
    dataset = []
    
    # Scenario 1: Leave Balance Calculation
    for i in range(1, 51):
        total_leaves = 21
        taken_leaves = random.randint(0, 21)
        balance = total_leaves - taken_leaves
        
        user_content = (
            f"I am EMP{i:03d}. How many Privilege Leaves do I have left?\n\n"
            f"--- Employee Profile (JSON) ---\n"
            f"{{\n  \"emp_id\": \"EMP{i:03d}\",\n  \"leaves_taken\": {taken_leaves}\n}}\n\n"
            f"--- Corporate Policy (PDF) ---\n"
            f"Bengaluru employees are entitled to a maximum of {total_leaves} Privilege Leaves per calendar year."
        )
        
        assistant_content = f"You are entitled to {total_leaves} Privilege Leaves. According to your profile, you have taken {taken_leaves}. You have a remaining balance of {balance} Privilege Leaves."
        
        dataset.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

    # Scenario 2: Policy Verification (No Math)
    locations = ["Bengaluru", "Hyderabad", "Pune"]
    for loc in locations:
        for i in range(15):
            user_content = (
                f"What is the remote work policy for my location?\n\n"
                f"--- Employee Profile (JSON) ---\n"
                f"{{\n  \"emp_id\": \"EMP_R{i}\",\n  \"location\": \"{loc}\"\n}}\n\n"
                f"--- Corporate Policy (PDF) ---\n"
                f"Employees based in {loc} are required to be in the office 3 days a week. Fully remote work is not permitted without VP approval."
            )
            
            assistant_content = f"According to the policy for {loc}, you are required to be in the office 3 days a week. Fully remote work requires VP approval."
            
            dataset.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            })

    # Shuffle the dataset to prevent the model from memorizing patterns sequentially
    random.shuffle(dataset)

    # Write to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully generated {len(dataset)} SFT training examples.")
    print(f"Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    print("Generating Supervised Fine-Tuning Dataset...")
    generate_sft_data()