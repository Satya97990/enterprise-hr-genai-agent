import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def train_hr_agent():
    print("Loading JSONL dataset...")
    # Load the perfectly formatted interactions we generated
    dataset = load_dataset("json", data_files="data/processed/sft_train.jsonl", split="train")

    print("Loading Hugging Face base model and tokenizer...")
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Chat models require a padding token to batch inputs during training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load raw PyTorch weights. 
    # Device map "auto" attempts to use GPU if available, otherwise falls back to CPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    print("Applying PEFT/LoRA Configuration...")
    # LoRA isolates the training to a tiny subset of attention weights (q_proj, v_proj), 
    # drastically reducing the memory footprint required for fine-tuning.
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules="all-linear", # Automatically targets Phi-3's specific layer names
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Define how the PyTorch training loop will behave
    training_args = TrainingArguments(
        output_dir="data/models/phi3-hr-agent-checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_torch",
        report_to="none" # Prevents WandB login prompts
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    print("Starting Supervised Fine-Tuning Loop...")
    trainer.train()

    print("Saving custom HR Agent weights...")
    final_model_path = "data/models/phi3-hr-agent-final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Phase 3 Complete! Model saved to {final_model_path}")

if __name__ == "__main__":
    train_hr_agent()