"""
Fine-tune Qwen2.5 7B on NakshaNirman house plan data.
Optimized for GTX 1650 (4GB VRAM) + 24GB RAM.
Run time: approximately 2-4 hours.
"""

from unsloth import FastLanguageModel
from datasets import Dataset
import json
import torch

print("=== NAKSHA-MASTER Fine-Tuning ===")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow!)'}")
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f}GB")

# Settings optimized for GTX 1650 (4GB VRAM)
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
OUTPUT_DIR = "./naksha_master_finetuned"
TRAINING_DATA = "./training_data.jsonl"
MAX_SEQ_LENGTH = 3072   # Reduced from 4096 to fit in 4GB VRAM
LORA_RANK = 8           # Reduced from 16 to fit in 4GB VRAM
BATCH_SIZE = 1          # Must be 1 for GTX 1650
GRAD_ACCUM = 8          # Simulate batch size of 8
EPOCHS = 3

print(f"\nLoading base model: {BASE_MODEL}")
print("This downloads ~4GB on first run...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves VRAM
    random_state=42,
)

print(f"Trainable parameters: {model.print_trainable_parameters()}")

# Load training data
print(f"\nLoading training data from {TRAINING_DATA}")
examples = []
with open(TRAINING_DATA, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = Dataset.from_list(examples)
dataset = dataset.map(format_example)
print(f"Training on {len(dataset)} examples")

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=1,  # Windows: keep at 1
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        dataloader_num_workers=0,  # Windows: must be 0
    ),
)

print("\nStarting training... (2-4 hours on GTX 1650)")
print("You can leave this running. Check loss every few minutes.")
print("Loss should go DOWN over time (start ~2.0, end ~0.3)\n")

trainer.train()

print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Converting to GGUF format for Ollama...")
model.save_pretrained_gguf(
    OUTPUT_DIR + "_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)

print(f"""
=== Training Complete! ===

Your fine-tuned model is saved in: {OUTPUT_DIR}_gguf

To use it in Ollama:
1. Create Modelfile:
   echo FROM ./{OUTPUT_DIR}_gguf/naksha-master.Q4_K_M.gguf > Modelfile
   echo PARAMETER temperature 0.05 >> Modelfile
   echo PARAMETER num_ctx 3072 >> Modelfile

2. Import into Ollama:
   ollama create naksha-master -f Modelfile

3. Update your .env:
   LOCAL_LLM_MODEL=naksha-master
   LOCAL_LLM_PLAN_MODEL=naksha-master

4. Test it:
   ollama run naksha-master "30x40 east 2BHK Pune"
""")
