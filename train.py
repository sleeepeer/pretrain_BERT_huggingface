# Training arguments
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

BLOCK_SIZE = 128  # Maximum number of tokens in an input sample
MAX_LENGTH = 128  # Maximum number of tokens in an input sample after padding

MLM_PROB = 0.15  # Probability with which tokens are masked in MLM

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4

# Accumulator, 用于节省显存
GRAD_ACC = 1

STEPS = 23767 / (BATCH_SIZE * GRAD_ACC)
# ------------------------------------------------------------------------------------------------------
# Load tokenizer
from transformers import BertTokenizer
tokenizer = BertTokenizer(vocab_file="./tokenizer_wikitxt/vocab.txt")

# Load model
from transformers import (
    CONFIG_MAPPING,MODEL_FOR_MASKED_LM_MAPPING, AutoConfig,
    BertForMaskedLM,
    AutoTokenizer,DataCollatorForLanguageModeling,HfArgumentParser,Trainer,TrainingArguments,set_seed,
)
config_kwargs = {
    "vocab_size" : 25000,
}
config = AutoConfig.from_pretrained("./config.json", **config_kwargs)
model = BertForMaskedLM(config)

# Load dataset
from datasets import load_from_disk
dataset = load_from_disk(dataset_path="./raw-wikitxt/")
column_names = dataset["train"].column_names

# tokenize the dataset
def tokenize_function(examples):
    # remove empty lines
    examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
    return tokenizer(
        examples["text"],
        padding="max_length", # 填充
        truncation=True, # 截断
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True,
    )
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=None,
    remove_columns="text",
    load_from_cache_file=True,
)

train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# Define trainer and train
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB)

training_args = TrainingArguments(
    output_dir='./outputs/',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=2000,
    save_total_limit=2,
    gradient_accumulation_steps=GRAD_ACC)

# 通过Trainer接口训练模型
trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

trainer.train(resume_from_checkpoint=False)
trainer.save_model("./outputs/")
