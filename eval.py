#%%
import transformers as t
import torch
import peft
import time
import safetensors

#%%
tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=True, torch_dtype=torch.float16)
tokenizer.pad_token_id = 0
#%%
config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")

model = peft.get_peft_model(model, config)

adapter_model_dict = safetensors.torch.load_file(
    "./output/checkpoint-800/adapter_model.safetensors", device="cpu"
)
torch.save(adapter_model_dict, "./output/checkpoint-800/adapter_model.bin")

peft.set_peft_model_state_dict(model, torch.load("./output/checkpoint-800/adapter_model.bin"))
#%%

TEMPLATE = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
INSTRUCTION = "Tell me about how a computer motherboard works."
prompt = TEMPLATE.format(instruction=INSTRUCTION)
#%%
pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
print("pipe(prompt)", pipe(prompt))