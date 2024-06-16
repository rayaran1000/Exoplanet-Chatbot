import pandas as pd
import torch
from trl import SFTTrainer
from datasets import Dataset
import bitsandbytes as bnb
from transformers import TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

from exoplanet_chatbot.entity import ModelTrainerConfig

class DebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step: {state.global_step}, Loss: {state.log_history[-1]['loss'] if state.log_history else 'N/A'}")

class ModelTrainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config = config
        self.bnb_config = BitsAndBytesConfig(load_in_4bit=self.config.load_in_4bit, bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant, 
                                             bnb_4bit_quant_type=self.config.bnb_4bit_quant_type, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_ckpt, quantization_config=self.bnb_config, device_map={"":0})
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_ckpt)
    
    # Dataset Train Test Split
    def train_test_split(self,data):

        data = data.train_test_split(test_size=0.2)
        train_data = data["train"]
        test_data = data["test"]

        return (train_data,test_data)

    # Dataset Creating and Tokenization for Finetuning
    def transform_and_tokenize(self):

        # Loading the finetuning dataset
        finetune_dataframe = pd.read_csv(self.config.data_path)

        # Converting the dataframe to dataset
        finetune_dataset = Dataset.from_pandas(finetune_dataframe)

        # Shuffling and Tokenization
        finetune_dataset = finetune_dataset.shuffle(seed=1234)  # Shuffle dataset here
        finetune_dataset = finetune_dataset.map(lambda samples: self.tokenizer(samples["prompt"]), batched=True)

        # Train Test Split of Dataset
        train_dataset,test_dataset = self.train_test_split(finetune_dataset)

        return (train_dataset,test_dataset)
    
    # Function for preparing the linear layers for training in LoRa
    def find_all_linear_names(self,model):
        cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if 'lm_head' in lora_module_names: # needed for 16-bit
                lora_module_names.remove('lm_head')
        return list(lora_module_names)

    # LoRa Configuration for training
    def Lora_config(self):

        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        modules = self.find_all_linear_names(self.model)

        lora_config = LoraConfig(
            r=self.config.r, #  Always keep it 2 times the lora_alpha
            lora_alpha=self.config.lora_alpha,
            target_modules=modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=self.config.task_type
        )

        self.model = get_peft_model(self.model, lora_config)

        return (self.model,lora_config)

    def Model_Config(self):

        training_args = TrainingArguments(
            warmup_steps = self.config.warmup_steps,
            per_device_train_batch_size = self.config.per_device_train_batch_size,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps,
            max_steps = self.config.max_steps,
            learning_rate = float(self.config.learning_rate),
            logging_steps = self.config.logging_steps,
            output_dir = self.config.output_dir,
            optim = self.config.optim,
            save_strategy = self.config.save_strategy
        )

        return training_args
    
    def tokenizer_and_model_save(self,model):

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_ckpt,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        merged_model= PeftModel.from_pretrained(base_model, model)
        merged_model= merged_model.merge_and_unload()

        # Save the merged model
        merged_model.save_pretrained(self.config.model_save_path,safe_serialization=True)
        self.tokenizer.save_pretrained(self.config.tokenizer_save_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def train(self):

        self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.cuda.empty_cache()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loading the tokenized datasets
        train_dataset_tokenized , validation_dataset_tokenized = self.transform_and_tokenize()

        # Loading the LoRa configured model
        model,lora_config = self.Lora_config()

        # Setting the Trainer       
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset_tokenized,
            eval_dataset=validation_dataset_tokenized,
            dataset_text_field="prompt",
            peft_config=lora_config,
            args=self.Model_Config(),
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),callbacks=[DebugCallback()]
        )

        # Training the model
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()

        #Saving the tokenizer and Model
        new_model = "gemma-Exochat-Instruct-Finetune-Step20"
        trainer.model.save_pretrained(new_model)
        self.tokenizer_and_model_save(new_model)
