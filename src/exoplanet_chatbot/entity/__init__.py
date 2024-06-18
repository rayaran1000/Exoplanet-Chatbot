from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig: # defined for the config components present in artifacts for data ingestion
    # Below are the return types for the components (root_dir is Path format , URL is string etc)
    root_dir : Path 
    source_URL : str
    local_data_file : Path
    unzip_dir : Path

@dataclass(frozen=True)
class DataTransformationConfig: # defined for the config components present in artifacts for data transformation
    root_dir : Path 
    data_path : Path
    data_path_transformed : Path

@dataclass(frozen=True)
class ModelTrainerConfig: # defined for the config components present in artifacts for model training
    root_dir : Path 
    data_path : Path
    tokenizer_ckpt : Path
    model_ckpt : Path
    model_save_path: Path
    tokenizer_save_path: Path
    warmup_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int 
    learning_rate: int
    logging_steps: int
    output_dir: str
    optim: str
    save_strategy: str
    r: int
    lora_alpha: int
    lora_dropout: int
    bias: int
    task_type: int
    load_in_4bit: bool
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_type: str

@dataclass(frozen=True)
class ModelEvaluatorConfig: # defined for the config components present in artifacts for model training
   root_dir: Path
   data_path: Path
   model_path: Path
   tokenizer_path: Path
   metric_file_name: Path
   evaluation_data_path: Path

@dataclass(frozen=True)
class PredictionConfig: # defined for the config components present in artifacts for model training
   finetuned_model_path: Path
   tokenizer_path: Path