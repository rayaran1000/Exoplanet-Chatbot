from exoplanet_chatbot.constants import *
from exoplanet_chatbot.utils.common import read_yaml,create_directories
from exoplanet_chatbot.entity import DataIngestionConfig
from exoplanet_chatbot.entity import DataTransformationConfig
from exoplanet_chatbot.entity import ModelTrainerConfig
from exoplanet_chatbot.entity import ModelEvaluatorConfig

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
    # Here we are reading the yaml file and we can now use the file paths present inside pararms and config.yaml        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root]) # Here we are calling the artifacts_root key values using '.' , which was the purpose of @ensure_annotations

    def get_data_ingestion_config(self) -> DataIngestionConfig: # Here we are using the entity to specify the return type classes to make sure proper output is returned
        config= self.config.data_ingestion # Calling the data_ingestion dictionary created in config.yaml file

        create_directories([config.root_dir]) # Creating a directory using the root directory

        data_ingestion_config = DataIngestionConfig( # Extracting the values from the config.yaml to here inside data_ingestion_config
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig: # Here we are using the entity to specify the return type classes to make sure proper output is returned
        config= self.config.data_transformation # Calling the data_validation dictionary created in config.yaml file

        create_directories([config.root_dir]) # Creating a directory using the root directory

        data_transformation_config = DataTransformationConfig( # Extracting the values from the config.yaml to here inside data_ingestion_config
            root_dir=config.root_dir,
            data_path=config.data_path,
            data_path_transformed=config.data_path_transformed
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config= self.config.model_trainer # Calling the model_trainer dictionary created in config.yaml file
        params_training=self.params.TrainingArguments # Calling the TrainingArguments dictionary in params.yaml file
        params_lora = self.params.LoraConfig # Calling the Lora Config dictionary in params.yaml file
        params_bnb = self.params.BitsandBytesConfig # Calling the BitsandBytesConfig dictionary in params.yaml file

        create_directories([config.root_dir]) # Creating a directory using the root directory

        model_trainer_config = ModelTrainerConfig( # Extracting the values from the config.yaml to here inside data_ingestion_config
            #Config parameters
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_ckpt=config.tokenizer_ckpt,
            model_ckpt=config.model_ckpt,
            model_save_path=config.model_save_path,
            tokenizer_save_path=config.tokenizer_save_path,

            #Training parameters
            warmup_steps=params_training.warmup_steps,
            per_device_train_batch_size=params_training.per_device_train_batch_size,
            gradient_accumulation_steps=params_training.gradient_accumulation_steps,
            max_steps=params_training.max_steps,
            learning_rate=params_training.learning_rate,
            logging_steps=params_training.logging_steps,
            output_dir=params_training.output_dir,
            optim=params_training.optim,
            save_strategy=params_training.save_strategy,

            #Lora parameters
            r=params_lora.r,
            lora_alpha=params_lora.lora_alpha,
            lora_dropout=params_lora.lora_dropout,
            bias=params_lora.bias,
            task_type=params_lora.task_type,

            #Bits and bytes Configuration
            load_in_4bit=params_bnb.load_in_4bit,
            bnb_4bit_use_double_quant=params_bnb.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=params_bnb.bnb_4bit_quant_type
        )

        return model_trainer_config

    def get_model_evaluator_config(self) -> ModelEvaluatorConfig:

        config= self.config.model_evaluation # Calling the model_trainer dictionary created in config.yaml file

        create_directories([config.root_dir]) # Creating a directory using the root directory

        model_evaluator_config = ModelEvaluatorConfig( # Extracting the values from the config.yaml to here inside data_ingestion_config

        root_dir=config.root_dir,
        data_path=config.data_path,
        model_path=config.model_path,
        tokenizer_path=config.tokenizer_path,
        metric_file_name=config.metric_file_name,
        evaluation_data_path=config.evaluation_data_path
        )
        return model_evaluator_config