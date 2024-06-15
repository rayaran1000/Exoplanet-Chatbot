from exoplanet_chatbot.constants import *
from exoplanet_chatbot.utils.common import read_yaml,create_directories
from exoplanet_chatbot.entity import DataIngestionConfig
from exoplanet_chatbot.entity import DataTransformationConfig

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