from exoplanet_chatbot.config.configuration import ConfigurationManager
from exoplanet_chatbot.components.data_ingestion import DataIngestion


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config() # Storing the configuration
            data_ingestion = DataIngestion(config=data_ingestion_config) # Using the configuration saved earlier to call data_ingestion
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
        except Exception as e:
            raise e