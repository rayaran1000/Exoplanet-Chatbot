from exoplanet_chatbot.config.configuration import ConfigurationManager
from exoplanet_chatbot.components.data_transformation import DataTransformation

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config() # Storing the configuration
            data_transformation = DataTransformation(config=data_transformation_config) # Using the configuration saved earlier to call model_transformation
            data_transformation.transform()
        except Exception as e:
            raise e