from exoplanet_chatbot.config.configuration import ConfigurationManager
from exoplanet_chatbot.components.model_evaluation import ModelEvaluator

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluator_config = config.get_model_evaluator_config() # Storing the configuration
            model_training = ModelEvaluator(config=model_evaluator_config) # Using the configuration saved earlier to call model_training
            model_training.evaluate()
        except Exception as e:
            raise e