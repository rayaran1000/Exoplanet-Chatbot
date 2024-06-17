from exoplanet_chatbot.logging import logger
from exoplanet_chatbot.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from exoplanet_chatbot.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from exoplanet_chatbot.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from exoplanet_chatbot.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main() # Calling the main method in pipeline
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main() # Calling the main method in pipeline
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Training stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_training = ModelTrainingPipeline()
   model_training.main() # Calling the main method in pipeline
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Evaluation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_evaluation = ModelEvaluationPipeline()
   model_evaluation.main() # Calling the main method in pipeline
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e