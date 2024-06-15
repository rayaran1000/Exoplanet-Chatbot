from exoplanet_chatbot.logging import logger
from exoplanet_chatbot.utils.common import get_size
import os

from pathlib import Path
from exoplanet_chatbot.entity import DataIngestionConfig
import urllib.request as request # To download data from URL
import zipfile # To unzip operation


class DataIngestion:
    def __init__(self,config:DataIngestionConfig): # It will take the configuration from DataIngestionConfig defined earlier , which will in turn use Configuration Manager to take data from config.yaml
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file): # If file does not exist
            filename, headers = request.urlretrieve( # Download the data from URL
                url= self.config.source_URL, # URL present in config.yaml
                filename = self.config.local_data_file # Path of the file getting saved
            )
            logger.info(f"{filename} download with following info: \n{headers}")
        
        else:
            logger.info(f"File already exists of size : {get_size(Path(self.config.local_data_file))}") # Checking file size present already in the path
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)