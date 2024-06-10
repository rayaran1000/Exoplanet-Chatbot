import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')

project_name = "exoplanet_chatbot"

list_of_files = [ #.gitkeep is given as a placeholder for empty folder

    ".github/workflows/.gitkeep", # Used for CI?CD deployment(whenever we will be committing the code, it will get deployed in cloud as well using this folder components)
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "application.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb" # Will contain the jupyter notebooks for experimentation
]

for file_path in list_of_files:
    file_path = Path(file_path) #Convert file path to windows path
    filedir , filename = os.path.split(file_path) # Split the path into folder paths and file paths for each file_path

    if filedir != '': # Only if the file directory is not empty, we create a folder using the path
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file:{filename}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)) == 0: # If the file_path doesn't exist or size is empty, then we will create that file
        with open(file_path,'w') as f:
            pass
            logging.info(f"Creating empty file : {file_path}")

    else: # The file is already present
        logging.info(f"{filename} already exists")