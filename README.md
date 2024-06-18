# ExoChat : Exoplanet Chatbot 

![image](https://github.com/rayaran1000/Exoplanet-Chatbot/assets/122597408/8991aa3a-b2f1-4c6f-9076-4632177fe6cb)


This project aims to provide accurate text information about exoplanets present in the NASA Public Exoplanet archive using the Google Gemini 2 billion Instruction tuned model from Hugging Face. Text generation is the process of generating a piece of text/ information using the prompt and the context. The Gemini 2billion model, based on transformer decoder architecture, has shown promising results in various natural language processing tasks, including text generation.

We are currently using the Gemini 2 billion parameter model, which is the smaller model in the Gemini model family. The model is instruction fine tuned so that it can be further finetuned based on the format of {prompt, context and response} 

By fine-tuning Gemini model on NASA Exoplanet Archive Dataset, which consists of multiple data points for the confirmed exoplanets discovered till data, we aim to create a model that can accurately generate correct information based on a prompt provided by the user for a discovered exoplanet. The project involves data preprocessing, model fine-tuning, and potentially deployment for real-world applications.


## Directory Structure

```plaintext
/project
│   README.md
│   requirements.txt
|   application.py
|   setup.py
|   template.py
|   Dockerfile
|   params.yaml
└───.github/workflows
|   └───main.yaml
└───research
|   └───Exoplanet_Chatbot_Dataset_Q_A_pair_creation.ipynb
|   └───Exoplanet_Chatbot_Model_Finetuning.ipynb
|   └───01_data_ingestion.ipynb
|   └───02_data_transformation.ipynb
|   └───03_model_trainer.ipynb 
|   └───03_model_evaluation.ipynb 
|   └───trials.ipynb 
└───src/exoplanet_chatbot
|   └───components
|       └───data_ingestion.py
|       └───data_transformation.py
|       └───model_trainer.py
|       └───model_evaluation.py
|   └───config
|       └───configuration.py
|   └───config
|       └───configuration.py
|   └───constants
|   └───entity
|   └───logging
|   └───utils
|   └───pipeline
|       └───prediction.py
|       └───stage_01_data_ingestion.py
|       └───stage_02_data_transformation.py
|       └───stage_03_model_trainer.py
|       └───stage_03_model_evaluation.py

```

# Installation
### STEPS:

Clone the repository

```bash
https://github.com/rayaran1000/Exoplanet-Chatbot
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n exochat python=3.8 -y
```

```bash
conda activate exochat
```


### STEP 02- Install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- Finally run the following command
```bash
python app.py
```

Now,
```bash
open up you local host and port 

URL -> localhost:8080
```


```bash
Author: Aranya Ray
Data Scientist
Email: aranya.ray1998@gmail.com

```
    
# Deployment

1. Created IAM user for deployment.
2. Open the command prompt
3. Ran the command "gcloud init" , to initialize the gcloud project
4. Gave the server as "Asia - South1"
5. Ran the command "gcloud app deploy app.yaml" to deploy the application on GCP

![image](https://github.com/rayaran1000/Exoplanet-Chatbot/assets/122597408/4194c2bd-126e-443b-90eb-50af79c90fd9)

## Dataset and Model Specifications

### Dataset 
NASA Exoplanet Archive - https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars

Documentation : https://exoplanetarchive.ipac.caltech.edu/docs/help.html

### Model
Model used -> Google Gemini 2b Instruction finetuned Model

Documentation : https://huggingface.co/google/gemma-2b-it
## Acknowledgements

I gratefully acknowledge the contributions of the developers and researchers behind the Hugging Face Transformers library, which provides access to state-of-the-art NLP models like Gemini. 

This research project has made use of the NASA Exoplanet Archive, which is operated by the California Institute of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet Exploration Program.
I would like to thank them for their approval of using Exoplanet Archive data for my project

Special thanks to the open-source community for their continuous support and valuable feedback.

