import pandas as pd
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_metric
from exoplanet_chatbot.entity import ModelEvaluatorConfig

class ModelEvaluator:
    def __init__(self,config: ModelEvaluatorConfig):
        self.config = config

    def sampler(self,data_path):

        finetune_dataset = pd.read_csv(data_path)
        evaluation_sample = finetune_dataset.sample(n=100, random_state=42)

        evaluation_sample.rename(columns={'output' : 'expected response'},inplace=True)

        return evaluation_sample
    
    def generate_response(self,prompt, model, tokenizer, max_length=350):
        inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


    def evaluate(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        evaluation_sample = self.sampler(self.config.data_path) # Calling the sampler function to generate the evaluation sample

        evaluation_sample.to_csv(self.config.evaluation_data_path) # Saving the evaluation dataset

        evaluation_sample = evaluation_sample[['prompt','expected response']] # Selecting the evaluation columns

        # Loading the model and tokenizer

        model = AutoModelForCausalLM.from_pretrained(self.config.model_path,torch_dtype=torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Generate predictions
        evaluation_sample['generated_response'] = evaluation_sample['prompt'].apply(lambda x: self.generate_response(x, model, tokenizer))

        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

        # Converting the evaluation sample to dataset
        evaluation_dataset = Dataset.from_pandas(evaluation_sample)

        # Load evaluation metrics
        rouge_metric = load_metric("rouge")
        bleu_metric = load_metric("bleu")

        # Extract references and predictions
        references = evaluation_dataset["expected_response"]
        predictions_texts = evaluation_dataset["generated_response"]

        # Calculate ROUGE scores
        rouge_result = rouge_metric.compute(predictions=predictions_texts, references=references)
        print("ROUGE Score:", rouge_result)

        # Extract individual ROUGE scores
        rouge_1 = rouge_result['rouge1']
        rouge_2 = rouge_result['rouge2']
        rouge_L = rouge_result['rougeL']
        rouge_Lsum = rouge_result['rougeLsum']

        # Calculate BLEU score
        bleu_result = bleu_metric.compute(predictions=[pred.split() for pred in predictions_texts], references=[[ref.split()] for ref in references])
        print("BLEU Score:", bleu_result)

        # Prepare data for CSV
        results = {
            "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum", "BLEU"],
            "Precision": [rouge_1['precision'], rouge_2['precision'], rouge_L['precision'], rouge_Lsum['precision'], None],
            "Recall": [rouge_1['recall'], rouge_2['recall'], rouge_L['recall'], rouge_Lsum['recall'], None],
            "F1": [rouge_1['f1'], rouge_2['f1'], rouge_L['f1'], rouge_Lsum['f1'], None],
            "Score": [None, None, None, None, bleu_result['bleu']]
        }

        metric_results = pd.DataFrame(results)
        metric_results.to_csv(self.config.metric_file_name, index=False)