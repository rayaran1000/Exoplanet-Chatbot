#Prediction Pipeline
from exoplanet_chatbot.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForCausalLM

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_prediction_pipeline_config() # Used to extract model path and tokenizer configuration

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(self.config.finetuned_model_path)

        gen_kwargs = {
            "max_length": 128,       # Maximum length of the generated response
            "num_beams": 5,          # Number of beams for beam search
            "early_stopping": True,  # Stop early if the model is confident
            "no_repeat_ngram_size": 2  # Avoid repeating n-grams of the specified size
        }

        inputs = tokenizer.encode(text, return_tensors='pt')
        output = model.generate(inputs, **gen_kwargs)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        print("User Input:")
        print(text)
        print("\nModel Response:")
        print(response)

        return response