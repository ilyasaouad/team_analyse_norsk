# config.py
import os
import dotenv
from dotenv import load_dotenv

load_dotenv()  
 
class Config:
    # Default settings
    output_dir = r"C:\Users\iao\Desktop\Patstat_TIP\Patent_family\applicants_inventors_analyse\dataTable_NO_2020_2020"  # Placeholder default path
    country_code = "NO"
    start_year = 2020
    end_year = 2020
    batch_size = 200  # Example static setting
    ollama_base_url = "http://localhost:11434"  # Ollama base URL
    model_name = "llama3.2:latest"  # Model name for Ollama
    openai_model_name = "gpt-4o"  # Model name for OpenAI
    # Load from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")  # OpenAI API key

    @classmethod
    def update(cls, **kwargs):
        """Update Config settings dynamically."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Unknown config attribute: {key}")
