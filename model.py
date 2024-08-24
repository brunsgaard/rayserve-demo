from ray import serve
from fastapi import FastAPI, Body
from transformers import pipeline
from typing import List, Dict, Any, cast

app = FastAPI()


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
@serve.ingress(app)
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    @app.post("/")
    def translate(self, text: str = Body(...)) -> str:
        # Run inference
        model_output = self.model(text)

        # Cast the model output to the expected type to satisfy Pyright
        model_output = cast(List[Dict[str, Any]], model_output)

        # Post-process output to return only the translation text
        translation: str = model_output[0]["translation_text"]

        return translation


translator_app = Translator.bind()
