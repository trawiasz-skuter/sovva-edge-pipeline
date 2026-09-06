from typing import Dict
from ai.schemas import VLMModelProfile, ModelPromptTemplate, OllamaOptions

MODEL_PROFILES: Dict[str, VLMModelProfile] = {
    "gemma4:e2b": VLMModelProfile(
        model_name="gemma4:e2b",
        prompt_template=ModelPromptTemplate(
            system_prompt="You are an edge AI vision assistant deployed on an autonomous UAV.",
            prefix=(
                "You are given a sequence of images captured by an aerial drone.\n"
                "The images are provided in the exact order listed below.\n"
                "Image N corresponds to the N-th image in the input.\n"
            ),
            suffix=(
                "Analyze the sequence in chronological order.\n"
                "For whole chunk:\n"
                "- describe the sequence of images scene,\n"
                "- infer the ongoing activity."
            ),
        ),
        options=OllamaOptions(seed=42, temperature=0.0),
    )
}


def get_model_profile(model_name: str) -> VLMModelProfile:
    """Retrieves the model profile or throws a clear error if the model is not registered."""
    if model_name not in MODEL_PROFILES:
        available = list(MODEL_PROFILES.keys())
        raise KeyError(
            f"Model '{model_name}' does not exist in the registry. Available: {available}"
        )
    return MODEL_PROFILES[model_name]
