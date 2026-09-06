from typing import Optional, Union
from pydantic import BaseModel, Field


class OllamaOptions(BaseModel):
    """
    Inference parameters passed directly to the Ollama engine.
    Defaults to None so as not to override the model's default values if none are provided.
    """

    seed: Optional[int] = Field(
        default=None, description="Random seed used for reproducible outputs."
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Controls randomness in generation (higher = more random).",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=0,
        description="Limits next token selection to the K most likely tokens.",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cumulative probability threshold for nucleus sampling.",
    )
    min_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold for token selection.",
    )
    stop: Optional[Union[list[str], str]] = Field(
        default=None, description="Stop sequences that will halt generation."
    )
    num_ctx: Optional[int] = Field(
        default=None, ge=1, description="Context length size (number of tokens)."
    )
    num_predict: Optional[int] = Field(
        default=None, description="Maximum number of tokens to generate."
    )


class ModelPromptTemplate(BaseModel):
    """
    A prompt template specific to the architecture of the given model.
    """

    system_prompt: Optional[str] = None
    prefix: str = Field(
        ..., description="Text preceding the dynamic list of frames and timestamps"
    )
    suffix: str = Field(
        ..., description="Task/instruction following the list of frames"
    )


class VLMModelProfile(BaseModel):
    """
    Complete profile of VLM model configuraton.
    """

    model_name: str
    prompt_template: ModelPromptTemplate
    options: OllamaOptions = Field(default_factory=OllamaOptions)
