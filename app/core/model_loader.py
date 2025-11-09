import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langsmith import traceable
from typing import Optional, Any

load_dotenv()


@traceable(name="LoadModel")
def load_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.3,
    **kwargs
) -> Any:
    """
    Load an LLM model based on available API keys or explicit provider.
    
    Args:
        provider: Explicit provider name ('google', 'openai', 'anthropic').
                 If None, auto-detects based on available API keys.
        model_name: Specific model name. If None, uses provider default.
        temperature: Temperature for model generation.
        **kwargs: Additional provider-specific parameters.
    
    Returns:
        Initialized LLM instance
        
    Raises:
        ValueError: If no valid API key is found or provider is unsupported.
    """
    # Provider detection and configuration
    provider_configs = {
        "google": {
            "env_key": "GOOGLE_API_KEY",
            "default_model": "gemini-2.0-flash-exp",
            "class": ChatGoogleGenerativeAI,
            "init_params": lambda api_key, model: {
                "model": model,
                "google_api_key": api_key,
                "temperature": temperature,
                **kwargs
            }
        },
        "openai": {
            "env_key": "OPENAI_API_KEY",
            "default_model": "gpt-4o-mini",
            "class": ChatOpenAI,
            "init_params": lambda api_key, model: {
                "model": model,
                "openai_api_key": api_key,
                "temperature": temperature,
                **kwargs
            }
        },
        "anthropic": {
            "env_key": "ANTHROPIC_API_KEY",
            "default_model": "claude-3-5-sonnet-20241022",
            "class": ChatAnthropic,
            "init_params": lambda api_key, model: {
                "model": model,
                "anthropic_api_key": api_key,
                "temperature": temperature,
                **kwargs
            }
        }
    }
    
    # Auto-detect provider if not specified
    if provider is None:
        for prov_name, config in provider_configs.items():
            if os.getenv(config["env_key"]):
                provider = prov_name
                break
        
        if provider is None:
            available_keys = [cfg["env_key"] for cfg in provider_configs.values()]
            raise ValueError(
                f"No API key found. Please set one of: {', '.join(available_keys)}"
            )
    
    # Validate provider
    provider = provider.lower()
    if provider not in provider_configs:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: {', '.join(provider_configs.keys())}"
        )
    
    config = provider_configs[provider]
    
    # Get API key
    api_key = os.getenv(config["env_key"])
    if not api_key:
        raise ValueError(f"Missing {config['env_key']} in environment variables.")
    
    # Use default model if not specified
    if model_name is None:
        model_name = config["default_model"]
    
    # Initialize model
    llm_class = config["class"]
    init_params = config["init_params"](api_key, model_name)
    
    llm = llm_class(**init_params)
    return llm


# Convenience functions for specific providers
@traceable(name="LoadGeminiModel")
def load_gemini_model(model_name: str = "gemini-2.0-flash-exp", temperature: float = 0.3, **kwargs):
    """Load a Google Gemini model."""
    return load_model(provider="google", model_name=model_name, temperature=temperature, **kwargs)


@traceable(name="LoadOpenAIModel")
def load_openai_model(model_name: str = "gpt-4o-mini", temperature: float = 0.3, **kwargs):
    """Load an OpenAI model."""
    return load_model(provider="openai", model_name=model_name, temperature=temperature, **kwargs)


@traceable(name="LoadAnthropicModel")
def load_anthropic_model(model_name: str = "claude-3-5-sonnet-20241022", temperature: float = 0.3, **kwargs):
    """Load an Anthropic model."""
    return load_model(provider="anthropic", model_name=model_name, temperature=temperature, **kwargs)


@traceable(name="ModelInference")
def call_model(llm, prompt: str, **kwargs):
    """
    Thin wrapper around LLM calls. Use this to ensure every model
    inference is traced consistently.
    """
    try:
        if hasattr(llm, "invoke"):
            resp = llm.invoke(prompt, **kwargs)
        else:
            resp = llm(prompt, **kwargs)
    except Exception:
        raise
    return resp