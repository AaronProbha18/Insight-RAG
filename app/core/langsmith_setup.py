import os
import logging

def init_langsmith():
    """
    Initialize LangSmith-related environment checks and optional SDK startup.

    Expected env vars:
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_PROJECT="AdaptiveRAG"
      - LANGCHAIN_API_KEY=<your_langsmith_api_key>

    This function warns when configuration is incomplete. If the installed
    LangSmith/Tracing SDK exposes an explicit startup function, add that
    call here.
    """
    required = ["LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGCHAIN_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        logging.warning("LangSmith: missing environment vars: %s. Tracing may be disabled.", ",".join(missing))

__all__ = ["init_langsmith"]
