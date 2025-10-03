"""
Typed Argument Parser
"""

__name__ = "nightjarpy"
__version__ = "0.1.0"
__description__ = "NightjarPy"
__url__ = "https://github.com/psg-mit/nightjarpy"
__author__ = "Ellie Cheng"
__author_email__ = "ellieyhc@csail.mit.edu"


from nightjarpy.configs import DEFAULT_CONFIG, Config, LLMConfig
from nightjarpy.decorators import fn
from nightjarpy.runtime import nj_llm_factory
from nightjarpy.types import EffectException, LLMUsage
from nightjarpy.utils import NJ_TELEMETRY

__all__ = [
    "__version__",
    "__name__",
    "__description__",
    "__url__",
    "__author__",
    "__author_email__",
    "fn",
    "EffectException",
    "NJ_TELEMETRY",
    "nj_llm_factory",
]
