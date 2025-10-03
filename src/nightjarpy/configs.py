import os
from enum import Enum
from typing import List, Literal, Optional

import dotenv
from pydantic import BaseModel

from nightjarpy.effects import (
    BASE_EFFECTS_NOREG,
    PYTHON_BASE_EFFECTS_NOREG,
    PYTHON_BASE_ISOLATED_EFFECTS_NOREG,
    PYTHON_EFFECTS_V1,
    EffectSet,
)
from nightjarpy.prompts.base import PromptTemplate
from nightjarpy.prompts.prompts import (
    COMPILER_AOT_V0_PROMPT,
    INTERPRETER_BASE_NOREG_V0_PROMPT,
    INTERPRETER_PYTHON_BASE_NOREG_V0_PROMPT,
    INTERPRETER_PYTHON_EAGER_V0_PROMPT,
    INTERPRETER_PYTHON_V0_PROMPT,
)

dotenv.load_dotenv()

ENABLE_CACHE_DEFAULT = os.getenv("NJ_ENABLE_CACHE", "False") == "True"


class LLMConfig(BaseModel):
    model: str = "openai/gpt-4.1"
    temperature: float = 1.0
    tool_choice: Literal["auto", "required"] = "required"
    parallel_tool_calls: bool = False
    max_tokens: int = 8192
    verbosity: Literal["low", "medium", "high"] = "low"
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal"
    cache: bool = ENABLE_CACHE_DEFAULT
    json_structured_output: bool = True
    parallelize_effects: bool = False
    container: bool = False

    model_config = {"frozen": True}


class ExecutionSubstrate(Enum):
    PYTHON = "python"
    BASE_NOREG = "base_noreg"
    PYTHON_BASE_ISOLATED_NOREG = "python_base_isolated_noreg"
    PYTHON_BASE_NOREG = "python_base_noreg"

    @classmethod
    def get_effect_set(cls, substrate: "ExecutionSubstrate") -> EffectSet:
        if substrate == ExecutionSubstrate.PYTHON:
            effect_set = PYTHON_EFFECTS_V1
        elif substrate == ExecutionSubstrate.BASE_NOREG:
            effect_set = BASE_EFFECTS_NOREG
        elif substrate == ExecutionSubstrate.PYTHON_BASE_ISOLATED_NOREG:
            effect_set = PYTHON_BASE_ISOLATED_EFFECTS_NOREG
        elif substrate == ExecutionSubstrate.PYTHON_BASE_NOREG:
            effect_set = PYTHON_BASE_EFFECTS_NOREG
        else:
            raise ValueError(f"Unknown execution substate {substrate}")
        return effect_set


class ExecutionStrategy(Enum):
    AOTCOMPILATION = "aotcompilation"
    INTERPRETER = "interpreter"


class CompilerConfig(BaseModel):
    execution_substrate: ExecutionSubstrate
    prompt_template: Optional[PromptTemplate] = None
    compute_prompt_template: Optional[PromptTemplate] = None
    include_source_code: bool = False
    use_nonce: bool = False
    var_generator_init: int = 0

    model_config = {"frozen": True}


class InterpreterConfig(BaseModel):
    execution_substrate: ExecutionSubstrate
    prompt_template: PromptTemplate
    compute_prompt_template: Optional[PromptTemplate] = None
    include_source_code: bool = False
    var_generator_init: int = 0
    jit: bool = False
    max_effects: int = 100
    max_effects_per_execute: Optional[int] = None
    max_iters: Optional[int] = None
    use_nonce: bool = False
    eager_load: bool = False
    show_effect_count: bool = False
    max_serialize_len: int = 1024
    steps_ahead: int = 1
    discard_steps_ahead: bool = True
    seed: Optional[int] = None

    model_config = {"frozen": True}


class Config(BaseModel):
    execution_strategy: ExecutionStrategy
    compiler_config: Optional[CompilerConfig]
    interpreter_config: Optional[InterpreterConfig]
    llm_config: LLMConfig
    use_functions: bool = False

    model_config = {"frozen": True}

    def disable_cache(self) -> "Config":
        return self.model_copy(update={"llm_config": self.llm_config.model_copy(update={"cache": False})})

    def enable_cache(self) -> "Config":
        return self.model_copy(update={"llm_config": self.llm_config.model_copy(update={"cache": True})})

    def with_interpreter_updates(
        self,
        max_effects: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
        compute_prompt_template: Optional[PromptTemplate] = None,
        var_generator_init: Optional[int] = None,
        steps_ahead: Optional[int] = None,
        use_nonce: Optional[bool] = None,
    ) -> "Config":
        """Create a copy of this config with updated interpreter settings."""
        if self.interpreter_config is None:
            return self.model_copy()

        updates = {}
        if max_effects is not None:
            updates["max_effects"] = max_effects
        if prompt_template is not None:
            updates["prompt_template"] = prompt_template
        if compute_prompt_template is not None:
            updates["compute_prompt_template"] = compute_prompt_template
        if var_generator_init is not None:
            updates["var_generator_init"] = var_generator_init
        if steps_ahead is not None:
            updates["steps_ahead"] = steps_ahead
        if use_nonce is not None:
            updates["use_nonce"] = use_nonce

        if not updates:
            return self.model_copy()

        return self.model_copy(update={"interpreter_config": self.interpreter_config.model_copy(update=updates)})

    def with_compiler_updates(
        self,
        prompt_template: Optional[PromptTemplate] = None,
        max_runtime_calls: Optional[int] = None,
        var_generator_init: Optional[int] = None,
    ) -> "Config":
        """Create a copy of this config with updated compiler settings."""
        if self.compiler_config is None:
            return self.model_copy()

        updates = {}
        if prompt_template is not None:
            updates["prompt_template"] = prompt_template

        if max_runtime_calls is not None:
            updates["max_runtime_calls"] = max_runtime_calls

        if var_generator_init is not None:
            updates["var_generator_init"] = var_generator_init

        if not updates:
            return self.model_copy()

        return self.model_copy(update={"compiler_config": self.compiler_config.model_copy(update=updates)})

    def with_llm_updates(self, model: Optional[str] = None, **kwargs) -> "Config":
        """Create a copy of this config with updated LLM settings."""
        updates = {}
        if model is not None:
            updates["model"] = model
        updates.update(kwargs)

        if not updates:
            return self.model_copy()

        return self.model_copy(update={"llm_config": self.llm_config.model_copy(update=updates)})


INTERPRETER_BASE_NOREG_JSON_CONFIG = Config(
    execution_strategy=ExecutionStrategy.INTERPRETER,
    compiler_config=None,
    interpreter_config=InterpreterConfig(
        execution_substrate=ExecutionSubstrate.BASE_NOREG,
        prompt_template=INTERPRETER_BASE_NOREG_V0_PROMPT,
        eager_load=False,
        show_effect_count=False,
    ),
    llm_config=LLMConfig(
        tool_choice="required",
        cache=False,
        json_structured_output=True,
    ),
)

INTERPRETER_PYTHON_BASE_ISOLATED_NOREG_JSON_CONFIG = Config(
    execution_strategy=ExecutionStrategy.INTERPRETER,
    compiler_config=None,
    interpreter_config=InterpreterConfig(
        execution_substrate=ExecutionSubstrate.PYTHON_BASE_ISOLATED_NOREG,
        prompt_template=INTERPRETER_PYTHON_BASE_NOREG_V0_PROMPT,
        eager_load=False,
        show_effect_count=False,
    ),
    llm_config=LLMConfig(
        tool_choice="required",
        cache=False,
        json_structured_output=True,
    ),
)

INTERPRETER_PYTHON_BASE_NOREG_JSON_CONFIG = Config(
    execution_strategy=ExecutionStrategy.INTERPRETER,
    compiler_config=None,
    interpreter_config=InterpreterConfig(
        execution_substrate=ExecutionSubstrate.PYTHON_BASE_NOREG,
        prompt_template=INTERPRETER_PYTHON_BASE_NOREG_V0_PROMPT,
        eager_load=False,
        show_effect_count=False,
    ),
    llm_config=LLMConfig(
        tool_choice="required",
        cache=False,
        json_structured_output=True,
    ),
)


INTERPRETER_PYTHON_JSON_CONFIG = Config(
    execution_strategy=ExecutionStrategy.INTERPRETER,
    compiler_config=None,
    interpreter_config=InterpreterConfig(
        execution_substrate=ExecutionSubstrate.PYTHON,
        prompt_template=INTERPRETER_PYTHON_V0_PROMPT,
        eager_load=False,
        show_effect_count=False,
    ),
    use_functions=True,
    llm_config=LLMConfig(
        tool_choice="required",
        cache=False,
        json_structured_output=True,
    ),
)

INTERPRETER_PYTHON_CACHE_JSON_CONFIG = Config(
    execution_strategy=ExecutionStrategy.INTERPRETER,
    compiler_config=None,
    interpreter_config=InterpreterConfig(
        execution_substrate=ExecutionSubstrate.PYTHON,
        prompt_template=INTERPRETER_PYTHON_V0_PROMPT,
        eager_load=False,
        show_effect_count=False,
    ),
    use_functions=True,
    llm_config=LLMConfig(
        tool_choice="required",
        cache=True,
        json_structured_output=True,
    ),
)

INTERPRETER_PYTHON_EAGER_CACHE_JSON_CONFIG = Config(
    execution_strategy=ExecutionStrategy.INTERPRETER,
    compiler_config=None,
    interpreter_config=InterpreterConfig(
        execution_substrate=ExecutionSubstrate.PYTHON,
        prompt_template=INTERPRETER_PYTHON_EAGER_V0_PROMPT,
        eager_load=True,
        show_effect_count=False,
    ),
    use_functions=True,
    llm_config=LLMConfig(
        tool_choice="required",
        cache=True,
        json_structured_output=True,
    ),
)


COMPILER_PYTHON_JSON_CONFIG = Config(
    execution_strategy=ExecutionStrategy.AOTCOMPILATION,
    compiler_config=CompilerConfig(
        execution_substrate=ExecutionSubstrate.PYTHON,
        prompt_template=COMPILER_AOT_V0_PROMPT,
        include_source_code=False,
    ),
    interpreter_config=None,
    llm_config=LLMConfig(
        cache=False,
        json_structured_output=True,
    ),
)

DEFAULT_CONFIG: Config = INTERPRETER_PYTHON_EAGER_CACHE_JSON_CONFIG
