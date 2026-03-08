"""Configuration loading from TOML files."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EndpointConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "no-key"


class ModelOptionConfig(BaseModel):
    tool_mode: str = "auto"  # "auto" | "native" | "prompt"
    max_tokens: int = 32768
    thinking: str = "off"
    stop_sequences: list[str] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    default: str = "gpt-oss-120b"
    compaction: str = ""  # falls back to default if empty
    embedding: str = "bge-m3"
    fallback: list[str] = Field(default_factory=list)  # failover model list
    options: dict[str, ModelOptionConfig] = Field(default_factory=dict)

    @property
    def compaction_model(self) -> str:
        return self.compaction or self.default


class EndpointsConfig(BaseModel):
    llm: EndpointConfig = Field(default_factory=EndpointConfig)
    embedding: EndpointConfig = Field(default_factory=EndpointConfig)


class ContextConfig(BaseModel):
    max_tokens: int = 32768
    compaction_threshold: float = 0.7
    reserve_tokens_floor: int = 20000
    tool_result_max_ratio: float = 0.3


class SessionConfig(BaseModel):
    dir: str = "~/.openclaw-py/sessions"

    @property
    def resolved_dir(self) -> Path:
        return Path(self.dir).expanduser()


class HybridMMRConfig(BaseModel):
    enabled: bool = True
    lambda_param: float = 0.7


class HybridTemporalDecayConfig(BaseModel):
    enabled: bool = True
    half_life_days: int = 30


class HybridConfig(BaseModel):
    enabled: bool = True
    vector_weight: float = 0.7
    text_weight: float = 0.3
    mmr: HybridMMRConfig = Field(default_factory=HybridMMRConfig)
    temporal_decay: HybridTemporalDecayConfig = Field(
        default_factory=HybridTemporalDecayConfig
    )


class MemoryCacheConfig(BaseModel):
    enabled: bool = True
    max_entries: int = 50000


class MemoryConfig(BaseModel):
    dir: str = "~/.openclaw-py/memory"
    chunk_size: int = 700
    chunk_overlap: int = 80
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    cache: MemoryCacheConfig = Field(default_factory=MemoryCacheConfig)

    @property
    def resolved_dir(self) -> Path:
        return Path(self.dir).expanduser()


class PruningConfig(BaseModel):
    mode: str = "cache-ttl"  # "cache-ttl" | "off"
    ttl_seconds: int = 300
    keep_last_assistants: int = 3
    soft_trim_chars: int = 4000
    hard_clear_ratio: float = 0.5
    min_prunable_tool_chars: int = 50000


class MemoryFlushConfig(BaseModel):
    enabled: bool = True
    soft_threshold_tokens: int = 4000


class CompactionConfig(BaseModel):
    mode: str = "safeguard"  # "default" | "safeguard"
    identifier_policy: str = "strict"  # "strict" | "off" | "custom"
    max_retries: int = 3
    memory_flush: MemoryFlushConfig = Field(default_factory=MemoryFlushConfig)


class BootstrapConfig(BaseModel):
    max_chars_per_file: int = 20000
    max_chars_total: int = 150000
    head_ratio: float = 0.7
    tail_ratio: float = 0.2


class SkillsConfig(BaseModel):
    dirs: list[str] = Field(default_factory=lambda: ["~/.openclaw-py/skills"])
    max_skills_in_prompt: int = 150
    max_prompt_chars: int = 30000

    @property
    def resolved_dirs(self) -> list[Path]:
        return [Path(d).expanduser() for d in self.dirs]


class WorkspaceConfig(BaseModel):
    dir: str = "~/.openclaw-py/workspace"

    @property
    def resolved_dir(self) -> Path:
        return Path(self.dir).expanduser()


class HooksConfig(BaseModel):
    pre_tool_call: str = ""
    post_tool_call: str = ""
    pre_message: str = ""
    post_message: str = ""
    on_error: str = ""
    timeout: int = 10


class AppConfig(BaseModel):
    """Root configuration."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from a TOML file.

    Resolution order:
    1. Explicit path argument
    2. OPENCLAW_PY_CONFIG env var
    3. ./config.toml
    4. ~/.openclaw-py/config.toml
    5. Default values
    """
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    if env_path := os.environ.get("OPENCLAW_PY_CONFIG"):
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path("config.toml"))
    candidates.append(Path("~/.openclaw-py/config.toml").expanduser())

    raw: dict[str, Any] = {}
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "rb") as f:
                raw = tomllib.load(f)
            break

    return AppConfig(**raw)
