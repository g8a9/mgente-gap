from dataclasses import dataclass, field


@dataclass
class AttributionUnit:
    """A unit of attribution."""

    tokens: list[str]
    span: list[(int, int)]
    metadata: dict[str, any] = None


@dataclass
class RowAttribution:
    """An attribution unit for a row."""

    rid: int = None
    translation_label: AttributionUnit = None
    translation: AttributionUnit = None


@dataclass
class AttributionOutput:
    """An output of attribution."""

    file_path: str
    model_name: str
    source: AttributionUnit = None
    system_prompt: AttributionUnit = None
    preamble: AttributionUnit = None
    guidelines: AttributionUnit = None
    demonstrations: list[(AttributionUnit, AttributionUnit)] = None
    rows: list[RowAttribution] = field(default_factory=list)
