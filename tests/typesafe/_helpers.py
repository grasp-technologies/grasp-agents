import json
from typing import Any

from typesafe_sdk import SystemOneResponse


def jev_response(
    answers: dict[str, dict[str, Any]],
    *,
    model: str = "jev-1.13.0",
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> SystemOneResponse:
    return SystemOneResponse.model_validate_json(
        json.dumps(
            {
                "model": model,
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }
        )
    )


def noul(probability: float) -> dict[str, Any]:
    return {"type": "noul", "noul": probability}


def choice(picked: str, probabilities: dict[str, float]) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": picked,
        "confidence": 0.8,
        "probabilities": probabilities,
    }


def score(
    expected: float, levels: list[str], probabilities: list[float]
) -> dict[str, Any]:
    return {
        "type": "score",
        "score": expected,
        "confidence": 0.7,
        "legend": {str(level): text for level, text in enumerate(levels)},
        "probabilities": {
            str(level): probability for level, probability in enumerate(probabilities)
        },
    }
