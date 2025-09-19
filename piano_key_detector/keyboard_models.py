"""Keyboard layout utilities and definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

BASE_WHITE_NOTES = ["C", "D", "E", "F", "G", "A", "B"]
# Indicates whether a black key exists after each white key in BASE_WHITE_NOTES
BASE_BLACK_PATTERN = [True, True, False, True, True, True, False]


def generate_black_key_pattern(num_white_keys: int) -> List[bool]:
    """Generate the repeating black key pattern for a given number of white keys."""
    if num_white_keys <= 1:
        return []

    pattern: List[bool] = []
    idx = 0
    while len(pattern) < num_white_keys - 1:
        pattern.append(BASE_BLACK_PATTERN[idx % len(BASE_BLACK_PATTERN)])
        idx += 1
    return pattern


@dataclass(frozen=True)
class KeyboardModel:
    """Representation of a keyboard layout."""

    name: str
    total_keys: int
    white_keys: int
    black_keys: int
    black_key_pattern: List[bool]

    def to_dict(self) -> Dict[str, int]:
        return {
            "name": self.name,
            "total_keys": self.total_keys,
            "white_keys": self.white_keys,
            "black_keys": self.black_keys,
        }


def _create_keyboard_model(total_keys: int, white_keys: int, *, name: Optional[str] = None) -> KeyboardModel:
    """Create a :class:`KeyboardModel` from counts."""
    black_keys = total_keys - white_keys
    if black_keys < 0:
        raise ValueError("White key count cannot exceed total keys")
    pattern = generate_black_key_pattern(white_keys)
    return KeyboardModel(
        name=name or f"{total_keys}_key",
        total_keys=total_keys,
        white_keys=white_keys,
        black_keys=black_keys,
        black_key_pattern=pattern,
    )


KEYBOARD_MODELS: Dict[str, KeyboardModel] = {
    "25": _create_keyboard_model(25, 15, name="25_key"),
    "37": _create_keyboard_model(37, 22, name="37_key"),
    "49": _create_keyboard_model(49, 29, name="49_key"),
    "61": _create_keyboard_model(61, 36, name="61_key"),
    "72": _create_keyboard_model(72, 42, name="72_key"),
}

# Convenience alias keyed by the canonical name as well (e.g. "25_key")
KEYBOARD_MODELS.update({model.name: model for model in KEYBOARD_MODELS.values()})


def get_model(name: str) -> Optional[KeyboardModel]:
    """Return the keyboard model for the provided name."""
    return KEYBOARD_MODELS.get(name)


def iter_models() -> Iterable[KeyboardModel]:
    """Yield all known keyboard models."""
    yielded = set()
    for key, model in KEYBOARD_MODELS.items():
        if model.name in yielded:
            continue
        yielded.add(model.name)
        yield model


def find_closest_model(white_key_count: int) -> Optional[KeyboardModel]:
    """Return the keyboard model whose white key count is closest to ``white_key_count``."""
    if white_key_count <= 0:
        return None

    closest: Optional[KeyboardModel] = None
    smallest_diff = float("inf")
    for model in iter_models():
        diff = abs(model.white_keys - white_key_count)
        if diff < smallest_diff:
            smallest_diff = diff
            closest = model
    return closest
