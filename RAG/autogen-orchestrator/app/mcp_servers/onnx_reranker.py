from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import (
    CrossEncoder,
)


@lru_cache(maxsize=1)
def _load_onnx_cross_encoder(
    model_name_or_path: str,
    model_file: str,
    max_length: int,
) -> CrossEncoder:
    return CrossEncoder(
        model_name_or_path,
        backend="onnx",
        max_length=max_length,
        model_kwargs={
            "file_name": model_file,
            "provider": (
                "CPUExecutionProvider"
            ),
            "export": False,
        },
        trust_remote_code=False,
    )


class OnnxCrossEncoderRerank:

    def __init__(
        self,
        *,
        model_name_or_path: str,
        model_file: str,
        batch_size: int = 2,
        max_length: int = 512,
    ) -> None:
        cleaned_model_path = (
            model_name_or_path.strip()
        )

        cleaned_model_file = (
            model_file.strip()
        )

        if not cleaned_model_path:
            raise ValueError(
                "model_name_or_path must "
                "not be empty"
            )

        if not cleaned_model_file:
            raise ValueError(
                "model_file must not be empty"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0"
            )

        if max_length < 64:
            raise ValueError(
                "max_length must be >= 64"
            )

        self.model_name_or_path = (
            cleaned_model_path
        )

        self.model_file = (
            cleaned_model_file
        )

        self.batch_size = batch_size
        self.max_length = max_length

        self._model = (
            _load_onnx_cross_encoder(
                self.model_name_or_path,
                self.model_file,
                self.max_length,
            )
        )

    def predict_pairs(
        self,
        pairs: Sequence[
            tuple[str, str]
        ],
    ) -> np.ndarray:
        if not pairs:
            return np.empty(
                shape=(0,),
                dtype=np.float32,
            )

        scores = self._model.predict(
            list(pairs),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        flattened = np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)

        if len(flattened) != len(pairs):
            raise RuntimeError(
                "Cross-encoder returned an "
                "unexpected number of scores: "
                f"scores={len(flattened)}, "
                f"pairs={len(pairs)}"
            )

        return flattened