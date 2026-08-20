from pathlib import Path

from sentence_transformers import (
    CrossEncoder,
    export_dynamic_quantized_onnx_model,
)


MODEL_ID = (
    "cross-encoder/"
    "ms-marco-MiniLM-L6-v2"
)

OUTPUT_DIR = Path(
    "models/ms-marco-MiniLM-L6-v2"
)

QUANTIZATION_CONFIG = "avx2"

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = CrossEncoder(
        MODEL_ID,
        backend="onnx",
        model_kwargs={
            "export": True,
            "provider": "CPUExecutionProvider",
        },
        trust_remote_code=False,
    )

    model.save_pretrained(
        str(OUTPUT_DIR)
    )

    export_dynamic_quantized_onnx_model(
        model=model,
        quantization_config=(
            QUANTIZATION_CONFIG
        ),
        model_name_or_path=str(
            OUTPUT_DIR
        ),
        file_suffix=(
            "qint8_avx2"
        ),
    )

if __name__ == "__main__":
    main()