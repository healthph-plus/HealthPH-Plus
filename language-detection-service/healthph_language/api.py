"""Run from the service folder: python -m healthph_language.api"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from .detector import (
    DEFAULT_SECONDARY_MIN_CONFIDENCE, MAX_BATCH_SIZE, MAX_TEXT_CHARACTERS,
    LanguageDetector, clean_valid_text,
)

logger = logging.getLogger(__name__)


def validate_raw_text(text: str) -> str:
    clean_valid_text(text)
    return text  # Keyword overrides must see the original text.


Text = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_TEXT_CHARACTERS),
                 AfterValidator(validate_raw_text)]


class PredictionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: Annotated[int, Field(strict=True, ge=1, le=2)] = 1
    secondary_min_confidence: Annotated[
        float, Field(strict=True, ge=0, le=1, allow_inf_nan=False),
    ] = DEFAULT_SECONDARY_MIN_CONFIDENCE


class PredictRequest(PredictionOptions):
    text: Text


class BatchPredictRequest(PredictionOptions):
    texts: Annotated[list[Text], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class FastTextPrediction(BaseModel):
    language: str
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Prediction(BaseModel):
    language: str
    cleaned_text: str
    is_supported: bool
    prediction_source: Literal["fasttext", "keyword_override"]
    fasttext_language: str
    fasttext_confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    fasttext_predictions: Annotated[list[FastTextPrediction], Field(min_length=1, max_length=2)] | None = None


class BatchPrediction(BaseModel):
    results: list[Prediction]


def create_app(detector_factory: Callable[[], LanguageDetector] = LanguageDetector) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        detector = detector_factory()
        Prediction.model_validate(detector.predict("language detection startup check"))
        app.state.detector = detector
        try:
            yield
        finally:
            app.state.detector = None

    app = FastAPI(title="HealthPH+ Language Detection", version="1.0.0", lifespan=lifespan)
    app.state.detector = None

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError):
        # Validation errors echo input; JSON cannot encode non-finite floats.
        return JSONResponse(status_code=422, content=jsonable_encoder(
            {"detail": exc.errors()},
            custom_encoder={float: lambda value: value if math.isfinite(value) else str(value)},
        ))

    def get_detector():
        if app.state.detector is None:
            raise HTTPException(status_code=503, detail="Language model is not ready")
        return app.state.detector

    def predict_many(texts, options: PredictionOptions):
        detector = get_detector()
        try:
            return detector.predict_many(
                texts, top_k=options.top_k,
                secondary_min_confidence=options.secondary_min_confidence,
            )
        except Exception:
            logger.exception("Language inference failed")
            raise HTTPException(status_code=500, detail="Language inference failed") from None

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        detector = get_detector()
        return {"status": "ready", "model": detector.model_path.name}

    @app.post("/v1/language/predict", response_model=Prediction, response_model_exclude_none=True)
    def predict(body: PredictRequest):
        return predict_many([body.text], body)[0]

    @app.post("/v1/language/predict-batch", response_model=BatchPrediction, response_model_exclude_none=True)
    def predict_batch(body: BatchPredictRequest):
        return {"results": predict_many(body.texts, body)}

    return app


app = create_app()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve local FastText language detection")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8001, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
