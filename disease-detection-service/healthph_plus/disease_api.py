"""Run with: python -m healthph_plus.disease_api"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from healthph_plus.disease_inference import (
    MAX_BATCH_SIZE,
    MAX_TEXT_CHARACTERS,
    DiseasePredictor,
    validate_text,
)

logger = logging.getLogger(__name__)
Text = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=MAX_TEXT_CHARACTERS),
    AfterValidator(validate_text),
]
DiseaseLabel = Literal["AURI", "PN", "TB", "COVID"]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Text


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: Annotated[list[Text], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class Prediction(BaseModel):
    labels: list[DiseaseLabel]
    probabilities: dict[DiseaseLabel, Probability]
    thresholds_used: dict[DiseaseLabel, Probability]
    model_version: str


class BatchPrediction(BaseModel):
    results: list[Prediction]


def create_app(predictor_factory: Callable[[], DiseasePredictor] = DiseasePredictor) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        predictor = predictor_factory()
        # Readiness requires a full tokenizer/forward/postprocessing pass.
        Prediction.model_validate(predictor.predict("disease inference startup check"))
        app.state.predictor = predictor
        try:
            yield
        finally:
            app.state.predictor = None

    app = FastAPI(title="HealthPH+ Disease Inference", version="1.0.0", lifespan=lifespan)
    app.state.predictor = None

    def get_predictor():
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Disease model is not ready")
        return app.state.predictor

    def predict_many(texts):
        predictor = get_predictor()
        try:
            return predictor.predict_many(texts)
        except Exception:
            logger.exception("Disease inference failed")
            raise HTTPException(status_code=500, detail="Disease inference failed") from None

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        predictor = get_predictor()
        return {"status": "ready", "model_version": predictor.model_version}

    @app.post("/v1/disease/predict", response_model=Prediction)
    def predict(body: PredictRequest):
        return predict_many([body.text])[0]

    @app.post("/v1/disease/predict-batch", response_model=BatchPrediction)
    def predict_batch(body: BatchPredictRequest):
        return {"results": predict_many(body.texts)}

    return app


app = create_app()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the local HealthPH+ disease model on CPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
