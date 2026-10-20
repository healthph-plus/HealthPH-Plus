"""Run with: python -m healthph_sentiment.api"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .inference import MAX_BATCH_SIZE, MAX_TEXT_CHARACTERS, SentimentPredictor, validate_text

logger = logging.getLogger(__name__)
Text = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=MAX_TEXT_CHARACTERS),
    AfterValidator(validate_text),
]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Text


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: Annotated[list[Text], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class Prediction(BaseModel):
    predicted_label: Literal[0, 1]
    predicted_label_name: Literal["calm", "worried"]
    worried_probability: Probability
    threshold_used: Probability
    model_version: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def consistent_prediction(self):
        label = int(self.worried_probability >= self.threshold_used)
        if self.predicted_label != label or self.predicted_label_name != ("calm", "worried")[label]:
            raise ValueError("Prediction must match the saved threshold and label mapping")
        return self


class BatchPrediction(BaseModel):
    results: list[Prediction]


def create_app(predictor_factory: Callable[[], SentimentPredictor] = SentimentPredictor) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        predictor = predictor_factory()
        Prediction.model_validate(predictor.predict("sentiment inference startup check"))
        app.state.predictor = predictor
        try:
            yield
        finally:
            app.state.predictor = None

    app = FastAPI(title="HealthPH+ Sentiment Inference", version="1.0.0", lifespan=lifespan)
    app.state.predictor = None

    def get_predictor():
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Sentiment model is not ready")
        return app.state.predictor

    def predict_many(texts):
        predictor = get_predictor()
        try:
            results = predictor.predict_many(texts)
            if len(results) != len(texts):
                raise RuntimeError("Model returned the wrong number of predictions")
            return [Prediction.model_validate(result) for result in results]
        except Exception:
            logger.exception("Sentiment inference failed")
            raise HTTPException(status_code=500, detail="Sentiment inference failed") from None

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        predictor = get_predictor()
        return {"status": "ready", "model_version": predictor.model_version}

    @app.post("/v1/sentiment/predict", response_model=Prediction)
    def predict(body: PredictRequest):
        return predict_many([body.text])[0]

    @app.post("/v1/sentiment/predict-batch", response_model=BatchPrediction)
    def predict_batch(body: BatchPredictRequest):
        return {"results": predict_many(body.texts)}

    return app


app = create_app()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the local HealthPH+ sentiment model on CPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8002, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
