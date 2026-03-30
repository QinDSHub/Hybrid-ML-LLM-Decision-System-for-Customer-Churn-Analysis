from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from app.core.config import get_settings
from app.schemas.predict import FeaturePredictRequest, HealthResponse, PredictResponse, RawPredictRequest
from app.services.feature_engineering import build_single_customer_feature_frame, coerce_feature_payload
from app.services.model import HybridKNNChurnModel


def _load_model(app: FastAPI) -> None:
    settings = get_settings()
    bundle_path = settings.model_bundle_path
    app.state.model = None
    app.state.model_load_error = None

    if not bundle_path.exists():
        app.state.model_load_error = f'模型文件不存在: {bundle_path}'
        return

    try:
        app.state.model = HybridKNNChurnModel.load(
            bundle_path,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime secrets/filesystem
        app.state.model = None
        app.state.model_load_error = f'{type(exc).__name__}: {exc}'


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model(app)
    yield



def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title='Churn Prediction API',
        version='1.0.0',
        description='基于 OpenAI/Hash embedding + KNN 的流失预测 API。',
        lifespan=lifespan,
    )

    @app.get('/', tags=['meta'])
    async def root() -> dict[str, str]:
        return {
            'message': 'Churn Prediction API is running.',
            'docs': '/docs',
            'health': '/health',
        }

    @app.get('/health', response_model=HealthResponse, tags=['meta'])
    async def health(request: Request) -> HealthResponse:
        model = getattr(request.app.state, 'model', None)
        return HealthResponse(
            status='ok' if model is not None else 'degraded',
            model_loaded=model is not None,
            model_bundle_path=str(settings.model_bundle_path),
            model_load_error=getattr(request.app.state, 'model_load_error', None),
            model_provider=model.config.embedding_provider if model is not None else None,
        )

    @app.post('/v1/predict/raw', response_model=PredictResponse, tags=['predict'])
    async def predict_raw(request: Request, payload: RawPredictRequest) -> PredictResponse:
        model: HybridKNNChurnModel | None = getattr(request.app.state, 'model', None)
        if model is None:
            raise HTTPException(status_code=503, detail=getattr(request.app.state, 'model_load_error', '模型尚未加载。'))

        try:
            feature_df = build_single_customer_feature_frame(payload.model_dump(), reference_date=payload.reference_date)
            result = model.predict_records(feature_df, return_neighbors=5)[0]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PredictResponse(**result, used_features=feature_df.iloc[0].to_dict())

    @app.post('/v1/predict/features', response_model=PredictResponse, tags=['predict'])
    async def predict_features(request: Request, payload: FeaturePredictRequest) -> PredictResponse:
        model: HybridKNNChurnModel | None = getattr(request.app.state, 'model', None)
        if model is None:
            raise HTTPException(status_code=503, detail=getattr(request.app.state, 'model_load_error', '模型尚未加载。'))

        try:
            feature_df = coerce_feature_payload(payload.model_dump())
            result = model.predict_records(feature_df, return_neighbors=5)[0]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PredictResponse(**result, used_features=feature_df.iloc[0].to_dict())

    @app.post('/v1/reload-model', tags=['meta'])
    async def reload_model(request: Request) -> dict[str, Any]:
        _load_model(request.app)
        model = getattr(request.app.state, 'model', None)
        return {
            'model_loaded': model is not None,
            'model_load_error': getattr(request.app.state, 'model_load_error', None),
        }

    return app


app = create_app()
