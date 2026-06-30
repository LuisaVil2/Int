from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.websocket.routes import router as websocket_router
from backend.monitoring.metrics import metrics
def create_app():
    app=FastAPI(title='Medical Interpretation Platform')
    app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
    app.include_router(websocket_router)
    @app.get('/health')
    async def health(): return {'status':'ok'}
    @app.get('/metrics')
    async def get_metrics(): return metrics.snapshot()
    return app
app=create_app()
