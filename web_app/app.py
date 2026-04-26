"""
Web 前端后端：为多模态图像检索系统提供 HTTP API 和 SSE 端点。
"""

import os
import sys
import json
import asyncio
from contextlib import asynccontextmanager

# 允许从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from agent_pipeline.pipeline import MultiModalAgentPipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "test_images")

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 pipeline 单例（模型加载 + 图像索引）。"""
    global pipeline
    loop = asyncio.get_event_loop()
    print("[Web] 正在初始化多模态检索流水线（加载模型 + 建立索引）...")
    pipeline = await loop.run_in_executor(None, MultiModalAgentPipeline)
    print("[Web] 流水线初始化完成，可以接受请求")
    yield
    print("[Web] 服务关闭")


app = FastAPI(title="多模态图像检索系统", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "initializing", "pipeline_initialized": False}
        )
    return {
        "status": "ok",
        "pipeline_initialized": True,
        "total_images": len(pipeline.regular_retrieval.offline_indexer.get_db()),
        "llm_model": pipeline.llm.model,
        "vl_model": pipeline.fine_grained.vl_manager.model_name
    }


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    try:
        body = await request.json()
    except Exception:
        raw = await request.body()
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = json.loads(raw.decode("gbk", errors="replace"))
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "查询不能为空"})
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "流水线尚未初始化"})

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def progress_callback(data):
            loop.call_soon_threadsafe(queue.put_nowait, ("progress", data))

        future = loop.run_in_executor(
            None, lambda: pipeline.chat_structured(query, progress_callback)
        )

        while True:
            if future.done():
                while not queue.empty():
                    event_type, data = await queue.get()
                    yield f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"

                try:
                    result = future.result()
                    structured = result.get("structured", {})
                    yield f"data: {json.dumps({'type': 'complete', 'output': result.get('output', ''), 'structured': structured}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
                break

            try:
                event_type, data = await asyncio.wait_for(queue.get(), timeout=15.0)
                yield f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# 静态文件挂载（需在 API 路由之后）
app.mount("/images", StaticFiles(directory=TEST_IMAGES_DIR), name="images")

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
