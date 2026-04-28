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
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from agent_pipeline.pipeline import MultiModalAgentPipeline
from data_load import discover_image_paths

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "test_images")

pipeline = None
CURRENT_IMAGE_DIR = DEFAULT_IMAGE_DIR


def _scan_image_dirs() -> list[dict]:
    """扫描项目内已知的图像目录，返回可用选项列表。"""
    known = [
        os.path.join(PROJECT_ROOT, "test_images"),
        os.path.join(PROJECT_ROOT, "experiment", "dataset"),
    ]
    dirs = []
    seen = set()
    for d in known:
        d = os.path.normpath(d)
        if d in seen:
            continue
        seen.add(d)
        if os.path.isdir(d):
            count = len(discover_image_paths(d))
            dirs.append({
                "path": d,
                "label": os.path.basename(d),
                "image_count": count,
            })
    return dirs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 pipeline 单例（模型加载 + 图像索引）。"""
    global pipeline, CURRENT_IMAGE_DIR
    loop = asyncio.get_event_loop()
    print(f"[Web] 图像目录: {CURRENT_IMAGE_DIR}")
    print("[Web] 正在初始化多模态检索流水线（加载模型 + 建立索引）...")
    pipeline = await loop.run_in_executor(
        None, lambda: MultiModalAgentPipeline(image_dir=CURRENT_IMAGE_DIR)
    )
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


# ── API 端点 ──

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
        "image_dir": CURRENT_IMAGE_DIR,
        "llm_model": pipeline.llm.model,
        "vl_model": pipeline.fine_grained.vl_manager.model_name
    }


@app.get("/api/image-dirs")
async def list_image_dirs():
    """返回可用的图像目录列表及当前选中的目录。"""
    dirs = _scan_image_dirs()
    return {
        "current": CURRENT_IMAGE_DIR,
        "directories": dirs,
    }


@app.post("/api/set-image-dir")
async def set_image_dir(request: Request):
    """切换图像目录并重建索引。"""
    global pipeline, CURRENT_IMAGE_DIR

    body = await request.json()
    new_dir = (body.get("path") or "").strip()
    if not new_dir:
        return JSONResponse(status_code=400, content={"error": "路径不能为空"})

    new_dir = os.path.normpath(new_dir)
    if not os.path.isdir(new_dir):
        return JSONResponse(status_code=400, content={"error": f"目录不存在: {new_dir}"})

    image_count = len(discover_image_paths(new_dir))
    if image_count == 0:
        return JSONResponse(status_code=400, content={"error": f"目录中无图像文件: {new_dir}"})

    print(f"[Web] 切换图像目录: {CURRENT_IMAGE_DIR} → {new_dir} ({image_count} 张)")
    CURRENT_IMAGE_DIR = new_dir

    loop = asyncio.get_event_loop()
    pipeline = await loop.run_in_executor(
        None, lambda: MultiModalAgentPipeline(image_dir=CURRENT_IMAGE_DIR)
    )
    print(f"[Web] 重建索引完成")

    return {
        "status": "ok",
        "image_dir": CURRENT_IMAGE_DIR,
        "image_count": image_count,
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


# ── 图像文件服务 ──

@app.get("/images/{filename:path}")
async def serve_image(filename: str):
    """从当前图像目录提供图片文件。"""
    file_path = os.path.join(CURRENT_IMAGE_DIR, filename)
    if not os.path.isfile(file_path):
        return JSONResponse(status_code=404, content={"error": "图片不存在"})
    return FileResponse(file_path)


# ── 前端静态文件 ──

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
