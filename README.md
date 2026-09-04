# verbatim-mem

[Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/) 文本赛道的记忆服务：提供 `Add` / `Search`，只交对话原话，不生成答案。

当前进度：`GET /health`、`POST /add`、`POST /search` 可用。Search 为 FTS5 ∪ 邻句块 ∪ FAISS，加权 RRF 后按问句实词覆盖 / 数字 / 专名 / 选项 / 人设 / 纠错加分，再按未覆盖问句词补证据、时间加分，只返回该用户下的原话。对照公开技术说明自行实现，未复制其它参赛仓库代码。

## 启动

Python 3.11+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

编辑 `.env`，填入 `MEMORY_API_KEY` 和百炼的 `EMBEDDING_API_KEY`。模型是 `qwen3.7-text-embedding`，维度 `2560`。然后：

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

请使用 `--workers 1`。

Linux / macOS 把激活和复制命令换成：

```bash
source .venv/bin/activate
cp .env.example .env
```

可选 CrossEncoder 重排（默认关闭）：

```powershell
pip install -r requirements-rerank.txt
```

在 `.env` 设置 `MEMORY_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`。

## Docker

```powershell
docker build -t verbatim-mem .
docker run --rm -p 8000:8000 -v verbatim-mem-data:/data `
  -e MEMORY_API_KEY=changeme `
  -e EMBEDDING_API_KEY=your-key `
  -e MEMORY_DB_PATH=/data/memory.db `
  verbatim-mem
```

容器内 `uvicorn` 监听 `0.0.0.0:8000`，`--workers 1`。`GET /health` 不鉴权，返回 2xx 即视为存活。公网评测请用 HTTPS 反代到该端口。

## 用 Swagger 查看和调试

服务起来后打开：

- Swagger UI：http://127.0.0.1:8000/docs
- ReDoc：http://127.0.0.1:8000/redoc

在 `/docs` 右上角点 **Authorize**，填入 `.env` 里的 `MEMORY_API_KEY`，再对 `POST /add`、`POST /search` 使用 Try it out。`GET /health` 不需要授权。

## 测试

```powershell
pytest -q
```
