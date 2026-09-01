# verbatim-mem

[Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/) 文本赛道的记忆服务：提供 `Add` / `Search`，只交对话原话，不生成答案。

当前进度：`GET /health`、`POST /add`、`POST /search` 可用。Search 为 FTS5 ∪ FAISS，只返回该用户下的原话。

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

## 用 Swagger 查看和调试

服务起来后打开：

- Swagger UI：http://127.0.0.1:8000/docs
- ReDoc：http://127.0.0.1:8000/redoc

在 `/docs` 右上角点 **Authorize**，填入 `.env` 里的 `MEMORY_API_KEY`，再对 `POST /add`、`POST /search` 使用 Try it out。`GET /health` 不需要授权。

## 测试

```powershell
pytest -q
```
