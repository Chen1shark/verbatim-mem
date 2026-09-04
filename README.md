# verbatim-mem

[Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/) 文本赛道的记忆服务：提供 `Add` / `Search`，只交对话原话，不生成答案。

当前进度：`GET /health`、`POST /add`、`POST /search` 可用。Search 为 FTS5 ∪ 邻句块 ∪ FAISS，加权 RRF 后按问句实词覆盖 / 数字 / 专名 / 选项加分；无时间意图时不加 recency。只返回该用户下的原话。对照公开技术说明自行实现，未复制其它参赛仓库代码。

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

## 接口

`POST /add`、`POST /search` 需要鉴权，三选一：`X-Api-Key`、`Authorization: Bearer <key>`、`Authorization: Token <key>`，值与 `.env` 的 `MEMORY_API_KEY` 相同。兼容路径：`POST /v1/memory/add`、`POST /v1/memory/search`。

### `GET /health`

不鉴权。

```json
{"status": "ok"}
```

### `POST /add`

```json
{
  "request_id": "eval:run:dataset:conv-0:chunk-0",
  "user_id": "eval:run:dataset:conv-0",
  "session_id": "eval:run:sample:0",
  "messages": [
    {
      "role": "user",
      "timestamp": 1704067200000,
      "content": "verbatim message one"
    },
    {
      "role": "assistant",
      "content": "verbatim message two"
    }
  ]
}
```

`messages[].timestamp` 为毫秒，可缺省；缺省入库为 SQL `NULL`，Search 的 `created_at` 为 `null`，不会写成 1970。同一 `request_id` 且内容相同返回 200；同一 `request_id` 且内容不同返回 409。

```json
{
  "success": true,
  "request_id": "eval:run:dataset:conv-0:chunk-0",
  "user_id": "eval:run:dataset:conv-0",
  "session_id": "eval:run:sample:0"
}
```

### `POST /search`

```json
{
  "query": "verbatim message",
  "user_id": "eval:run:dataset:conv-0",
  "top_k": 100
}
```

选择题可带 `options`，每项与 `query` 分路召回后再融合；`options` 可缺省。

```json
{
  "query": "verbatim message",
  "user_id": "eval:run:dataset:conv-0",
  "top_k": 100,
  "options": ["A. alpha", "B. beta"]
}
```

`content` 为入库原话。`created_at` 来自 `messages.timestamp` 的 UTC ISO-8601；无原始时间则为 `null`。

```json
{
  "data": [
    {
      "id": "eval:run:dataset:conv-0:chunk-0:0",
      "content": "verbatim message one",
      "score": 1.23,
      "created_at": "2024-01-01T00:00:00Z",
      "role": "user"
    }
  ]
}
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
