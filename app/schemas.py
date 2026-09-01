from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = Field(min_length=1, description="user / assistant 等", examples=["user"])
    timestamp: int = Field(description="毫秒时间戳", examples=[1704067200000])
    content: str = Field(description="原话")


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(
        min_length=1,
        description="幂等键：同内容 200，不同内容 409",
        examples=["eval:run:dataset:conv-0:chunk-0"],
    )
    user_id: str = Field(
        min_length=1,
        description="用户隔离",
        examples=["eval:run:dataset:conv-0"],
    )
    session_id: str = Field(
        min_length=1,
        description="会话 ID",
        examples=["eval:run:sample:0"],
    )
    messages: list[Message] = Field(min_length=1, description="至少一条原话")


class AddResponse(BaseModel):
    success: bool = Field(default=True, description="已提交或幂等命中")
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        min_length=1,
        description="自然语言问句；只用来召回，不当答案",
        examples=["What is the name of my cat?"],
    )
    user_id: str = Field(
        min_length=1,
        description="只在该用户下搜",
        examples=["eval:run:dataset:conv-0"],
    )
    top_k: int = Field(default=100, ge=1, le=100, description="最多返回条数")
    options: list[str] | None = Field(
        default=None,
        description="选择题选项；第一版忽略，官网多给不 422",
    )


class SearchItem(BaseModel):
    id: str
    content: str = Field(description="入库原话，不摘要、不答题")
    score: float
    created_at: str = Field(description="对话时间，UTC ISO-8601")


class SearchResponse(BaseModel):
    data: list[SearchItem]
