# Python Agent Learning Summary

这份笔记总结了目前在 `Mio` 里已经搭起来的最小 LLM/Agent 骨架，以及我们一路学过的核心概念。

目标不是一次做完整 agent，而是先把最重要的分层和抽象搞清楚，再逐步接 OpenAI、事件流、agent loop。

## 1. 现在的目录结构

目前和这次学习最相关的文件有：

```text
main.py
ai/src/llm/
  __init__.py
  types.py
  models.py
  provider.py
  registry.py
  client.py
  adapters/
    __init__.py
    dummy.py
```

每个文件的职责：

- `main.py`
  入口文件，只负责组装和运行，不负责放具体实现。
- `ai/src/llm/types.py`
  放消息、上下文、事件这些基础数据结构。
- `ai/src/llm/models.py`
  放模型定义 `ModelSpec` 和模型注册表。
- `ai/src/llm/provider.py`
  放 `ApiAdapter` 协议，也就是“adapter 应该长什么样”。
- `ai/src/llm/registry.py`
  放 adapter 注册表，负责 `api -> adapter` 的查找。
- `ai/src/llm/client.py`
  放统一调用入口 `LLMClient`，把“找 adapter 并调用”的细节藏起来。
- `ai/src/llm/adapters/dummy.py`
  一个假的 adapter，用来帮助理解架构和流式过程。

## 2. 我们已经学过的核心概念

### 2.1 `ModelSpec`

`ModelSpec` 是“模型配置卡”。

它描述的不是消息内容，而是“我要调用哪个模型、怎么调用”。

例如：

- `id`
- `name`
- `api`
- `provider`
- `base_url`
- `reasoning`

这相当于 Go 里的一个 `struct`：

```go
type ModelSpec struct {
    ID       string
    Name     string
    API      string
    Provider string
    BaseURL  string
}
```

### 2.2 `provider` 和 `api` 的区别

这是目前最重要的架构概念之一。

- `provider`
  表示“谁提供服务”，比如 `openai`、`deepseek`、`zai`
- `api`
  表示“怎么和它说话”，比如 `openai-responses`

它们不是一回事。

例子：

- `provider = "openai"`，`api = "openai-responses"`
- `provider = "deepseek"`，`api = "openai-responses"`

这表示：

- OpenAI 和 DeepSeek 是不同公司
- 但它们可能都兼容同一套调用协议

所以系统应该按 `api` 找 adapter，而不是按 `provider` 找 adapter。

### 2.3 `Message` 和 `Context`

`types.py` 里现在有最小版的消息类型：

- `UserMessage`
- `AssistantMessage`
- `ToolResultMessage`

以及：

- `Message`
- `Context`

可以这样理解：

- `Message`
  是对话中的一条记录
- `Context`
  是“一次请求发送给模型的完整上下文”

### 2.4 `ApiAdapter`

`ApiAdapter` 在 `provider.py` 里定义。

它不是具体实现，而是一个协议规则。

可以把它理解成 Go 的 `interface`：

```go
type ApiAdapter interface {
    Complete(model ModelSpec, ctx Context) AssistantMessage
    Stream(model ModelSpec, ctx Context) ...
}
```

在 Python 里我们用 `Protocol` 表达这个意思。

重点：

- `ApiAdapter` 不负责真正调用某个模型
- 它只是规定“一个 adapter 至少应该提供哪些能力”

### 2.5 `Protocol` 是什么

`Protocol` 很像 Go 的隐式接口实现。

也就是说，一个类不需要显式继承 `ApiAdapter`，只要它：

- 有 `api`
- 有 `complete(...)`
- 有 `stream(...)`

它就可以被当成一个 `ApiAdapter` 使用。

这就是为什么 `DummyAdapter` 没写成：

```python
class DummyAdapter(ApiAdapter):
    ...
```

但仍然可以被 `registry` 接收。

### 2.6 `registry`

`registry.py` 是 adapter 注册表。

它保存的是：

```text
api -> adapter
```

比如：

```python
"openai-responses" -> DummyAdapter()
```

核心函数：

- `register_api_adapter(adapter)`
- `get_api_adapter(api)`
- `list_api_adapters()`
- `clear_api_adapters()`

### 2.7 `LLMClient`

`LLMClient` 是一个门面层。

它的作用不是实现 OpenAI，而是把这些重复动作封装起来：

1. 根据 `model.api` 去 registry 找 adapter
2. 调 `adapter.complete(...)` 或 `adapter.stream(...)`

这样 `main.py` 就不用自己直接碰 registry 细节。

可以把它理解成“前台 / 调度员 / facade”。

## 3. `complete()` 和 `stream()` 的区别

### `complete()`

表示：

“别管过程，最后把完整结果给我。”

调用方式：

```python
reply = await client.complete(model, context)
```

### `stream()`

表示：

“不要等最后一次性返回，过程中的内容也逐步给我。”

调用方式：

```python
async for event in client.stream(model, context):
    ...
```

### 为什么 `stream()` 更底层

因为真实 LLM 交互天然是一个过程：

1. 开始响应
2. 逐步输出内容
3. 可能产生 tool call
4. 正常或异常结束

`complete()` 只是把整个过程折叠成最后一个结果。

所以：

- `stream()` 更接近真实过程
- `complete()` 更适合简单调用

## 4. 从“消息流”升级到“事件流”

一开始的 `DummyAdapter.stream()` 只是返回 `AssistantMessage`。

这虽然也算流，但只有完整消息，不够细。

后来我们升级成了“事件流”。

现在 `types.py` 里已经有最小版事件：

- `StartEvent`
- `TextDeltaEvent`
- `DoneEvent`
- `ErrorEvent`
- `AssistantEvent`

### 为什么事件流更强

因为它不仅能表达“最终消息长什么样”，还能表达：

- 什么时候开始
- 每次增量输出是什么
- 什么时候结束
- 是否发生错误

这对 agent、UI、日志、回放都很重要。

## 5. 现在的 `DummyAdapter` 做了什么

`DummyAdapter` 现在是一个教学用的假实现。

它的作用不是调真实模型，而是帮助理解调用链和事件流。

它现在会：

1. `complete()`
   直接生成一条完整的 `AssistantMessage`
2. `stream()`
   分阶段产出事件：
   - `StartEvent`
   - 多次 `TextDeltaEvent`
   - `DoneEvent`

这样就能真实看到：

```text
start -> text_delta -> text_delta -> done
```

## 6. 现在整条调用链是什么样

当前最小链路：

```text
main.py
  -> 创建 ModelSpec
  -> register_model(model)
  -> register_api_adapter(DummyAdapter())
  -> 创建 Context
  -> LLMClient.complete(...) / LLMClient.stream(...)
  -> registry 按 model.api 找 adapter
  -> DummyAdapter 执行
  -> 返回 AssistantMessage 或 AssistantEvent
```

这已经是一个非常标准的最小 LLM 调用架构了。

## 7. 目前学到的 Python 关键点

### `BaseModel`

`pydantic.BaseModel` 很适合做结构化数据定义。

可以把它理解成：

- Python 版的“带校验的 struct”

### `Field(default_factory=...)`

用于安全地给 `list`、`dict` 这种可变对象设默认值。

例如：

```python
headers: dict[str, str] = Field(default_factory=dict)
```

### `Literal[...]`

用来限制某个字段只能取固定值。

例如：

```python
role: Literal["user"] = "user"
```

### `Protocol`

像 Go 的 interface。

### `async def`

表示异步函数，常用于网络调用和异步流程。

### `yield`

表示“吐出一个值，但函数还没真正结束”。

### `async for`

用来消费异步迭代器。

可以粗略理解成：

- Python 版的“异步 range channel”

## 8. 为什么 `main.py` 不该放太多实现

我们已经把 `DummyAdapter` 从 `main.py` 拆到了 `ai/src/llm/adapters/dummy.py`。

这是因为：

- `main.py` 应该负责组装和启动
- 具体 adapter 实现应该单独放文件

这和 Go 项目里把具体实现放到不同 package / file 的思路一致。

## 9. 现在暂时还没学，但后面会学的内容

接下来会逐步进入这些内容：

1. 真正的 `EventStream` 类
   不是只靠 `AsyncIterator`，而是支持 `.result()`、统一结束语义的流对象
2. 真实的 OpenAI adapter
   用 OpenAI SDK 替换 `DummyAdapter`
3. message transform
   把不同 provider / model 的消息做安全转换
4. agent loop
   让系统能处理 tool call、继续下一轮、形成真正的 agent 行为
5. OAuth
   为未来短期 token / provider 登录预留能力

## 10. 当前最重要的理解

到目前为止，最重要的不是 Python 语法本身，而是这些架构边界：

- `ModelSpec` 是模型配置，不是消息
- `provider` 和 `api` 不是一回事
- `registry` 按 `api` 找 adapter
- `client` 是统一调用入口
- `complete()` 给最终结果
- `stream()` 给过程
- `AssistantMessage` 是结果对象
- `AssistantEvent` 是过程对象

如果这些边界清楚了，后面接真实模型和搭 agent loop 就会顺很多。

## 11. 一句话版本

我们现在已经从“直接写一个脚本调模型”，走到了“有 model、adapter、registry、client、event stream 这些分层的最小 agent 基础架构”。
