# openclaw-py 아키텍처 완전 해설서

> 이 문서는 LLM 이론은 알지만 대규모 Python 프로젝트 경험이 적은 분을 위해,
> 모든 모듈의 코드를 **한 줄 한 줄** 해설합니다.

---

## 목차

1. [전체 그림: 에이전트가 뭘 하는 건가?](#1-전체-그림)
2. [데이터 타입 (`agent/types.py`)](#2-데이터-타입)
3. [에이전트 루프 (`agent/loop.py`)](#3-에이전트-루프)
4. [모델 프로바이더 (`model/provider.py`)](#4-모델-프로바이더)
5. [에러 분류와 페일오버 (`model/failover.py`)](#5-에러-분류와-페일오버)
6. [Thinking 레벨 (`model/thinking.py`)](#6-thinking-레벨)
7. [세션 관리 (`session/manager.py`)](#7-세션-관리)
8. [컴팩션 (`session/compaction.py`)](#8-컴팩션)
9. [프루닝 (`session/pruning.py`)](#9-프루닝)
10. [세션 레인 (`session/lanes.py`)](#10-세션-레인)
11. [메모리 플러시 (`session/memory_flush.py`)](#11-메모리-플러시)
12. [컨텍스트 가드 (`context/guard.py`)](#12-컨텍스트-가드)
13. [메모리 스토어 (`memory/store.py`)](#13-메모리-스토어)
14. [하이브리드 검색 (`memory/search.py`)](#14-하이브리드-검색)
15. [임베딩 프로바이더 (`memory/embeddings.py`)](#15-임베딩-프로바이더)
16. [시스템 프롬프트 빌더 (`prompt/builder.py`)](#16-시스템-프롬프트-빌더)
17. [부트스트랩 파일 (`prompt/bootstrap.py`)](#17-부트스트랩-파일)
18. [인젝션 방어 (`prompt/sanitize.py`)](#18-인젝션-방어)
19. [도구 레지스트리 (`tools/registry.py`)](#19-도구-레지스트리)
20. [내장 도구들 (`tools/builtins/`)](#20-내장-도구들)
21. [서브에이전트 (`subagent/spawn.py`)](#21-서브에이전트)
22. [Hook 시스템 (`hooks/`)](#22-hook-시스템)
23. [Cron 스케줄러 (`cron/`)](#23-cron-스케줄러)
24. [스킬 로더 (`skills/loader.py`)](#24-스킬-로더)
25. [설정 (`config.py`)](#25-설정)
26. [REPL과 Python API (`repl.py`)](#26-repl과-python-api)
27. [핵심 개념 용어집](#27-핵심-개념-용어집)

---

## 1. 전체 그림

### 에이전트란?

주피터 노트북에서 `model.generate("질문")` 하면 답변이 하나 나오죠.
에이전트는 이걸 **루프**로 감쌉니다:

```
사용자 → [모델 호출] → 답변에 "도구 호출"이 포함됨?
                           ├─ 아니오 → 답변 반환 (끝)
                           └─ 예 → 도구 실행 → 결과를 대화에 추가 → 다시 모델 호출
```

이 루프를 **에이전트 루프**라고 합니다. 핵심은 모델이 "파일을 읽어라", "명령을 실행해라" 같은 **도구(tool)**를 호출할 수 있다는 것입니다.

### 왜 이렇게 복잡한가?

단순 루프는 쉽지만, 실제 운용에는 이런 문제가 생깁니다:

| 문제 | 해결 모듈 |
|------|-----------|
| 대화가 길어지면 컨텍스트 윈도우 초과 | `compaction.py`, `pruning.py`, `guard.py` |
| 모델 API가 에러나면? | `failover.py` |
| 도구가 무한 루프에 빠지면? | `registry.py` (루프 감지) |
| 대화 내용을 다음 세션에서 기억? | `memory/` |
| 악의적 입력이 프롬프트를 조작? | `sanitize.py` |
| 병렬 작업이 필요하면? | `subagent/`, `lanes.py` |

이 모든 게 ~9,600줄의 코드에 구현되어 있습니다.

### 데이터 흐름 전체도

```
사용자 입력
    │
    ▼
┌─────────────────────────────────────────────┐
│  repl.py (Agent 클래스)                      │
│  ┌─────────────────────────────────────────┐ │
│  │  agent/loop.py  run()                   │ │
│  │   ├── sanitize_text(입력)               │ │
│  │   ├── session.append(사용자 메시지)      │ │
│  │   └── _attempt_loop()                   │ │
│  │        ├── context_guard.check()        │ │
│  │        ├── compact_session() (필요시)    │ │
│  │        ├── prune_messages() (캐시 TTL)  │ │
│  │        ├── build_system_prompt()        │ │
│  │        ├── provider.stream()  ← LLM 호출│ │
│  │        │    └── OpenAI API              │ │
│  │        ├── 도구 호출 파싱               │ │
│  │        ├── tool_registry.execute()      │ │
│  │        │    └── bash/read/write/...     │ │
│  │        ├── loop_detector.record()       │ │
│  │        └── (루프: 도구 결과 → 재호출)   │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  메모리: SQLite + FTS5 + 벡터 임베딩         │
│  세션: JSONL append-only 파일                │
└─────────────────────────────────────────────┘
```

---

## 2. 데이터 타입

**파일**: `openclaw/agent/types.py` (274줄)

이 파일은 전체 시스템에서 사용하는 **데이터 구조**를 정의합니다.
Pydantic의 `BaseModel`을 사용하는데, 이건 Python 딕셔너리보다 안전한 데이터 클래스입니다.

### ContentBlock (메시지의 구성 요소)

LLM API에서 하나의 메시지는 여러 "블록"으로 구성됩니다:

```python
class TextBlock(BaseModel):
    type: Literal["text"] = "text"   # 항상 "text" 문자열
    text: str                        # 실제 텍스트 내용
```

`Literal["text"]`는 타입 힌트로, 이 필드가 오직 `"text"` 값만 가질 수 있다는 뜻입니다.
JSON 직렬화할 때 `{"type": "text", "text": "안녕하세요"}` 형태가 됩니다.

```python
class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:24]}")
    name: str                        # 도구 이름 (예: "bash")
    input: dict[str, Any] = Field(default_factory=dict)  # 도구 인자
```

`default_factory=lambda: ...`는 인스턴스가 생성될 때마다 새로운 고유 ID를 만든다는 뜻입니다.
`uuid.uuid4().hex[:24]`는 랜덤 16진수 24자리를 생성합니다.

```python
class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str    # 어떤 ToolUseBlock에 대한 응답인지
    content: str = ""   # 도구 실행 결과
    is_error: bool = False
```

이 네 종류의 블록을 **Union 타입**으로 묶습니다:

```python
ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock
```

`|`는 Python 3.10+의 Union 문법입니다. "이 중 하나"라는 뜻입니다.

### AgentMessage (대화의 한 턴)

```python
class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: Literal["user", "assistant", "system"]
    content: list[ContentBlock] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
```

하나의 메시지 안에 여러 블록이 들어갈 수 있습니다. 예를 들어 assistant 메시지에
텍스트 블록 + 도구 호출 블록이 함께 올 수 있죠:

```python
# assistant가 "파일을 읽어볼게요"라고 말하면서 동시에 read 도구를 호출한 경우
msg = AgentMessage(
    role="assistant",
    content=[
        TextBlock(text="파일을 읽어볼게요"),
        ToolUseBlock(name="read", input={"file_path": "/tmp/data.txt"}),
    ]
)
```

### ToolDefinition (도구 스키마)

```python
class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI API의 function calling 형식으로 변환"""
        ...
```

이 메서드는 도구 정의를 OpenAI API가 이해하는 JSON Schema로 바꿔줍니다:

```json
{
  "type": "function",
  "function": {
    "name": "read",
    "description": "Read file contents",
    "parameters": {
      "type": "object",
      "properties": {
        "file_path": {"type": "string", "description": "..."}
      },
      "required": ["file_path"]
    }
  }
}
```

### ThinkingLevel (사고 레벨)

```python
class ThinkingLevel(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    ...
    XHIGH = "xhigh"

    def fallback(self) -> ThinkingLevel:
        """한 단계 낮은 레벨 반환"""
```

이건 모델에게 "얼마나 깊이 생각해라"를 지시하는 설정입니다.
`fallback()`은 모델이 해당 레벨을 지원하지 않으면 자동으로 한 단계 낮추는 용도입니다.

### RunResult (실행 결과)

```python
class RunResult(BaseModel):
    text: str = ""              # 최종 텍스트 응답
    messages: list[AgentMessage] = ...  # 전체 대화 이력
    usage: TokenUsage = ...     # 토큰 사용량
    tool_calls_count: int = 0   # 도구 호출 횟수
    compacted: bool = False     # 컴팩션 발생 여부
    error: str | None = None    # 에러 메시지
```

---

## 3. 에이전트 루프

**파일**: `openclaw/agent/loop.py` (478줄)

이 파일이 전체 시스템의 **심장**입니다. 모든 모듈이 여기서 조립됩니다.

### AgentContext (실행 컨텍스트)

```python
@dataclass
class AgentContext:
    config: AppConfig           # 전체 설정
    provider: ModelProvider     # LLM API 클라이언트
    session: SessionManager     # 세션(대화 이력) 관리자
    tool_registry: ToolRegistry # 등록된 도구 목록
    context_guard: ContextGuard # 컨텍스트 윈도우 감시
    failover: FailoverManager   # 에러 시 다른 모델로 전환
    loop_detector: ToolLoopDetector  # 도구 무한 루프 감지
    ...
```

`@dataclass`는 Pydantic의 `BaseModel`보다 가벼운 데이터 클래스입니다.
주로 내부에서만 사용하고 JSON 직렬화가 불필요할 때 씁니다.

### run() — 진입점

```python
async def run(ctx: AgentContext, user_input: str) -> RunResult:
```

`async`는 이 함수가 **비동기**라는 뜻입니다. LLM API 호출처럼 네트워크를 기다리는
동안 다른 작업을 할 수 있게 해줍니다. 주피터 노트북에서 `await`로 호출합니다.

```python
    # 1. 세션 로드 (JSONL 파일에서 이전 대화 불러오기)
    ctx.session.load()

    # 2. 메모리 플러시 (컴팩션 전에 중요한 내용 저장)
    if await should_flush(...):
        await execute_memory_flush(...)

    # 3. 입력 살균 (유니코드 공격 방어)
    clean_input = sanitize_text(user_input)

    # 4. 사용자 메시지를 세션에 추가
    user_msg = AgentMessage(role="user", content=[TextBlock(text=clean_input)])
    ctx.session.append(user_msg)

    # 5. 핵심 루프 실행
    result = await _attempt_loop(ctx, model)

    return result
```

### _attempt_loop() — 핵심 루프

여기서 실제 "모델 호출 → 도구 실행 → 반복" 루프가 돌아갑니다:

```python
async def _attempt_loop(ctx: AgentContext, model: str) -> RunResult:
    while turn_count < MAX_TURNS_PER_RUN:  # 최대 50턴
        turn_count += 1

        # (A) 컨텍스트 예산 확인
        estimated_tokens = ctx.session.estimate_tokens()
        status = ctx.context_guard.check(estimated_tokens)
        if status.action == ContextAction.COMPACT:
            await compact_session(...)  # 오래된 대화 요약

        # (B) 프루닝 (메모리 내에서만, 디스크 안 건드림)
        pruned_messages = prune_messages(...)

        # (C) 시스템 프롬프트 조립
        system_prompt = build_system_prompt(...)

        # (D) LLM 스트리밍 호출
        async for chunk in ctx.provider.stream(...):
            if chunk.text:
                accumulated_text += chunk.text    # 텍스트 누적
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)  # 도구 호출 누적
```

`async for`는 스트리밍입니다. LLM이 토큰을 하나씩 생성할 때마다
이 루프가 한 번 돌면서 실시간으로 텍스트를 받습니다.

```python
        # (E) 도구 호출이 없으면 → 완료!
        if not tool_calls:
            result.text = accumulated_text.strip()
            break

        # (F) 도구 실행
        for tc in tool_calls:
            tool_result = await ctx.tool_registry.execute(tc.name, tc.input)

            # 결과 크기 제한
            tool_result.content = truncate_tool_result(tool_result.content, max_chars)

            # 루프 감지
            warning = ctx.loop_detector.record(tc.name, tc.input, tool_result.content)
            if warning and "CRITICAL" in warning:
                return result  # 강제 종료

        # 도구 결과를 대화에 추가 → while 루프 처음으로 돌아감
        # → 모델이 도구 결과를 보고 다시 응답
```

**핵심 패턴**: 모델 응답에 도구 호출이 있으면 실행하고, 그 결과를 대화에 넣고,
다시 모델을 호출합니다. 도구 호출이 없을 때까지 반복합니다.

### Thinking 에러 폴백

```python
        except Exception as stream_err:
            if ctx.thinking != ThinkingLevel.OFF and _is_thinking_error(stream_err):
                lower = ctx.thinking.fallback()  # 한 단계 낮춤
                if lower != ctx.thinking:
                    ctx.thinking = lower
                    turn_count -= 1  # 이 턴 카운트 안 셈
                    continue         # while 루프 처음으로
```

모델이 "thinking 기능을 지원 안 함" 에러를 내면, 자동으로 레벨을 낮춰서 재시도합니다.
예: `HIGH → MEDIUM → LOW → OFF`

---

## 4. 모델 프로바이더

**파일**: `openclaw/model/provider.py` (401줄)

LLM API를 호출하는 클라이언트입니다. OpenAI 호환 API (vLLM, OpenRouter 등)를 지원합니다.

### 초기화

```python
class ModelProvider:
    def __init__(self, config: AppConfig) -> None:
        self.client = AsyncOpenAI(
            base_url=config.endpoints.llm.base_url,  # 예: "http://localhost:8000/v1"
            api_key=config.endpoints.llm.api_key,
        )
```

`AsyncOpenAI`는 OpenAI Python SDK의 비동기 클라이언트입니다.
`base_url`을 바꾸면 vLLM이든 OpenRouter든 아무 OpenAI 호환 서버에 연결됩니다.

### 듀얼 도구 호출 모드

LLM에게 도구를 사용하라고 알려주는 방식이 두 가지입니다:

**1. Native 모드**: API 레벨에서 `tools` 파라미터로 전달
```python
kwargs["tools"] = [t.to_openai_schema() for t in tools]
kwargs["tool_choice"] = "auto"
```

**2. Prompt 모드**: 시스템 프롬프트에 도구 설명을 텍스트로 삽입
```python
# 모델에게 이런 텍스트를 보냄:
# "다음 도구를 사용할 수 있습니다:
#  <tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>"
```

`tool_mode = "auto"`이면 먼저 Native를 시도하고, 실패하면 Prompt로 폴백합니다.
vLLM의 오래된 버전은 Native를 지원하지 않으므로 이 폴백이 중요합니다.

### 스트리밍 도구 호출 누적

스트리밍에서 도구 호출은 한 번에 오지 않고 **조각조각** 옵니다:

```
청크1: tool_calls=[{index:0, id:"toolu_abc", name:"bash"}]
청크2: tool_calls=[{index:0, arguments:"{\"com"}]
청크3: tool_calls=[{index:0, arguments:"mand\": \"ls\"}"}]
청크4: finish_reason="tool_calls"
```

이걸 하나로 합치는 게 `pending_tool_calls` 딕셔너리입니다:

```python
pending_tool_calls: dict[int, dict[str, str]] = {}  # index → {id, name, arguments}

# 청크마다 누적
pending_tool_calls[idx]["arguments"] += tc_delta.function.arguments

# finish_reason이 "tool_calls"일 때 한 번에 파싱
if chunk.choices[0].finish_reason == "tool_calls":
    for idx in sorted(pending_tool_calls):
        args = json.loads(tc["arguments"])
        parsed.tool_calls.append(ToolUseBlock(...))
```

이 버그를 고치는 데 가장 오래 걸렸습니다. 처음에는 청크마다 파싱해서
같은 도구 호출이 중복 생성되는 문제가 있었습니다.

### Prompt 기반 도구 호출 파싱

모델이 텍스트 안에 `<tool_call>JSON</tool_call>`을 출력하면 정규식으로 파싱합니다:

```python
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)",
    re.DOTALL,
)
```

`re.DOTALL`은 `.`이 줄바꿈도 매칭하게 합니다 (JSON이 여러 줄일 수 있으므로).

---

## 5. 에러 분류와 페일오버

**파일**: `openclaw/model/failover.py` (829줄)

LLM API는 다양한 이유로 실패합니다. 이 모듈은 에러 메시지를 분석해서
**어떤 종류의 에러인지** 분류하고, 적절한 대응을 합니다.

### 에러 분류 (`classify_error`)

약 60개의 정규식 패턴으로 에러를 12종으로 분류합니다:

```python
# 예: Rate limit 패턴들
_RATE_LIMIT_PATTERNS = [
    re.compile(r"rate[_ ]limit|too many requests|429"),
    "exceeded your current quota",
    ...
]
```

분류 순서가 중요합니다 (더 구체적인 것을 먼저 검사):
1. HTTP 상태 코드 (402, 429, 401 등)
2. 세션 만료
3. 모델 미존재
4. Rate limit
5. 과부하
6. 과금 문제
7. 타임아웃
8. 컨텍스트 오버플로
9. 인증 문제

### 페일오버 결정 (`should_failover`)

```python
def should_failover(reason: FailoverReason) -> bool:
    if reason == FailoverReason.UNKNOWN:
        return False
    return reason not in _NO_FAILOVER_REASONS  # CONTEXT_OVERFLOW는 제외
```

컨텍스트 오버플로는 다른 모델로 바꿔도 해결이 안 됩니다 (컴팩션이 필요).
나머지 에러는 다른 API 키나 모델로 전환하면 해결될 수 있습니다.

### ProfileCooldown (쿨다운 관리)

```python
@dataclass
class ProfileCooldown:
    error_count: int = 0
    cooldown_until: float = 0.0  # 이 시간까지 사용 금지

    def mark_failure(self, reason: FailoverReason) -> None:
        # 타임아웃은 카운터만 증가, 쿨다운 설정 안 함
        if reason == FailoverReason.TIMEOUT:
            return
        # 과금 에러: 5시간 → 10시간 → 24시간
        # 일반 에러: 1분 → 5분 → 25분 → 1시간
```

지수 백오프(exponential backoff)를 사용합니다. 에러가 반복될수록
대기 시간이 기하급수적으로 늘어납니다.

### API 키 로테이션

```python
class ApiKeyRotator:
    keys: list[str]  # ["sk-aaa", "sk-bbb", "sk-ccc"]

    def rotate_on_error(self, reason) -> str | None:
        """현재 키에 에러 → 쿨다운 설정 → 다음 키로 이동"""
        self._cooldowns[self._current_idx].mark_failure(reason)
        self._current_idx = (self._current_idx + 1) % len(self.keys)
        return self.get_current_key()  # 쿨다운 안 된 키 반환
```

Round-robin 방식으로 키를 돌려가며 사용합니다.

### 상태 영속화

```python
def save_state(self) -> None:
    """fcntl 파일 잠금으로 안전하게 JSON 저장"""
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)  # 배타적 잠금
    os.write(fd, json.dumps(state).encode())
    os.replace(str(tmp), str(path))  # 원자적 교체
```

`fcntl.flock`은 Unix 파일 잠금입니다. 여러 프로세스가 동시에 상태를 쓰는 것을 방지합니다.
`os.replace`는 원자적(atomic) 파일 교체로, 쓰는 도중에 중단돼도 파일이 깨지지 않습니다.

---

## 6. Thinking 레벨

**파일**: `openclaw/model/thinking.py` (80줄)

일부 모델은 "생각하는 과정"을 보여주는 기능이 있습니다.
이 모듈은 요청한 thinking 레벨이 모델에서 지원되지 않을 때 자동으로 한 단계씩 낮추는 폴백 로직입니다.

```python
def resolve_thinking(requested: ThinkingLevel, model_supported: set | None) -> ThinkingLevel:
    level = requested
    while level != ThinkingLevel.OFF:
        if level in model_supported:
            return level
        level = level.fallback()  # HIGH → MEDIUM → LOW → ...
    return ThinkingLevel.OFF
```

사용자가 `/think`이나 `/thinking high` 같은 디렉티브를 메시지에 넣으면
`parse_thinking_directive()`가 파싱합니다.

---

## 7. 세션 관리

**파일**: `openclaw/session/manager.py` (301줄)

대화 이력을 **JSONL 파일**로 관리합니다.

### JSONL이란?

한 줄에 하나의 JSON 객체가 들어가는 형식입니다:

```jsonl
{"role": "user", "content": [{"type": "text", "text": "안녕"}], "timestamp": 1710000000}
{"role": "assistant", "content": [{"type": "text", "text": "안녕하세요!"}], "timestamp": 1710000001}
```

장점: **append-only** (파일 끝에 추가만 하면 됨) → 매우 빠르고 안전합니다.

### SessionWriteLock (파일 잠금)

```python
class SessionWriteLock:
    def acquire(self) -> bool:
        # 1. 기존 잠금 파일이 있으면 → 프로세스가 살아있나 확인
        os.kill(pid, 0)  # 신호 0: 프로세스 존재 확인만, 실제 종료 안 함

        # 2. 새 잠금 파일 생성 (O_EXCL: 이미 있으면 실패)
        self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

        # 3. 배타적 잠금 (논블로킹)
        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
```

`O_CREAT | O_EXCL`은 파일이 이미 존재하면 에러를 내는 플래그입니다.
이것으로 두 프로세스가 동시에 잠금을 획득하는 것을 방지합니다.

### 메시지 직렬화/역직렬화

```python
def _serialize_block(block: ContentBlock) -> dict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    ...
```

Pydantic 모델을 JSON으로 바꾸는 단순한 변환입니다. 역방향도 마찬가지.

### 도구 페어링 수리

세션 파일이 깨질 수 있습니다 (비정상 종료 등). 이때 assistant가 도구를 호출했는데
그 결과가 없으면 API가 에러를 냅니다. `_repair_tool_pairing()`이 빠진 결과를 채워넣습니다:

```python
def _repair_tool_pairing(self) -> None:
    for tu in tool_uses:
        if tu.id not in existing_ids:
            next_msg.content.append(
                ToolResultBlock(
                    tool_use_id=tu.id,
                    content="[Tool result missing — session repaired]",
                    is_error=True,
                )
            )
```

### 토큰 추정

```python
def estimate_tokens(self) -> int:
    """4글자 = 1토큰 휴리스틱"""
    return total_chars // 4
```

정확한 토큰 수를 세려면 tokenizer가 필요하지만, 빠른 추정으로 4:1 비율을 사용합니다.
영어는 ~4자/토큰, 한국어는 ~2-3자/토큰이므로 보수적 추정입니다.

---

## 8. 컴팩션

**파일**: `openclaw/session/compaction.py` (1023줄)

컨텍스트 윈도우가 가득 차면 오래된 대화를 **요약**해서 줄입니다.
이 파일이 전체 코드베이스에서 가장 복잡합니다.

### 왜 필요한가?

LLM의 컨텍스트 윈도우는 유한합니다 (예: 32K 토큰).
대화가 길어지면 초과하므로, 오래된 부분을 요약으로 바꿔야 합니다:

```
[턴 1] "서버 배포해줘" → (도구 실행 10번) → "완료"
[턴 2] "로그 확인해줘" → (도구 실행 5번) → "에러 없음"
[턴 3] "DB 마이그레이션" → ...
                                    ↓ 컴팩션
[요약] "서버 배포 완료, 로그 정상, DB 마이그레이션 진행 중"
[턴 3 계속] ...
```

### 다단계 컴팩션 흐름

```python
async def compact_session(...) -> CompactionEntry | None:
    # 1. 최소 4개 이상 메시지가 있어야 의미 있음
    if len(messages) < 4:
        return None

    # 2. 최근 4개는 보존, 나머지를 요약 대상으로
    summarize_msgs = messages[:-keep_count]

    # 3. 큰 도구 결과를 잘라냄 (요약기에 보내기 전)
    stripped = strip_tool_result_details(summarize_msgs)

    # 4. 식별자 추출 (URL, 파일 경로, 해시 등)
    identifiers = _extract_identifiers_from_recent(stripped)

    # 5. 메시지가 매우 크면 → 다단계 요약
    if total_tokens > chunk_budget * 2:
        summary = await summarize_in_stages(...)
    else:
        summary = await summarize_with_fallback(...)

    # 6. 품질 검증 (safeguard)
    summary = await _safeguard_validate(...)
```

### 다단계 요약 (`summarize_in_stages`)

메시지가 너무 많으면 한 번에 요약할 수 없습니다. 이때 3-단계로 나눕니다:

```
[파트1: 턴 1-30] → 요약1
[파트2: 턴 31-60] → 요약2
[파트3: 턴 61-80] → 요약3
                      ↓ 병합
            [최종 요약]
```

```python
# 토큰 기준으로 균등 분할
splits = split_messages_by_token_share(messages, normalized_parts)

# 각 파트를 독립적으로 요약
for chunk in splits:
    partial = await summarize_with_fallback(chunk, ...)
    partial_summaries.append(partial)

# 부분 요약들을 하나로 합침
merged = await _llm_complete_with_retry(provider, merge_prompt)
```

### Safeguard 검증

요약이 잘 됐는지 자동으로 검증합니다:

1. **필수 섹션 존재**: `## Decisions`, `## Open TODOs`, `## Constraints`, `## Pending user asks`, `## Exact identifiers`
2. **순서 검증**: 섹션들이 올바른 순서로 나열되는지
3. **식별자 보존**: URL, 파일 경로 등이 요약에 포함되는지
4. **Ask 오버랩**: 마지막 사용자 요청이 요약에 반영되는지

검증 실패 시 **원본 메시지에서 다시 요약**합니다 (최대 3회):

```python
async def _safeguard_validate(...) -> str:
    for attempt in range(capped_retries + 1):
        ok, reasons = _audit_summary_quality(summary, ...)
        if ok:
            return summary
        # 다시 요약 (LLM에게 피드백 제공)
        summary = await _llm_complete_with_retry(...)
```

---

## 9. 프루닝

**파일**: `openclaw/session/pruning.py` (372줄)

컴팩션이 **요약으로 대체**하는 거라면, 프루닝은 **도구 결과를 줄이는** 것입니다.
**메모리 내에서만** 동작하고 디스크 파일은 건드리지 않습니다.

### 두 단계 전략

```
1단계 (Soft Trim): 큰 도구 결과의 앞/뒤만 남기고 중간 자르기
   원본: "line 1\nline 2\n... (10만 줄) ...\nline 100000"
   결과: "line 1\n...\n... [100000 chars, soft-trimmed] ...\n...\nline 100000"

2단계 (Hard Clear): 오래된 도구 결과를 완전히 제거
   원본: (도구 결과 전체)
   결과: "[Old tool result content cleared]"
```

### 비율 기반 게이팅

```python
ratio = total_chars / char_window
if ratio < soft_trim_ratio:  # 0.3
    return messages  # 아직 여유 있으니 프루닝 안 함
```

컨텍스트 사용률이 30% 미만이면 프루닝하지 않습니다.
50% 이상이면 하드 클리어까지 수행합니다.

### 보호 영역

- **부트스트랩 보호**: 첫 번째 user 메시지 이전은 절대 프루닝 안 함 (SOUL.md 등)
- **최근 메시지 보호**: 마지막 3개 assistant 메시지부터는 프루닝 안 함

### 이미지 프루닝

```python
def prune_processed_images(messages, keep_last_turns=2):
    """오래된 이미지를 텍스트 플레이스홀더로 교체"""
    # base64 이미지는 컨텍스트를 엄청나게 소비 (~8000자)
    # 모델이 이미 본 이미지는 더 이상 필요 없음
```

---

## 10. 세션 레인

**파일**: `openclaw/session/lanes.py` (120줄)

하나의 세션 안에서 **병렬 대화 스레드**를 관리합니다.

```python
class LaneManager:
    def __init__(self):
        self._lanes = {"main": Lane(id="main")}  # 기본 레인

    def create(self, name="") -> Lane:
        """새 레인 생성 (서브에이전트용)"""

    def merge_into_main(self, lane_id) -> str:
        """레인의 대화 요약을 메인 레인에 합침"""
```

서브에이전트가 별도 레인에서 작업하고, 완료되면 결과를 메인 레인에 합칩니다.

---

## 11. 메모리 플러시

**파일**: `openclaw/session/memory_flush.py` (100줄)

컴팩션 직전에 중요한 정보를 **영구 메모리 파일**에 저장합니다.

```python
FLUSH_USER_PROMPT = (
    "The session is approaching context compaction. "
    "Write any durable facts, decisions, or important context to "
    "memory/{date}.md before they are lost."
)
```

에이전트에게 "곧 오래된 대화가 요약될 거니까, 중요한 건 미리 메모해둬"라고 하는 겁니다.
`NO_REPLY`를 받으면 저장할 게 없다는 뜻입니다.

---

## 12. 컨텍스트 가드

**파일**: `openclaw/context/guard.py` (268줄)

컨텍스트 윈도우 사용량을 감시하고 조치를 결정합니다.

### 3단계 에스컬레이션

```python
class ContextAction(Enum):
    OK = auto()       # 여유 있음
    COMPACT = auto()  # 컴팩션 필요 (70% 이상)
    ERROR = auto()    # 오버플로 (100% 이상)
```

```python
def check(self, estimated_tokens) -> ContextStatus:
    total = int(estimated_tokens * SAFETY_MARGIN)  # 20% 안전 마진
    utilization = total / self.config.max_tokens

    if total >= self.config.max_tokens:
        return ContextStatus(action=ERROR, ...)
    if utilization >= 0.7:
        return ContextStatus(action=COMPACT, ...)
    return ContextStatus(action=OK, ...)
```

### 도구 결과 트렁케이션

```python
def _truncate_text_to_budget(text, max_chars):
    """끝부분에 중요한 패턴이 있으면 head+tail, 아니면 head만"""
    if _has_important_tail(text):
        # error, traceback, summary 등이 끝에 있으면 tail도 보존
        return text[:head_cut] + "[truncated]" + text[tail_start:]
    else:
        return text[:cut_point] + "[truncated]"
```

에러 메시지가 출력 끝에 있는 경우가 많으므로, tail을 보존하는 것이 중요합니다.

### enforce_budget (인플레이스 변환)

```python
def enforce_budget(self, messages):
    """메시지 리스트를 직접 수정해서 예산 내로 맞춤"""
    # Pass 1: 개별 도구 결과가 너무 크면 자르기
    # Pass 2: 그래도 초과하면 오래된 것부터 제거
```

`enforce_budget`은 메시지 리스트를 **인플레이스**로 수정합니다 (새 리스트를 만들지 않음).
이건 성능을 위한 선택입니다 — 대화 이력이 클 때 복사 비용이 큽니다.

---

## 13. 메모리 스토어

**파일**: `openclaw/memory/store.py` (243줄)

**SQLite + FTS5**로 에이전트의 장기 기억을 저장합니다.

### 스키마

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,      -- 원본 파일 경로
    line_start INTEGER NOT NULL,  -- 시작 줄 번호
    line_end INTEGER NOT NULL,    -- 끝 줄 번호
    text TEXT NOT NULL,           -- 청크 텍스트
    embedding BLOB,               -- float32 벡터 (바이너리)
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- FTS5: 전문 검색 인덱스 (BM25 지원)
CREATE VIRTUAL TABLE fts_search USING fts5(text, ...);

-- 임베딩 캐시 (같은 텍스트를 다시 임베딩하지 않기 위해)
CREATE TABLE embedding_cache (
    text_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);
```

### 벡터 저장

```python
embedding_blob = chunk.embedding.tobytes()  # numpy → bytes
```

numpy 배열을 바이트로 변환해서 SQLite BLOB에 저장합니다.
읽을 때는 `np.frombuffer(blob, dtype=np.float32)`로 복원합니다.

### 배치 코사인 유사도 캐시

```python
def get_all_embeddings_cached(self) -> tuple[list[int], np.ndarray] | None:
    """모든 임베딩을 하나의 행렬로 캐싱"""
    if self._emb_cache_matrix is not None:
        return (self._emb_cache_ids, self._emb_cache_matrix)
    matrix = np.vstack([emb for _, emb in all_embs])  # N×D 행렬
    ...
```

검색할 때마다 DB에서 읽으면 느리므로, 전체 임베딩을 메모리에 행렬로 캐싱합니다.
데이터가 변경되면 (`_invalidate_embedding_cache()`) 캐시를 무효화합니다.

---

## 14. 하이브리드 검색

**파일**: `openclaw/memory/search.py` (817줄)

이 파일은 **정보 검색(IR)** 시스템입니다. 두 가지 검색을 합칩니다:

### 검색 파이프라인

```
쿼리 → [1. 벡터 검색] → 코사인 유사도 상위 N개
     → [2. BM25 검색]  → 키워드 매칭 상위 N개
     → [3. 점수 합산]  → vector_weight×벡터 + text_weight×BM25
     → [4. MMR 재랭킹] → 다양성 확보
     → [5. 시간 감쇠]  → 오래된 결과 페널티
     → 최종 결과
```

### 1. 벡터 검색 (Batch Cosine Similarity)

```python
# 쿼리 임베딩과 전체 임베딩 행렬의 코사인 유사도를 한 번에 계산
similarities = matrix @ query_embedding / (safe_norms * query_norm)

# 상위 K개를 효율적으로 선택 (O(n) vs O(n log n))
top_indices = np.argpartition(similarities, -candidates)[-candidates:]
```

`np.argpartition`은 전체 정렬 대신 상위 K개만 찾는 알고리즘입니다.
데이터가 많을 때 `np.argsort`보다 빠릅니다.

### 2. BM25 검색

SQLite FTS5의 내장 BM25 랭킹을 사용합니다:

```python
rows = conn.execute(
    "SELECT rowid, rank FROM fts_search WHERE fts_search MATCH ? ORDER BY rank",
    (query,)
)
```

FTS5의 `rank`는 음수 (낮을수록 좋음)이므로, 0~1 스코어로 변환합니다:

```python
def bm25_rank_to_score(rank: float) -> float:
    return 1.0 / (1.0 + abs(rank))
```

### 3. 하이브리드 점수 합산

```python
final_score = 0.7 * vector_score + 0.3 * bm25_score
```

벡터는 **의미 유사도** (동의어, 비슷한 개념), BM25는 **키워드 매칭** (정확한 단어).
둘을 합치면 더 좋은 결과가 나옵니다.

### 4. MMR (Maximal Marginal Relevance)

검색 결과가 다 비슷한 내용이면 쓸모없습니다. MMR은 **다양성**을 보장합니다:

```python
mmr_score = λ × relevance - (1-λ) × max_similarity_to_selected
```

λ=0.7이면 관련성 70%, 다양성 30%의 비율입니다.
이미 선택된 결과와 너무 비슷한 후보는 페널티를 받습니다.

유사도는 Jaccard 유사도 (토큰 집합의 교집합/합집합)를 사용합니다:

```python
def _jaccard_similarity(a, b):
    tokens_a = set(re.findall(r"[a-z0-9_]+", a.lower()))
    tokens_b = set(re.findall(r"[a-z0-9_]+", b.lower()))
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
```

### 5. 시간 감쇠

```python
decay = exp(-λ × age_in_days)  # 30일 반감기
final_score *= decay
```

30일 전 정보는 점수가 절반으로, 60일 전은 1/4로 줄어듭니다.
단, `MEMORY.md` 같은 에버그린 파일은 감쇠하지 않습니다.

### 다국어 쿼리 확장

한국어, 중국어, 일본어 불용어(stop words)를 제거하고,
한국어 조사를 분리합니다:

```python
# "서버에서" → "서버" + "서버에서" (조사 "에서" 분리)
stem = _strip_korean_trailing_particle("서버에서")  # → "서버"
```

중국어는 바이그램으로 분해합니다:

```python
# "机器学习" → ["机", "器", "学", "习", "机器", "器学", "学习"]
```

---

## 15. 임베딩 프로바이더

**파일**: `openclaw/memory/embeddings.py` (116줄)

OpenAI 호환 `/v1/embeddings` 엔드포인트를 호출합니다.

### L2 정규화

```python
def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    sanitized = np.where(np.isfinite(vec), vec, 0.0)  # NaN/Inf → 0
    magnitude = np.linalg.norm(sanitized)
    return sanitized / magnitude  # 단위 벡터로 변환
```

L2 정규화하면 벡터의 크기가 1이 됩니다. 이때 코사인 유사도 = 내적이므로,
계산이 더 빨라집니다 (나눗셈 불필요).

### 임베딩 캐시

```python
@staticmethod
def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]
```

같은 텍스트를 다시 임베딩하지 않도록 해시를 키로 캐싱합니다.

---

## 16. 시스템 프롬프트 빌더

**파일**: `openclaw/prompt/builder.py` (314줄)

LLM에게 보내는 시스템 프롬프트를 **13개 섹션**으로 동적 조립합니다.

### 섹션 순서

1. **Identity**: "You are a personal assistant running inside OpenClaw."
2. **Tooling**: 사용 가능한 도구 목록
3. **Safety**: 안전 규칙 (자기 보존 금지, 감시 준수)
4. **Skills**: 스킬 목록 (nano-pdf, himalaya 등)
5. **Memory Recall**: 메모리 검색 지침
6. **Workspace**: 작업 디렉토리
7. **Project Context**: 부트스트랩 파일 (SOUL.md 등)
8. **Silent Replies**: `NO_REPLY` 규칙
9. **Heartbeats**: 하트비트 응답 규칙
10. **Date/Time**: 현재 날짜/시간
11. **Runtime**: OS, Python 버전, 모델명
12. **Reasoning Format**: `<think>...</think>` 형식 (thinking 활성 시)
13. **Compaction context**: 이전 컴팩션 요약 (있을 경우)

### 프롬프트 모드

```python
if prompt_mode == "none":      # 최소 (테스트용)
    return "You are a personal assistant..."
if prompt_mode == "minimal":   # 서브에이전트용 (Identity + Tooling + Safety만)
    return "\n\n".join(sections)
# "full": 모든 섹션 포함
```

---

## 17. 부트스트랩 파일

**파일**: `openclaw/prompt/bootstrap.py` (180줄)

워크스페이스에서 8종의 표준 파일을 읽어 시스템 프롬프트에 주입합니다:

```python
BOOTSTRAP_FILENAMES = [
    "AGENTS.md",     # 에이전트 지침
    "SOUL.md",       # 페르소나/톤
    "TOOLS.md",      # 도구 사용 가이드
    "IDENTITY.md",   # 정체성
    "USER.md",       # 사용자 정보
    "HEARTBEAT.md",  # 하트비트 설정
    "BOOTSTRAP.md",  # 초기화 지침
    "MEMORY.md",     # 영구 메모리
]
```

### 예산 관리

```python
# 파일당 최대 20,000자, 전체 최대 150,000자
config.max_chars_per_file = 20000
config.max_chars_total = 150000
```

초과 시 head 70% + tail 20%를 남기고 중간을 잘라냅니다:

```python
def _truncate_content(content, max_chars, head_ratio, tail_ratio):
    head_size = int(max_chars * 0.7)
    tail_size = int(max_chars * 0.2)
    return content[:head_size] + "... [truncated] ..." + content[-tail_size:]
```

---

## 18. 인젝션 방어

**파일**: `openclaw/prompt/sanitize.py` (406줄)

악의적 사용자가 프롬프트를 조작하는 것을 막습니다.

### 4계층 방어

**1. 유니코드 살균**

```python
def sanitize_text(text: str) -> str:
    """제어 문자(Cc), 포맷 문자(Cf), 특수 분리자 제거"""
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in {"Cc", "Cf"} and ch not in ("\n", "\r", "\t"):
            continue  # 이 문자를 건너뜀 (제거)
```

보이지 않는 유니코드 문자로 프롬프트 구조를 깨뜨리는 공격을 방어합니다.

**2. 의심 패턴 감지**

```python
_SUSPICIOUS_PATTERNS = [
    ("ignore_previous_instructions",
     re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)")),
    ("you_are_now",
     re.compile(r"you\s+are\s+now\s+(a|an)\s+")),
    ("system_tags",
     re.compile(r"</?system>")),
    ...
]
```

13가지 프롬프트 인젝션 패턴을 감지합니다. 감지되면 WARNING 로그를 남깁니다.

**3. 호모글리프 폴딩**

```python
_ANGLE_BRACKET_MAP = {
    0xFF1C: "<",   # 전각 < (＜) → 반각 <
    0x3008: "<",   # CJK 괄호 (〈) → <
    0x27E8: "<",   # 수학 괄호 (⟨) → <
    ...
}
```

공격자가 `＜system＞`처럼 유사한 유니코드 문자를 사용해서
`<system>` 태그를 위장하는 것을 방지합니다.

**4. 암호화 경계 마커**

```python
def wrap_external_content(content, source="unknown"):
    marker_id = secrets.token_hex(8)  # 예: "a3f2b7c4e1d9f083"
    start = f'<<<EXTERNAL_UNTRUSTED_CONTENT id="{marker_id}">>>'
    end = f'<<<END_EXTERNAL_UNTRUSTED_CONTENT id="{marker_id}">>>'
```

외부 콘텐츠(이메일, 웹 검색 결과)를 랜덤 ID가 포함된 마커로 감쌉니다.
공격자가 마커를 위조해서 일찍 닫는 것을 방지합니다.

---

## 19. 도구 레지스트리

**파일**: `openclaw/tools/registry.py` (628줄)

### 도구 등록과 실행

```python
class ToolRegistry:
    def register(self, definition, executor, group="custom"):
        self._tools[definition.name] = RegisteredTool(...)

    async def execute(self, name, arguments) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(content=f"Error: Unknown tool '{name}'", is_error=True)
        return await tool.executor(arguments)
```

### 스마트 트렁케이션

도구 결과가 너무 크면 잘라야 하는데, 단순히 앞부분만 남기면 에러 메시지를 놓칩니다:

```python
def truncate_tool_result(content, max_chars):
    if _has_important_tail(content):
        # 끝에 error, traceback, summary가 있으면 → head 70% + tail 30%
        return content[:head] + "[...omitted...]" + content[tail:]
    else:
        # 끝이 중요하지 않으면 → head만
        return content[:budget] + "[truncated]"
```

### 루프 감지 (4종)

에이전트가 같은 도구를 반복 호출하는 것을 감지합니다:

**1. generic_repeat**: 같은 도구+인자 10회 반복 → WARNING (차단 안 함)

**2. known_poll_no_progress**: 폴링 도구(process poll 등)가 같은 결과를 계속 반환
   - 10회 → WARNING
   - 20회 → CRITICAL (차단)

**3. ping_pong**: A→B→A→B→A 패턴으로 두 도구를 번갈아 호출
   - 10회 → WARNING
   - 20회 + 결과 변화 없음 → CRITICAL (차단)

**4. global_circuit_breaker**: 특정 도구가 30회 연속 같은 결과 → CRITICAL (차단)

```python
def record(self, name, args, result) -> str | None:
    """도구 호출을 기록하고 루프 여부 확인"""
    self.record_call(name, args)
    self.record_outcome(result)
    detection = self.detect(name, args)
    if detection.stuck:
        return detection.message  # WARNING 또는 CRITICAL 메시지
    return None
```

호출과 결과를 **해시**로 비교합니다:

```python
def _hash_call(name, args):
    stable = json.dumps({"name": name, "args": args}, sort_keys=True)
    return f"{name}:{sha256(stable)[:16]}"
```

`sort_keys=True`는 딕셔너리 키 순서가 달라도 같은 해시를 만들기 위함입니다.

---

## 20. 내장 도구들

**디렉토리**: `openclaw/tools/builtins/`

각 도구는 `DEFINITION` (스키마)과 `execute()` (실행 함수)를 가집니다.

### read.py — 파일 읽기

```python
async def execute(args, workspace=""):
    path = Path(file_path)
    if not path.is_absolute() and workspace:
        path = Path(workspace) / path  # 상대 경로 → 절대 경로

    lines = text.splitlines()
    selected = lines[start:end]

    # 줄 번호 포맷
    for i, line in enumerate(selected, start=start + 1):
        truncated = line[:2000]  # 한 줄 최대 2000자
        numbered.append(f"{i:>6}\t{truncated}")
```

### bash.py — 쉘 명령 실행

```python
async def execute(args, workspace=""):
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
```

`create_subprocess_shell`은 비동기 서브프로세스를 만듭니다.
`wait_for`로 타임아웃을 걸어서 무한 실행을 방지합니다.

### write.py — 파일 생성/덮어쓰기

```python
path.parent.mkdir(parents=True, exist_ok=True)  # 부모 디렉토리 자동 생성
path.write_text(content, encoding="utf-8")
```

### edit.py — 파일 편집 (찾아 바꾸기)

```python
count = content.count(old_string)
if count == 0:
    return ToolResult(content="Error: old_string not found", is_error=True)
if count > 1 and not replace_all:
    return ToolResult(content=f"Error: found {count} times. Use replace_all=true", is_error=True)
```

안전장치: `old_string`이 여러 번 나오면 `replace_all=true`를 명시적으로 요구합니다.

### pdf_tool.py — PDF 읽기

PyPDF2로 텍스트 추출. 한 번에 최대 20페이지까지.

### web_fetch.py — 웹 페이지 가져오기

httpx + BeautifulSoup으로 웹 페이지를 가져와서 readable text로 변환.

### hancom_tool.py — 한컴 오피스 파일 읽기

olefile로 HWP (한글), HWPX (한글 XML), 한쇼, 한셀 파일 텍스트 추출.

### memory_tool.py — 메모리 검색/저장

`memory_search`: 하이브리드 검색으로 관련 메모리 찾기
`memory_save`: 텍스트를 메모리 파일에 append

---

## 21. 서브에이전트

**파일**: `openclaw/subagent/spawn.py` (216줄)

메인 에이전트가 하위 에이전트를 생성해서 병렬 작업을 수행합니다.

### 깊이 제한

```python
DEFAULT_MAX_SPAWN_DEPTH = 1  # 서브에이전트의 서브에이전트는 불가

def can_spawn(self, parent_depth, parent_session_key):
    if parent_depth >= self.config.max_spawn_depth:
        return False, "Max spawn depth reached"
```

무한 재귀 방지를 위해 깊이를 제한합니다.

### 도구 정책

```python
def get_tools_for_depth(self, depth, all_tools):
    # 깊이 0 (메인): 모든 도구
    # 깊이 1 (리프): 세션 관리 도구 제외
    # 깊이 2+: spawn 도구까지 제거
```

서브에이전트는 메인보다 제한된 도구만 사용할 수 있습니다.

### 계단식 중지

```python
def cascade_stop(self, session_key):
    """부모가 중지되면 모든 자식도 중지"""
    for entry in self._entries.values():
        if entry.parent_session_key == session_key:
            entry.status = FAILED
            self.cascade_stop(entry.session_key)  # 재귀
```

---

## 22. Hook 시스템

**파일**: `openclaw/hooks/__init__.py` (100줄)

에이전트 이벤트에 셸 명령을 연결합니다.

```toml
# config.toml
[hooks]
pre_tool_call = "echo 'Tool: {tool_name}' >> /tmp/agent.log"
post_tool_call = "echo '{status} in {duration}s' >> /tmp/agent.log"
on_error = "curl -X POST https://slack.webhook/... -d '{error_message}'"
```

```python
class HookRunner:
    async def fire(self, event_name, **kwargs):
        command = command_template.format(**kwargs)  # 변수 치환
        proc = await asyncio.create_subprocess_shell(command, ...)
        await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
```

**Fire-and-forget**: 훅 실패는 로그만 남기고 에이전트를 중단하지 않습니다.

---

## 23. Cron 스케줄러

**파일**: `openclaw/cron/__init__.py` (210줄)

주기적 작업을 asyncio 백그라운드 태스크로 실행합니다.

```python
class CronScheduler:
    def register(self, name, callback, interval=60.0, one_shot=False):
        """60초마다 callback을 실행하는 태스크 등록"""

    async def _run_loop(self, task):
        while self._running:
            await asyncio.sleep(task.interval_seconds)
            await task.callback()  # 실행!
```

### 내장 하트비트

- **model_ping**: 모델 서버가 살아있는지 확인
- **memory_check**: SQLite DB가 정상인지 확인

---

## 24. 스킬 로더

**파일**: `openclaw/skills/loader.py` (205줄)

SKILL.md 파일에서 YAML frontmatter를 읽어서 스킬 목록을 만듭니다.

```yaml
---
name: nano-pdf
description: "PDF 편집 도구"
metadata:
  openclaw:
    requires:
      bins: ["nano-pdf"]  # 이 바이너리가 PATH에 있어야 활성화
---
```

### 번들 스킬

```python
def _bundled_skills_dir() -> Path:
    return Path(__file__).parent / "bundled"  # 패키지에 포함된 스킬
```

`Path(__file__).parent`는 현재 파일의 디렉토리입니다.
번들 스킬은 최저 우선순위로, 사용자 스킬이 같은 이름이면 덮어씁니다.

### 적격성 검사

```python
def _is_eligible(entry):
    # OS 필터 (darwin/linux/win32)
    # 필수 바이너리 존재 여부 (shutil.which)
    # 환경 변수 설정 여부
```

`shutil.which("nano-pdf")`는 `which nano-pdf`와 동일합니다.
바이너리가 PATH에 있으면 `True`.

---

## 25. 설정

**파일**: `openclaw/config.py` (202줄)

TOML 파일에서 설정을 로딩합니다.

### Pydantic 모델 계층

```python
class AppConfig(BaseModel):
    models: ModelsConfig       # LLM 모델 설정
    endpoints: EndpointsConfig # API 엔드포인트
    context: ContextConfig     # 컨텍스트 윈도우
    session: SessionConfig     # 세션 저장 경로
    memory: MemoryConfig       # 메모리 DB 설정
    pruning: PruningConfig     # 프루닝 규칙
    compaction: CompactionConfig  # 컴팩션 규칙
    bootstrap: BootstrapConfig # 부트스트랩 파일 제한
    skills: SkillsConfig       # 스킬 디렉토리
    hooks: HooksConfig         # Hook 명령어
```

Pydantic 모델은 **타입 검증**을 자동으로 해줍니다.
TOML에서 `max_tokens = "abc"` 같은 잘못된 값이 오면 에러를 냅니다.

### 설정 해상 순서

```python
def load_config(path=None) -> AppConfig:
    candidates = [
        Path(path),                          # 1. 명시적 경로
        Path(os.environ["OPENCLAW_PY_CONFIG"]),  # 2. 환경 변수
        Path("config.toml"),                 # 3. 현재 디렉토리
        Path("~/.openclaw-py/config.toml"),  # 4. 홈 디렉토리
    ]
```

---

## 26. REPL과 Python API

**파일**: `openclaw/repl.py` (517줄)

### Agent 클래스 (고수준 API)

```python
from openclaw.repl import Agent

agent = Agent.from_config()           # config.toml 자동 로딩
result = await agent.run("서버 상태 확인해줘")
print(result.text)                    # 답변
print(result.tool_calls_count)        # 도구 호출 횟수
```

### 커스텀 도구 등록

```python
@agent.tool("weather", description="날씨 조회")
async def get_weather(args):
    city = args["city"]
    return ToolResult(content=f"{city}: 맑음 25도")
```

### 모듈 조립

`_build_context()`에서 모든 모듈이 연결됩니다:

```python
def _build_context(self, session_id, thinking):
    # 부트스트랩 파일 로딩
    bootstrap_ctx = load_bootstrap_files(self.workspace, self.config.bootstrap)

    # 스킬 로딩
    skills_snapshot = load_skills(skill_dirs)
    skills_prompt = build_skills_prompt(skills_snapshot)

    # 메모리 도구 연결
    self._wire_memory_tools(searcher)

    return AgentContext(
        provider=self.provider,
        session=session,
        tool_registry=self.tool_registry,
        context_guard=ContextGuard(self.config.context),
        failover=FailoverManager(...),
        ...
    )
```

---

## 27. 핵심 개념 용어집

| 용어 | 설명 |
|------|------|
| **에이전트 루프** | 모델 호출 → 도구 실행 → 반복. 도구 호출이 없을 때까지 계속 |
| **컨텍스트 윈도우** | LLM이 한 번에 처리할 수 있는 최대 토큰 수 (예: 32K) |
| **컴팩션** | 오래된 대화를 LLM으로 요약해서 컨텍스트 절약 |
| **프루닝** | 도구 결과를 자르거나 제거해서 컨텍스트 절약 (메모리 내) |
| **네이티브 도구 호출** | OpenAI API의 `tools` 파라미터로 도구를 전달하는 방식 |
| **프롬프트 기반 도구 호출** | 시스템 프롬프트에 도구 설명을 넣고 `<tool_call>JSON</tool_call>`으로 파싱 |
| **페일오버** | API 에러 시 다른 모델/키로 자동 전환 |
| **부트스트랩 파일** | 워크스페이스의 SOUL.md, AGENTS.md 등 — 에이전트 초기 설정 |
| **세션** | JSONL 파일로 저장되는 대화 이력 |
| **메모리** | SQLite + 벡터 임베딩으로 저장되는 장기 기억 |
| **BM25** | 키워드 기반 검색 알고리즘 (TF-IDF의 개선판) |
| **MMR** | Maximal Marginal Relevance — 검색 결과의 다양성 확보 |
| **하트비트** | 주기적으로 모델 서버 상태를 확인하는 ping |
| **레인** | 세션 내 병렬 대화 스레드 (서브에이전트용) |
| **JSONL** | 한 줄에 하나의 JSON — append-only 로그에 적합 |
| **fcntl** | Unix 파일 잠금 — 동시 접근 방지 |
| **L2 정규화** | 벡터를 단위 벡터로 변환 (크기=1) |
| **호모글리프** | 시각적으로 비슷한 유니코드 문자 (＜ vs <) |
| **시간 감쇠** | 오래된 검색 결과의 점수를 낮추는 지수 함수 |

---

## 부록: Python 문법 빠른 참조

주피터 노트북에서 짧은 코드만 써봤다면 낯설 수 있는 문법들:

```python
# 1. async/await — 비동기 프로그래밍
async def fetch():          # 비동기 함수 선언
    result = await api()    # 비동기 호출 (네트워크 대기)
    async for chunk in stream():  # 비동기 이터레이터

# 2. @dataclass — 데이터 클래스 (자동 __init__ 생성)
@dataclass
class Point:
    x: float
    y: float
# Point(1.0, 2.0)으로 생성 가능

# 3. Union 타입
str | None               # 문자열 또는 None
list[int | str]           # int 또는 str의 리스트

# 4. Field(default_factory=...)
items: list = Field(default_factory=list)  # 각 인스턴스마다 새 리스트 생성
# items: list = [] 하면 모든 인스턴스가 같은 리스트를 공유! (버그)

# 5. TYPE_CHECKING — 순환 임포트 방지
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openclaw.config import AppConfig  # 타입 체크 시에만 임포트

# 6. Path — 파일 경로 객체
from pathlib import Path
path = Path("~/.config") / "app" / "config.toml"
path.expanduser()        # ~ → /Users/username
path.exists()            # 존재 여부
path.read_text()         # 파일 읽기

# 7. 컨텍스트 매니저
with open("file.txt") as f:  # __enter__/__exit__ 자동 호출
    ...                       # 에러가 나도 파일이 닫힘
```
