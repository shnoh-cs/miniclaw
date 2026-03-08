# openclaw-py

OpenClaw Agent Harness의 Python 최소 포트. 모든 "지능(intelligence)" 기능을 100% 유지하면서 채널 통합(WhatsApp, Telegram, Discord 등)은 제거한 버전.

## 주요 특징

- **OpenAI-compatible API**: vLLM, OpenRouter 등 OpenAI 호환 엔드포인트와 동작
- **Native/Prompt 듀얼 툴 콜링**: 네이티브 function calling 지원 모델은 네이티브 모드, 미지원 모델은 `<tool_call>` XML 프롬프트 기반 모드로 자동 전환
- **세션 관리**: JSONL append-only 파일 기반, 다중 세션 지원
- **컨텍스트 윈도우 관리**: 토큰 예산, 자동 컴팩션, 프루닝 (cache-TTL)
- **하이브리드 메모리 검색**: BM25 (FTS5) + 벡터 코사인 유사도 + MMR 다양성 + 시간 감쇠
- **도구 루프 감지**: 4가지 감지기 (generic_repeat, known_poll_no_progress, ping_pong, global_circuit_breaker)
- **프롬프트 인젝션 방어**: 유니코드 제어 문자 제거, `<untrusted-text>` 래핑, HTML 이스케이핑
- **에러 분류 & 페일오버**: auth/billing/rate_limit/timeout/context_overflow 분류, 지수 백오프
- **스킬 시스템**: SKILL.md YAML frontmatter, OS/바이너리 게이팅
- **부트스트랩 파일**: AGENTS.md, SOUL.md, TOOLS.md 등 8종 자동 로딩
- **Thinking 레벨**: off/minimal/low/medium/high/xhigh/adaptive + 자동 폴백 체인

## 빠른 시작

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 설치
pip install -e .

# 설정 파일 작성 (config.example.toml 참고)
cp config.example.toml config.toml
# config.toml에서 API 키와 엔드포인트 수정

# 대화형 REPL 실행
openclaw-py
```

## Python API 사용법

```python
import asyncio
from openclaw.repl import Agent
from openclaw.config import load_config

config = load_config()
agent = Agent(config)

# 기본 실행
result = asyncio.run(agent.run("Hello!"))
print(result.text)

# 스트리밍
async def main():
    async for chunk in agent.stream("Count from 1 to 5"):
        print(chunk, end="")

asyncio.run(main())

# 커스텀 도구 등록
@agent.tool("get_weather", description="Get weather for a city", parameters=[
    {"name": "city", "type": "string", "description": "City name", "required": True},
])
def get_weather(city: str) -> str:
    return f"Sunny, 22°C in {city}"
```

## 프로젝트 구조

```
openclaw/
├── agent/
│   ├── types.py          # 핵심 타입 정의 (AgentMessage, ToolDefinition, RunResult 등)
│   └── loop.py           # 에이전트 루프 (run → attempt → stream → tool dispatch)
├── model/
│   ├── provider.py       # OpenAI 호환 API 클라이언트 (스트리밍, 툴 콜링)
│   ├── failover.py       # 에러 분류 & 페일오버 매니저
│   └── thinking.py       # Thinking 레벨 해석
├── session/
│   ├── manager.py        # JSONL 세션 관리 (파일 잠금, 복구)
│   ├── compaction.py     # 다단계 요약 컴팩션
│   ├── pruning.py        # Cache-TTL 기반 프루닝
│   └── memory_flush.py   # 컴팩션 전 메모리 플러시
├── context/
│   └── guard.py          # 컨텍스트 윈도우 가드 (토큰 예산)
├── memory/
│   ├── store.py          # SQLite + FTS5 메모리 스토어
│   ├── search.py         # 하이브리드 검색 (BM25 + 벡터 + MMR + 시간 감쇠)
│   └── embeddings.py     # 임베딩 프로바이더
├── prompt/
│   ├── builder.py        # 11-섹션 시스템 프롬프트 조립
│   ├── bootstrap.py      # 부트스트랩 파일 로딩
│   └── sanitize.py       # 프롬프트 인젝션 방어
├── skills/
│   └── loader.py         # 스킬 디스커버리 & 로딩
├── tools/
│   ├── registry.py       # 도구 레지스트리 + 루프 감지 + 결과 트렁케이션
│   └── builtins/         # 10개 내장 도구
│       ├── read.py       # 파일 읽기
│       ├── write.py      # 파일 쓰기
│       ├── edit.py       # 파일 편집 (find & replace)
│       ├── apply_patch.py # Unified diff 패치 적용
│       ├── bash.py       # 셸 명령 실행
│       ├── process_tool.py # 백그라운드 프로세스 관리
│       ├── web_fetch.py  # URL 페치 (HTML→텍스트 변환)
│       ├── pdf_tool.py   # PDF 텍스트 추출
│       ├── image_tool.py # 이미지 분석 (비전 모델 필요)
│       └── memory_tool.py # 메모리 저장/검색
├── subagent/
│   └── spawn.py          # 서브에이전트 레지스트리 & 깊이 제한
├── config.py             # TOML 설정 로딩
├── repl.py               # 대화형 REPL & Agent Python API
└── __init__.py
```

## 설정

`config.toml` 예시 (사내 vLLM 서버):

```toml
[models]
default = "gpt-oss-120b"
compaction = "gpt-oss-120b"
embedding = "bge-m3"

[endpoints.llm]
base_url = "http://vllm-server:8000/v1"
api_key = "dummy"

[endpoints.embedding]
base_url = "http://vllm-server:8000/v1"
api_key = "dummy"

[models.options."gpt-oss-120b"]
tool_mode = "auto"      # auto | native | prompt
max_tokens = 4096
thinking = "off"
```

`config.toml` 예시 (OpenRouter 테스트):

```toml
[models]
default = "anthropic/claude-sonnet-4.6"

[endpoints.llm]
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-or-v1-..."
```

---

## 테스트 결과

> 테스트 환경: macOS Darwin 24.5.0 (Apple Silicon), Python 3.11.12
> 모델: `anthropic/claude-sonnet-4.6` via OpenRouter
> 날짜: 2026-03-08

### Test 1: 단순 질문 (도구 미사용)

**입력:** `What is 2+2? Answer in one word.`
**결과:**
- 응답: `Four.`
- 도구 호출: 0회
- 에러: 없음
- 토큰: input=2262, output=5
- **PASS** — 도구 없이 순수 LLM 응답 정상 동작

### Test 2: 단일 파일 읽기 (Read 도구)

**입력:** `Read /Users/sh/Desktop/openclaw-py/pyproject.toml and list the project name and version.`
**결과:**
- 응답: 프로젝트 이름 `openclaw-py`, 버전 `0.1.0` 정확히 추출
- 도구 호출: 1회 (Read)
- 에러: 없음
- **PASS** — Read 도구로 파일 읽고 내용 분석 정상

### Test 3: 다중 파일 읽기

**입력:** `Read both pyproject.toml and config.example.toml. Compare their contents briefly.`
**결과:**
- 응답: 두 파일의 역할과 내용 차이를 정확히 비교 설명
- 도구 호출: 2회 (Read × 2)
- 에러: 없음
- **PASS** — 한 번의 요청에서 여러 파일을 순차적으로 읽고 종합 분석

### Test 4: Bash 명령 실행

**입력:** `Run "uname -a" and "python3 --version" and tell me the OS and Python version.`
**결과:**
- 응답: macOS Darwin 24.5.0 ARM64, Python 3.11.12 정확히 보고
- 도구 호출: 2회 (Bash × 2)
- 에러: 없음
- **PASS** — 셸 명령 실행 및 결과 해석 정상

### Test 5: 파일 쓰기 + 읽기 검증

**입력:** `Create a file at /tmp/openclaw-test.txt with the content "Hello from openclaw-py!" then read it back to verify.`
**결과:**
- Write 도구로 파일 생성 후 Read 도구로 내용 일치 확인
- 도구 호출: 2회 (Write → Read)
- 에러: 없음
- **PASS** — 쓰기/읽기 라운드트립 검증 완료

### Test 6: 파일 편집 (Find & Replace) + 다중 턴 세션

**입력 (턴 1):** `Write a file /tmp/openclaw-edit-test.txt with content: name = "old_name" ...`
**입력 (턴 2):** `Now edit the file and change "old_name" to "new_name", then read it to confirm.`
**결과:**
- 턴 1: Write로 파일 생성 (1회 도구)
- 턴 2: Edit으로 문자열 교체 후 Read로 확인 (2회 도구)
- 같은 세션 ID(`t6`)로 컨텍스트 유지
- 에러: 없음
- **PASS** — 다중 턴 세션에서 편집 + 검증 정상

### Test 7: 웹 페치

**입력:** `Fetch the URL https://httpbin.org/json and tell me what data it contains.`
**결과:**
- 응답: httpbin.org의 JSON 응답(슬라이드쇼 데이터) 정확히 파싱 및 설명
- 도구 호출: 1회 (WebFetch)
- 에러: 없음
- **PASS** — HTTP 요청 및 응답 분석 정상

### Test 8: 커스텀 도구 (`@agent.tool` 데코레이터)

**코드:**
```python
@agent.tool("get_weather", description="Get weather for a city", parameters=[
    {"name": "city", "type": "string", "description": "City name", "required": True},
])
def get_weather(city: str) -> str:
    weather_db = {"Seoul": "Sunny, 22°C", "Tokyo": "Cloudy, 18°C"}
    return weather_db.get(city, f"No data for {city}")
```
**입력:** `What is the weather in Seoul?`
**결과:**
- 응답: `Sunny, 22°C` — 커스텀 도구의 반환값을 정확히 활용
- 도구 호출: 1회 (get_weather)
- 에러: 없음
- **PASS** — `@agent.tool` 데코레이터를 통한 커스텀 도구 등록 및 호출 정상

### Test 9: 다중 턴 대화 (세션 지속성)

**입력 (턴 1):** `My name is Alice and I am working on a project called Project Falcon.`
**입력 (턴 2):** `What is my name and what project am I working on?`
**결과:**
- 턴 2에서 `Alice`와 `Project Falcon`을 정확히 기억하여 응답
- 같은 세션 ID(`t9-session`) 사용
- **PASS** — 세션 내 대화 컨텍스트 유지 정상

### Test 10: 복합 코드 분석

**입력:** `Read loop.py and tell me: 1) max turns, 2) context overflow handling, 3) main functions`
**결과:**
- MAX_TURNS_PER_RUN = 50 정확히 식별
- ContextAction.COMPACT → ERROR 에스컬레이션 과정 정확히 설명
- run(), _attempt_loop(), _parse_thinking(), stream_run() 함수 목록 정확
- 도구 호출: 1회 (Read)
- **PASS** — 복잡한 코드 분석 및 다중 질문 응답 정상

### Test 11: 스트리밍 API

**코드:**
```python
async for chunk in agent.stream("Count from 1 to 5"):
    chunks.append(chunk)
```
**결과:**
- 2개 청크 수신
- 전체 텍스트: `1\n2\n3\n4\n5`
- **PASS** — `agent.stream()` 비동기 이터레이터 정상 동작

### Test 12: 에러 처리 (존재하지 않는 파일)

**입력:** `Read the file /tmp/this-file-does-not-exist-12345.txt`
**결과:**
- 에이전트가 에러를 우아하게 처리하고 파일 부재를 설명
- 도구 호출: 1회 (Read — 에러 반환)
- `result.error`: None (에이전트 레벨 에러 아님, 도구 에러를 해석하여 사용자에게 설명)
- **PASS** — 도구 에러 처리 및 사용자 친화적 응답

### Test 13: 패치 적용 (Apply Patch + Edit 폴백)

**입력:** `Apply a patch to change print("hello") to print("hello world")`
**결과:**
- apply_patch 시도 후 edit 도구로 폴백하여 성공
- 도구 호출: 4회 (Write → ApplyPatch → Edit → Read)
- 에러: 없음
- **PASS** — 패치 실패 시 자동으로 대안 도구 사용

### Test 14: 연쇄 추론 (다중 도구 순차 실행)

**입력:** `1) Run date, 2) Write log file with the date, 3) Read back to confirm`
**결과:**
- 날짜 획득 → 파일 생성 → 검증까지 3단계 자동 수행
- 도구 호출: 3회 (Bash → Write → Read)
- 에러: 없음
- **PASS** — 도구 결과를 다음 도구 입력으로 활용하는 연쇄 추론 정상

### Test 15: 컨텍스트 가드 (토큰 예산 관리)

직접 `ContextGuard` 유닛 테스트:

| 토큰 수 | Utilization | Action |
|---------|-------------|--------|
| 1,000 | 0.04 | OK |
| 22,937 (70% threshold) | 0.84 | COMPACT |
| 33,768 (overflow) | 1.24 | ERROR |

- Tool result max chars: 39,321
- **PASS** — OK → COMPACT → ERROR 에스컬레이션 정상, 안전 마진(1.2x) 반영

### Test 16: 도구 루프 감지

동일한 Read 호출을 6회 반복 시뮬레이션:
- 호출 1-3: 정상
- 호출 4-6: `WARNING: Tool 'Read' polling with no progress (4 identical results).`
- **PASS** — `known_poll_no_progress` 감지기가 4회차부터 경고 발생

### Test 17: 도구 결과 트렁케이션

| 입력 크기 | 최대 허용 | 결과 크기 | 트렁케이션 |
|----------|----------|----------|-----------|
| 11 chars | 1,000 | 11 | 없음 |
| 50,000 chars | 10,000 | 10,053 | TRUNCATED 표시 |

- **PASS** — head/tail (70/30) 비율로 트렁케이션, 초과 시 명확한 표시

### Test 18: 프롬프트 인젝션 방어

| 테스트 | 입력 | 결과 |
|--------|------|------|
| 유니코드 제어 문자 제거 | `Hello\x00World\x01Test\x0b` | `HelloWorldTest` |
| Untrusted 래핑 | `<script>alert(1)</script>` | `<untrusted-text>` 태그로 래핑 |
| HTML 이스케이핑 | `<script>` | `&lt;script&gt;`로 변환 |

- **PASS** — 3단계 방어(제어 문자 제거 → 태그 래핑 → HTML 이스케이핑) 모두 동작

### Test 19: Thinking 레벨 폴백 체인

```
off      → (self)
minimal  → off
low      → minimal
medium   → low
high     → medium
xhigh    → high
adaptive → off
```

- `from_str("invalid")` → `off` (안전한 기본값)
- **PASS** — 모든 레벨의 폴백 체인 및 문자열 파싱 정상

### Test 20: 에러 분류 (페일오버)

| 에러 메시지 | 분류 결과 |
|------------|----------|
| `Rate limit exceeded` | `rate_limit` |
| `Request timed out` | `timeout` |
| `billing quota reached` | `billing` |
| `maximum context length exceeded` | `context_overflow` |
| `random server error 500` | `unknown` |

- **PASS** — 핵심 에러 패턴(rate_limit, timeout, billing, context_overflow) 정확히 분류

### Test 21: 부트스트랩 파일 로딩

- 프로젝트 디렉토리에 AGENTS.md/SOUL.md 등이 없으면 빈 목록 반환 (정상)
- `has_soul: False`
- **PASS** — 파일 부재 시 안전하게 빈 컨텍스트 반환

### Test 22: 세션 JSONL 영속성

```
Written 2 messages → Loaded 2 messages
  [user] Hello
  [assistant] Hi there!
```

- 임시 디렉토리에 쓰기 후 새 SessionManager 인스턴스로 다시 로딩
- **PASS** — JSONL append-only 세션 저장/복원 정상

### Test 23: 프롬프트 기반 도구 호출 파싱

| 케이스 | 결과 |
|--------|------|
| `<tool_call>{"name":"Read",...}</tool_call>` | 1개 파싱 성공 |
| 연속 2개 `<tool_call>` | 2개 모두 파싱 성공 |
| 닫는 태그 없음 (스트리밍 컷오프) | 1개 파싱 성공 |

- **PASS** — 네이티브 function calling 미지원 모델에서 사용할 프롬프트 기반 파싱 정상

### Test 24: 스트리밍 콜백 시스템

**입력:** `Run echo hello and tell me what it outputs.`
**콜백 기록:**
- 스트림 청크: 6개 수신
- 도구 이벤트: `[('start', 'bash'), ('end', 'bash', True)]`
- 최종 텍스트: `echo hello` 출력 결과 설명
- **PASS** — `on_stream`, `on_tool_start`, `on_tool_end` 콜백 모두 정상 호출

### 테스트 요약

| 카테고리 | 테스트 수 | 통과 | 실패 |
|----------|----------|------|------|
| LLM 기본 응답 | 1 | 1 | 0 |
| 내장 도구 (Read/Write/Edit/Bash/WebFetch/Patch) | 8 | 8 | 0 |
| 커스텀 도구 (@agent.tool) | 1 | 1 | 0 |
| 다중 턴 세션 | 2 | 2 | 0 |
| 스트리밍 API | 2 | 2 | 0 |
| 에러 처리 | 2 | 2 | 0 |
| 컨텍스트 관리 (Guard/Pruning/Truncation) | 2 | 2 | 0 |
| 보안 (인젝션 방어/Sanitize) | 1 | 1 | 0 |
| 인프라 (Session/Failover/Thinking/Bootstrap) | 4 | 4 | 0 |
| 프롬프트 기반 도구 파싱 | 1 | 1 | 0 |
| **합계** | **24** | **24** | **0** |

---

## gpt-oss-120b 테스트 결과

> 테스트 환경: macOS Darwin 24.5.0 (Apple Silicon), Python 3.11.12
> 모델: `openai/gpt-oss-120b` (117B MoE, OpenAI 오픈웨이트) via OpenRouter
> 날짜: 2026-03-08

### Test 1: 단순 질문 (도구 미사용)

**입력:** `What is 2+2? Answer in one word only.`
**출력:**
```
Text: Four
Tool calls: 0
Error: None
Time: 1.6s
Usage: in=2262, out=3
```
- **PASS** — 도구 없이 즉답, 1.6초

### Test 2: 단일 파일 읽기 (Read 도구)

**입력:** `Read pyproject.toml and tell me the project name and version.`
**출력:**
```
Text: **Project Name:** `openclaw-py`
      **Version:** `0.1.0`
Tool calls: 1
Error: None
Time: 4.2s
```
- **PASS** — Read 도구로 파일 읽고 정확히 추출

### Test 3: Bash 명령 실행

**입력:** `Run the command "echo hello world" and tell me its output.`
**출력:**
```
Text: The command printed:
      hello world
Tool calls: 1
Error: None
Time: 2.8s
```
- **PASS** — Bash 도구 실행 정상

### Test 4: 파일 쓰기 + 읽기 검증

**입력:** `Write a file /tmp/oss-test.txt with "gpt-oss-120b test successful" then read it back.`
**출력:**
```
Text: The file was created and contains the expected text:
      gpt-oss-120b test successful
Tool calls: 2
Error: None
Time: 4.1s
```
- **PASS** — Write → Read 라운드트립 정상

### Test 5: 다중 파일 읽기 + 분석

**입력:** `Read pyproject.toml and config.example.toml. List the dependencies.`
**출력:**
```
Text: **Dependencies listed in `pyproject.toml`**
      - openai>=1.30.0
      - tiktoken>=0.7.0
      - numpy>=1.26.0
      - pydantic>=2.7.0
      - httpx>=0.27.0
      - rich>=13.7.0
      - beautifulsoup4>=4.12.0
      - PyPDF2>=3.0.0
      - PyYAML>=6.0.0
Tool calls: 2
Error: None
Time: 4.3s
```
- **PASS** — 2개 파일 순차 읽기 + 의존성 정확히 나열

### Test 6: 파일 편집 (Edit)

**입력:** `Write "name = old_value", edit to "new_value", read to confirm.`
**출력:**
```
Text: The file has been successfully created, edited, and verified.
      Contents: name = new_value
Tool calls: 3
Error: None
Time: 4.7s
```
- **PASS** — Write → Edit → Read 3단계 정상

### Test 7: 연쇄 추론 (4단계 도구 체이닝)

**입력:** `Run date, write log file with the date, read it back.`
**출력:**
```
Text: Here is the content of /tmp/oss-log.txt:
      Log entry: 2026-03-08 12:26:10 KST
Tool calls: 4
Error: None
Time: 7.8s
```
- **PASS** — Bash(date) → Write → Read까지 4회 도구 연쇄, 날짜 정확히 포함

### Test 8: 웹 페치

**입력:** `Fetch https://httpbin.org/json and tell me what data it contains.`
**출력:**
```
Text: {
        "slideshow": {
          "author": "Yours Truly",
          "title": "Sample Slide Show",
          "slides": [...]
        }
      }
Tool calls: 1
Error: None
Time: 7.3s
```
- **PASS** — WebFetch 도구로 JSON 데이터 정확히 파싱

### Test 9: 코드 분석

**입력:** `Read loop.py, tell me MAX_TURNS_PER_RUN and list the main functions.`
**출력:**
```
Text: MAX_TURNS_PER_RUN = 50
      Main functions: run(), _attempt_loop(), _parse_thinking(), stream_run()
Tool calls: 2
Error: None
Time: 8.1s
```
- **PASS** — 상수값 정확, 함수 목록 정확

### Test 10: 에러 처리 (존재하지 않는 파일)

**입력:** `Read /tmp/this-does-not-exist-99999.txt`
**출력:**
```
Text: The file /tmp/this-does-not-exist-99999.txt does not exist,
      so there's nothing to read.
Tool calls: 1
Error: None
Time: 3.2s
```
- **PASS** — 도구 에러를 해석하여 사용자 친화적 응답

### Test 11: 스트리밍 API

**코드:** `async for chunk in agent.stream("Count from 1 to 5")`
**출력:**
```
Chunks: 9
Text: 1
      2
      3
      4
      5
Time: 1.8s
```
- **PASS** — 스트리밍 청크 9개 수신, 내용 정확

### Test 12: 커스텀 도구 (@agent.tool)

**입력:** `What is the weather in Seoul?` (커스텀 get_weather 도구 등록)
**출력:**
```
Text: The current weather in Seoul is sunny with a temperature around 22 °C.
Tool calls: 1
Error: None
Time: 5.4s
Usage: in=2503, out=93
```
- **PASS** — @agent.tool 데코레이터 커스텀 도구 호출 정상

### Test 13: 다중 턴 대화 (세션 지속성)

**턴 1 입력:** `My name is Alice and I work on Project Falcon.`
**턴 1 출력:**
```
Text: Got it! I've noted that your name is Alice and you're working on
      Project Falcon. Let me know how I can help.
Tool calls: 1
Time: 4.5s
```
**턴 2 입력:** `What is my name and what project am I working on?`
**턴 2 출력:**
```
Text: Your name is Alice, and you're working on Project Falcon.
Tool calls: 1
Time: 4.6s
```
- **PASS** — 세션 내 컨텍스트 정확히 유지

### Test 14: 파일 편집 (Apply Patch / Edit)

**입력:** `Edit /tmp/oss-patch.py to change print("hello") to print("hello world"), read to confirm.`
**출력:**
```
Text: Updated /tmp/oss-patch.py content:
      def greet():
          print("hello world")
Tool calls: 2
Error: None
Time: 3.5s
```
- **PASS** — Edit 도구로 정확히 수정 후 확인

### Test 15: 컨텍스트 가드 (토큰 예산 관리)

```
1000 tokens:  OK      (util=0.04)
threshold:    COMPACT (util=0.84)
overflow:     ERROR   (util=1.24)
Tool result max chars: 39321
```
- **PASS**

### Test 16: 도구 루프 감지

```
Call 1: ok
Call 2: ok
Call 3: ok
Call 4: WARNING - Tool 'Read' polling with no progress (4 identical results).
Call 5: WARNING
Call 6: WARNING
```
- **PASS** — 4회차부터 경고 발생

### Test 17: 도구 결과 트렁케이션

```
Small (5 chars):              truncated=False
Large (50000 chars, max=10000): len=10053, has_notice=True
```
- **PASS**

### Test 18: 프롬프트 인젝션 방어

```
Control chars: "HelloWorld" (제거됨)
HTML escaped: <script> → &lt;script&gt;
Untrusted wrapping: <untrusted-text> 태그 적용
```
- **PASS**

### Test 19: Thinking 레벨 폴백 체인

```
off->off | minimal->off | low->minimal | medium->low | high->medium | xhigh->high | adaptive->off
from_str("invalid"): off
```
- **PASS**

### Test 20: 에러 분류 (페일오버)

```
"Rate limit exceeded"                 → rate_limit       [PASS]
"Request timed out"                   → timeout          [PASS]
"maximum context length exceeded"     → context_overflow  [PASS]
```
- **PASS**

### Test 21: 부트스트랩 파일 로딩

```
Files: [], has_soul=False
```
- **PASS** — 파일 부재 시 빈 컨텍스트 반환

### Test 22: 세션 JSONL 영속성

```
Written: 2 → Loaded: 2
  [user] Hello
  [assistant] Hi!
```
- **PASS**

### Test 23: 프롬프트 기반 도구 호출 파싱

```
Single <tool_call>:  1 call, name=Read
Multi <tool_call>:   2 calls, names=[Bash, Read]
No closing tag:      1 call (streaming cutoff 처리)
```
- **PASS**

### Test 24: 스트리밍 콜백 시스템

**입력:** `Run echo test123 and tell me the output.`
**출력:**
```
Stream chunks: 10
Tool events: [('start', 'bash'), ('end', 'bash', True)]
Text: The command printed: test123
Time: 2.5s
```
- **PASS** — on_stream, on_tool_start, on_tool_end 콜백 모두 정상

### gpt-oss-120b 테스트 요약

| # | 테스트 | 도구 호출 | 시간 | 결과 |
|---|--------|----------|------|------|
| 1 | 단순 질문 | 0 | 1.6s | PASS |
| 2 | 파일 읽기 | 1 | 4.2s | PASS |
| 3 | Bash 명령 | 1 | 2.8s | PASS |
| 4 | 쓰기+읽기 | 2 | 4.1s | PASS |
| 5 | 다중 파일 분석 | 2 | 4.3s | PASS |
| 6 | 파일 편집 | 3 | 4.7s | PASS |
| 7 | 연쇄 추론 (4단계) | 4 | 7.8s | PASS |
| 8 | 웹 페치 | 1 | 7.3s | PASS |
| 9 | 코드 분석 | 2 | 8.1s | PASS |
| 10 | 에러 처리 | 1 | 3.2s | PASS |
| 11 | 스트리밍 API | - | 1.8s | PASS |
| 12 | 커스텀 도구 | 1 | 5.4s | PASS |
| 13 | 다중 턴 세션 | 2 | 9.1s | PASS |
| 14 | 파일 편집 (패치) | 2 | 3.5s | PASS |
| 15 | 컨텍스트 가드 | - | - | PASS |
| 16 | 루프 감지 | - | - | PASS |
| 17 | 결과 트렁케이션 | - | - | PASS |
| 18 | 인젝션 방어 | - | - | PASS |
| 19 | Thinking 폴백 | - | - | PASS |
| 20 | 에러 분류 | - | - | PASS |
| 21 | 부트스트랩 로딩 | - | - | PASS |
| 22 | 세션 영속성 | - | - | PASS |
| 23 | 프롬프트 파싱 | - | - | PASS |
| 24 | 스트리밍 콜백 | 1 | 2.5s | PASS |
| | **합계** | | | **24/24** |

### 모델 비교: Claude Sonnet 4.6 vs gpt-oss-120b

| 항목 | Claude Sonnet 4.6 | gpt-oss-120b |
|------|-------------------|--------------|
| 전체 통과율 | 24/24 (100%) | 24/24 (100%) |
| 네이티브 tool calling | 지원 | 지원 |
| 응답 품질 | 매우 상세 (마크다운 테이블, 이모지) | 간결하고 정확 |
| 평균 응답 시간 (도구 1회) | ~3-5s | ~3-5s |
| 연쇄 추론 (3-4단계) | 정상 | 정상 |
| 다중 턴 기억 | 정상 | 정상 |
| 에러 해석 능력 | 상세한 제안 포함 | 간단하고 명확 |
| 코드 분석 | 상세 설명 | 핵심 정확히 추출 |

**결론:** `gpt-oss-120b`는 117B MoE 오픈웨이트 모델임에도 모든 에이전트 기능(네이티브 tool calling, 연쇄 추론, 다중 턴, 에러 처리)을 완벽히 수행. 사내 vLLM 환경에서도 문제없이 동작할 것으로 예상.

---

## 개발 중 발견 & 수정된 버그

### Bug 1: 스트리밍 도구 호출 중복 (duplicate tool_use ID)

**증상:** `messages.1.content.1: 'tool_use' ids must be unique` 에러

**원인:** OpenAI 스트리밍 API는 도구 호출을 여러 청크에 걸쳐 전송함 (이름은 첫 청크, 인자는 이후 청크에 분산). `_parse_chunk()`가 매 청크마다 부분적 `ToolUseBlock`을 생성하고, 별도의 누적 로직도 완성된 블록을 생성하여 중복 발생.

**수정:** `_parse_chunk(chunk, native=False)`로 호출하여 청크별 파싱을 비활성화. 누적 로직(`pending_tool_calls`)만 사용하여 `finish_reason == "tool_calls"` 시점에 완성된 블록 1회만 생성.

### Bug 2: 다중 도구 결과 누락

**증상:** 여러 도구를 호출한 후 두 번째 이후 결과가 API에 전달되지 않음

**원인:** `_convert_message()`가 tool result 리스트를 구성했지만 `return results[0] if len(results) == 1 else None`으로 단일 결과만 반환.

**수정:** 전체 리스트를 반환하고, `_build_api_messages()`에서 `isinstance(api_msg, list)` 체크 후 `extend()`로 평탄화.

### Bug 3: 프롬프트 기반 도구 파싱 실패

**증상:** `<tool_call>{"name":"Read", "arguments": {"file_path": "/tmp/test"}}</tool_call>` 파싱 불가

**원인:** 정규식 `\{.*?\}`가 non-greedy라서 중첩 `{}`의 첫 번째 `}`에서 매칭 종료. `{"arguments": {"file_path": ...}}`의 내부 `}`에서 끊김.

**수정:** 정규식을 `<tool_call>\s*(.*?)\s*(?:</tool_call>|$)`로 변경하여 태그 사이의 전체 내용을 캡처하고, `json.loads()`가 JSON 파싱을 담당.

## 라이선스

MIT
