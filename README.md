# miniclaw (openclaw-py)

[OpenClaw](https://github.com/openclaw/openclaw) Agent Harness의 Python 포트.
모든 **지능(intelligence)** 기능을 100% 유지하면서 채널 통합(WhatsApp, Telegram, Discord 등)은 제거한 경량 버전.

**사내 격리망(air-gapped network)** 에서 vLLM + 오픈소스 LLM으로 운용할 수 있는 개인 AI 비서 하니스.

## 주요 특징

| 기능 | 설명 |
|------|------|
| **Agent Python API** | `Agent.from_config()` → `agent.run()` / `agent.stream()` |
| **듀얼 툴 콜링** | 네이티브 function calling + `<tool_call>` XML 프롬프트 자동 전환 |
| **11개 내장 도구** | Read, Write, Edit, ApplyPatch, Bash, Process, WebFetch, PDF, Hancom, Image, Memory |
| **다단계 컴팩션** | split→summarize→merge로 장시간 대화에서도 맥락 유지 |
| **하이브리드 메모리** | BM25 + 벡터 코사인 + MMR 다양성 + 시간 감쇠 (30일 반감기) |
| **메모리 큐레이션** | 임베딩 기반 반복 패턴 탐지 → MEMORY.md 자동 승격 |
| **auto-recall** | 매 턴 자동 메모리 검색, 장기/단기 스코프 분리 |
| **컨텍스트 자가 진단** | 토큰 사용량 분석, 70%/85%에서 설정 자동 조정 |
| **컴팩션 후 복원** | 체크포인트 자동 생성·복원으로 맥락 손실 최소화 |
| **Failover** | 12종 에러 분류, API 키 로테이션, 상태 영속화, 프로브 메커니즘 |
| **인젝션 방어** | 13종 패턴 감지, 호모글리프 폴딩, 암호화 경계 마커 |
| **루프 감지** | 4종 감지기 (repeat, circuit breaker, poll, ping-pong) |
| **Thinking 레벨** | off→minimal→low→medium→high→xhigh + 자동 폴백 체인 |
| **Hook 시스템** | pre/post tool_call, pre/post message, on_error |
| **Cron/Heartbeat** | 주기적 모델 핑, 메모리 체크, 에이전트 루프 기반 실행 |
| **서브에이전트** | 깊이 제한(max 5), 독립 세션, 배치 동시 실행 (Semaphore 3) |
| **스킬 시스템** | YAML frontmatter, OS/바이너리 게이팅 |

## 빠른 시작

```bash
# 1. 클론 & 가상환경
git clone https://github.com/shnoh-cs/miniclaw.git
cd miniclaw
python3 -m venv .venv
source .venv/bin/activate

# 2. 설치
pip install -e .

# 3. 설정
cp config.example.toml config.toml
# config.toml에서 API 엔드포인트와 키 수정

# 4. 실행
openclaw-py
```

## Python API

```python
import asyncio
from openclaw.agent.api import Agent
from openclaw.config import load_config

agent = Agent(load_config())

# 기본 실행
result = asyncio.run(agent.run("Hello!"))
print(result.text)

# 스트리밍
async def main():
    async for chunk in agent.stream("1부터 10까지 세줘"):
        print(chunk, end="")
asyncio.run(main())

# 커스텀 도구
@agent.tool("get_weather", description="Get weather for a city", parameters=[
    {"name": "city", "type": "string", "description": "City name", "required": True},
])
def get_weather(city: str) -> str:
    return f"Sunny, 22C in {city}"

# 다중 세션
result = await agent.run("안녕!", session_id="session-a")

# Thinking 레벨 지정
from openclaw.agent.types import ThinkingLevel
result = await agent.run("복잡한 문제", thinking=ThinkingLevel.HIGH)
```

## 설정

### 사내 vLLM 서버 (격리망)

```toml
[models]
default = "gpt-oss-120b"
compaction = "gpt-oss-7b"
embedding = "bge-m3"
fallback = ["gpt-oss-70b", "gpt-oss-7b"]

[models.options.gpt-oss-120b]
tool_mode = "auto"       # auto | native | prompt
max_tokens = 32768
thinking = "adaptive"

[endpoints.llm]
base_url = "http://vllm.internal:8000/v1"
api_key = "internal-key"

[endpoints.embedding]
base_url = "http://vllm.internal:8001/v1"
api_key = "internal-key"
```

### OpenRouter (외부 테스트)

```toml
[models]
default = "anthropic/claude-sonnet-4"

[endpoints.llm]
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-or-v1-..."
```

### 전체 설정 옵션

```toml
[context]
max_tokens = 32768            # 컨텍스트 윈도우 크기
compaction_threshold = 0.7    # 70%에서 컴팩션 발동
reserve_tokens_floor = 20000  # 최소 여유 토큰
tool_result_max_ratio = 0.3   # 단일 도구 결과 최대 비율

[session]
dir = "~/.openclaw-py/sessions"

[memory]
dir = "~/.openclaw-py/memory"
chunk_size = 1600
chunk_overlap = 320

[memory.hybrid]
vector_weight = 0.7
text_weight = 0.3

[memory.hybrid.mmr]
lambda_param = 0.7

[memory.hybrid.temporal_decay]
half_life_days = 30

[pruning]
mode = "cache-ttl"
ttl_seconds = 300
keep_last_assistants = 3

[compaction]
mode = "safeguard"
identifier_policy = "strict"
max_retries = 3

[hooks]
pre_tool_call = ""
post_tool_call = ""
pre_message = ""
post_message = ""
on_error = ""
timeout = 10
```

## 아키텍처

```
사용자 입력
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Agent (agent/api.py)                                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Agent Loop (agent/loop.py)                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │ Context  │→│  Model   │→│  Tool Registry   │ │  │
│  │  │  Guard   │  │ Provider │  │  (11 built-in)   │ │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │  │
│  │       │            │              │                │  │
│  │       ▼            ▼              ▼                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │Compacti- │  │ Failover │  │    Memory        │ │  │
│  │  │on/Prune  │  │ Manager  │  │ (SQLite+FTS5+Vec)│ │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ auto-recall│  │  Diagnosis  │  │   Curation      │   │
│  │ (per-turn) │  │ (auto-tune) │  │ (daily→MEMORY)  │   │
│  └────────────┘  └─────────────┘  └─────────────────┘   │
└──────────────────────────────────────────────────────────┘
    │
    ▼
  응답 출력 (텍스트 / 스트리밍)
```

### 에이전트 루프 흐름

```
run()
 ├── _ensure_initialized() (1회: 메모리 인덱싱, 세션 인덱싱, 큐레이션 체크)
 ├── auto-recall (장기/단기 스코프 분리 메모리 검색)
 └── _attempt_loop()
      ├── Context Guard 체크 (OK / COMPACT / ERROR)
      │    ├── 85%+ → 메모리 플러시 (에이전트 루프 기반, tool access)
      │    ├── 체크포인트 저장 (.context-checkpoint.md)
      │    └── 컴팩션 실행 → 체크포인트 복원
      ├── 컨텍스트 진단 → 설정 자동 조정 (70%+)
      ├── 프루닝 적용 (in-memory)
      ├── 시스템 프롬프트 조립 (13-섹션 + auto-recall + recovery)
      ├── 모델 스트리밍 호출
      │    ├── 텍스트 누적 + 스트리밍 콜백
      │    ├── 도구 호출 감지 (native / prompt)
      │    └── Thinking 태그 파싱
      ├── 도구 실행
      │    ├── Hook (pre/post)
      │    ├── 결과 트렁케이션 (context guard)
      │    └── 루프 감지 (4종)
      └── 재진입 (도구 호출 있으면 반복, 최대 50턴)
```

### 지능 기능 상세

#### 메모리 플러시 (Pre-Compaction)
컴팩션 직전, 에이전트 루프를 통해 `memory_search`/`memory_save` 도구를 사용하여 중요 정보를 영구 저장. 일반 LLM completion이 아닌 full agent loop으로 실행되어 도구 접근 가능.

#### 메모리 큐레이션 (Daily → MEMORY.md)
임베딩 기반 2단계 접근:
1. **교차일 반복 탐지**: 일별 노트의 문단을 임베딩, 다른 날짜의 문단과 코사인 유사도 > 0.75인 것을 "반복 패턴"으로 분류
2. **신규성 필터링**: 후보를 MEMORY.md와 비교, 유사도 < 0.80인 것만 "신규"로 판정
3. **모델 합성**: 신규 반복 패턴을 간결한 bullet point로 요약하여 MEMORY.md에 추가

임베딩 미사용 시 프롬프트 기반으로 폴백. 24시간 디바운싱.

#### auto-recall (Per-Turn Memory Retrieval)
매 턴 사용자 입력으로 메모리 자동 검색, 스코프 분리:
- **Long-term Memory**: MEMORY.md 소스, 1.2x score boost
- **Recent Context**: 일별 노트 (episodic)
- **Session**: 필터링 (이미 컨텍스트에 존재)

#### 컨텍스트 자가 진단 & 자동 조정
토큰 사용량을 카테고리별로 분해 (시스템 프롬프트, 도구 스키마, 세션 히스토리, 컴팩션 요약).
임계치 도달 시 설정 자동 변경:
- **70%+**: `compaction_threshold` 하향 → 더 이른 컴팩션
- **85%+**: `reserve_tokens_floor` 상향 + `tool_result_max_ratio` 하향

REPL에서 `/context` 명령으로 진단 보고서 확인 가능.

#### 컴팩션 후 복원 (Post-Compaction Recovery)
컴팩션 전 `.context-checkpoint.md`에 최근 대화 요약을 저장. 컴팩션 후 시스템 프롬프트에 자동 주입하여 맥락 손실 최소화.

## 테스트

```bash
# 전체 테스트 (45개: 오프라인 32 + 라이브 13)
python test_live.py
```

### 테스트 구성

#### 1. 단위 테스트 — 오프라인 12개
| 테스트 | 검증 내용 |
|--------|-----------|
| ContextGuard | 토큰 예산 → OK/COMPACT 에스컬레이션 |
| ToolRegistry | 도구 등록·조회, 필수 도구 존재 |
| Session Lanes | 병렬 대화 스레드 생성·병합 |
| Cron 스케줄러 | 주기 실행·상태 관리 |
| HookRunner | pre/post hook 호출 |
| 프롬프트 인젝션 방어 | 13종 패턴 정리 |
| ThinkingLevel 파싱 | 문자열→enum 변환·폴백 |
| 세션 영속성 | JSONL 저장·로드 |
| 도구 결과 트렁케이션 | head+tail 70/30 분할 |
| 도구 루프 감지 | N회 반복 경고 |
| ApplyPatch | 패치 도구 등록 확인 |
| Failover 설정 | 폴백 체인 구성 |

#### 2. 배관 공사 검증 — 오프라인 13개
memory_get 등록·실행, 메모리 플러시, Thinking API, FileWatcher, subagent_batch, AgentContext.memory_searcher, prunable_tools, 컴팩션 체크포인트, heartbeat_from_file, compact_session, prompt builder, flush 안전마진

#### 3. 지능 갭 검증 — 오프라인 7개
메모리 플러시 에이전트 루프, auto_recall 필드, 체크포인트 복원, 컨텍스트 진단, 큐레이션 모듈, 진단 자동 조정, auto-recall 스코프 분리

#### 4. 라이브 테스트 — API 호출 13개
단순 대화, Read 도구, Bash 도구, Write→Read 체인, 다중 턴 대화, WebFetch, Edit 도구, 커스텀 도구, 스트리밍 API, 에러 처리, 한국어 응답, 긴 출력 처리, 수학 추론

## 프로젝트 구조

```
miniclaw/
├── openclaw/               # 메인 패키지 (~10,600줄, 49 모듈)
│   ├── agent/              #   Agent API·에이전트 루프·타입
│   │   ├── api.py          #     Agent 클래스 (진입점)
│   │   ├── loop.py         #     메인 루프
│   │   └── types.py        #     메시지·도구·결과 타입
│   ├── model/              #   LLM 프로바이더·페일오버·thinking
│   ├── session/            #   세션·컴팩션·프루닝·lanes·메모리 플러시
│   ├── context/            #   컨텍스트 가드·자가 진단
│   ├── memory/             #   SQLite+FTS5+벡터 메모리·큐레이션
│   ├── prompt/             #   시스템 프롬프트·인젝션 방어
│   ├── tools/              #   도구 레지스트리·11개 내장 도구
│   ├── skills/             #   스킬 디스커버리
│   ├── subagent/           #   서브에이전트
│   ├── hooks/              #   Hook 시스템
│   ├── cron/               #   Cron/Heartbeat
│   ├── config.py           #   TOML 설정
│   └── repl.py             #   대화형 REPL
├── test_live.py            # 테스트 (45개)
├── config.example.toml     # 설정 예시
├── pyproject.toml          # 빌드 설정
└── CLAUDE.md               # 개발 컨텍스트
```

## 라이선스

MIT
