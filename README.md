# openclaw-py

[OpenClaw](https://github.com/openclaw/openclaw) Agent Harness의 Python 포트.
모든 **지능(intelligence)** 기능을 100% 유지하면서 채널 통합(WhatsApp, Telegram, Discord 등)은 제거한 경량 버전.

**사내 격리망(air-gapped network)** 에서 vLLM + 오픈소스 LLM으로 운용할 수 있는 개인 AI 비서 하니스.

## 주요 특징

| 기능 | 설명 |
|------|------|
| **듀얼 툴 콜링** | 네이티브 function calling + `<tool_call>` XML 프롬프트 자동 전환 |
| **14개 내장 도구** | Read, Write, Edit, Bash, WebFetch, PDF, Image, Memory, Subagent 등 |
| **다단계 컴팩션** | split→summarize→merge로 장시간 대화에서도 맥락 유지 |
| **하이브리드 메모리** | BM25 + 벡터 코사인 + MMR 다양성 + 시간 감쇠 (30일 반감기) |
| **Failover** | 12종 에러 분류, API 키 로테이션, 상태 영속화, 프로브 메커니즘 |
| **인젝션 방어** | 13종 패턴 감지, 호모글리프 폴딩, 암호화 경계 마커 |
| **루프 감지** | 4종 감지기 (repeat, circuit breaker, poll, ping-pong) |
| **Thinking 레벨** | off→minimal→low→medium→high→xhigh + 자동 폴백 체인 |
| **Hook 시스템** | pre/post tool_call, pre/post message, on_error |
| **Cron/Heartbeat** | 주기적 모델 핑, 메모리 체크, 커스텀 스케줄 |
| **서브에이전트** | 깊이 제한(max 5), 독립 세션, 도구 정책 |
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
from openclaw.repl import Agent
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
default = "anthropic/claude-sonnet-4.6"

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
┌─────────────────────────────────────────────┐
│  Agent Loop (agent/loop.py)                 │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Context │→│  Model   │→│   Tool    │  │
│  │  Guard  │  │ Provider │  │ Registry  │  │
│  └─────────┘  └──────────┘  └───────────┘  │
│       │            │              │         │
│       ▼            ▼              ▼         │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │Compacti-│  │ Failover │  │  14 Built │  │
│  │on/Prune │  │ Manager  │  │  -in Tools│  │
│  └─────────┘  └──────────┘  └───────────┘  │
│       │                           │         │
│       ▼                           ▼         │
│  ┌─────────┐                ┌───────────┐   │
│  │ Session │                │  Memory   │   │
│  │ Manager │                │  Search   │   │
│  └─────────┘                └───────────┘   │
└─────────────────────────────────────────────┘
    │
    ▼
  응답 출력 (텍스트 / 스트리밍)
```

### 에이전트 루프 흐름

```
run() → _attempt_loop()
         ├── Context Guard 체크 (OK / COMPACT / ERROR)
         ├── 프루닝 적용 (in-memory)
         ├── 이미지 프루닝 (오래된 base64 제거)
         ├── 시스템 프롬프트 조립 (13-섹션)
         ├── 모델 스트리밍 호출
         │    ├── 텍스트 누적 + 스트리밍 콜백
         │    ├── 도구 호출 감지 (native / prompt)
         │    └── Thinking 태그 파싱
         ├── 도구 실행
         │    ├── Hook (pre/post)
         │    ├── 결과 트렁케이션 (context guard)
         │    ├── 세션 캡 (400K)
         │    └── 루프 감지 (4종)
         └── 재진입 (도구 호출 있으면 반복, 최대 50턴)
```

## 테스트

```bash
# 단위·통합 테스트 (25개, 오프라인 12 + 라이브 13)
python test_live.py

# 장시간 운용 내구성 테스트
python test_endurance.py
```

### 테스트 결과 요약

| 테스트 | 항목 수 | 결과 |
|--------|---------|------|
| 단위·통합 (`test_live.py`) | 25 | **25/25 PASS** |
| 내구성 (`test_endurance.py`) | 88턴, 10페이즈 | **PASS** |

### 내구성 테스트 상세

| 항목 | 결과 |
|------|------|
| 총 턴 수 | 88 |
| 소요 시간 | 30.3분 |
| 컴팩션 발동 | 13회 |
| 도구 호출 | 91회 |
| 에러 | 0건 |
| **기억 보존율** | **97.9%** |

10개 페이즈에서 9개 핵심 사실(이름, 생일, 프로젝트명, 팀원, 서버IP, DB 테이블 수, 배포 코드네임, 버그 티켓, 스프린트 목표)을 반복 검증. 컴팩션 13회를 거쳐도 거의 완벽하게 보존.

## 프로젝트 구조

```
openclaw-py/
├── openclaw/               # 메인 패키지 (~9,600줄)
│   ├── agent/              #   에이전트 루프·타입
│   ├── model/              #   LLM 프로바이더·페일오버·thinking
│   ├── session/            #   세션·컴팩션·프루닝·lanes
│   ├── context/            #   컨텍스트 가드
│   ├── memory/             #   SQLite+FTS5+벡터 메모리
│   ├── prompt/             #   시스템 프롬프트·인젝션 방어
│   ├── tools/              #   도구 레지스트리·14개 내장 도구
│   ├── skills/             #   스킬 디스커버리
│   ├── subagent/           #   서브에이전트
│   ├── hooks/              #   Hook 시스템
│   ├── cron/               #   Cron/Heartbeat
│   ├── config.py           #   TOML 설정
│   └── repl.py             #   REPL & Agent API
├── test_live.py            # 단위·통합 테스트 (25개)
├── test_endurance.py       # 장시간 내구성 테스트 (88턴)
├── config.example.toml     # 설정 예시
├── pyproject.toml          # 빌드 설정
└── CLAUDE.md               # 개발 컨텍스트
```

## 라이선스

MIT
