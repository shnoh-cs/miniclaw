# miniclaw (openclaw-py)

[OpenClaw](https://github.com/openclaw/openclaw) Agent Harness의 Python 포트.
모든 **지능(intelligence)** 기능을 100% 유지하면서 채널 통합(WhatsApp, Telegram, Discord 등)은 제거한 경량 버전.

**사내 격리망(air-gapped network)** 에서 vLLM + 오픈소스 LLM으로 운용할 수 있는 개인 AI 비서 하니스.
**Rocket.Chat** DM을 통해 다중 사용자가 각자 독립된 세션으로 에이전트와 대화할 수 있다.

## 주요 특징

| 기능 | 설명 |
|------|------|
| **Rocket.Chat 연동** | DM 폴링 → 유저별 독립 세션 (`rc-{room_id}`) → 응답 전송 |
| **Agent Python API** | `Agent.from_config()` → `agent.run()` / `agent.stream()` |
| **듀얼 툴 콜링** | 네이티브 function calling + `<tool_call>` XML 프롬프트 자동 전환 |
| **14개 내장 도구** | Read, Write, Edit, ApplyPatch, Bash, Process, WebFetch, PDF, Hancom, Image, Memory(3), Cron, SessionStatus, Browser |
| **다단계 컴팩션** | split→summarize→merge로 장시간 대화에서도 맥락 유지 |
| **하이브리드 메모리** | BM25 + 벡터 코사인 + MMR 다양성 + 시간 감쇠 (30일 반감기) |
| **메모리 큐레이션** | 임베딩 기반 반복 패턴 탐지 → MEMORY.md 자동 승격 |
| **auto-recall** | 매 턴 자동 메모리 검색, 장기/단기 스코프 분리 |
| **컨텍스트 자가 진단** | 토큰 사용량 분석, 70%/85%에서 설정 자동 조정 |
| **Failover** | 12종 에러 분류, API 키 로테이션, 상태 영속화 |
| **인젝션 방어** | 13종 패턴 감지, 호모글리프 폴딩, 암호화 경계 마커 |
| **루프 감지** | 4종 감지기 (repeat, circuit breaker, poll, ping-pong) |
| **브라우저 자동화** | Playwright 기반 풀 브라우저 제어 |
| **Cron/Heartbeat** | 3종 스케줄(every/cron/at), 에페머럴 세션, HEARTBEAT.md |
| **서브에이전트** | 깊이 제한(max 5), 독립 세션, 배치 동시 실행 |
| **스킬 시스템** | YAML frontmatter, OS/바이너리 게이팅 |

## 빠른 시작

```bash
# 1. 클론 & 설치
git clone https://github.com/shnoh-cs/miniclaw.git
cd miniclaw
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. 설정
cp config.example.toml config.toml
# config.toml에서 API 엔드포인트, 키, Rocket.Chat 설정 수정

# 3. Rocket.Chat 실행 (Docker)
docker compose -f docker-compose.rocketchat.yml up -d

# 4. 에이전트 실행
openclaw
```

## Rocket.Chat 연동

### Docker Compose로 Rocket.Chat 실행

```bash
docker compose -f docker-compose.rocketchat.yml up -d
```

MongoDB 8.0 + Rocket.Chat이 시작된다. 기본 포트는 `docker-compose.rocketchat.yml`에서 설정.

### config.toml 설정

```toml
[rocketchat]
enabled = true
url = "http://localhost:9101"       # Rocket.Chat 서버 URL
user = "openclaw-bot"               # 봇 계정 username
password = "bot1234"                # 봇 계정 password
poll_interval = 2.0                 # 폴링 간격 (초)
channels = []                      # 모니터링할 채널 (DM 전용이면 비워둠)
notify_channel = ""                 # 크론 알림 채널
```

### 동작 방식

1. 에이전트가 시작되면 봇 계정으로 Rocket.Chat에 로그인
2. DM 룸을 2초 간격으로 폴링하여 새 메시지 감지
3. 각 유저의 DM은 `rc-{room_id}` 세션으로 독립 관리
4. 에이전트 응답을 같은 DM 룸에 전송
5. 크론 알림은 `notify_channel`로 전송 (설정 시)

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

## 아키텍처

```
Rocket.Chat DM
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  RocketChatBridge (rocketchat.py)                        │
│  폴링 → 새 메시지 감지 → agent.run() → 응답 전송         │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Agent (agent/api.py)                                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Agent Loop (agent/loop.py)                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │ Context  │→│  Model   │→│  Tool Registry   │ │  │
│  │  │  Guard   │  │ Provider │  │  (14 built-in)   │ │  │
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
```

## 테스트

```bash
# 전체 테스트 (58개: 오프라인 45 + 라이브 13)
python tests/test_live.py

# 오프라인만
python tests/test_live.py --offline

# Intelligence Eval Suite (8개 시나리오, 평균 96%)
python tests/eval/eval_intelligence.py

# Golden Parity Eval (원본 OpenClaw 대비 8개 섹션, 100%)
python tests/eval/eval_parity.py
```

## 프로젝트 구조

```
miniclaw/
├── openclaw/                  # 메인 패키지
│   ├── agent/                 #   Agent API·에이전트 루프·타입
│   ├── model/                 #   LLM 프로바이더·페일오버·thinking
│   ├── session/               #   세션·컴팩션·프루닝·lanes·메모리 플러시
│   ├── context/               #   컨텍스트 가드·자가 진단
│   ├── memory/                #   SQLite+FTS5+벡터 메모리·큐레이션
│   ├── browser/               #   Playwright 브라우저 자동화
│   ├── prompt/                #   시스템 프롬프트·인젝션 방어
│   ├── tools/                 #   도구 레지스트리·14개 내장 도구
│   ├── skills/                #   스킬 디스커버리
│   ├── subagent/              #   서브에이전트
│   ├── hooks/                 #   Hook 시스템
│   ├── cron/                  #   Cron/Heartbeat
│   ├── rocketchat.py          #   Rocket.Chat REST API 폴링 브릿지
│   ├── main.py                #   진입점 (agent + cron + RC 브릿지)
│   └── config.py              #   TOML 설정 로딩
├── tests/                     # 테스트 & Eval Suites
├── docker-compose.rocketchat.yml  # Rocket.Chat + MongoDB
├── config.example.toml        # 설정 예시
├── pyproject.toml             # 빌드 설정
└── CLAUDE.md                  # 개발 컨텍스트
```

## 라이선스

MIT
