# miniclaw (openclaw-py)

OpenClaw Agent Harness의 Python 포트. 원본의 모든 "지능(intelligence)" 기능을 100% 유지하면서 채널 통합(WhatsApp, Telegram, Discord 등)은 제거.

## 프로젝트 목적

- **사내 격리망(air-gapped network)** 환경에서 vLLM 서버 + 오픈소스 LLM으로 운용할 에이전트 하니스
- **Rocket.Chat** DM을 통해 다중 사용자가 에이전트와 독립 세션으로 대화
- 현재 OpenRouter API로 테스트 중
- GitHub: https://github.com/shnoh-cs/miniclaw

## 기술 스택

- Python 3.11+, 가상환경: `.venv/`
- 빌드: hatchling
- 의존성: openai, tiktoken, numpy, pydantic, httpx, rich, beautifulsoup4, PyPDF2, PyYAML, olefile, croniter, python-dateutil, playwright
- 설정: `config.toml` (gitignore됨) / `config.example.toml` (vLLM 예시)
- Rocket.Chat: Docker Compose (`docker-compose.rocketchat.yml`) — MongoDB 8.0 + Rocket.Chat

## 디렉토리 구조

```
openclaw/
├── agent/          # Agent API, 에이전트 루프, 타입 정의
│   ├── api.py           Agent 클래스 (Python API 진입점)
│   ├── loop.py          메인 루프 (run → attempt → stream → tool dispatch)
│   └── types.py         AgentMessage, ToolDefinition, RunResult 등
├── model/          # LLM 프로바이더, 페일오버
│   ├── provider.py      OpenAI 호환 API 클라이언트
│   ├── failover.py      FailoverManager (프로필 로테이션·상태 영속화)
│   ├── error_classify.py 에러 패턴 분류·should_failover 판정
│   ├── cooldown.py      ProfileCooldown·ApiKeyRotator·백오프
│   └── thinking.py      Thinking 레벨 해석·폴백
├── session/        # 세션 관리, 컴팩션, 프루닝
│   ├── manager.py       JSONL append-only 세션 (ephemeral 지원)
│   ├── compaction.py    다단계 컴팩션 (split→summarize→merge)
│   ├── identifiers.py   식별자 추출·정규화
│   ├── safeguard.py     컴팩션 품질 검증·도구 실패 추적
│   ├── pruning.py       Cache-TTL 프루닝·이미지 프루닝
│   ├── lanes.py         병렬 대화 스레드
│   └── memory_flush.py  컴팩션 전 메모리 플러시
├── context/        # 컨텍스트 윈도우 관리
│   ├── guard.py         토큰 예산·트렁케이션·enforce_budget
│   └── diagnosis.py     컨텍스트 자가 진단·설정 자동 조정
├── memory/         # 하이브리드 메모리 시스템
│   ├── store.py         SQLite + FTS5
│   ├── search.py        MemorySearcher (하이브리드 검색 오케스트레이터)
│   ├── ranking.py       cosine·BM25·Jaccard·MMR·시간 감쇠
│   ├── query.py         다국어 쿼리 토큰화·확장·FTS 빌더
│   ├── watchers.py      FileWatcher·Reranker·SessionSyncWatcher
│   ├── embeddings.py    임베딩 프로바이더
│   └── curation.py      일별 노트 → MEMORY.md 자동 승격
├── prompt/         # 시스템 프롬프트·인젝션 방어
│   ├── builder.py       13-섹션 프롬프트 조립 (부트스트랩 파일 소독)
│   ├── bootstrap.py     부트스트랩 파일 8종 로딩
│   └── sanitize.py      인젝션 방어 (13종 패턴·호모글리프·경계마커)
├── browser/        # Playwright 브라우저 자동화
│   └── __init__.py      BrowserManager (스냅샷·ref·클릭·타이핑·멀티탭)
├── tools/          # 도구 레지스트리·14개 내장 도구
│   ├── registry.py      ToolRegistry·RegisteredTool
│   ├── loop_detector.py 4종 루프 감지 (repeat·poll·ping-pong·breaker)
│   ├── truncation.py    도구 결과 트렁케이션·세션 가드
│   └── builtins/        Read, Write, Edit, ApplyPatch, Bash, Process,
│                         WebFetch, PDF, Hancom, Image, Memory(3), Cron, SessionStatus, Browser
├── skills/         # 스킬 디스커버리
│   ├── loader.py        YAML frontmatter, OS/바이너리 게이팅, 번들 스킬 자동 로드
│   └── bundled/         번들 스킬 (nano-pdf, himalaya)
├── subagent/       # 서브에이전트
│   └── spawn.py         깊이 제한(max 5), 도구 정책
├── hooks/          # Hook 시스템 (shlex 인젝션 방어)
│   └── __init__.py      pre/post tool_call, pre/post message, on_error
├── cron/           # Cron/Heartbeat (3종 스케줄: every, cron, at)
│   ├── __init__.py      CronScheduler, 모델 핑, 메모리 체크, HEARTBEAT.md
│   └── persistence.py   크론잡 JSON 영속화·복원·미실행 감지
├── rocketchat.py   Rocket.Chat REST API 폴링 브릿지
├── main.py         진입점 (agent + cron + RC 브릿지)
└── config.py       TOML 설정 로딩
```

## 핵심 아키텍처

### Rocket.Chat 연동 (`rocketchat.py`)
- REST API 폴링 브릿지 — httpx 기반, 추가 의존성 없음
- `RocketChatClient`: 로그인, 메시지 송수신, 채널/DM 히스토리 조회
- `RocketChatBridge`: 폴링 루프 → 새 메시지 감지 → agent.run() → 응답 전송
- 크론 알림도 `_notification_callbacks`를 통해 Rocket.Chat으로 전송
- 세션 ID: `rc-{room_id}`로 채널/DM별 독립 세션

### Agent API (`agent/api.py`)
- `Agent` 클래스: 모든 기능의 진입점
- `Agent.from_config()` → `agent.run()` / `agent.stream()` 패턴
- `@agent.tool` 데코레이터로 커스텀 도구 등록
- 에페머럴 세션: `cron-`, `heartbeat` 접두사 세션은 디스크 I/O 없이 메모리만 사용

### 에이전트 루프 (`agent/loop.py`)
- `run()` → `_attempt_loop()` → stream → tool dispatch → re-entry (최대 50턴)
- 컨텍스트 초과 시 자동 컴팩션·프루닝 적용
- Hook 연동, 고아 tool_use 감지·합성 결과 주입

### 지능 기능 (Intelligence Features)
- **메모리 플러시**: 컴팩션 전 에이전트 루프를 통해 tool access와 함께 기억 저장
- **메모리 큐레이션**: 임베딩 기반 교차일 유사도로 반복 패턴 탐지 → MEMORY.md 승격
- **auto-recall**: 매 턴 자동 메모리 검색, 장기(MEMORY.md 1.2x boost) / 단기(일별노트) 스코프 분리
- **컴팩션 후 복원**: `.context-checkpoint.md` 자동 생성 → 컴팩션 후 시스템 프롬프트에 주입
- **컨텍스트 자가 진단**: 토큰 사용량 분석, 70%/85%에서 설정 자동 조정

### 모델 프로바이더 (`model/provider.py`)
- OpenAI 호환 API 클라이언트 (vLLM, OpenRouter 등)
- **Native/Prompt 듀얼 툴 콜링**: `tool_mode = auto|native|prompt`

## 테스트

```bash
cd ~/miniclaw
pip install -e .
openclaw                              # 에이전트 실행 (크론 + Rocket.Chat)
python tests/test_live.py [--offline]  # 테스트 (58개, --offline: 오프라인만)
```

## 주의사항

- `config.toml`은 `.gitignore`에 포함됨 (API 키 보호)
- `config.example.toml`을 복사하여 `config.toml` 생성 후 사용
- Docker: `docker compose -f docker-compose.rocketchat.yml up -d`로 Rocket.Chat 실행
- 사용자는 한국어 선호
