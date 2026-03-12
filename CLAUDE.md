# miniclaw (openclaw-py)

OpenClaw Agent Harness의 Python 포트. 원본의 모든 "지능(intelligence)" 기능을 100% 유지하면서 채널 통합(WhatsApp, Telegram, Discord 등)은 제거.

## 프로젝트 목적

- **사내 격리망(air-gapped network)** 환경에서 vLLM 서버 + 오픈소스 LLM으로 운용할 에이전트 하니스
- 현재 OpenRouter API (`anthropic/claude-sonnet-4`)로 테스트 중
- GitHub: https://github.com/shnoh-cs/miniclaw

## 기술 스택

- Python 3.11+, 가상환경: `.venv/`
- 빌드: hatchling
- 의존성: openai, tiktoken, numpy, pydantic, httpx, rich, beautifulsoup4, PyPDF2, PyYAML, olefile, croniter, python-dateutil, playwright, fastapi, uvicorn
- 설정: `config.toml` (gitignore됨) / `config.example.toml` (vLLM 예시)
- 총 소스: ~11,100줄 코어 + ~1,530줄 builtins (65개 모듈)

## 디렉토리 구조

```
openclaw/
├── agent/          # Agent API, 에이전트 루프, 타입 정의
│   ├── api.py           967줄  Agent 클래스 (Python API 진입점)
│   ├── loop.py          685줄  메인 루프 (run → attempt → stream → tool dispatch)
│   └── types.py         288줄  AgentMessage, ToolDefinition, RunResult 등
├── model/          # LLM 프로바이더, 페일오버
│   ├── provider.py      410줄  OpenAI 호환 API 클라이언트
│   ├── failover.py      300줄  FailoverManager (프로필 로테이션·상태 영속화)
│   ├── error_classify.py 329줄  에러 패턴 분류·should_failover 판정
│   ├── cooldown.py      162줄  ProfileCooldown·ApiKeyRotator·백오프
│   └── thinking.py       62줄  Thinking 레벨 해석·폴백
├── session/        # 세션 관리, 컴팩션, 프루닝
│   ├── manager.py       350줄  JSONL append-only 세션 (ephemeral 지원)
│   ├── compaction.py    708줄  다단계 컴팩션 (split→summarize→merge)
│   ├── identifiers.py    73줄  식별자 추출·정규화
│   ├── safeguard.py     326줄  컴팩션 품질 검증·도구 실패 추적
│   ├── pruning.py       372줄  Cache-TTL 프루닝·이미지 프루닝
│   ├── lanes.py         150줄  병렬 대화 스레드
│   └── memory_flush.py  127줄  컴팩션 전 메모리 플러시
├── context/        # 컨텍스트 윈도우 관리
│   ├── guard.py         268줄  토큰 예산·트렁케이션·enforce_budget
│   └── diagnosis.py     223줄  컨텍스트 자가 진단·설정 자동 조정
├── memory/         # 하이브리드 메모리 시스템
│   ├── store.py         312줄  SQLite + FTS5
│   ├── search.py        352줄  MemorySearcher (하이브리드 검색 오케스트레이터)
│   ├── ranking.py       219줄  cosine·BM25·Jaccard·MMR·시간 감쇠
│   ├── query.py         177줄  다국어 쿼리 토큰화·확장·FTS 빌더
│   ├── watchers.py      146줄  FileWatcher·Reranker·SessionSyncWatcher
│   ├── embeddings.py     61줄  임베딩 프로바이더
│   └── curation.py      281줄  일별 노트 → MEMORY.md 자동 승격
├── prompt/         # 시스템 프롬프트·인젝션 방어
│   ├── builder.py       333줄  13-섹션 프롬프트 조립 (부트스트랩 파일 소독)
│   ├── bootstrap.py     138줄  부트스트랩 파일 8종 로딩
│   └── sanitize.py      406줄  인젝션 방어 (13종 패턴·호모글리프·경계마커)
├── browser/        # Playwright 브라우저 자동화
│   └── __init__.py      549줄  BrowserManager (스냅샷·ref·클릭·타이핑·멀티탭)
├── tools/          # 도구 레지스트리·14개 내장 도구
│   ├── registry.py       88줄  ToolRegistry·RegisteredTool
│   ├── loop_detector.py 401줄  4종 루프 감지 (repeat·poll·ping-pong·breaker)
│   ├── truncation.py    130줄  도구 결과 트렁케이션·세션 가드
│   └── builtins/       1532줄  Read, Write, Edit, ApplyPatch, Bash, Process,
│                                WebFetch, PDF, Hancom, Image, Memory(3), Cron, SessionStatus, Browser
├── skills/         # 스킬 디스커버리
│   ├── loader.py        207줄  YAML frontmatter, OS/바이너리 게이팅, 번들 스킬 자동 로드
│   └── bundled/         번들 스킬 (nano-pdf, himalaya)
├── subagent/       # 서브에이전트
│   └── spawn.py         216줄  깊이 제한(max 5), 도구 정책
├── hooks/          # Hook 시스템 (shlex 인젝션 방어)
│   └── __init__.py       87줄  pre/post tool_call, pre/post message, on_error
├── cron/           # Cron/Heartbeat (3종 스케줄: every, cron, at)
│   ├── __init__.py      377줄  CronScheduler, 모델 핑, 메모리 체크, HEARTBEAT.md
│   └── persistence.py   143줄  크론잡 JSON 영속화·복원·미실행 감지
├── server.py       229줄  FastAPI 웹 서버 (send+poll 채팅, 크론, 세션 API)
├── static/
│   └── index.html       391줄  단일 파일 채팅 UI (HTML/CSS/JS)
└── config.py       207줄  TOML 설정 로딩
```

## 핵심 아키텍처

### Agent API (`agent/api.py`)
- `Agent` 클래스: 모든 기능의 진입점
- `Agent.from_config()` → `agent.run()` / `agent.stream()` 패턴
- `@agent.tool` 데코레이터로 커스텀 도구 등록
- `_ensure_initialized()`: 메모리 인덱싱, 세션 인덱싱, 큐레이션 체크 1회 실행
- `_build_context()`: AgentContext 조립 (도구, 메모리, 세션, 프롬프트)
- 서브에이전트 도구 (spawn, batch, list, read) 자동 등록
- 에페머럴 세션: `cron-`, `heartbeat` 접두사 세션은 자동으로 디스크 I/O 없이 메모리만 사용

### 웹 서버 (`server.py` + `static/index.html`)
- FastAPI 기반, code-server 리버스 프록시 호환 (GET only, 루트 경로, 상대 URL)
- **send+poll 패턴**: `?action=send` → 즉시 반환, 백그라운드 실행 → 프론트엔드가 `?action=history` 폴링
- `?action=status` — 에이전트 처리 중 여부 확인
- 단일 HTML 파일 채팅 UI (다크 테마, 세션 전환, 크론 패널)
- 크론잡 결과는 `_post_cron_notification()`으로 세션에 기록 → 브라우저 오프라인이어도 나중에 확인 가능

### 크론잡 영속화 (`cron/persistence.py`)
- `jobs.json`에 사용자 크론잡 저장 (시스템 태스크 제외)
- 서버 재시작 시 `restore_cron_jobs()`로 복원 + `check_missed_jobs()`로 미실행 감지·catch-up
- 콜백 재구성: `_make_cron_callback()`으로 에페머럴 세션에서 실행 → 결과를 web 세션에 알림

### 에이전트 루프 (`agent/loop.py`)
- `run()` → `_attempt_loop()` → stream → tool dispatch → re-entry (최대 50턴)
- 컨텍스트 초과 시 자동 컴팩션·프루닝 적용
- Hook 연동, 고아 tool_use 감지·합성 결과 주입
- 이미지 프루닝 (오래된 base64 자동 제거)
- Thinking 레벨 폴백 (에러 시 자동 다운그레이드)

### 지능 기능 (Intelligence Features)
- **메모리 플러시**: 컴팩션 전 에이전트 루프를 통해 tool access와 함께 기억 저장
- **메모리 큐레이션**: 임베딩 기반 교차일 유사도로 반복 패턴 탐지 → MEMORY.md 승격
- **auto-recall**: 매 턴 자동 메모리 검색, 장기(MEMORY.md 1.2x boost) / 단기(일별노트) 스코프 분리
- **컴팩션 후 복원**: `.context-checkpoint.md` 자동 생성 → 컴팩션 후 시스템 프롬프트에 주입
- **컨텍스트 자가 진단**: 토큰 사용량 분석, 70%/85%에서 설정 자동 조정

### 모델 프로바이더 (`model/provider.py`)
- OpenAI 호환 API 클라이언트 (vLLM, OpenRouter 등)
- **Native/Prompt 듀얼 툴 콜링**: `tool_mode = auto|native|prompt`
- 스트리밍 도구 호출: `pending_tool_calls` 청크 누적 → `finish_reason == "tool_calls"` 시 1회 생성

### 다단계 컴팩션 (`session/compaction.py` + `identifiers.py` + `safeguard.py`)
- split→summarize→merge (adaptive chunk ratio, SAFETY_MARGIN=1.2)
- 3단계 프로그레시브 폴백 (`summarize_with_fallback`)
- safeguard 검증: 식별자 보존, 정렬된 섹션, ask 중복 검사
- `_estimate_message_tokens()` 통합 헬퍼 (5개 중복 패턴 제거)

### 메모리 (`memory/search.py` + `ranking.py` + `query.py` + `watchers.py`)
- SQLite + FTS5 + 벡터 임베딩 (L2 정규화)
- 하이브리드 검색: BM25 + cosine + MMR + 시간 감쇠 (30일 반감기)
- 다국어 쿼리 확장 (EN/KO/ZH/JA)
- FileWatcher (30초 디바운싱), 세션 인덱싱, 캐시 임베딩
- `_batch_embed_with_cache()` 통합 헬퍼 (중복 캐시 로직 제거)

### 컨텍스트 자가 진단 (`context/diagnosis.py`)
- 토큰 사용량 카테고리별 분해 (시스템/도구/세션/컴팩션)
- 설정 자동 조정:
  - 70%+: `compaction_threshold` 하향 (더 이른 컴팩션)
  - 85%+: `reserve_tokens_floor` 상향 + `tool_result_max_ratio` 하향

## 테스트

### `tests/test_live.py` — 58개 테스트
- 오프라인 16개: ContextGuard, ToolRegistry, SessionLanes, Cron, Hook, 인젝션 방어, ThinkingLevel, 세션 영속성, 트렁케이션, 루프 감지, ApplyPatch, Failover, Cron Expression, Ephemeral Session, Browser 등록/기본동작, Browser 스냅샷/상호작용
- 배관 공사 13개: memory_get, 플러시, Thinking API, FileWatcher, subagent_batch, AgentContext, 프루닝, 체크포인트, heartbeat, compact_session, prompt builder, flush 안전마진
- 지능 갭 7개: 플러시 에이전트 루프, auto_recall, 체크포인트 복원, 컨텍스트 진단, 큐레이션, 진단 자동 조정, auto-recall 스코프
- 라이브 13개: 대화, Read, Bash, Write→Read, 다중 턴, WebFetch, Edit, 커스텀 도구, 스트리밍, 에러 처리, 한국어, 긴 출력, 수학 추론

## 실행 방법

```bash
cd ~/miniclaw
source .venv/bin/activate
pip install -e .
openclaw-serve --reload               # 웹 서버 (기본 포트 8089)
python tests/test_live.py [--offline]  # 테스트 (58개, --offline: 오프라인만)
```

## 주의사항

- `config.toml`은 `.gitignore`에 포함됨 (API 키 보호)
- `config.example.toml`을 복사하여 `config.toml` 생성 후 사용
- 사용자는 한국어 선호
