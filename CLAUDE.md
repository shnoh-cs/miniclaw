# openclaw-py

OpenClaw Agent Harness의 Python 포트. 원본의 모든 "지능(intelligence)" 기능을 100% 유지하면서 채널 통합(WhatsApp, Telegram, Discord 등)은 제거.

## 프로젝트 목적

- **사내 격리망(air-gapped network)** 환경에서 vLLM 서버 + 오픈소스 LLM으로 운용할 에이전트 하니스
- 현재 MacBook에서 OpenRouter API (`anthropic/claude-sonnet-4.6`)로 테스트 중
- GitHub: https://github.com/shnoh-cs/miniclaw

## 기술 스택

- Python 3.11+, 가상환경: `.venv/`
- 빌드: hatchling
- 의존성: openai, tiktoken, numpy, pydantic, httpx, rich, beautifulsoup4, PyPDF2, PyYAML, olefile
- 설정: `config.toml` (gitignore됨) / `config.example.toml` (vLLM 예시)
- 총 소스: ~9,600줄 (40개 모듈)

## 디렉토리 구조

```
openclaw/
├── agent/          # 에이전트 루프, 타입 정의
│   ├── loop.py          478줄  메인 루프 (run → attempt → stream → tool dispatch)
│   └── types.py         274줄  AgentMessage, ToolDefinition, RunResult 등
├── model/          # LLM 프로바이더, 페일오버
│   ├── provider.py      401줄  OpenAI 호환 API 클라이언트
│   ├── failover.py      829줄  에러 분류·페일오버·키 로테이션·상태 영속화
│   └── thinking.py       80줄  Thinking 레벨 해석·폴백
├── session/        # 세션 관리, 컴팩션, 프루닝
│   ├── manager.py       301줄  JSONL append-only 세션
│   ├── compaction.py   1023줄  다단계 컴팩션 (split→summarize→merge)
│   ├── pruning.py       372줄  Cache-TTL 프루닝·이미지 프루닝
│   ├── lanes.py         120줄  병렬 대화 스레드
│   └── memory_flush.py  100줄  컴팩션 전 메모리 플러시
├── context/        # 컨텍스트 윈도우 관리
│   └── guard.py         268줄  토큰 예산·트렁케이션·enforce_budget
├── memory/         # 하이브리드 메모리 시스템
│   ├── store.py         243줄  SQLite + FTS5
│   ├── search.py        817줄  BM25 + 벡터 + MMR + 시간 감쇠
│   └── embeddings.py    116줄  임베딩 프로바이더
├── prompt/         # 시스템 프롬프트·인젝션 방어
│   ├── builder.py       314줄  13-섹션 프롬프트 조립
│   ├── bootstrap.py     180줄  부트스트랩 파일 8종 로딩
│   └── sanitize.py      406줄  인젝션 방어 (13종 패턴·호모글리프·경계마커)
├── tools/          # 도구 레지스트리·14개 내장 도구
│   ├── registry.py      628줄  등록·실행·루프 감지(4종)·트렁케이션
│   └── builtins/        ~900줄  Read, Write, Edit, ApplyPatch, Bash, Process,
│                                WebFetch, PDF, Hancom, Image, Memory(3), Subagent
├── skills/         # 스킬 디스커버리
│   └── loader.py        197줄  YAML frontmatter, OS/바이너리 게이팅
├── subagent/       # 서브에이전트
│   └── spawn.py         216줄  깊이 제한(max 5), 도구 정책
├── hooks/          # Hook 시스템
│   └── __init__.py      100줄  pre/post tool_call, pre/post message, on_error
├── cron/           # Cron/Heartbeat
│   └── __init__.py      210줄  주기적 모델 핑·메모리 체크
├── config.py       202줄  TOML 설정 로딩
└── repl.py         517줄  대화형 REPL & Agent Python API
```

## 핵심 아키텍처

### 에이전트 루프 (`agent/loop.py`)
- `run()` → `_attempt_loop()` → stream → tool dispatch → re-entry (최대 50턴)
- 컨텍스트 초과 시 자동 컴팩션·프루닝 적용
- Hook 연동, 고아 tool_use 감지·합성 결과 주입
- 이미지 프루닝 (오래된 base64 자동 제거)
- Thinking 레벨 폴백 (에러 시 자동 다운그레이드)

### 모델 프로바이더 (`model/provider.py`)
- OpenAI 호환 API 클라이언트 (vLLM, OpenRouter 등)
- **Native/Prompt 듀얼 툴 콜링**: `tool_mode = auto|native|prompt`
- 스트리밍 도구 호출: `pending_tool_calls` 청크 누적 → `finish_reason == "tool_calls"` 시 1회 생성
- 도구 결과 리스트 반환 + `extend()` 평탄화

### 다단계 컴팩션 (`session/compaction.py`)
- split→summarize→merge (adaptive chunk ratio, SAFETY_MARGIN=1.2)
- 3단계 프로그레시브 폴백 (`summarize_with_fallback`)
- safeguard 검증: 식별자 보존, 정렬된 섹션, ask 중복 검사
- 파일 작업 추적, 도구 실패 추적 (최대 8건)
- LLM 호출 재시도: 3회 + 지수 백오프

### 프루닝 (`session/pruning.py`)
- Cache-TTL 기반, 비율 게이팅 (softTrimRatio=0.3)
- 반복 하드 클리어 + 부트스트랩 보호
- 이미지 콘텐츠 추정, `prunable_tools` 필터링

### 메모리 (`memory/`)
- SQLite + FTS5 + 벡터 임베딩 (L2 정규화)
- 하이브리드 검색: BM25 + cosine + MMR + 시간 감쇠 (30일 반감기)
- 다국어 쿼리 확장 (EN/KO/ZH/JA)
- FileWatcher (30초 디바운싱), 세션 인덱싱, 캐시 임베딩

### 도구 시스템 (`tools/`)
- 14개 내장: Read, Write, Edit, ApplyPatch, Bash, Process, WebFetch, PDF, Hancom, Image, MemorySearch, MemorySave, MemoryGet, Subagent
- 루프 감지 4종: generic_repeat, global_circuit_breaker, known_poll_no_progress, ping_pong
- 스마트 트렁케이션 (head+tail 70/30, important tail 감지, 400K 하드캡)
- `@agent.tool` 데코레이터로 커스텀 도구 등록

### 시스템 프롬프트 (`prompt/builder.py`)
- 원본 `buildAgentSystemPrompt` 포팅, 13-섹션 동적 조립
- `prompt_mode`: full / minimal (서브에이전트) / none
- 인젝션 방어: 13종 패턴, 호모글리프 폴딩, 암호화 경계 마커

### Failover (`model/failover.py`)
- 에러 분류 12종, 정규식 ~60개
- 쿨다운 윈도우, API 키 로테이션, 상태 영속화 (fcntl)
- 프로브 메커니즘 (300초 간격)

### 컨텍스트 가드 (`context/guard.py`)
- OK → COMPACT → ERROR 에스컬레이션
- enforce_budget(): 인플레이스 도구 결과 트렁케이션 + 반복 컴팩션

## 테스트

### 단위·통합 테스트 (`test_live.py`) — 25/25 PASS
- 오프라인 12개: ContextGuard, ToolRegistry, SessionLanes, Cron, Hook, 인젝션 방어, ThinkingLevel, 세션 영속성, 트렁케이션, 루프 감지, ApplyPatch, Failover
- 라이브 13개: 대화, Read, Bash, Write→Read, 다중 턴, WebFetch, Edit, 커스텀 도구, 스트리밍, 에러 처리, 한국어, 긴 출력, 수학 추론

### 장시간 운용 내구성 테스트 (`test_endurance.py`) — PASS
- **88턴, 30분, 컴팩션 13회, 기억 보존율 97.9%**
- 10개 페이즈: 앵커 팩트 심기 → 컨텍스트 가속 충전 → 기억 검증 → 세션 복원 → 복합 추론 → 스트레스

### 수정된 버그 3건
1. 스트리밍 도구 호출 중복 ID → `_parse_chunk(native=False)` + 누적 로직 분리
2. 다중 도구 결과 누락 → 리스트 반환 + `extend()` 평탄화
3. 프롬프트 기반 파싱 정규식 → 중첩 JSON 처리 가능하도록 수정

## 실행 방법

```bash
cd ~/Desktop/openclaw-py
source .venv/bin/activate
pip install -e .
openclaw-py              # 대화형 REPL
python test_live.py      # 단위·통합 테스트 (25개)
python test_endurance.py # 장시간 내구성 테스트 (88턴)
```

## 주의사항

- `config.toml`은 `.gitignore`에 포함됨 (API 키 보호)
- `config.example.toml`을 복사하여 `config.toml` 생성 후 사용
- 사용자는 한국어 선호
