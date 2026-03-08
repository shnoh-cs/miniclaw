# openclaw-py

OpenClaw Agent Harness의 Python 최소 포트. 모든 "지능(intelligence)" 기능을 100% 유지하면서 채널 통합(WhatsApp, Telegram, Discord 등)은 제거.

## 프로젝트 목적

- **사내 격리망(air-gapped network)** 환경에서 vLLM 서버 + 오픈소스 LLM으로 운용할 에이전트 하니스
- 현재 MacBook에서 OpenRouter API (`anthropic/claude-sonnet-4.6`)로 테스트 중
- GitHub: https://github.com/shnoh-cs/miniclaw

## 기술 스택

- Python 3.11+, 가상환경: `.venv/`
- 빌드: hatchling, 의존성: openai, tiktoken, numpy, pydantic, httpx, rich, beautifulsoup4, PyPDF2, PyYAML
- 설정: `config.toml` (gitignore됨, API 키 포함) / `config.example.toml` (vLLM 예시)

## 아키텍처 핵심

### 에이전트 루프 (`openclaw/agent/loop.py`)
- `run()` → `_attempt_loop()` → stream → tool dispatch → re-entry (최대 50턴)
- 컨텍스트 초과 시 자동 컴팩션, 프루닝 적용
- Hook 시스템 연동: pre/post tool_call, pre/post message, on_error
- 고아 tool_use 감지 → 합성 결과 주입
- 이미지 프루닝: 오래된 base64 이미지 자동 제거
- Thinking 레벨 폴백: thinking 관련 에러 시 자동 레벨 다운그레이드

### 모델 프로바이더 (`openclaw/model/provider.py`)
- OpenAI 호환 API 클라이언트 (vLLM, OpenRouter 등)
- **Native/Prompt 듀얼 툴 콜링**: `tool_mode = auto|native|prompt`
- 스트리밍 도구 호출은 `pending_tool_calls` dict로 청크 누적 후 `finish_reason == "tool_calls"` 시점에 완성된 블록 1회 생성 (중복 방지)
- `_convert_message()`는 tool result를 리스트로 반환, `_build_api_messages()`에서 `extend()`로 평탄화

### 세션 (`openclaw/session/`)
- JSONL append-only, 파일 잠금(fcntl), 다중 세션 지원
- **다단계 컴팩션**: split→summarize→merge (adaptive chunk ratio, SAFETY_MARGIN=1.2)
  - `split_messages_by_token_share()`: 토큰 비율 기반 균등 분할
  - `summarize_in_stages()`: 부분 요약 → 병합 요약
  - `summarize_with_fallback()`: 3단계 프로그레시브 폴백
  - `compute_adaptive_chunk_ratio()`: 대규모 메시지 자동 축소
  - safeguard 검증: 식별자 보존, 정렬된 섹션, ask 중복 검사
  - 파일 작업 추적 (`<read-files>`, `<modified-files>`), 도구 실패 추적 (최대 8건)
  - LLM 호출 재시도: 3회 + 지수 백오프
- Cache-TTL 프루닝 (in-memory only)
  - 비율 기반 게이팅 (softTrimRatio=0.3)
  - 반복 하드 클리어 + 부트스트랩 보호
  - 이미지 콘텐츠 추정 (ImageBlock당 8000자)
  - `prunable_tools` 필터링
- Session lanes (`openclaw/session/lanes.py`): 병렬 대화 스레드, 서브에이전트 격리
- 컴팩션 전 메모리 플러시

### 메모리 (`openclaw/memory/`)
- SQLite + FTS5 + 벡터 임베딩 (L2 정규화)
- 하이브리드 검색: BM25 + cosine similarity + MMR 다양성 + 시간 감쇠 (30일 반감기)
- 점수 정규화 (min-max → [0,1]) 후 MMR 적용
- 다국어 쿼리 확장 (EN/KO/ZH/JA 불용어 + 한국어 조사 제거 + CJK n-gram)
- 4배 후보 멀티플라이어, FTS 쿼리 빌딩 (토큰화→인용→AND)
- `FileWatcher`: 30초 디바운싱 자동 리인덱스
- `clamp_results_by_chars()`: 주입 컨텍스트 크기 제한
- `index_session_jsonl()`: 세션 트랜스크립트 메모리 인덱싱
- 캐시 임베딩: `get_all_embeddings_cached()` (numpy 매트릭스 + 캐시 무효화)
- `source_type` 필드로 소스 구분

### 도구 (`openclaw/tools/`)
- 14개 내장: Read, Write, Edit, ApplyPatch, Bash, Process, WebFetch, PDF, Hancom, Image, MemorySearch, MemorySave, MemoryGet, Subagent
- `ToolRegistry`: 등록/실행/루프 감지(4종)/결과 트렁케이션(head+tail 70/30)
  - `generic_repeat`: 경고 전용 (critical 아님)
  - `global_circuit_breaker`: 도구별 무진전 스트릭 (총 호출 카운터 아님)
  - `known_poll_no_progress`: 도구 특화 식별 (process, command_status)
  - `ping_pong`: 동적 스트릭 + 무진전 증거로 critical 판단
  - 경고 중복 제거 (버킷 기반, 10건당 1회)
  - 2단계 기록 (`record_call` + `record_outcome`)
  - `_has_important_tail()`: 에러/트레이스백/JSON 패턴 감지로 스마트 트렁케이션
  - `cap_tool_result_for_session()`: 400K 하드캡 (JSONL 저장 전)
  - `synthesize_missing_tool_result()`: 고아 tool_use 합성 결과
- `@agent.tool` 데코레이터로 커스텀 도구 등록 가능

### 프롬프트 (`openclaw/prompt/`)
- 13-섹션 시스템 프롬프트 동적 조립 (원본 OpenClaw `buildAgentSystemPrompt` 포팅)
  1. Identity
  2. Tooling + Tool Call Style
  3. Safety
  4. Skills (mandatory)
  5. Memory Recall
  6. Workspace
  7. Project Context (bootstrap files + SOUL.md 페르소나)
  8. Silent Replies
  9. Heartbeats
  10. Current Date & Time
  11. Runtime (os/python/model/thinking)
  12. Reasoning Format (`<think>`/`<final>` 구조)
  13. Compaction context
- `prompt_mode`: full (기본) / minimal (서브에이전트용) / none (bare identity)
- 부트스트랩 파일 8종 (AGENTS.md, SOUL.md 등) 자동 로딩
- 프롬프트 인젝션 방어:
  - 13종 의심 패턴 감지 (`detect_suspicious_patterns`)
  - 유니코드 Cc/Cf 제거, HTML 이스케이핑
  - 암호화 랜덤 경계 마커 + SECURITY NOTICE (`wrap_external_content`)
  - 전각 ASCII + 24종 꺾쇠 호모글리프 폴딩 (`fold_marker_text`)
  - `<untrusted-text>` 래핑 (빈 입력 가드, 트렁케이트 선 이스케이프 후)
  - 웹 콘텐츠 차별화 (`wrap_web_content`: fetch vs search)

### 컨텍스트 가드 (`openclaw/context/guard.py`)
- `enforce_budget()`: 인플레이스 도구 결과 트렁케이션 + 반복 컴팩션
- `_has_important_tail()`: 에러/트레이스백/JSON 패턴 감지
- 개행 경계 컷, 4K 꼬리 캡, 400K 하드 맥스
- `TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE=2`, `CONTEXT_INPUT_HEADROOM_RATIO=0.75`

### Failover (`openclaw/model/failover.py`)
- 에러 분류 12종: auth, auth_permanent, billing, rate_limit, overloaded, timeout, context_overflow, format, session_expired, server, network, unknown
- 정규식 패턴 ~60개 커버리지
- 불변 쿨다운 윈도우, 타임아웃 안전 로테이션, 오버로드 페이싱
- 만료 쿨다운 자동 정리, 재시도 반복 가드 (MAX=32)
- `ApiKeyRotator`: 프로바이더별 다중 키 로테이션
- 상태 영속화: `save_state`/`load_state` (fcntl 파일 잠금)
- 프로브 메커니즘: `should_probe_primary`/`probe_primary` (300초 간격)
- Thinking 레벨 폴백: thinking 관련 에러 시 자동 다운그레이드

### 기타
- Thinking 레벨: off→minimal→low→medium→high→xhigh (폴백 체인)
- 스킬 시스템: YAML frontmatter, OS/바이너리 게이팅
- 서브에이전트: 깊이 제한(max 5), 도구 정책
- Hook 시스템 (`openclaw/hooks/`): pre/post tool_call, pre/post message, on_error + timeout
- Cron/heartbeat (`openclaw/cron/`): 주기적 모델 핑, 메모리 체크, 커스텀 스케줄 작업

## 테스트

### 테스트 스위트 (`test_live.py`) — 25개 전체 PASS
- 오프라인 12개: ContextGuard, ToolRegistry, SessionLanes, Cron, Hook, 인젝션 방어, ThinkingLevel, 세션 저장/로드, 트렁케이션, 루프 감지, ApplyPatch, Failover
- 라이브 13개: 단순 대화, Read, Bash, Write→Read, 다중 턴, WebFetch, Edit, 커스텀 도구, 스트리밍, 에러 처리, 한국어, 긴 출력, 수학 추론

### 수정된 버그 3건
1. 스트리밍 도구 호출 중복 ID → `_parse_chunk(native=False)` + 누적 로직 분리
2. 다중 도구 결과 누락 → 리스트 반환 + `extend()` 평탄화
3. 프롬프트 기반 파싱 정규식 → 중첩 JSON 처리 가능하도록 수정

## 실행 방법

```bash
cd ~/Desktop/openclaw-py
source .venv/bin/activate
pip install -e .
openclaw-py          # 대화형 REPL
python test_live.py  # 전체 테스트
```

## 주의사항

- `config.toml`은 `.gitignore`에 포함됨 (API 키 보호)
- `config.example.toml`을 복사하여 `config.toml` 생성 후 사용
- 사용자는 한국어 선호
