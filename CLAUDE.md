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

### 모델 프로바이더 (`openclaw/model/provider.py`)
- OpenAI 호환 API 클라이언트 (vLLM, OpenRouter 등)
- **Native/Prompt 듀얼 툴 콜링**: `tool_mode = auto|native|prompt`
- 스트리밍 도구 호출은 `pending_tool_calls` dict로 청크 누적 후 `finish_reason == "tool_calls"` 시점에 완성된 블록 1회 생성 (중복 방지)
- `_convert_message()`는 tool result를 리스트로 반환, `_build_api_messages()`에서 `extend()`로 평탄화

### 세션 (`openclaw/session/`)
- JSONL append-only, 파일 잠금(fcntl), 다중 세션 지원
- 다단계 컴팩션 (식별자 보존, safeguard 검증)
- Cache-TTL 프루닝 (in-memory only)
- 컴팩션 전 메모리 플러시

### 메모리 (`openclaw/memory/`)
- SQLite + FTS5 + 벡터 임베딩
- 하이브리드 검색: BM25 + cosine similarity + MMR 다양성 + 시간 감쇠 (30일 반감기)

### 도구 (`openclaw/tools/`)
- 10개 내장: Read, Write, Edit, ApplyPatch, Bash, Process, WebFetch, PDF, Image, Memory
- `ToolRegistry`: 등록/실행/루프 감지(4종)/결과 트렁케이션(head+tail 70/30)
- `@agent.tool` 데코레이터로 커스텀 도구 등록 가능

### 프롬프트 (`openclaw/prompt/`)
- 11-섹션 시스템 프롬프트 동적 조립
- 부트스트랩 파일 8종 (AGENTS.md, SOUL.md 등) 자동 로딩
- 프롬프트 인젝션 방어: 유니코드 Cc/Cf 제거, `<untrusted-text>` 래핑, HTML 이스케이핑

### 기타
- Thinking 레벨: off→minimal→low→medium→high→xhigh (폴백 체인)
- 에러 분류 & 페일오버: auth/billing/rate_limit/timeout/context_overflow
- 스킬 시스템: YAML frontmatter, OS/바이너리 게이팅
- 서브에이전트: 깊이 제한(max 5), 도구 정책

## 개발 현황

### 완료 (24개 테스트 전체 PASS)
- 모든 핵심 모듈 구현 완료
- Read/Write/Edit/Bash/WebFetch/Patch 도구 동작 확인
- 커스텀 도구, 다중 턴 세션, 스트리밍 API, 에러 처리 확인
- 컨텍스트 가드, 루프 감지, 트렁케이션, 인젝션 방어 확인

### 수정된 버그 3건
1. 스트리밍 도구 호출 중복 ID → `_parse_chunk(native=False)` + 누적 로직 분리
2. 다중 도구 결과 누락 → 리스트 반환 + `extend()` 평탄화
3. 프롬프트 기반 파싱 정규식 → 중첩 JSON 처리 가능하도록 수정

### 최근 추가 기능
- 한컴오피스 도구 (HWP/HWPX/Show/Cell 파일 읽기) — `hancom_tool.py`
- 이미지 분석 (비전 모델 연동, base64 인코딩) — `image_tool.py`
- 서브에이전트 (spawn → run → result 연결 완료) — `subagent` 도구
- 총 13개 내장 도구: Read, Write, Edit, ApplyPatch, Bash, Process, WebFetch, PDF, Hancom, Image, MemorySearch, MemorySave, Subagent
- Hook 시스템 (`openclaw/hooks/`) — pre/post tool_call, pre/post message, on_error 이벤트
- Session lanes (`openclaw/session/lanes.py`) — 병렬 대화 스레드, 서브에이전트 격리
- Cron/heartbeat (`openclaw/cron/`) — 주기적 모델 핑, 메모리 체크, 커스텀 스케줄 작업
- Failover 설정 (`config.toml`의 `models.fallback` 리스트로 장애 시 모델 전환)

### 미구현 / 추후 과제
- Failover 실환경 테스트 (다중 모델 전환)
- Cron 실환경 테스트

## 실행 방법

```bash
cd ~/Desktop/openclaw-py
source .venv/bin/activate
pip install -e .
openclaw-py          # 대화형 REPL
```

## 주의사항

- `config.toml`은 `.gitignore`에 포함됨 (API 키 보호)
- `config.example.toml`을 복사하여 `config.toml` 생성 후 사용
- 사용자는 한국어 선호
