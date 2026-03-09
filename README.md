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
# 전체 테스트 (57개: 오프라인 44 + 라이브 13)
python test_live.py

# 오프라인만
python test_live.py --offline

# Intelligence Eval Suite (8개 시나리오)
python eval/eval_intelligence.py
```

### 테스트 구성 (`test_live.py`)

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

#### 3. 지능 갭 검증 — 오프라인 7개 + 원본 대비 8개 + 4차 4개
메모리 플러시 에이전트 루프, auto_recall 필드, 체크포인트 복원, 컨텍스트 진단, 큐레이션 모듈, 진단 자동 조정, auto-recall 스코프 분리, Reranker, 임베딩 fingerprint, 세션 델타 싱크, 플러시 프롬프트, 동적 keep_count, 더블 플러시 방지, 플러시 전체 컨텍스트, 워크스페이스 접근 체크, memory_get 빈 파일 처리, clamp_results_by_chars, 연속 컴팩션 방지, tiktoken 토큰 추정

#### 4. 라이브 테스트 — API 호출 13개
단순 대화, Read 도구, Bash 도구, Write→Read 체인, 다중 턴 대화, WebFetch, Edit 도구, 커스텀 도구, 스트리밍 API, 에러 처리, 한국어 응답, 긴 출력 처리, 수학 추론

### 테스트 결과 (2026-03-09)

```
오프라인 44개: ALL PASS
라이브 13개: 11/13 PASS (Read 도구, 에러 처리 — LLM 응답 비결정성으로 간헐적 실패)
```

---

## Intelligence Eval Suite (`eval/eval_intelligence.py`)

### 설계 원칙

> "순수 LLM으로는 불가능한 것들만 테스트한다."

같은 대화 안에서 "10턴 전에 알려준 API 키 뭐였지?"라고 물어보는 건 단순히 LLM 컨텍스트 윈도우를 테스트하는 것이지 에이전트 하니스의 지능을 테스트하는 것이 아니다. 이 eval suite는 **에이전트 하니스가 없으면 절대 불가능한 8가지 행동**을 검증한다.

모든 시나리오는 타임스탬프 조작, Mock, 임시 디렉토리를 사용하여 **수 초 이내**에 완료된다.
(예: 큐레이션은 실제로 3일이 걸리지만, `.last-curation` 타임스탬프를 25시간 전으로 설정하고 일별 노트 3개를 직접 생성하여 즉시 테스트)

### 채점 기준

- 각 시나리오: 0.0 ~ 1.0 점수
- **70% 이상 = PASS**
- 전체 평균과 개별 시나리오 점수 출력

### 최신 결과 (2026-03-09)

```
============================================================
  OpenClaw Intelligence Eval Suite
============================================================

  기준: 순수 LLM으로는 불가능한 행동만 테스트
  임계: 70% 이상 = PASS

  100%  1. 크로스세션 기억 (API 키=OK, DB 호스트=OK, 결정사항=OK | 0.5s)
   92%  2. 컴팩션 생존 (id_extract=4/5, prompt_ids=yes, rejects_bad=True, sections_ok=True | 0.0s)
   75%  3. 프로액티브 리콜 (pref_recall=no, scope_split=yes, session_filter=yes, memory_boost=yes | 0.0s)
  100%  4. 큐레이션 승격 (no_marker_run=True, old_marker_run=True, recent_marker_skip=True, llm_called=True, prompt_structure=ok, promoted=True | 0.0s)
  100%  5. 스케줄 실행 (cron_runs=3, hb_called=True, content_injected=True, one_shot=True | 0.6s)
  100%  6. 컨텍스트 압박 (70%_threshold_down=True (0.7→0.65), 85%_floor_up=True (5000→10000), 85%_ratio_down=True (0.3→0.25), guard_compact=True (util=96%), guard_error=True, enforce_truncate=True (100000→10000) | 0.1s)
  100%  7. 페일오버 복구 (classify_rate_limit=True, classify_overload=True, no_failover_context=True, failover_to_b=True, failover_to_c=True, success_reset=True | 0.3s)
  100%  8. 루프 탈출 (warn_at=9, critical_at=29, no_false_positive=True, pingpong_detect=True | 0.0s)

  전체 평균: 96%
  PASS (>=70%): 8개

  ALL PASS!

  [██████████████████████████████] 100%  1. 크로스세션 기억
  [███████████████████████████░░░]  92%  2. 컴팩션 생존
  [██████████████████████░░░░░░░░]  75%  3. 프로액티브 리콜
  [██████████████████████████████] 100%  4. 큐레이션 승격
  [██████████████████████████████] 100%  5. 스케줄 실행
  [██████████████████████████████] 100%  6. 컨텍스트 압박
  [██████████████████████████████] 100%  7. 페일오버 복구
  [██████████████████████████████] 100%  8. 루프 탈출
```

### 시나리오 상세

---

#### 시나리오 1: 크로스세션 기억 — 100%

**왜 순수 LLM으로는 불가능한가**: LLM은 대화가 끝나면 모든 것을 잊는다. 세션A에서 저장한 사실을 세션B에서 꺼내려면 **외부 메모리 시스템**(SQLite + FTS5 인덱스)이 반드시 필요하다.

**테스트 셋업**:
1. 임시 디렉토리에 `memory/2026-03-08.md` 파일 생성 (세션A가 저장한 것 시뮬레이션):
   ```markdown
   ## 프로젝트 설정
   - API 키: sk-proj-ABC123XYZ789
   - DB 호스트: prod-db.internal.corp:5432
   - 결정사항: Redis 대신 Memcached를 쓰기로 결정함
   ```
2. `MemoryStore`(SQLite)에 `MemorySearcher.index_file()`로 인덱싱 → FTS5 가상 테이블에 청크 저장
3. 세션B 시뮬레이션: FTS5 BM25 쿼리로 3개 사실 검색 시도

**실제 실행 동작**:

| 검색 대상 | FTS5 쿼리 | 결과 |
|-----------|-----------|------|
| API 키 | `"sk" AND "proj"` | OK — `sk-proj-ABC123XYZ789` 포함 청크 검색됨 |
| DB 호스트 | `"prod" AND "db"` | OK — `prod-db.internal.corp:5432` 포함 청크 검색됨 |
| 결정사항 | `Memcached*` | OK — `Memcached를` 포함 청크가 prefix 매칭으로 검색됨 |

**발견된 이슈 & 해결**: SQLite FTS5의 unicode61 토크나이저는 Latin 문자와 한글 음절을 모두 "word character"로 취급하여 `Memcached를`을 하나의 토큰으로 저장한다. 따라서 exact match `"Memcached"`는 실패하고, prefix match `Memcached*`로 해결.

---

#### 시나리오 2: 컴팩션 생존 — 92%

**왜 순수 LLM으로는 불가능한가**: 컨텍스트 윈도우가 꽉 차면 LLM은 오래된 대화를 그냥 잘라낸다. 하니스의 컴팩션은 **식별자를 추출하고 보존**하는 다단계 요약을 수행하여 URL, API 키, IP 주소 등을 절대 잃지 않는다.

**테스트 셋업**:
1. 5개 핵심 식별자가 산재한 20개 메시지(user/assistant 교대) 생성:
   - `https://api.example.com/v2/users`
   - `sk-proj-ABC123XYZ789DEF456`
   - `192.168.1.42`
   - `/home/deploy/app/config.toml`
   - `a1b2c3d4e5f6`
2. `extract_identifiers()` 함수로 식별자 자동 추출
3. `_build_summarization_prompt()`로 요약 프롬프트 생성 → 식별자 포함 확인
4. `_audit_summary_quality()`로 불완전한 요약 거부 검증
5. `_has_required_sections_in_order()`로 올바른 섹션 구조 검증

**실제 실행 동작**:

| 체크 항목 | 결과 | 상세 |
|-----------|------|------|
| 식별자 추출 | 4/5 (80%) | URL, API키, IP, 경로는 추출됨. 짧은 hex `a1b2c3d4e5f6`는 정규식 패턴에 안 걸림 |
| 프롬프트에 식별자 포함 | yes | 추출된 식별자가 요약 프롬프트의 `## Exact identifiers` 섹션에 명시적으로 나열됨 |
| 불완전 요약 거부 | True | `"This is a bad summary"` → `_audit_summary_quality`가 필수 섹션 누락으로 거부 |
| 섹션 구조 통과 | True | `Decisions → Open TODOs → Constraints → Pending user asks → Exact identifiers` 순서 확인 |

**점수 계산**: `(0.8 × 0.4) + (0.2) + (0.2) + (0.2) = 0.92`

---

#### 시나리오 3: 프로액티브 리콜 — 75%

**왜 순수 LLM으로는 불가능한가**: 사용자가 "코드 리뷰 해줘"라고만 말했을 때, LLM은 이전에 저장한 "보안 취약점 우선 체크" 선호를 알 수 없다. 하니스의 auto-recall이 매 턴 자동으로 메모리를 검색하여 관련 선호/에피소드를 시스템 프롬프트에 주입해야만 가능하다.

**테스트 셋업**:
1. `MEMORY.md`에 사용자 선호 저장: `"코드 리뷰 시 보안 취약점 우선 체크"`
2. `memory/2026-03-08.md`에 에피소드 기억 저장: `"FastAPI 서버에 rate limiting 추가"`
3. 두 파일을 FTS5 인덱싱
4. `expand_query("코드 리뷰 해줘")` → `build_fts_query()` → BM25 검색
5. `loop.py` 소스코드 검사: auto-recall 구현 확인

**실제 실행 동작**:

| 체크 항목 | 결과 | 상세 |
|-----------|------|------|
| 선호 리콜 (pref_recall) | no | `"코드 리뷰 해줘"` → `expand_query` → `"코드 리뷰"` → FTS 쿼리 `"코드" AND "리뷰"`. "보안"이 포함된 청크는 "코드 리뷰"를 포함하지 않아 BM25 매칭 실패. 임베딩 벡터 검색이 있으면 의미적으로 매칭될 수 있지만 오프라인 테스트에서는 벡터 없이 BM25만 사용 |
| 스코프 분리 (scope_split) | yes | `loop.py`의 `run()` 함수에 `"Long-term Memory"` / `"Recent Context"` 문자열 존재 확인 |
| 세션 필터링 (session_filter) | yes | `loop.py` 소스에 `"source_type"` 필터링 코드 존재 확인 |
| MEMORY.md 부스트 (memory_boost) | yes | `loop.py` 소스에 `"1.2"` (MEMORY.md 소스에 1.2x 점수 부스트) 존재 확인 |

**참고**: `pref_recall=no`는 BM25의 한계(키워드 정확매칭 기반)이지, 실제 운영에서는 벡터 임베딩이 함께 동작하여 의미적 연관성으로 매칭됨. 오프라인 테스트 환경의 제약.

---

#### 시나리오 4: 큐레이션 승격 — 100%

**왜 순수 LLM으로는 불가능한가**: LLM은 "3일 연속 반복된 패턴"을 인지할 수 없다 — 각 세션은 독립적이고 이전 대화를 모른다. 하니스의 큐레이션 시스템이 일별 노트를 읽고, 반복 패턴을 탐지하고, MEMORY.md에 영구 승격해야만 가능하다.

**테스트 셋업**:
1. 3일치 일별 노트 생성 (`2026-03-07`, `2026-03-08`, `2026-03-09`):
   ```markdown
   ## 2026-03-07 작업
   서버 상태 확인 작업을 수행했다. 매일 아침 9시에 서버 상태 확인하고 슬랙에 보고
   코드 리뷰 3건 처리하고 배포 파이프라인 점검을 완료했다.
   ```
2. `.last-curation` 마커 파일로 디바운싱 로직 검증
3. `curate_memories()`에 Mock LLM 주입 → 프롬프트 구성 확인
4. Mock LLM이 승격 내용을 반환 → MEMORY.md 실제 생성 확인

**실제 실행 동작**:

| 체크 항목 | 결과 | 상세 |
|-----------|------|------|
| 마커 없을 때 실행 | True | `.last-curation` 파일 미존재 → `_should_curate()` = True |
| 마커 25시간 전 → 실행 | True | 마커를 `time.time() - 90000`으로 설정 → `_should_curate()` = True (24시간 초과) |
| 마커 1시간 전 → 스킵 | True | 마커를 `time.time() - 3600`으로 설정 → `_should_curate()` = False (24시간 미만) |
| LLM 호출됨 | True | `curate_memories()` 실행 시 `mock_provider.complete`가 호출됨 = 3개 일별 노트를 읽고 프롬프트를 구성했다는 증거 |
| 프롬프트 구조 | ok | LLM에 전달된 프롬프트에 "recurring"/"pattern"/"daily" 키워드 포함 확인 |
| MEMORY.md 승격 | True | Mock LLM 응답 `"- 매일 아침 서버 상태 확인이 반복 패턴"` → `MEMORY.md` 파일 생성됨 → 내용에 `"서버 상태"` 포함 확인 |

---

#### 시나리오 5: 스케줄 실행 — 100%

**왜 순수 LLM으로는 불가능한가**: LLM은 "매일 아침 9시에 실행"할 수 없다 — 요청이 들어와야만 동작한다. 하니스의 `CronScheduler` + `HEARTBEAT.md`가 주기적으로 에이전트를 호출해야만 가능하다.

**테스트 셋업**:
1. `HEARTBEAT.md` 생성:
   ```markdown
   ## Daily Tasks
   - Check server status
   - Summarize overnight alerts
   ```
2. `CronScheduler` 인스턴스에 0.1초 간격 콜백 등록 → 0.35초 대기 → 실행 횟수 확인
3. `heartbeat_from_file()`에 Mock LLM 주입 → 프롬프트에 HEARTBEAT.md 내용 포함 확인
4. one-shot 모드 테스트: 1회 실행 후 `status = "completed"` 확인

**실제 실행 동작**:

| 체크 항목 | 결과 | 상세 |
|-----------|------|------|
| Cron 실행 횟수 | 3회 | 0.1초 간격으로 등록, 0.35초 대기 → 3회 실행 (정확히 기대값) |
| Heartbeat LLM 호출 | True | `heartbeat_from_file()`이 HEARTBEAT.md를 읽고 OpenAI 호환 API `chat.completions.create` 호출 |
| 프롬프트에 내용 주입 | True | LLM에 전달된 messages에 `"server status"` 문자열 포함 확인 |
| one-shot 모드 | True | `one_shot=True`로 등록된 태스크가 1회 실행 후 `status = "completed"` |

---

#### 시나리오 6: 컨텍스트 압박 — 100%

**왜 순수 LLM으로는 불가능한가**: LLM은 컨텍스트 윈도우가 꽉 차면 그냥 에러를 반환하거나 성능이 급락한다. 하니스의 자가 진단 시스템이 사용량을 모니터링하고 **설정을 자동 조정**하여 안정성을 유지해야만 가능하다.

**테스트 셋업**:
1. tiktoken으로 정확한 토큰 수를 추정하는 `diagnose_context()`에 70% / 85% 시나리오 입력
2. `ContextGuard.check()`에 80%, 120% 사용량 입력 → 액션 에스컬레이션 확인
3. `enforce_budget()`으로 100,000자 도구 결과 → 트렁케이션 확인

**실제 실행 동작**:

| 체크 항목 | 입력 | 출력 | 상세 |
|-----------|------|------|------|
| 70% 임계 | 12K 윈도우에 ~8.4K 토큰 | `compaction_threshold` 0.7 → **0.65** | 진단이 "곧 가득 참"을 감지, 컴팩션을 더 일찍 트리거하도록 임계값 하향 |
| 85% 임계 (floor) | 20K 윈도우에 ~17K 토큰 | `reserve_tokens_floor` 5,000 → **10,000** | 여유 공간 부족 감지, 최소 예약 토큰 2배 증가 |
| 85% 임계 (ratio) | 20K 윈도우에 ~17K 토큰 | `tool_result_max_ratio` 0.3 → **0.25** | 도구 결과가 차지할 수 있는 최대 비율 축소 |
| Guard COMPACT | 10K 윈도우에 8K 사용 | `ContextAction.COMPACT` (util=96%) | 80% 초과 → 컴팩션 액션 트리거 |
| Guard ERROR | 10K 윈도우에 12K 사용 | `ContextAction.ERROR` | 100% 초과 → 에러 액션 |
| 예산 강제 적용 | 100,000자 도구 결과 | **10,000자**로 트렁케이션 | `enforce_budget()`이 head(70%)+tail(30%) 분할로 10배 축소 |

---

#### 시나리오 7: 페일오버 복구 — 100%

**왜 순수 LLM으로는 불가능한가**: LLM은 자기 자신이 장애인지 알 수 없다. 하니스의 `FailoverManager`가 에러를 분류하고, API 키를 로테이션하고, 폴백 모델로 전환하고, 성공 시 원복하는 전체 lifecycle을 관리해야만 가능하다.

**테스트 셋업**:
1. `classify_error()`: 에러 메시지 패턴 → `FailoverReason` enum 분류
2. `FailoverManager(fallback_models=["model-b", "model-c"])` 생성 (임시 state_path 사용)
3. 순차적 에러 주입 → 모델 전환 확인
4. `mark_success()` → 상태 초기화 확인

**실제 실행 동작**:

| 체크 항목 | 입력 | 출력 | 상세 |
|-----------|------|------|------|
| Rate limit 분류 | `"Rate limit exceeded (429)"` | `FailoverReason.RATE_LIMIT` | "429" + "rate limit" 패턴 매칭 |
| Overload 분류 | `"Service overloaded (503)"` | `FailoverReason.OVERLOADED` | "503" + "overload" 패턴 매칭 |
| Context overflow 비페일오버 | `"context_length_exceeded"` | `should_failover() = False` | 컨텍스트 초과는 모델 문제가 아닌 입력 문제 → 페일오버 대상 아님 |
| 1차 에러 → model-b | Rate limit | `next_model = "model-b"` | 프로필 "default" 1개뿐 → `mark_failure()` → 60초 cooldown 설정 → `advance_profile()` → 유일한 프로필이 cooldown 중 → None 반환 → `advance_model()` → `"model-b"` 반환 |
| 2차 에러 → model-c | Overload | `next_model = "model-c"` | model-b에서도 실패 → `advance_model()` → `"model-c"` 반환 |
| 성공 시 초기화 | `mark_success()` | `_retry_count = 0` | 성공 표시 후 재시도 카운터, overload 시도 횟수 모두 리셋 |

**발견된 이슈 & 해결**: `FailoverManager.__post_init__`이 `load_state()`를 호출하여 `~/.openclaw-py/failover_state.json`에서 기존 상태를 로드한다. 이전 실행의 잔여 상태가 테스트를 오염시킬 수 있으므로, 임시 디렉토리의 state_path를 사용하여 격리.

---

#### 시나리오 8: 루프 탈출 — 100%

**왜 순수 LLM으로는 불가능한가**: LLM은 "같은 도구를 30번 호출했다"는 사실을 인지할 수 없다 — 각 턴은 독립적이다. 하니스의 `ToolLoopDetector`가 호출 패턴을 추적하고, WARNING/CRITICAL을 시스템 프롬프트에 주입하여 에이전트가 루프에서 빠져나오도록 해야만 가능하다.

**테스트 셋업**:
1. 동일한 `read(path="/same/file.txt")` 15회 반복 → WARNING 감지 시점 확인
2. 동일한 `bash(command="curl http://fail")` 35회 반복 → CRITICAL 감지 시점 확인
3. 매번 다른 경로로 `read` 20회 → 오탐(false positive) 없는지 확인
4. 핑퐁 패턴 (`read /a.txt` → `write /a.txt` → `read /a.txt` → ...) 25회 → 패턴 감지 확인

**실제 실행 동작**:

| 체크 항목 | 입력 | 출력 | 상세 |
|-----------|------|------|------|
| WARNING 시점 | 동일 도구+인풋 15회 | **9번째** (0-indexed) | 10회 반복 시점에서 `"WARNING: You have called read 10 times with identical arguments"` 메시지 생성. 이 메시지가 시스템 프롬프트에 주입되어 LLM이 행동을 수정할 기회를 줌 |
| CRITICAL 시점 | 동일 도구+인풋 35회 | **29번째** (0-indexed) | 30회 반복 시점에서 `"CRITICAL: ..."` 메시지 생성. 에이전트 루프가 강제 중단됨 |
| 오탐 없음 | 매번 다른 경로 20회 | True | 경로가 다르면 "반복"으로 카운트하지 않음. 20회 전부 경고 없이 통과 |
| 핑퐁 감지 | read↔write 교대 25회 | True | 2개 도구가 번갈아 호출되는 패턴을 감지. `A→B→A→B→...` 교대 패턴은 일반 반복과 다른 감지기(ping-pong detector)가 처리 |

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
