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

> 테스트 환경: macOS Darwin 24.5.0 (Apple Silicon), Python 3.11.12
> 모델: `anthropic/claude-sonnet-4.6` via OpenRouter
> 날짜: 2026-03-08

---

### 1. 단위·통합 테스트 (`test_live.py`) — 25/25 PASS

#### 오프라인 테스트 (API 호출 없음)

```
  ✓ PASS  ContextGuard 동작       (1K→OK, 30K→COMPACT)
  ✓ PASS  ToolRegistry 등록/조회  (등록된 도구 13개, 필수 6/6)
  ✓ PASS  Session Lanes           (레인 2개→병합→main 1메시지)
  ✓ PASS  Cron 스케줄러            (실행 횟수: 1, 상태: completed)
  ✓ PASS  HookRunner              (로그 내용: hook:bash)
  ✓ PASS  프롬프트 인젝션 방어      (정리 후: HelloWorldTest<script>alert(1)</script>)
  ✓ PASS  ThinkingLevel 파싱      (off=OFF, ultrathink=HIGH, garbage=OFF, HIGH.fallback=MEDIUM)
  ✓ PASS  세션 파일 저장/로드       (저장 2개 → 로드 2개)
  ✓ PASS  도구 결과 트렁케이션      (10000자 → 2217자)
  ✓ PASS  도구 루프 감지            (15회 반복, 경고 1건: WARNING: You have called read 10 times...)
  ✓ PASS  ApplyPatch 등록 확인     (apply_patch 등록: True)
  ✓ PASS  Failover 설정           (1차→model-a, 2차→model-b)
```

#### 라이브 테스트 (API 호출)

```
  ✓ PASS  단순 대화     "안녕! 1+1은?"
                       → 에이전트: "1+1은 2입니다! 안녕하세요, 무엇을 도와드릴까요?"
                       → 도구 호출: 0회 | 2.2s

  ✓ PASS  Read 도구    "pyproject.toml 파일의 첫 5줄만 읽어줘"
                       → 에이전트: read 도구로 파일 읽기 → 내용 출력
                       → 도구 호출: 1회 (read) | 4.3s

  ✓ PASS  Bash 도구    "echo 'hello openclaw' 실행해줘"
                       → 에이전트: bash 도구 실행 → "hello openclaw" 출력 확인
                       → 도구 호출: 1회 (bash) | 4.6s

  ✓ PASS  Write→Read   "/tmp/openclaw_test_file.txt에 '테스트 성공!' 쓰고 다시 읽어서 확인해줘"
                       → 에이전트: write로 파일 생성 → read로 내용 확인
                       → 도구 호출: 2회 (write → read) | 8.0s

  ✓ PASS  다중 턴 대화  턴1: "내 이름은 민수야. 기억해!"
                       턴2: "내 이름이 뭐라고 했지?"
                       → 에이전트: "민수님이라고 하셨습니다."
                       → 세션 컨텍스트 유지 확인 | 3.7s

  ✓ PASS  WebFetch     "https://httpbin.org/get에 접속해서 응답 내용 요약해줘"
                       → 에이전트: web_fetch 도구로 HTTP GET → JSON 응답 파싱·요약
                       → 도구 호출: 1회 (web_fetch) | 8.0s

  ✓ PASS  Edit 도구    턴1: "파일에 'apple banana cherry' 써줘"
                       턴2: "'banana'를 'mango'로 바꿔줘"
                       → 에이전트: write로 생성 → edit으로 치환 → 결과 확인
                       → 도구 호출: 2회 (write + edit) | 8.2s

  ✓ PASS  커스텀 도구   @agent.tool("dice_roll") 등록 → "주사위를 굴려줘!"
                       → 에이전트: dice_roll 도구 호출 → "4가 나왔습니다"
                       → 도구 호출: 1회 (dice_roll) | 4.7s

  ✓ PASS  스트리밍 API  async for chunk in agent.stream("하늘은 왜 파란색이야?")
                       → 15개 청크 수신, 총 51자 | 2.5s

  ✓ PASS  에러 처리     "존재하지 않는 파일 읽어줘"
                       → 에이전트: read 도구 에러 수신 → "파일이 존재하지 않습니다" 안내
                       → 도구 에러를 우아하게 처리 | 4.8s

  ✓ PASS  한국어 응답   "대한민국의 수도는 어디야?"
                       → 에이전트: "대한민국의 수도는 **서울**입니다."
                       → 1.6s

  ✓ PASS  긴 출력 처리  "1부터 20까지 숫자를 각각 별도 줄에 출력해"
                       → 에이전트: 1~20 전부 출력 (50자) | 2.2s

  ✓ PASS  수학 추론     "127 x 83의 정확한 값은?"
                       → 에이전트: "10541"
                       → 1.5s
```

---

### 2. 장시간 운용 내구성 테스트 (`test_endurance.py`) — PASS

88턴에 걸쳐 컨텍스트를 의도적으로 과부하시키고, 컴팩션이 반복 발동되는 상황에서도 에이전트가 핵심 사실을 기억하는지 검증하는 테스트.

**심은 앵커 팩트 9개:**
- 이름: 김태양, 생일: 1993.8.15, 프로젝트: 오로라
- 팀원: 이수현/박지민/최은비, 서버IP: 10.42.7.100, DB테이블: 287개
- 배포 코드네임: 썬라이즈, 버그: ORA-4521 (P0), 목표: 200ms

#### Phase 1: 앵커 팩트 심기 (T001-T004)

```
  T001  "내 이름은 '김태양'이야. 꼭 기억해."          →  63자 ⚙1  (7.0s)
  T002  "내 생일은 1993년 8월 15일이야."              →  73자 ⚙2  (8.5s)
  T003  "나는 '프로젝트 오로라'라는 AI 시스템을 만들고 있어." → 129자 ⚙1  (6.5s)
  T004  "내 이름, 생일, 프로젝트 이름이 뭐였지?"         → 115자 ⚙1  (5.7s)
  → 기억 확인: 3/3 ✓ (김태양, 1993, 오로라)
```

#### Phase 2: 컨텍스트 가속 충전 → 1차 컴팩션 (T005-T020)

15개 소스 파일을 연속으로 읽어 컨텍스트를 가득 채움. 컴팩션 5회 발동.

```
  T005  find로 Python 파일 목록                    → 109자 ⚙1  (6.3s)
  T006  pyproject.toml 전체 읽기                   → 106자 ⚙1  (5.3s)
  T007  agent/loop.py 전체 읽기                    → 116자 ⚙2  (7.5s)
  T008  repl.py 전체 읽기                          → 469자 ⚙3  (59.4s)
  T009  config.py 전체 읽기                        → 647자 ⚙1  (11.2s)
  T010  model/provider.py 전체 읽기                → 751자 ⚙1  (12.1s)
  T011  session/compaction.py 전체 읽기            → 2545자 ⚙2  [COMPACTED] (44.0s)
  T012  tools/registry.py 전체 읽기                → 2153자 ⚙1  (30.3s)
  T013  prompt/builder.py 전체 읽기                → 2183자 ⚙1  [COMPACTED] (38.3s)
  T014  context/guard.py 전체 읽기                 → 2100자 ⚙1  (26.2s)
  T015  memory/search.py 전체 읽기                 → 3101자 ⚙1  [COMPACTED] (59.6s)
  T016  model/failover.py 전체 읽기                → 3554자 ⚙1  (42.1s)
  T017  ls -la 결과                               →  268자 ⚙1  [COMPACTED] (29.9s)
  T018  prompt/sanitize.py 전체 읽기               → 3001자 ⚙1  (32.5s)
  T019  session/pruning.py 전체 읽기               → 2937자 ⚙1  (35.6s)

  T020  "아까 내가 알려준 내 이름, 생일, 프로젝트 이름 기억해?"
        → 93자 [COMPACTED] (41.2s)
  → 1차 충전 후 기억 확인: 3/3 ✓ (컴팩션 5회 후에도 전부 기억)
```

#### Phase 3: 2차 앵커 팩트 심기 — 팀 정보 (T021-T025)

```
  T021  "팀원은 이수현, 박지민, 최은비 3명이야."       → 100자 ⚙2  (10.8s)
  T022  "서버는 AWS ap-northeast-2. IP는 10.42.7.100." →  84자 ⚙1  (7.3s)
  T023  "DB는 PostgreSQL 15, 테이블 287개."          →  51자 ⚙1  (5.3s)
  T024  "팀원 이름들, 서버 IP, DB 테이블 수가 뭐였지?"    → 149자      (2.7s)
  → 기억 확인: 3/3 ✓
  T025  "내 이름이랑 프로젝트 이름도 아직 기억해?"        →  50자      (2.0s)
  → 1차 팩트 교차 확인: 2/2 ✓ (김태양, 오로라)
```

#### Phase 4: 컨텍스트 가속 충전 → 2차 컴팩션 (T026-T041)

15개 파일 + bash 출력으로 재충전. 컴팩션 1회 추가 발동.

```
  T026  python3 -c "import this" 실행               → 267자 ⚙1  (6.3s)
  T027  env 환경변수 목록                            → 162자      (4.0s)
  T028-T040  나머지 소스 파일 13개 연속 읽기           (10-30s each)
  T036  test_live.py 전체 읽기                       → 2730자 ⚙2 [COMPACTED] (131.6s)

  T041  "지금까지 알려준 정보 총정리: 이름, 생일, 프로젝트명, 팀원, 서버IP, DB 테이블 수"
        → 386자 ⚙1  (9.2s)
  → 2차 충전 후 전체 기억 확인: 6/6 ✓ (6개 핵심 사실 전부 보존)
```

#### Phase 5: 3차 앵커 팩트 심기 — 프로젝트 세부 (T042-T044)

```
  T042  "다음 배포는 3월 28일. 코드네임은 '썬라이즈'."   → 113자 ⚙1  (8.8s)
  T043  "버그 ORA-4521은 메모리 릭. P0."              → 155자 ⚙1  (8.6s)
  T044  "스프린트 목표는 API 응답시간 200ms 이하."       → 151자 ⚙1  (8.8s)
```

#### Phase 6: 컨텍스트 가속 충전 → 3차 컴팩션 (T045-T060)

bash 대량 출력 + 추가 소스 파일. 컴팩션 2회 추가 발동.

```
  T045  python3으로 200줄 출력                       →  64자 ⚙1  [COMPACTED] (67.4s)
  T046  pip list                                    → 477자 ⚙1  (7.2s)
  T047-T059  소스 파일 + bash 출력 13개               (5-40s each)
  T055  memory_tool.py 전체 읽기                     → 1181자 ⚙1 [COMPACTED] (146.2s)

  T060  "내 이름, 프로젝트명, 배포 코드네임, P0 버그 번호, 서버 IP 말해봐."
        → 158자 ⚙1  (8.2s)
  → 3차 충전 후 핵심 기억 확인: 5/5 ✓ (김태양, 오로라, 썬라이즈, ORA-4521, 10.42.7.100)
```

#### Phase 7: 세션 복원 후 연속성 확인 (T061-T062)

에이전트 인스턴스를 완전히 재생성하고, 같은 세션 파일을 로드하여 연속성 검증.

```
  [에이전트 재생성 (세션 복원 시뮬레이션)]

  T061  "세션이 복원됐어. 내가 누군지, 무슨 프로젝트 하는지 기억해?"
        → 378자  (5.3s)
  → 세션 복원 후 기억 확인: 2/2 ✓ (김태양, 오로라)

  T062  "팀원 이름이랑 다음 배포 일정도 말해봐."
        → 149자  (3.6s)
  → 세션 복원 세부 정보: 3/3 ✓ (이수현, 3월 28일, 썬라이즈)
```

#### Phase 8: 복합 추론 과제 (T063-T066)

기억 + 도구 호출 + 논리 추론을 결합하는 복합 과제.

```
  T063  "프로젝트 오로라 배포까지 남은 시간 계산해. bash로 date 확인 후 계산."
        → 176자 ⚙2 (bash+계산)  (10.0s)
        → 추론 기억: 2/2 ✓ (3월 28일, 썬라이즈)

  T064  "ORA-4521 우선순위가 뭐였지? 어떤 팀원한테 맡기면 좋을지 추천해."
        → 420자 (기억+추론)  (8.0s)
        → 추론 기억: 2/2 ✓ (P0, ORA-4521 — 이수현에게 할당 추천)

  T065  "서버 IP가 뭐였지? bash로 서브넷 계산해줘."
        → 161자 ⚙1 (bash)  (7.9s)
        → 추론 기억: 2/2 ✓ (10.42.7.100, 10.42.7.0/24)

  T066  "내가 알려준 모든 숫자형 데이터 정리: 생년, 테이블 수, 목표 응답시간 등"
        → 533자 ⚙1  (12.8s)
        → 추론 기억: 3/4 (1993, 287, 200ms 기억. 마지막 옥텟 100은 미포함)
```

#### Phase 9: 4차 컴팩션 유도 → 최종 기억 확인 (T067-T077)

대용량 bash 출력 + 소스 파일로 4차 충전. 컴팩션 5회 추가 발동.

```
  T067  python3으로 100-key JSON 출력               →  92자 ⚙1  (7.1s)
  T068  provider.py 재읽기                          → 968자 ⚙1  [COMPACTED] (105.5s)
  T070  python3으로 500줄 로그 출력                  → 254자 ⚙1  [COMPACTED] (24.1s)
  T072  python3으로 200줄 랜덤 레코드 출력            →  91자 ⚙1  [COMPACTED] (31.4s)
  T074  failover.py 재읽기                          →   0자 ⚙2  [COMPACTED] (44.2s)
  T076  memory/search.py 재읽기                     → 1389자 ⚙1 [COMPACTED] (53.7s)

  T077  "마지막 테스트야. 기억하고 있는 내 정보를 전부 말해봐:
         이름, 생일, 프로젝트명, 팀원들, 서버IP, DB테이블수,
         배포일, 코드네임, P0버그, 스프린트목표. 하나도 빠뜨리지 마."
        → 449자 ⚙1  (10.7s)
  → 최종 기억 확인: 9/9 ✓ (9개 핵심 사실 전부 보존!)
```

#### Phase 10: 빠른 연속 대화 스트레스 (T078-T088)

10개 질문을 빠르게 연속 투하 (일반 지식 + 기억 확인 혼합).

```
  T078  "1부터 10까지 합은?"                  →  25자  (2.1s)  → "55"
  T079  "리스트 컴프리헨션으로 짝수 필터링"       →  56자  (2.1s)  → "[x for x in lst if x%2==0]"
  T080  "HTTP 418의 의미는?"                  → 140자  (4.6s)  → "I'm a Teapot"
  T081  "김태양이 누구야?"                     → 120자  (11.7s) → 기억 확인 ✓
  T082  "Base64 인코딩의 원리"                 →  83자  (3.4s)
  T083  "프로젝트 오로라 배포일이 언제야?"        →  62자  (2.2s)  → 기억 확인 ✓
  T084  "echo hello world 실행"              →  24자 ⚙1 (4.5s)
  T085  "TCP와 UDP 차이"                      →  74자  (3.8s)
  T086  "ORA-4521 버그가 뭐였지?"              → 162자  (4.5s)  → 기억 확인 ✓
  T087  "python3 -c print(2**64) 실행"       →  36자 ⚙1 (4.7s) → "18446744073709551616"
  T088  "내 이름이랑 프로젝트 이름 한 번만 더 확인."  →  34자  (2.7s)
  → 스트레스 후 기억: 2/2 ✓ (김태양, 오로라)
```

#### 내구성 테스트 결과 요약

```
  총 턴 수:       88
  소요 시간:      30.3분
  컴팩션 횟수:    13
  도구 호출 수:   91
  에러 수:        0

  기억 보존율 상세:
    phase1-immediate            3/3  ████████████████████ 100%
    phase1-after-fill           3/3  ████████████████████ 100%
    phase2-immediate            3/3  ████████████████████ 100%
    phase1-cross-check          2/2  ████████████████████ 100%
    all-after-fill2             6/6  ████████████████████ 100%
    critical-after-fill3        5/5  ████████████████████ 100%
    session-restore             2/2  ████████████████████ 100%
    session-restore-detail      3/3  ████████████████████ 100%
    reasoning-T63               2/2  ████████████████████ 100%
    reasoning-T64               2/2  ████████████████████ 100%
    reasoning-T65               2/2  ████████████████████ 100%
    reasoning-T66               3/4  ███████████████░░░░░  75%
    FINAL                       9/9  ████████████████████ 100%
    post-rapid                  2/2  ████████████████████ 100%

  전체 기억 보존율: 97.9%

  컴팩션 발동 지점:
    T011 (fill-1)  T013 (fill-1)  T015 (fill-1)  T017 (fill-1)  T020 (verify)
    T036 (fill-2)  T045 (fill-3)  T055 (fill-3)
    T068 (fill-4)  T070 (fill-4)  T072 (fill-4)  T074 (fill-4)  T076 (fill-4)

  ✅ 장시간 운용 내구성 테스트 PASS
```

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
