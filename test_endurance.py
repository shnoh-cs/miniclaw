#!/usr/bin/env python3
"""장시간 운용 내구성 테스트.

컨텍스트를 의도적으로 빠르게 채워서 컴팩션을 여러 번 발동시키고,
매 단계마다 이전 대화 맥락을 기억하는지 검증한다.

목표:
- 컴팩션 3회 이상 발동
- 70턴 이상 연속 대화
- 매 페이즈마다 기억 유지 확인
- 세션 저장/복원 후 연속성 확인
- 대용량 도구 결과 처리 확인

실행: source .venv/bin/activate && python test_endurance.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import random
import string
from pathlib import Path
from dataclasses import dataclass, field

# ── 색상 ──────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ── 앵커 팩트: 모델이 기억해야 할 사실들 ─────────────────
# 각 페이즈에서 심고, 이후 페이즈에서 검증
ANCHOR_FACTS = {
    "phase1": {
        "plant": [
            ("내 이름은 '김태양'이야. 꼭 기억해.", "김태양"),
            ("내 생일은 1993년 8월 15일이야.", "1993"),
            ("나는 '프로젝트 오로라'라는 AI 시스템을 만들고 있어.", "오로라"),
        ],
        "verify_prompt": "내 이름, 생일, 진행 중인 프로젝트 이름이 뭐였지?",
        "verify_keys": ["김태양", "1993", "오로라"],
    },
    "phase2": {
        "plant": [
            ("우리 팀원은 이수현(백엔드), 박지민(프론트), 최은비(디자인) 3명이야.", "이수현"),
            ("서버는 AWS ap-northeast-2 리전에 있어. IP는 10.42.7.100이야.", "10.42.7.100"),
            ("데이터베이스는 PostgreSQL 15 사용하고, 테이블 287개야.", "287"),
        ],
        "verify_prompt": "우리 팀원 이름들, 서버 IP, DB 테이블 수가 뭐였지?",
        "verify_keys": ["이수현", "10.42.7.100", "287"],
    },
    "phase3": {
        "plant": [
            ("다음 배포 일정은 3월 28일이야. 코드네임은 '썬라이즈'야.", "썬라이즈"),
            ("버그 티켓 ORA-4521은 메모리 릭 문제야. 우선순위 P0.", "ORA-4521"),
            ("이번 스프린트 목표는 API 응답시간 200ms 이하로 줄이는 거야.", "200ms"),
        ],
        "verify_prompt": "다음 배포 코드네임, P0 버그 티켓 번호, 스프린트 목표치가 뭐였지?",
        "verify_keys": ["썬라이즈", "ORA-4521", "200ms"],
    },
}


@dataclass
class TurnRecord:
    """개별 턴 기록."""
    turn: int
    phase: str
    prompt: str
    response_len: int
    tool_calls: int
    compacted: bool
    elapsed: float
    error: str | None = None


@dataclass
class EnduranceStats:
    """내구성 테스트 통계."""
    turns: list[TurnRecord] = field(default_factory=list)
    compaction_count: int = 0
    total_tool_calls: int = 0
    memory_checks: list[tuple[str, int, int]] = field(default_factory=list)  # (phase, found, total)
    errors: list[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def elapsed_minutes(self) -> float:
        return (self.end_time - self.start_time) / 60

    @property
    def memory_retention_pct(self) -> float:
        if not self.memory_checks:
            return 0.0
        total_found = sum(f for _, f, _ in self.memory_checks)
        total_possible = sum(t for _, _, t in self.memory_checks)
        return (total_found / total_possible * 100) if total_possible > 0 else 0.0


def banner(text: str) -> None:
    width = 64
    print(f"\n{BOLD}{CYAN}{'━' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'━' * width}{RESET}\n")


def phase_header(num: int, title: str) -> None:
    print(f"\n{BOLD}{MAGENTA}── Phase {num}: {title} {'─' * (40 - len(title))}{RESET}\n")


def log_turn(turn: int, prompt_preview: str, result_len: int, tools: int,
             compacted: bool, elapsed: float, error: str | None = None) -> None:
    status = f"{GREEN}✓{RESET}" if not error else f"{RED}✗{RESET}"
    compact_tag = f" {YELLOW}[COMPACTED]{RESET}" if compacted else ""
    tool_tag = f" ⚙{tools}" if tools > 0 else ""
    err_tag = f" {RED}{error[:60]}{RESET}" if error else ""
    print(f"  {status} T{turn:03d} {DIM}{prompt_preview[:50]:50s}{RESET}"
          f" → {result_len:5d}자{tool_tag}{compact_tag}{err_tag}"
          f" {DIM}({elapsed:.1f}s){RESET}")


def check_memory(response: str, keys: list[str]) -> tuple[int, int]:
    """응답에서 기억해야 할 키워드가 몇 개 포함됐는지 확인."""
    found = sum(1 for k in keys if k in response)
    return found, len(keys)


async def send(agent, session_id: str, prompt: str, turn: int,
               phase: str, stats: EnduranceStats) -> str:
    """한 턴을 실행하고 기록."""
    t0 = time.time()
    try:
        result = await agent.run(prompt, session_id=session_id)
        elapsed = time.time() - t0
        text = result.text or ""
        compacted = result.compacted
        tool_calls = result.tool_calls_count
        error = result.error

        if compacted:
            stats.compaction_count += 1
        stats.total_tool_calls += tool_calls

        record = TurnRecord(
            turn=turn, phase=phase, prompt=prompt[:80],
            response_len=len(text), tool_calls=tool_calls,
            compacted=compacted, elapsed=elapsed, error=error,
        )
        stats.turns.append(record)
        log_turn(turn, prompt[:50], len(text), tool_calls, compacted, elapsed, error)

        if error:
            stats.errors.append(f"T{turn}: {error}")

        return text
    except Exception as e:
        elapsed = time.time() - t0
        stats.errors.append(f"T{turn}: EXCEPTION {e}")
        record = TurnRecord(
            turn=turn, phase=phase, prompt=prompt[:80],
            response_len=0, tool_calls=0, compacted=False,
            elapsed=elapsed, error=str(e),
        )
        stats.turns.append(record)
        log_turn(turn, prompt[:50], 0, 0, False, elapsed, str(e))
        return ""


async def main():
    from openclaw.repl import Agent
    from openclaw.agent.types import RunResult

    config_path = Path("config.toml")
    if not config_path.exists():
        print(f"{RED}Error: config.toml not found{RESET}")
        sys.exit(1)

    agent = Agent.from_config(str(config_path))
    model = agent.config.models.default
    max_tokens = agent.config.context.max_tokens

    banner("OpenClaw-Py 장시간 운용 내구성 테스트")
    print(f"  {DIM}모델: {model}{RESET}")
    print(f"  {DIM}컨텍스트 윈도우: {max_tokens:,} 토큰{RESET}")
    print(f"  {DIM}컴팩션 임계값: {agent.config.context.compaction_threshold:.0%}{RESET}")
    print(f"  {DIM}세션: endurance-test{RESET}")

    session_id = "endurance-test"
    stats = EnduranceStats(start_time=time.time())
    turn = 0

    # 이전 세션 정리
    session_dir = agent.config.session.resolved_dir
    old_session = session_dir / f"{session_id}.jsonl"
    if old_session.exists():
        old_session.unlink()
        print(f"  {DIM}이전 세션 파일 삭제{RESET}")

    # 임시 파일 생성용 디렉토리
    workspace = agent.config.workspace.resolved_dir
    workspace.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Phase 1: 앵커 팩트 심기 (기본 사실)
    # ================================================================
    phase_header(1, "앵커 팩트 심기 — 기본 정보")

    for prompt, _ in ANCHOR_FACTS["phase1"]["plant"]:
        turn += 1
        await send(agent, session_id, prompt, turn, "plant-1", stats)

    # 심은 직후 즉시 확인
    turn += 1
    resp = await send(agent, session_id,
                      ANCHOR_FACTS["phase1"]["verify_prompt"],
                      turn, "verify-1", stats)
    found, total = check_memory(resp, ANCHOR_FACTS["phase1"]["verify_keys"])
    stats.memory_checks.append(("phase1-immediate", found, total))
    print(f"  {CYAN}→ 기억 확인: {found}/{total}{RESET}")

    # ================================================================
    # Phase 2: 컨텍스트 가속 충전 — 1차 컴팩션 유도
    # ================================================================
    phase_header(2, "컨텍스트 가속 충전 → 1차 컴팩션")

    # 대용량 도구 호출로 컨텍스트를 빠르게 채움
    heavy_prompts = [
        "현재 디렉토리의 모든 Python 파일 목록을 보여줘. find 명령어로.",
        "pyproject.toml 파일 전체를 읽어줘.",
        "openclaw/agent/loop.py 파일 전체를 읽어줘.",
        "openclaw/repl.py 파일 전체를 읽어줘.",
        "openclaw/config.py 파일 전체를 읽어줘.",
        "openclaw/model/provider.py 파일 전체를 읽어줘.",
        "openclaw/session/compaction.py 파일 전체를 읽어줘.",
        "openclaw/tools/registry.py 파일 전체를 읽어줘.",
        "openclaw/prompt/builder.py 파일 전체를 읽어줘.",
        "openclaw/context/guard.py 파일 전체를 읽어줘.",
        "openclaw/memory/search.py 파일 전체를 읽어줘.",
        "openclaw/model/failover.py 파일 전체를 읽어줘.",
        "ls -la 결과를 보여줘.",
        "openclaw/prompt/sanitize.py 파일 전체를 읽어줘.",
        "openclaw/session/pruning.py 파일 전체를 읽어줘.",
    ]

    for prompt in heavy_prompts:
        turn += 1
        await send(agent, session_id, prompt, turn, "fill-1", stats)

    # 1차 컴팩션 후 기억 확인
    turn += 1
    resp = await send(agent, session_id,
                      "잠깐, 아까 내가 알려준 내 이름, 생일, 프로젝트 이름 기억해?",
                      turn, "verify-1-post", stats)
    found, total = check_memory(resp, ANCHOR_FACTS["phase1"]["verify_keys"])
    stats.memory_checks.append(("phase1-after-fill", found, total))
    print(f"  {CYAN}→ 1차 충전 후 기억 확인: {found}/{total}{RESET}")

    # ================================================================
    # Phase 3: 2차 앵커 팩트 심기 (팀 정보)
    # ================================================================
    phase_header(3, "2차 앵커 팩트 심기 — 팀 정보")

    for prompt, _ in ANCHOR_FACTS["phase2"]["plant"]:
        turn += 1
        await send(agent, session_id, prompt, turn, "plant-2", stats)

    # 즉시 확인
    turn += 1
    resp = await send(agent, session_id,
                      ANCHOR_FACTS["phase2"]["verify_prompt"],
                      turn, "verify-2", stats)
    found, total = check_memory(resp, ANCHOR_FACTS["phase2"]["verify_keys"])
    stats.memory_checks.append(("phase2-immediate", found, total))
    print(f"  {CYAN}→ 기억 확인: {found}/{total}{RESET}")

    # 1차 팩트도 여전히 기억하는지
    turn += 1
    resp = await send(agent, session_id,
                      "그리고 내 이름이랑 프로젝트 이름도 아직 기억해?",
                      turn, "cross-verify-1", stats)
    found, total = check_memory(resp, ANCHOR_FACTS["phase1"]["verify_keys"][:1] +
                                      ANCHOR_FACTS["phase1"]["verify_keys"][2:])
    stats.memory_checks.append(("phase1-cross-check", found, total))
    print(f"  {CYAN}→ 1차 팩트 교차 확인: {found}/{total}{RESET}")

    # ================================================================
    # Phase 4: 컨텍스트 가속 충전 — 2차 컴팩션 유도
    # ================================================================
    phase_header(4, "컨텍스트 가속 충전 → 2차 컴팩션")

    # Bash로 대용량 출력 생성
    heavy_prompts_2 = [
        "bash로 python3 -c \"import this\" 실행해줘.",
        "bash로 env 명령어 실행해서 환경변수 목록 보여줘.",
        "openclaw/tools/builtins/bash.py 파일 전체를 읽어줘.",
        "openclaw/tools/builtins/read.py 파일 전체를 읽어줘.",
        "openclaw/tools/builtins/write.py 파일 전체를 읽어줘.",
        "openclaw/tools/builtins/edit.py 파일 전체를 읽어줘.",
        "openclaw/tools/builtins/web_fetch.py 파일 전체를 읽어줘.",
        "openclaw/session/manager.py 파일 전체를 읽어줘.",
        "openclaw/memory/store.py 파일 전체를 읽어줘.",
        "openclaw/memory/embeddings.py 파일 전체를 읽어줘.",
        "test_live.py 파일 전체를 읽어줘.",
        "openclaw/hooks/__init__.py 파일 전체를 읽어줘.",
        "openclaw/session/lanes.py 파일 전체를 읽어줘.",
        "openclaw/cron/__init__.py 파일 전체를 읽어줘.",
        "openclaw/agent/types.py 파일 전체를 읽어줘.",
    ]

    for prompt in heavy_prompts_2:
        turn += 1
        await send(agent, session_id, prompt, turn, "fill-2", stats)

    # 2차 컴팩션 후 기억 확인 — 모든 팩트
    turn += 1
    resp = await send(agent, session_id,
                      "지금까지 내가 알려준 정보 총정리해줘: 내 이름, 생일, 프로젝트명, "
                      "팀원 이름들, 서버 IP, DB 테이블 수",
                      turn, "verify-all-mid", stats)
    all_keys = (ANCHOR_FACTS["phase1"]["verify_keys"] +
                ANCHOR_FACTS["phase2"]["verify_keys"])
    found, total = check_memory(resp, all_keys)
    stats.memory_checks.append(("all-after-fill2", found, total))
    print(f"  {CYAN}→ 2차 충전 후 전체 기억 확인: {found}/{total}{RESET}")

    # ================================================================
    # Phase 5: 3차 앵커 팩트 심기 (프로젝트 세부사항)
    # ================================================================
    phase_header(5, "3차 앵커 팩트 심기 — 프로젝트 세부")

    for prompt, _ in ANCHOR_FACTS["phase3"]["plant"]:
        turn += 1
        await send(agent, session_id, prompt, turn, "plant-3", stats)

    # ================================================================
    # Phase 6: 컨텍스트 가속 충전 — 3차 컴팩션 유도
    # ================================================================
    phase_header(6, "컨텍스트 가속 충전 → 3차 컴팩션")

    # 대용량 bash 출력으로 컨텍스트 충전
    heavy_prompts_3 = [
        "bash로 python3 -c \"for i in range(200): print(f'line {i}: ' + 'x'*80)\" 실행해줘.",
        "bash로 pip list 실행해줘.",
        "bash로 python3 -c \"import sys; print(sys.version_info); import os; print(os.uname())\" 실행해줘.",
        "openclaw/subagent/spawn.py 파일 전체를 읽어줘.",
        "openclaw/skills/loader.py 파일 전체를 읽어줘.",
        "openclaw/prompt/bootstrap.py 파일 전체를 읽어줘.",
        "openclaw/tools/builtins/apply_patch.py 파일 전체를 읽어줘.",
        "openclaw/model/thinking.py 파일 전체를 읽어줘.",
        "CLAUDE.md 파일 전체를 읽어줘.",
        "bash로 wc -l openclaw/**/*.py 실행해줘.",
        "openclaw/tools/builtins/memory_tool.py 파일 전체를 읽어줘.",
        "openclaw/tools/builtins/pdf_tool.py 파일 전체를 읽어줘.",
        "openclaw/session/memory_flush.py 파일이 있으면 읽어줘.",
        "bash로 python3 -c \"for i in range(300): print(f'data-{i:04d}: ' + '='*60)\" 실행해줘.",
        "bash로 du -sh * 실행해줘.",
    ]

    for prompt in heavy_prompts_3:
        turn += 1
        await send(agent, session_id, prompt, turn, "fill-3", stats)

    # 3차 컴팩션 후 전체 기억 확인
    turn += 1
    resp = await send(agent, session_id,
                      "다시 한번 확인할게. 내 이름, 프로젝트명, 배포 코드네임, "
                      "P0 버그 티켓 번호, 서버 IP 말해봐.",
                      turn, "verify-all-post3", stats)
    critical_keys = ["김태양", "오로라", "썬라이즈", "ORA-4521", "10.42.7.100"]
    found, total = check_memory(resp, critical_keys)
    stats.memory_checks.append(("critical-after-fill3", found, total))
    print(f"  {CYAN}→ 3차 충전 후 핵심 기억 확인: {found}/{total}{RESET}")

    # ================================================================
    # Phase 7: 세션 저장 → 복원 → 연속성 확인
    # ================================================================
    phase_header(7, "세션 복원 후 연속성 확인")

    # 새 Agent 인스턴스로 같은 세션 로드
    print(f"  {DIM}에이전트 재생성 (세션 복원 시뮬레이션)...{RESET}")
    agent2 = Agent.from_config(str(config_path))

    turn += 1
    resp = await send(agent2, session_id,
                      "세션이 복원됐어. 내가 누군지, 무슨 프로젝트 하는지 기억해?",
                      turn, "restore-verify", stats)
    found, total = check_memory(resp, ["김태양", "오로라"])
    stats.memory_checks.append(("session-restore", found, total))
    print(f"  {CYAN}→ 세션 복원 후 기억 확인: {found}/{total}{RESET}")

    # 복원 후에도 세부 정보 기억하는지
    turn += 1
    resp = await send(agent2, session_id,
                      "팀원 이름이랑 다음 배포 일정도 말해봐.",
                      turn, "restore-detail", stats)
    found, total = check_memory(resp, ["이수현", "3월 28일", "썬라이즈"])
    stats.memory_checks.append(("session-restore-detail", found, total))
    print(f"  {CYAN}→ 세션 복원 세부 정보: {found}/{total}{RESET}")

    agent = agent2  # 이후 agent2 사용

    # ================================================================
    # Phase 8: 복합 추론 과제 (기억 + 도구 + 논리)
    # ================================================================
    phase_header(8, "복합 추론 과제")

    reasoning_prompts = [
        ("프로젝트 오로라의 다음 배포까지 남은 시간을 계산해줘. "
         "오늘 날짜는 bash로 date 명령어로 확인하고, 배포일은 내가 말해준 걸 기억해서 계산해.",
         ["3월 28일", "썬라이즈"]),

        ("ORA-4521 버그의 우선순위가 뭐였지? "
         "그리고 이 버그를 해결하려면 어떤 팀원한테 맡기면 좋을지 추천해줘.",
         ["P0", "ORA-4521"]),

        ("우리 서버 IP가 뭐였지? 그 IP에 대해 bash로 "
         "python3 -c \"ip='10.42.7.100'; parts=ip.split('.'); print(f'subnet: {parts[0]}.{parts[1]}.{parts[2]}.0/24')\" "
         "실행해줘.",
         ["10.42.7.100", "10.42.7.0"]),

        ("지금까지 내가 알려준 모든 숫자형 데이터를 정리해줘: "
         "생년, 테이블 수, 목표 응답시간, 서버 IP의 마지막 옥텟 등",
         ["1993", "287", "200ms", "100"]),
    ]

    for prompt, keys in reasoning_prompts:
        turn += 1
        resp = await send(agent, session_id, prompt, turn, "reasoning", stats)
        found, total = check_memory(resp, keys)
        stats.memory_checks.append((f"reasoning-T{turn}", found, total))
        print(f"  {CYAN}→ 추론 기억: {found}/{total}{RESET}")

    # ================================================================
    # Phase 9: 4차 컴팩션 유도 + 최종 기억 확인
    # ================================================================
    phase_header(9, "4차 컴팩션 유도 → 최종 기억 확인")

    heavy_prompts_4 = [
        "bash로 python3 -c \"import json; d = {f'key_{i}': f'value_{i}' * 20 for i in range(100)}; print(json.dumps(d, indent=2))\" 실행해줘.",
        "openclaw/model/provider.py 파일 다시 읽어줘.",
        "openclaw/session/compaction.py 파일 다시 읽어줘.",
        "bash로 python3 -c \"for i in range(500): print(f'log-entry-{i:04d}: timestamp={i*1000} level=INFO msg=processing_request_{i}')\" 실행해줘.",
        "openclaw/tools/registry.py 파일 다시 읽어줘.",
        "bash로 python3 -c \"import string; [print(f'record_{i:03d}|' + ''.join(__import__('random').choices(string.ascii_letters, k=100))) for i in range(200)]\" 실행해줘.",
        "openclaw/agent/loop.py 파일 다시 읽어줘.",
        "openclaw/model/failover.py 파일 다시 읽어줘.",
        "bash로 python3 -c \"for i in range(400): print(f'metric_{i}: cpu={i%100}% mem={i%64}GB disk={i%500}GB')\" 실행해줘.",
        "openclaw/memory/search.py 파일 다시 읽어줘.",
    ]

    for prompt in heavy_prompts_4:
        turn += 1
        await send(agent, session_id, prompt, turn, "fill-4", stats)

    # 최종 종합 기억 확인
    turn += 1
    resp = await send(agent, session_id,
                      "마지막 테스트야. 지금까지 기억하고 있는 내 정보를 전부 말해봐: "
                      "이름, 생일, 프로젝트명, 팀원들, 서버IP, DB테이블수, "
                      "배포일, 코드네임, P0버그, 스프린트목표. 하나도 빠뜨리지 마.",
                      turn, "final-verify", stats)
    all_keys_final = [
        "김태양", "1993", "오로라",
        "이수현", "10.42.7.100", "287",
        "썬라이즈", "ORA-4521", "200ms",
    ]
    found, total = check_memory(resp, all_keys_final)
    stats.memory_checks.append(("FINAL", found, total))
    print(f"  {CYAN}→ 최종 기억 확인: {found}/{total}{RESET}")

    # ================================================================
    # Phase 10: 빠른 연속 대화 (내구성 스트레스)
    # ================================================================
    phase_header(10, "빠른 연속 대화 스트레스")

    rapid_prompts = [
        "1부터 10까지 합은?",
        "파이썬에서 리스트 컴프리헨션으로 짝수만 필터링하는 코드 한줄만.",
        "HTTP 상태코드 418의 의미는?",
        "김태양이 누구야?",  # 기억 확인
        "Base64 인코딩의 원리를 한 줄로 설명해.",
        "프로젝트 오로라 배포일이 언제야?",  # 기억 확인
        "bash로 echo hello world 실행해줘.",
        "TCP와 UDP의 핵심 차이 한 줄.",
        "ORA-4521 버그가 뭐였지?",  # 기억 확인
        "bash로 python3 -c \"print(2**64)\" 실행해줘.",
    ]

    for prompt in rapid_prompts:
        turn += 1
        await send(agent, session_id, prompt, turn, "rapid", stats)

    # 스트레스 후 기억 보존 확인
    turn += 1
    resp = await send(agent, session_id,
                      "내 이름이랑 프로젝트 이름 한 번만 더 확인.",
                      turn, "rapid-verify", stats)
    found, total = check_memory(resp, ["김태양", "오로라"])
    stats.memory_checks.append(("post-rapid", found, total))
    print(f"  {CYAN}→ 스트레스 후 기억: {found}/{total}{RESET}")

    # ================================================================
    # 결과 요약
    # ================================================================
    stats.end_time = time.time()

    banner("내구성 테스트 결과 요약")

    print(f"  총 턴 수:       {BOLD}{stats.total_turns}{RESET}")
    print(f"  소요 시간:      {BOLD}{stats.elapsed_minutes:.1f}분{RESET}")
    print(f"  컴팩션 횟수:    {BOLD}{stats.compaction_count}{RESET}")
    print(f"  도구 호출 수:   {BOLD}{stats.total_tool_calls}{RESET}")
    print(f"  에러 수:        {BOLD}{len(stats.errors)}{RESET}")
    print()

    # 기억 보존율 상세
    print(f"  {BOLD}기억 보존율 상세:{RESET}")
    for phase_name, found, total in stats.memory_checks:
        pct = found / total * 100 if total > 0 else 0
        color = GREEN if pct >= 80 else YELLOW if pct >= 50 else RED
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {phase_name:30s} {color}{found}/{total}{RESET} "
              f"{DIM}{bar} {pct:.0f}%{RESET}")

    print()
    overall_pct = stats.memory_retention_pct
    color = GREEN if overall_pct >= 80 else YELLOW if overall_pct >= 50 else RED
    print(f"  {BOLD}전체 기억 보존율: {color}{overall_pct:.1f}%{RESET}")
    print()

    # 컴팩션 이벤트 타임라인
    compact_turns = [t for t in stats.turns if t.compacted]
    if compact_turns:
        print(f"  {BOLD}컴팩션 발동 지점:{RESET}")
        for ct in compact_turns:
            print(f"    T{ct.turn:03d} ({ct.phase}) — {ct.elapsed:.1f}s")
    print()

    # 에러 목록
    if stats.errors:
        print(f"  {BOLD}{RED}에러 목록:{RESET}")
        for err in stats.errors[:10]:
            print(f"    {RED}• {err}{RESET}")
        if len(stats.errors) > 10:
            print(f"    {DIM}... 외 {len(stats.errors) - 10}건{RESET}")
    print()

    # 최종 판정
    if stats.compaction_count >= 3 and overall_pct >= 60:
        print(f"  {GREEN}{BOLD}✅ 장시간 운용 내구성 테스트 PASS{RESET}")
        print(f"  {DIM}컴팩션 {stats.compaction_count}회 발동, "
              f"기억 보존율 {overall_pct:.0f}%{RESET}")
    elif stats.compaction_count >= 2 and overall_pct >= 50:
        print(f"  {YELLOW}{BOLD}⚠️  부분 PASS (개선 필요){RESET}")
        print(f"  {DIM}컴팩션 {stats.compaction_count}회, "
              f"기억 보존율 {overall_pct:.0f}%{RESET}")
    else:
        print(f"  {RED}{BOLD}❌ 장시간 운용 내구성 테스트 FAIL{RESET}")
        if stats.compaction_count < 3:
            print(f"  {DIM}컴팩션 {stats.compaction_count}회 (목표: 3회 이상){RESET}")
        if overall_pct < 50:
            print(f"  {DIM}기억 보존율 {overall_pct:.0f}% (목표: 50% 이상){RESET}")

    print()

    # JSON 리포트 저장
    report_path = Path("endurance_report.json")
    report = {
        "total_turns": stats.total_turns,
        "elapsed_minutes": round(stats.elapsed_minutes, 1),
        "compaction_count": stats.compaction_count,
        "total_tool_calls": stats.total_tool_calls,
        "error_count": len(stats.errors),
        "memory_retention_pct": round(overall_pct, 1),
        "memory_checks": [
            {"phase": p, "found": f, "total": t}
            for p, f, t in stats.memory_checks
        ],
        "compaction_turns": [t.turn for t in compact_turns],
        "errors": stats.errors[:20],
        "turns": [
            {
                "turn": t.turn, "phase": t.phase,
                "response_len": t.response_len,
                "tool_calls": t.tool_calls,
                "compacted": t.compacted,
                "elapsed": round(t.elapsed, 1),
            }
            for t in stats.turns
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"  {DIM}리포트 저장: {report_path}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
