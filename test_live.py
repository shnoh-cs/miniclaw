#!/usr/bin/env python3
"""비대화형 라이브 테스트 스크립트.

Agent Python API를 직접 사용하여 다양한 시나리오를 테스트한다.
실행: source .venv/bin/activate && python test_live.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# ── 색상 헬퍼 ─────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def passed(name: str, detail: str = "") -> None:
    extra = f" {DIM}({detail}){RESET}" if detail else ""
    print(f"  {GREEN}✓ PASS{RESET}  {name}{extra}")


def failed(name: str, detail: str = "") -> None:
    extra = f" {DIM}({detail}){RESET}" if detail else ""
    print(f"  {RED}✗ FAIL{RESET}  {name}{extra}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


# ── 테스트 러너 ──────────────────────────────────────────
results: list[tuple[str, bool, str]] = []


async def run_test(name: str, coro):
    """테스트를 실행하고 결과를 기록한다."""
    t0 = time.time()
    try:
        ok, detail = await coro
        elapsed = f"{time.time() - t0:.1f}s"
        if ok:
            passed(name, f"{detail} | {elapsed}")
        else:
            failed(name, f"{detail} | {elapsed}")
        results.append((name, ok, detail))
    except Exception as e:
        elapsed = f"{time.time() - t0:.1f}s"
        failed(name, f"{e} | {elapsed}")
        results.append((name, False, str(e)))


# ── 개별 테스트 ──────────────────────────────────────────

async def test_simple_chat(agent) -> tuple[bool, str]:
    """단순 대화 테스트."""
    result = await agent.run("안녕! 1+1은?", session_id="test-chat")
    ok = result.success and result.text and len(result.text) > 0
    return ok, f"응답 길이: {len(result.text or '')}자"


async def test_tool_read(agent) -> tuple[bool, str]:
    """Read 도구 사용 테스트."""
    result = await agent.run(
        "pyproject.toml 파일의 첫 5줄만 읽어줘", session_id="test-read"
    )
    ok = result.success and result.tool_calls_count > 0
    return ok, f"도구 호출 {result.tool_calls_count}회"


async def test_tool_bash(agent) -> tuple[bool, str]:
    """Bash 도구 사용 테스트."""
    result = await agent.run("echo 'hello openclaw' 실행해줘", session_id="test-bash")
    ok = result.success and result.tool_calls_count > 0
    has_hello = "hello" in (result.text or "").lower() or result.tool_calls_count > 0
    return ok and has_hello, f"도구 호출 {result.tool_calls_count}회"


async def test_tool_write_and_read(agent) -> tuple[bool, str]:
    """Write → Read 다중 도구 체인 테스트."""
    result = await agent.run(
        "/tmp/openclaw_test_file.txt 에 '테스트 성공!' 이라고 쓰고 다시 읽어서 내용 확인해줘",
        session_id="test-write-read",
    )
    ok = result.success and result.tool_calls_count >= 2
    return ok, f"도구 호출 {result.tool_calls_count}회"


async def test_multi_turn(agent) -> tuple[bool, str]:
    """다중 턴 대화 테스트."""
    sid = "test-multiturn"
    r1 = await agent.run("내 이름은 민수야. 기억해!", session_id=sid)
    r2 = await agent.run("내 이름이 뭐라고 했지?", session_id=sid)
    ok = r1.success and r2.success and "민수" in (r2.text or "")
    return ok, f"2턴 완료, 이름 기억: {'민수' in (r2.text or '')}"


async def test_web_fetch(agent) -> tuple[bool, str]:
    """WebFetch 도구 테스트."""
    result = await agent.run(
        "https://httpbin.org/get 에 접속해서 응답 내용 요약해줘",
        session_id="test-webfetch",
    )
    ok = result.success and result.tool_calls_count > 0
    return ok, f"도구 호출 {result.tool_calls_count}회"


async def test_edit_tool(agent) -> tuple[bool, str]:
    """Edit 도구 테스트."""
    # 먼저 파일 생성
    r1 = await agent.run(
        "/tmp/openclaw_edit_test.txt 에 'apple banana cherry' 라고 써줘",
        session_id="test-edit",
    )
    # 수정
    r2 = await agent.run(
        "/tmp/openclaw_edit_test.txt 에서 'banana'를 'mango'로 바꿔줘",
        session_id="test-edit",
    )
    ok = r1.success and r2.success and r2.tool_calls_count > 0
    return ok, f"도구 호출 {r1.tool_calls_count + r2.tool_calls_count}회"


async def test_custom_tool(agent) -> tuple[bool, str]:
    """커스텀 도구 등록 & 사용 테스트."""

    @agent.tool("dice_roll", description="Roll a dice and return the result")
    def dice_roll(sides: str = "6") -> str:
        return f"Rolled a {sides}-sided dice: result is 4"

    result = await agent.run("주사위를 굴려줘!", session_id="test-custom-tool")
    ok = result.success and ("4" in (result.text or "") or result.tool_calls_count > 0)
    return ok, f"커스텀 도구 호출 {result.tool_calls_count}회"


async def test_streaming(agent) -> tuple[bool, str]:
    """스트리밍 API 테스트."""
    chunks: list[str] = []
    async for chunk in agent.stream("하늘은 왜 파란색이야? 한 문장으로", session_id="test-stream"):
        chunks.append(chunk)
    full = "".join(chunks)
    ok = len(chunks) > 0 and len(full) > 10
    return ok, f"{len(chunks)}개 청크, 총 {len(full)}자"


async def test_error_handling(agent) -> tuple[bool, str]:
    """존재하지 않는 파일 읽기 → 에러 처리 테스트."""
    result = await agent.run(
        "/tmp/this_file_absolutely_does_not_exist_12345.txt 읽어줘",
        session_id="test-error",
    )
    # 에이전트가 에러를 받아서 사용자에게 알려줘야 함
    ok = result.success  # 도구 에러는 있지만 agent run 자체는 성공
    has_error_mention = any(
        kw in (result.text or "").lower()
        for kw in ["not found", "존재하지", "없", "error", "찾을 수"]
    )
    return ok and has_error_mention, f"에러 메시지 포함: {has_error_mention}"


async def test_context_guard(agent) -> tuple[bool, str]:
    """ContextGuard 동작 확인 (토큰 추정)."""
    from openclaw.context.guard import ContextGuard, ContextAction

    guard = ContextGuard(agent.config.context)
    status = guard.check(1000)
    ok = status.action == ContextAction.OK
    status2 = guard.check(30000)
    ok2 = status2.action in (ContextAction.COMPACT, ContextAction.ERROR)
    return ok and ok2, f"1K→{status.action.value}, 30K→{status2.action.value}"


async def test_tool_registry(agent) -> tuple[bool, str]:
    """ToolRegistry 등록 & 조회."""
    defs = agent.tool_registry.get_definitions()
    names = [d.name for d in defs]
    expected = {"read", "write", "edit", "bash", "web_fetch", "image"}
    found = expected & set(names)
    ok = len(found) == len(expected)
    return ok, f"등록된 도구 {len(names)}개, 필수 {len(found)}/{len(expected)}"


async def test_session_lanes(agent) -> tuple[bool, str]:
    """Session lanes 생성/관리."""
    from openclaw.session.lanes import LaneStatus

    lm = agent.get_lane_manager("test-lanes")
    lane = lm.create(name="sub-task", parent_lane_id="main")
    from openclaw.agent.types import AgentMessage, TextBlock

    lane.append(AgentMessage(role="user", content=[TextBlock(text="서브태스크 메시지")]))
    lane.append(AgentMessage(role="assistant", content=[TextBlock(text="완료!")]))

    ok1 = len(lm.list_active()) == 2
    summary = lm.merge_into_main(lane.id)
    ok2 = summary is not None and "서브태스크" in summary
    ok3 = lane.status == LaneStatus.COMPLETED
    ok4 = lm.main.message_count == 1
    return ok1 and ok2 and ok3 and ok4, f"레인 2개→병합→main {lm.main.message_count}메시지"


async def test_cron_scheduler(agent) -> tuple[bool, str]:
    """Cron 스케줄러 등록/상태 확인."""
    call_count = 0

    async def dummy_task() -> None:
        nonlocal call_count
        call_count += 1

    agent.scheduler.register("test_job", dummy_task, interval=0.2, one_shot=True)
    await agent.scheduler.start()
    await asyncio.sleep(0.5)  # one-shot이 한 번 실행될 시간
    await agent.scheduler.stop()

    status = agent.scheduler.status()
    test_job = next((s for s in status if s["name"] == "test_job"), None)
    ok = test_job is not None and test_job["run_count"] >= 1
    return ok, f"실행 횟수: {call_count}, 상태: {test_job['status'] if test_job else '?'}"


async def test_hook_runner(agent) -> tuple[bool, str]:
    """HookRunner fire-and-forget 테스트."""
    from openclaw.hooks import HookRunner
    from openclaw.config import HooksConfig

    log_file = "/tmp/openclaw_hook_test.log"
    Path(log_file).unlink(missing_ok=True)

    config = HooksConfig(
        pre_tool_call=f"echo 'hook:{{tool_name}}' >> {log_file}",
        timeout=5,
    )
    runner = HookRunner(config)
    await runner.fire("pre_tool_call", tool_name="bash")
    await asyncio.sleep(0.3)

    content = Path(log_file).read_text().strip() if Path(log_file).exists() else ""
    ok = "hook:bash" in content
    Path(log_file).unlink(missing_ok=True)
    return ok, f"로그 내용: {content[:50]}"


async def test_prompt_sanitize(agent) -> tuple[bool, str]:
    """프롬프트 인젝션 방어."""
    from openclaw.prompt.sanitize import sanitize_text

    dirty = "Hello\x00World\x08Test<script>alert(1)</script>"
    clean = sanitize_text(dirty)
    ok = "\x00" not in clean and "\x08" not in clean
    return ok, f"정리 후: {clean[:60]}"


async def test_thinking_levels(agent) -> tuple[bool, str]:
    """ThinkingLevel 파싱 & 폴백."""
    from openclaw.agent.types import ThinkingLevel

    t1 = ThinkingLevel.from_str("off")
    t2 = ThinkingLevel.from_str("ultrathink")
    t3 = ThinkingLevel.from_str("garbage")
    fb = ThinkingLevel.HIGH.fallback()
    ok = t1 == ThinkingLevel.OFF and t2 == ThinkingLevel.HIGH and t3 == ThinkingLevel.OFF
    ok2 = fb == ThinkingLevel.MEDIUM
    return ok and ok2, f"off={t1}, ultrathink={t2}, garbage={t3}, HIGH.fallback={fb}"


async def test_session_persistence(agent) -> tuple[bool, str]:
    """세션 파일 저장/로드."""
    from openclaw.session.manager import SessionManager

    tmp_dir = Path("/tmp/openclaw_test_sessions")
    tmp_dir.mkdir(exist_ok=True)
    sm = SessionManager(tmp_dir, "persist-test")
    from openclaw.agent.types import AgentMessage, TextBlock

    sm.load()
    sm.append(AgentMessage(role="user", content=[TextBlock(text="persist test")]))
    sm.append(AgentMessage(role="assistant", content=[TextBlock(text="ok!")]))

    # 다시 로드
    sm2 = SessionManager(tmp_dir, "persist-test")
    sm2.load()
    ok = len(sm2.messages) == 2 and sm2.messages[0].text == "persist test"

    # 정리
    sm.file_path.unlink(missing_ok=True)
    return ok, f"저장 {len(sm.messages)}개 → 로드 {len(sm2.messages)}개"


async def test_tool_truncation(agent) -> tuple[bool, str]:
    """도구 결과 트렁케이션 (head+tail)."""
    from openclaw.tools.registry import truncate_tool_result

    big = "A" * 10000
    truncated = truncate_tool_result(big, 500)
    ok = len(truncated) < len(big) and "truncated" in truncated.lower()
    return ok, f"10000자 → {len(truncated)}자"


async def test_loop_detection(agent) -> tuple[bool, str]:
    """도구 루프 감지."""
    from openclaw.tools.registry import ToolLoopDetector

    detector = ToolLoopDetector()
    for i in range(10):
        w = detector.record("read", {"path": "/same"}, "same content")
        if w and "CRITICAL" in w:
            return True, f"{i+1}회 반복 후 CRITICAL 감지"
    # 경고만이라도
    w = detector.record("read", {"path": "/same"}, "same content")
    ok = w is not None
    return ok, f"11회 반복, 경고: {(w or '')[:60]}"


async def test_korean_response(agent) -> tuple[bool, str]:
    """한국어 응답 테스트."""
    result = await agent.run(
        "대한민국의 수도는 어디야? 한 문장으로 답해줘.", session_id="test-korean"
    )
    ok = result.success and "서울" in (result.text or "")
    return ok, f"응답: {(result.text or '')[:80]}"


async def test_long_output(agent) -> tuple[bool, str]:
    """긴 출력 처리."""
    result = await agent.run(
        "1부터 20까지 숫자를 각각 별도 줄에 출력해. 반드시 모든 숫자를 빠짐없이 써줘:\n1\n2\n3\n...\n20\n이런 식으로.",
        session_id="test-long",
    )
    text = result.text or ""
    ok = result.success and len(text) > 20
    # 최소한 1, 10, 20이 포함되면 OK
    has_numbers = "1" in text and "20" in text
    return ok and has_numbers, f"응답 길이: {len(text)}자"


async def test_math_reasoning(agent) -> tuple[bool, str]:
    """수학 추론 테스트."""
    result = await agent.run(
        "127 x 83의 정확한 값은? 숫자만 답해줘.", session_id="test-math"
    )
    ok = result.success and "10541" in (result.text or "").replace(",", "").replace(" ", "")
    return ok, f"응답: {(result.text or '')[:50]}"


async def test_patch_tool(agent) -> tuple[bool, str]:
    """ApplyPatch 도구 확인 (등록 여부)."""
    defs = agent.tool_registry.get_definitions()
    names = [d.name for d in defs]
    ok = "apply_patch" in names
    return ok, f"apply_patch 등록: {ok}"


async def test_failover_config(agent) -> tuple[bool, str]:
    """Failover 설정 확인."""
    from openclaw.model.failover import FailoverManager

    fm = FailoverManager(fallback_models=["model-a", "model-b"])
    # should_failover가 True인 에러 (rate_limit, timeout 등)를 사용해야 함
    reason, next_model = fm.handle_error(Exception("rate limit exceeded 429"))
    ok = next_model == "model-a"
    reason2, next_model2 = fm.handle_error(Exception("timeout error 504"))
    ok2 = next_model2 == "model-b"
    return ok and ok2, f"1차→{next_model}, 2차→{next_model2}"


# ── 메인 ─────────────────────────────────────────────────


async def main() -> None:
    from openclaw.repl import Agent

    header("OpenClaw-Py 비대화형 라이브 테스트")

    agent = Agent.from_config()
    info(f"모델: {agent.config.models.default}")
    info(f"워크스페이스: {agent.workspace}")
    info(f"엔드포인트: {agent.config.endpoints.llm.base_url}")
    print()

    # ── 단위 테스트 (API 호출 없음) ──
    header("1. 단위 테스트 (오프라인)")
    await run_test("ContextGuard 동작", test_context_guard(agent))
    await run_test("ToolRegistry 등록/조회", test_tool_registry(agent))
    await run_test("Session Lanes", test_session_lanes(agent))
    await run_test("Cron 스케줄러", test_cron_scheduler(agent))
    await run_test("HookRunner", test_hook_runner(agent))
    await run_test("프롬프트 인젝션 방어", test_prompt_sanitize(agent))
    await run_test("ThinkingLevel 파싱", test_thinking_levels(agent))
    await run_test("세션 파일 저장/로드", test_session_persistence(agent))
    await run_test("도구 결과 트렁케이션", test_tool_truncation(agent))
    await run_test("도구 루프 감지", test_loop_detection(agent))
    await run_test("ApplyPatch 등록 확인", test_patch_tool(agent))
    await run_test("Failover 설정", test_failover_config(agent))

    # ── 라이브 테스트 (API 호출) ──
    header("2. 라이브 테스트 (API 호출)")
    await run_test("단순 대화", test_simple_chat(agent))
    await run_test("Read 도구", test_tool_read(agent))
    await run_test("Bash 도구", test_tool_bash(agent))
    await run_test("Write → Read 체인", test_tool_write_and_read(agent))
    await run_test("다중 턴 대화", test_multi_turn(agent))
    await run_test("WebFetch 도구", test_web_fetch(agent))
    await run_test("Edit 도구", test_edit_tool(agent))
    await run_test("커스텀 도구", test_custom_tool(agent))
    await run_test("스트리밍 API", test_streaming(agent))
    await run_test("에러 처리", test_error_handling(agent))
    await run_test("한국어 응답", test_korean_response(agent))
    await run_test("긴 출력 처리", test_long_output(agent))
    await run_test("수학 추론", test_math_reasoning(agent))

    # ── 결과 요약 ──
    header("결과 요약")
    total = len(results)
    pass_count = sum(1 for _, ok, _ in results if ok)
    fail_count = total - pass_count

    print(f"  전체: {total}개")
    print(f"  {GREEN}PASS: {pass_count}개{RESET}")
    if fail_count:
        print(f"  {RED}FAIL: {fail_count}개{RESET}")
        print(f"\n  {RED}실패 목록:{RESET}")
        for name, ok, detail in results:
            if not ok:
                print(f"    - {name}: {detail}")
    else:
        print(f"\n  {GREEN}{BOLD}🎉 전체 PASS!{RESET}")

    # 정리
    Path("/tmp/openclaw_test_file.txt").unlink(missing_ok=True)
    Path("/tmp/openclaw_edit_test.txt").unlink(missing_ok=True)

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
