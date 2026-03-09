#!/usr/bin/env python3
"""OpenClaw Intelligence Eval Suite.

순수 LLM으로는 불가능한 에이전트 하니스 고유 행동 8가지를 검증한다.
모든 시나리오는 시간/상태를 조작하여 수 초 이내에 실행된다.

실행: .venv/bin/python eval/eval_intelligence.py
"""

from __future__ import annotations

import asyncio
import datetime
import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── 색상 헬퍼 ─────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

results: list[tuple[str, float, str]] = []


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def report(name: str, score: float, detail: str = "") -> None:
    """0.0~1.0 점수 기록. 0.7 이상이면 PASS."""
    results.append((name, score, detail))
    color = GREEN if score >= 0.7 else (YELLOW if score >= 0.4 else RED)
    extra = f" {DIM}({detail}){RESET}" if detail else ""
    print(f"  {color}{score:.0%}{RESET}  {name}{extra}")


async def run_eval(name: str, coro) -> None:
    t0 = time.time()
    try:
        score, detail = await coro
        elapsed = f"{time.time() - t0:.1f}s"
        report(name, score, f"{detail} | {elapsed}")
    except Exception as e:
        elapsed = f"{time.time() - t0:.1f}s"
        report(name, 0.0, f"ERROR: {e} | {elapsed}")


# ══════════════════════════════════════════════════════════
# 시나리오 1: 크로스세션 기억
# 세션A에서 사실 저장 → 세션B에서 auto-recall로 회수
# ══════════════════════════════════════════════════════════

async def eval_cross_session_memory() -> tuple[float, str]:
    """세션 간 기억 연속성: 세션A 저장 → 세션B에서 회수."""
    from openclaw.memory.embeddings import EmbeddingProvider
    from openclaw.memory.search import MemorySearcher
    from openclaw.memory.store import MemoryStore
    from openclaw.config import MemoryConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_dir = Path(tmpdir) / "memory"
        memory_dir.mkdir()

        # 세션A: 사실 3개 저장 (파일에 직접 기록)
        facts = {
            "api_key": "sk-proj-ABC123XYZ789",
            "db_host": "prod-db.internal.corp:5432",
            "decision": "Redis 대신 Memcached를 쓰기로 결정함",
        }
        note_file = memory_dir / "2026-03-08.md"
        note_file.write_text(
            f"## 프로젝트 설정\n"
            f"- API 키: {facts['api_key']}\n"
            f"- DB 호스트: {facts['db_host']}\n"
            f"- 결정사항: {facts['decision']}\n",
            encoding="utf-8",
        )

        # 메모리 인덱싱 (세션B 시작 시 일어나는 것과 동일)
        db_path = Path(tmpdir) / "memory.sqlite"
        store = MemoryStore(db_path)
        config = MemoryConfig(dir=str(memory_dir))

        # 임베딩 없이 BM25만으로 검색 (오프라인 테스트)
        searcher = MemorySearcher(store, AsyncMock(), config)

        # 파일 인덱싱 (BM25용 FTS5 인덱스)
        await searcher.index_file(note_file)

        # 세션B: 키워드 검색으로 회수
        queries = [
            ("API 키", "sk-proj-ABC123XYZ789", '"sk" AND "proj"'),
            ("DB 호스트", "prod-db.internal.corp", '"prod" AND "db"'),
            # FTS5 unicode61 merges Latin+Hangul into one token ("memcached를"),
            # so we need prefix matching to find "Memcached" in "Memcached를"
            ("결정사항", "Memcached", 'Memcached*'),
        ]

        found = 0
        details = []
        for label, expected, fts_query in queries:
            results_list = store.bm25_search(fts_query, limit=5)
            if results_list:
                chunk_ids = [cid for cid, _ in results_list]
                chunks = store.get_chunks_by_ids(chunk_ids)
                text = " ".join(c.text for c in chunks)
                if expected.split(":")[0] in text or expected.split("-")[0] in text:
                    found += 1
                    details.append(f"{label}=OK")
                else:
                    details.append(f"{label}=text_miss")
            else:
                details.append(f"{label}=no_hit")

        score = found / len(queries)
        return score, ", ".join(details)


# ══════════════════════════════════════════════════════════
# 시나리오 2: 컴팩션 생존
# 50+ 메시지 → 컴팩션 → 식별자 보존 확인
# ══════════════════════════════════════════════════════════

async def eval_compaction_survival() -> tuple[float, str]:
    """컴팩션 후 핵심 식별자 보존."""
    from openclaw.session.compaction import (
        extract_identifiers,
        _build_summarization_prompt,
        _has_required_sections_in_order,
        _audit_summary_quality,
    )
    from openclaw.agent.types import AgentMessage, TextBlock

    # 핵심 식별자가 포함된 대화 생성
    identifiers = [
        "https://api.example.com/v2/users",
        "sk-proj-ABC123XYZ789DEF456",
        "192.168.1.42",
        "/home/deploy/app/config.toml",
        "a1b2c3d4e5f6",
    ]

    messages = []
    for i in range(20):
        if i % 5 == 0:
            # 식별자 포함 메시지
            idx = i // 5 % len(identifiers)
            text = f"Turn {i}: Working with {identifiers[idx]} for the deployment."
        else:
            text = f"Turn {i}: This is a filler conversation message about various topics."
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(AgentMessage(role=role, content=[TextBlock(text=text)]))

    # 식별자 추출 테스트
    extracted = extract_identifiers(
        " ".join(m.text for m in messages)
    )

    # 추출된 식별자 중 원본과 매칭되는 수
    matched = 0
    for ident in identifiers:
        for ext in extracted:
            if ext in ident or ident in ext:
                matched += 1
                break

    # 요약 프롬프트 빌드 테스트
    context_text = "\n".join(f"[{m.role.upper()}]: {m.text}" for m in messages)
    prompt = _build_summarization_prompt(context_text, "strict", extracted)

    checks = []
    # 1. 식별자 추출 정확도
    id_score = matched / len(identifiers)
    checks.append(f"id_extract={matched}/{len(identifiers)}")

    # 2. 프롬프트에 식별자 목록 포함
    prompt_has_ids = all(
        any(ext in prompt for ext in extracted)
        for _ in identifiers[:3]  # 상위 3개만 확인
    ) if extracted else False
    checks.append(f"prompt_ids={'yes' if prompt_has_ids else 'no'}")

    # 3. 감사 함수가 불완전한 요약을 거부하는지
    bad_summary = "This is a bad summary without required sections."
    ok, reasons = _audit_summary_quality(bad_summary, extracted, "deploy stuff", "strict")
    rejects_bad = not ok
    checks.append(f"rejects_bad={rejects_bad}")

    # 4. 올바른 섹션이 있는 요약은 통과하는지
    good_summary = (
        "## Decisions\nUsed API endpoint for deployment.\n"
        "## Open TODOs\nFinalize config.\n"
        "## Constraints\nMust use internal network.\n"
        "## Pending user asks\nNone.\n"
        "## Exact identifiers\n"
        + "\n".join(f"- {e}" for e in extracted[:5])
    )
    has_sections = _has_required_sections_in_order(good_summary)
    checks.append(f"sections_ok={has_sections}")

    total = (id_score * 0.4) + (0.2 if prompt_has_ids else 0) + (0.2 if rejects_bad else 0) + (0.2 if has_sections else 0)
    return total, ", ".join(checks)


# ══════════════════════════════════════════════════════════
# 시나리오 3: 프로액티브 리콜
# 메모리에 저장만 하고 질문 안 함 → auto-recall 코드가 꺼내는가
# ══════════════════════════════════════════════════════════

async def eval_proactive_recall() -> tuple[float, str]:
    """auto-recall이 관련 기억을 자동으로 꺼내는지."""
    from openclaw.memory.search import MemorySearcher, SearchResult
    from openclaw.memory.store import MemoryChunk, MemoryStore
    from openclaw.config import MemoryConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_dir = Path(tmpdir) / "memory"
        memory_dir.mkdir()

        # MEMORY.md에 장기 기억 저장
        memory_md = Path(tmpdir) / "MEMORY.md"
        memory_md.write_text(
            "## User Preferences\n"
            "- 사용자는 항상 한국어로 응답받기를 선호\n"
            "- 코드 리뷰 시 보안 취약점 우선 체크\n"
            "- 커밋 메시지는 conventional commits 형식\n",
            encoding="utf-8",
        )

        # 일별 노트에 에피소드 기억
        daily = memory_dir / "2026-03-08.md"
        daily.write_text(
            "## 작업 내역\n"
            "- FastAPI 서버에 rate limiting 추가\n"
            "- Redis 세션 스토어 연결 완료\n",
            encoding="utf-8",
        )

        db_path = Path(tmpdir) / "memory.sqlite"
        store = MemoryStore(db_path)
        config = MemoryConfig(dir=str(memory_dir))
        searcher = MemorySearcher(store, AsyncMock(), config)

        await searcher.index_file(memory_md)
        await searcher.index_file(daily)

        # auto-recall 시뮬레이션: 사용자 입력과 관련 있는 기억 검색
        # (실제 loop.py에서는 매 턴 자동 실행)
        user_input = "코드 리뷰 해줘"

        # BM25 검색 (벡터 없이)
        from openclaw.memory.search import build_fts_query, expand_query
        expanded = expand_query(user_input)
        fts_query = build_fts_query(expanded)

        found_preference = False
        found_episode = False

        if fts_query:
            bm25_results = store.bm25_search(fts_query, limit=10)
            if bm25_results:
                chunk_ids = [cid for cid, _ in bm25_results]
                chunks = store.get_chunks_by_ids(chunk_ids)
                all_text = " ".join(c.text for c in chunks)
                found_preference = "보안" in all_text or "한국어" in all_text
                found_episode = "FastAPI" in all_text or "rate limiting" in all_text

        # auto-recall 코드에서 스코프 분리 확인
        import inspect
        from openclaw.agent import loop as loop_mod
        source = inspect.getsource(loop_mod.run)
        has_scope_split = "Long-term Memory" in source and "Recent Context" in source
        has_session_filter = "source_type" in source
        has_boost = "1.2" in source  # MEMORY.md 1.2x boost

        checks = []
        score = 0.0

        if found_preference:
            score += 0.25
            checks.append("pref_recall=yes")
        else:
            checks.append("pref_recall=no")

        if has_scope_split:
            score += 0.25
            checks.append("scope_split=yes")
        else:
            checks.append("scope_split=no")

        if has_session_filter:
            score += 0.25
            checks.append("session_filter=yes")
        else:
            checks.append("session_filter=no")

        if has_boost:
            score += 0.25
            checks.append("memory_boost=yes")
        else:
            checks.append("memory_boost=no")

        return score, ", ".join(checks)


# ══════════════════════════════════════════════════════════
# 시나리오 4: 큐레이션 승격
# 3일간 같은 패턴 → MEMORY.md에 자동 승격
# ══════════════════════════════════════════════════════════

async def eval_curation_promotion() -> tuple[float, str]:
    """3일 반복 패턴이 MEMORY.md로 자동 승격되는지."""
    from openclaw.memory.curation import (
        _should_curate,
        _mark_curated,
        curate_memories,
        CURATION_INTERVAL_SECONDS,
        _CURATION_MARKER,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_dir = Path(tmpdir) / "memory"
        memory_dir.mkdir()
        workspace_dir = tmpdir  # _should_curate uses workspace_dir for marker

        # 3일간 같은 패턴 반복
        recurring_pattern = "매일 아침 9시에 서버 상태 확인하고 슬랙에 보고"
        for day in ["2026-03-07", "2026-03-08", "2026-03-09"]:
            note = memory_dir / f"{day}.md"
            note.write_text(
                f"## {day} 작업\n\n"
                f"서버 상태 확인 작업을 수행했다. {recurring_pattern}\n\n"
                f"코드 리뷰 3건 처리하고 배포 파이프라인 점검을 완료했다.\n",
                encoding="utf-8",
            )

        checks = []

        # 1. 큐레이션 실행 조건 확인 (.last-curation 타임스탬프 조작)
        marker = Path(workspace_dir) / _CURATION_MARKER
        # 마커 없으면 실행 가능
        ok1 = _should_curate(workspace_dir)
        checks.append(f"no_marker_run={ok1}")

        # 마커를 25시간 전으로 설정 → 실행 가능
        marker.write_text(str(time.time() - 90000))
        ok2 = _should_curate(workspace_dir)
        checks.append(f"old_marker_run={ok2}")

        # 마커를 1시간 전으로 설정 → 실행 불가
        marker.write_text(str(time.time() - 3600))
        ok3 = not _should_curate(workspace_dir)
        checks.append(f"recent_marker_skip={ok3}")

        # 2. curate_memories 프롬프트 검증 (LLM 모킹)
        marker.unlink(missing_ok=True)  # 마커 제거하여 실행 허용
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="NO_REPLY")

        await curate_memories(
            memory_dir, workspace_dir, mock_provider, "test-model",
        )
        # LLM이 호출되었는지 (= 일별 노트를 읽고 프롬프트를 구성했는지)
        ok4 = mock_provider.complete.called
        checks.append(f"llm_called={ok4}")

        # 프롬프트 내용 확인
        prompt_ok = False
        if ok4:
            call_args = mock_provider.complete.call_args
            msgs = call_args.kwargs.get("messages", []) or (call_args.args[1] if len(call_args.args) > 1 else [])
            if msgs:
                prompt_text = msgs[0].text if hasattr(msgs[0], "text") else ""
                prompt_ok = "recurring" in prompt_text.lower() or "pattern" in prompt_text.lower() or "daily" in prompt_text.lower()
        checks.append(f"prompt_structure={'ok' if prompt_ok else 'bad'}")

        # 3. 실제 승격 시뮬 (LLM이 내용 반환)
        marker2 = Path(workspace_dir) / _CURATION_MARKER
        marker2.unlink(missing_ok=True)
        mock_provider2 = MagicMock()
        mock_provider2.complete = AsyncMock(
            return_value="- 매일 아침 서버 상태 확인이 반복 패턴\n- 코드 리뷰 일상 업무"
        )

        promoted = await curate_memories(
            memory_dir, workspace_dir, mock_provider2, "test-model",
        )
        memory_md = Path(workspace_dir) / "MEMORY.md"
        ok5 = promoted and memory_md.exists()
        if ok5:
            content = memory_md.read_text()
            ok5 = "서버 상태" in content
        checks.append(f"promoted={ok5}")

        # 점수 계산
        score = 0.0
        if ok1: score += 0.2
        if ok2: score += 0.2
        if ok3: score += 0.15
        if ok4: score += 0.15
        if prompt_ok: score += 0.15
        if ok5: score += 0.15

        return score, ", ".join(checks)


# ══════════════════════════════════════════════════════════
# 시나리오 5: 스케줄 실행
# HEARTBEAT.md 작성 → cron 실행 → 콜백 호출 확인
# ══════════════════════════════════════════════════════════

async def eval_heartbeat_schedule() -> tuple[float, str]:
    """HEARTBEAT.md 스케줄이 실제로 실행되는지."""
    from openclaw.cron import CronScheduler, heartbeat_from_file

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # HEARTBEAT.md 작성
        hb_file = workspace / "HEARTBEAT.md"
        hb_file.write_text(
            "## Daily Tasks\n"
            "- Check server status\n"
            "- Summarize overnight alerts\n",
            encoding="utf-8",
        )

        checks = []

        # 1. CronScheduler 기본 동작
        scheduler = CronScheduler()
        call_count = 0

        async def test_callback():
            nonlocal call_count
            call_count += 1

        scheduler.register("test_task", test_callback, interval=0.1)
        await scheduler.start()
        await asyncio.sleep(0.35)  # 0.1초 간격 × 3회
        await scheduler.stop()

        ok1 = call_count >= 2  # 최소 2회 실행
        checks.append(f"cron_runs={call_count}")

        # 2. heartbeat_from_file이 파일을 읽고 실행하는지
        # (LLM 없이 구조만 확인)
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "NO_REPLY"
        mock_provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        await heartbeat_from_file(
            str(hb_file), mock_provider, "test-model", str(workspace)
        )
        ok2 = mock_provider.client.chat.completions.create.called
        checks.append(f"hb_called={ok2}")

        # 3. 프롬프트에 HEARTBEAT.md 내용이 포함되는지
        if ok2:
            call_args = mock_provider.client.chat.completions.create.call_args
            messages = call_args.kwargs.get("messages", [])
            content = " ".join(m.get("content", "") for m in messages)
            ok3 = "server status" in content.lower()
            checks.append(f"content_injected={ok3}")
        else:
            ok3 = False
            checks.append("content_injected=skip")

        # 4. one-shot 모드
        scheduler2 = CronScheduler()
        one_shot_ran = False

        async def one_shot_cb():
            nonlocal one_shot_ran
            one_shot_ran = True

        scheduler2.register("one_shot", one_shot_cb, interval=0.05, one_shot=True)
        await scheduler2.start()
        await asyncio.sleep(0.2)
        await scheduler2.stop()

        status = scheduler2.status()
        task_status = status[0]["status"] if status else "unknown"
        ok4 = one_shot_ran and task_status == "completed"
        checks.append(f"one_shot={ok4}")

        score = (0.3 if ok1 else 0) + (0.3 if ok2 else 0) + (0.2 if ok3 else 0) + (0.2 if ok4 else 0)
        return score, ", ".join(checks)


# ══════════════════════════════════════════════════════════
# 시나리오 6: 컨텍스트 압박 하 안정성
# 85%까지 채움 → 자동 조정 → 에이전트 안 죽는가
# ══════════════════════════════════════════════════════════

async def eval_context_pressure() -> tuple[float, str]:
    """컨텍스트 85%+ 압박 하에서 자동 조정 및 안정성."""
    from openclaw.context.diagnosis import diagnose_context
    from openclaw.context.guard import ContextGuard, ContextAction
    from openclaw.config import ContextConfig
    from openclaw.agent.types import AgentMessage, TextBlock, ToolResultBlock
    import random

    checks = []

    # 1. 70% 시나리오: compaction_threshold 하향
    random.seed(70)
    words = ["hello", "world", "test", "data", "func", "var", "cls", "method"]
    text_70 = " ".join(random.choice(words) for _ in range(10000))
    config_70 = ContextConfig(
        max_tokens=12000, compaction_threshold=0.7,
        reserve_tokens_floor=2000, tool_result_max_ratio=0.3,
    )
    diag_70 = diagnose_context(
        [AgentMessage(role="user", content=[TextBlock(text=text_70)])],
        "system", max_tokens=12000,
    )
    old_threshold = config_70.compaction_threshold
    diag_70.apply_adjustments(config_70)
    ok1 = config_70.compaction_threshold < old_threshold
    checks.append(f"70%_threshold_down={ok1} ({old_threshold}→{config_70.compaction_threshold})")

    # 2. 85% 시나리오: reserve_floor 상향 + ratio 하향
    random.seed(85)
    text_85 = " ".join(random.choice(words) for _ in range(20000))
    config_85 = ContextConfig(
        max_tokens=20000, compaction_threshold=0.7,
        reserve_tokens_floor=5000, tool_result_max_ratio=0.3,
    )
    diag_85 = diagnose_context(
        [AgentMessage(role="user", content=[TextBlock(text=text_85)])],
        "sys" * 50, max_tokens=20000,
    )
    old_floor = config_85.reserve_tokens_floor
    old_ratio = config_85.tool_result_max_ratio
    diag_85.apply_adjustments(config_85)
    ok2 = config_85.reserve_tokens_floor > old_floor
    ok3 = config_85.tool_result_max_ratio < old_ratio
    checks.append(f"85%_floor_up={ok2} ({old_floor}→{config_85.reserve_tokens_floor})")
    checks.append(f"85%_ratio_down={ok3} ({old_ratio}→{config_85.tool_result_max_ratio})")

    # 3. ContextGuard COMPACT 액션 트리거
    guard_config = ContextConfig(max_tokens=10000, compaction_threshold=0.7)
    guard = ContextGuard(guard_config)
    status = guard.check(8000)  # 80% → COMPACT
    ok4 = status.action == ContextAction.COMPACT
    checks.append(f"guard_compact={ok4} (util={status.utilization:.0%})")

    # 4. ERROR 액션 (초과)
    status_err = guard.check(12000)
    ok5 = status_err.action == ContextAction.ERROR
    checks.append(f"guard_error={ok5}")

    # 5. enforce_budget이 도구 결과를 트렁케이트
    big_result = ToolResultBlock(tool_use_id="t1", content="x" * 100000)
    msgs = [AgentMessage(role="user", content=[big_result])]
    guard.enforce_budget(msgs)
    truncated_len = len(msgs[0].content[0].content)
    ok6 = truncated_len < 100000
    checks.append(f"enforce_truncate={ok6} ({100000}→{truncated_len})")

    score = sum([ok1, ok2, ok3, ok4, ok5, ok6]) / 6
    return score, ", ".join(checks)


# ══════════════════════════════════════════════════════════
# 시나리오 7: 페일오버 복구
# 1차 모델 에러 → 2차 전환 → 1차 복귀
# ══════════════════════════════════════════════════════════

async def eval_failover_recovery() -> tuple[float, str]:
    """모델 장애 시 자동 전환 및 원래 모델 복귀."""
    import os
    from openclaw.model.failover import (
        FailoverManager,
        classify_error,
        should_failover,
        FailoverReason,
    )

    checks = []

    # 1. 에러 분류
    rate_limit_err = Exception("Rate limit exceeded (429)")
    reason = classify_error(rate_limit_err)
    ok1 = reason == FailoverReason.RATE_LIMIT
    checks.append(f"classify_rate_limit={ok1}")

    overload_err = Exception("Service overloaded (503)")
    reason2 = classify_error(overload_err)
    ok2 = reason2 == FailoverReason.OVERLOADED
    checks.append(f"classify_overload={ok2}")

    # context overflow는 페일오버 대상 아님
    ctx_err = Exception("context_length_exceeded")
    reason3 = classify_error(ctx_err)
    ok3 = not should_failover(reason3)
    checks.append(f"no_failover_context={ok3}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2. 페일오버 매니저 모델 전환
        # temp state_path로 기존 상태 간섭 방지
        fm = FailoverManager(
            fallback_models=["model-b", "model-c"],
            state_path=os.path.join(tmpdir, "state.json"),
        )
        # 첫 에러: 프로필 1개(default)이므로 cooldown 후 바로 모델 전환
        reason_fo, next_model = fm.handle_error(rate_limit_err)
        ok4 = next_model == "model-b"
        checks.append(f"failover_to_b={ok4}")

        # 3. 두 번째 실패 → model-c
        _, next_model2 = fm.handle_error(overload_err)
        ok5 = next_model2 == "model-c"
        checks.append(f"failover_to_c={ok5}")

        # 4. 성공 표시 후 상태 초기화
        fm.mark_success()
        ok6 = fm._retry_count == 0
        checks.append(f"success_reset={ok6}")

    score = sum([ok1, ok2, ok3, ok4, ok5, ok6]) / 6
    return score, ", ".join(checks)


# ══════════════════════════════════════════════════════════
# 시나리오 8: 루프 탈출
# 동일 도구 30회 반복 → 감지·중단
# ══════════════════════════════════════════════════════════

async def eval_loop_escape() -> tuple[float, str]:
    """무한루프 감지 및 자동 중단."""
    from openclaw.tools.registry import ToolLoopDetector

    checks = []
    detector = ToolLoopDetector()

    # 1. 10회 반복 → WARNING
    warnings = []
    for i in range(15):
        w = detector.record("read", {"path": "/same/file.txt"}, "same content")
        if w and "WARNING" in w:
            warnings.append(i)

    ok1 = len(warnings) >= 1
    first_warn = warnings[0] if warnings else -1
    checks.append(f"warn_at={first_warn}")

    # 2. 30회 반복 → CRITICAL
    detector2 = ToolLoopDetector()
    critical_found = False
    critical_turn = -1
    for i in range(35):
        w = detector2.record("bash", {"command": "curl http://fail"}, "connection refused")
        if w and "CRITICAL" in w:
            critical_found = True
            critical_turn = i
            break

    ok2 = critical_found
    checks.append(f"critical_at={critical_turn}")

    # 3. 다른 도구/인풋은 카운트 안 됨
    detector3 = ToolLoopDetector()
    false_positive = False
    for i in range(20):
        w = detector3.record("read", {"path": f"/file_{i}.txt"}, f"content {i}")
        if w:
            false_positive = True
            break

    ok3 = not false_positive
    checks.append(f"no_false_positive={ok3}")

    # 4. 핑퐁 패턴 감지 (A→B→A→B)
    detector4 = ToolLoopDetector()
    pingpong_warn = False
    for i in range(25):
        if i % 2 == 0:
            w = detector4.record("read", {"path": "/a.txt"}, "content a")
        else:
            w = detector4.record("write", {"path": "/a.txt"}, "ok")
        if w:
            pingpong_warn = True
            break

    ok4 = pingpong_warn
    checks.append(f"pingpong_detect={ok4}")

    score = sum([ok1, ok2, ok3, ok4]) / 4
    return score, ", ".join(checks)


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════

async def main() -> None:
    header("OpenClaw Intelligence Eval Suite")
    print(f"  기준: 순수 LLM으로는 불가능한 행동만 테스트")
    print(f"  임계: 70% 이상 = PASS\n")

    await run_eval("1. 크로스세션 기억", eval_cross_session_memory())
    await run_eval("2. 컴팩션 생존", eval_compaction_survival())
    await run_eval("3. 프로액티브 리콜", eval_proactive_recall())
    await run_eval("4. 큐레이션 승격", eval_curation_promotion())
    await run_eval("5. 스케줄 실행", eval_heartbeat_schedule())
    await run_eval("6. 컨텍스트 압박", eval_context_pressure())
    await run_eval("7. 페일오버 복구", eval_failover_recovery())
    await run_eval("8. 루프 탈출", eval_loop_escape())

    # 결과 요약
    header("결과 요약")
    total_score = sum(s for _, s, _ in results) / len(results) if results else 0
    pass_count = sum(1 for _, s, _ in results if s >= 0.7)
    fail_count = len(results) - pass_count

    print(f"  전체 평균: {total_score:.0%}")
    print(f"  {GREEN}PASS (>=70%): {pass_count}개{RESET}")
    if fail_count:
        print(f"  {RED}FAIL (<70%): {fail_count}개{RESET}")
        for name, score, detail in results:
            if score < 0.7:
                print(f"    - {name}: {score:.0%}")
    else:
        print(f"\n  {GREEN}{BOLD}ALL PASS!{RESET}")

    print()
    for name, score, detail in results:
        bar_len = 30
        filled = int(score * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        color = GREEN if score >= 0.7 else (YELLOW if score >= 0.4 else RED)
        print(f"  {color}[{bar}] {score:.0%}{RESET}  {name}")


if __name__ == "__main__":
    asyncio.run(main())
