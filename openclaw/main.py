"""OpenClaw agent runner — starts agent, cron, and Rocket.Chat bridge."""

from __future__ import annotations

import asyncio
import logging
import signal

from openclaw.agent.api import Agent


async def run() -> None:
    agent = Agent.from_config()
    log = logging.getLogger("openclaw")
    log.info("Agent initialized: model=%s", agent.config.models.default)

    await agent.start_heartbeat()
    await agent.restore_cron_jobs()

    rc_bridge = None
    if agent.config.rocketchat.enabled:
        from openclaw.rocketchat import RocketChatBridge

        rc_bridge = RocketChatBridge(agent, agent.config.rocketchat)
        await rc_bridge.start()
        agent._notification_callbacks.append(
            lambda _name, text, _err, _reply_to="": asyncio.create_task(
                rc_bridge.send_notification(text, reply_to=_reply_to)
            )
        )
        log.info("Rocket.Chat bridge: %s", agent.config.rocketchat.url)

    # Wait until interrupted
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    log.info("Running. Press Ctrl+C to stop.")
    await stop.wait()

    # Shutdown
    if rc_bridge:
        await rc_bridge.stop()
    agent._save_cron_jobs()
    await agent.stop_heartbeat()
    log.info("Shutdown complete")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
