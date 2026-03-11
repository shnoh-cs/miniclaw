"""Browser automation via Playwright for web interaction.

Provides a BrowserManager that wraps Playwright to give the agent
full browser control: navigation, clicking, typing, form filling,
screenshots, and multi-tab management.

Interactive elements are assigned ref numbers via a DOM snapshot.
The LLM sees ``[ref] role "name"`` and can say ``click ref=5``.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_MAX_ELEMENTS = 200

# --------------------------------------------------------------------------- #
# JavaScript injected into the page to build an AI-friendly snapshot.
# Assigns ``data-ocref`` attributes to interactive elements so that
# subsequent click/type actions can target them reliably.
# --------------------------------------------------------------------------- #
_SNAPSHOT_JS = r"""
(() => {
    const results = [];
    let nextRef = 1;

    // Clear old refs
    document.querySelectorAll('[data-ocref]').forEach(
        el => el.removeAttribute('data-ocref')
    );

    const SKIP = new Set([
        'SCRIPT','STYLE','NOSCRIPT','SVG','PATH','META','LINK','BR','HR','TEMPLATE'
    ]);
    const INTERACTIVE = new Set([
        'A','BUTTON','INPUT','SELECT','TEXTAREA','SUMMARY'
    ]);
    const INTERACTIVE_ROLES = new Set([
        'button','link','textbox','checkbox','radio','combobox','listbox',
        'menuitem','option','searchbox','slider','spinbutton','switch','tab',
        'treeitem','menuitemcheckbox','menuitemradio'
    ]);

    function vis(el) {
        if (el.offsetParent === null && el.tagName !== 'BODY'
            && getComputedStyle(el).position !== 'fixed') return false;
        const s = getComputedStyle(el);
        return s.display !== 'none' && s.visibility !== 'hidden';
    }

    function inter(el) {
        if (INTERACTIVE.has(el.tagName)) return true;
        const r = el.getAttribute('role');
        if (r && INTERACTIVE_ROLES.has(r)) return true;
        if (el.contentEditable === 'true') return true;
        if (el.hasAttribute('onclick') || el.hasAttribute('tabindex')) return true;
        return false;
    }

    function role(el) {
        const r = el.getAttribute('role');
        if (r) return r;
        switch (el.tagName) {
            case 'A': return el.href ? 'link' : 'text';
            case 'BUTTON': return 'button';
            case 'INPUT': {
                const t = (el.type||'text').toLowerCase();
                if (t==='checkbox') return 'checkbox';
                if (t==='radio') return 'radio';
                if (t==='submit'||t==='button'||t==='reset') return 'button';
                if (t==='file') return 'file';
                if (t==='range') return 'slider';
                if (t==='number') return 'spinbutton';
                if (t==='search') return 'searchbox';
                if (t==='hidden') return 'hidden';
                return 'textbox';
            }
            case 'SELECT': return 'combobox';
            case 'TEXTAREA': return 'textbox';
            case 'IMG': return 'img';
            case 'H1':case 'H2':case 'H3':case 'H4':case 'H5':case 'H6':
                return 'heading';
            case 'NAV': return 'navigation';
            case 'FORM': return 'form';
            default: return null;
        }
    }

    function name(el) {
        let n = el.getAttribute('aria-label');
        if (n) return n.trim();
        const lby = el.getAttribute('aria-labelledby');
        if (lby) { const l = document.getElementById(lby); if (l) return l.textContent.trim(); }
        if (['INPUT','TEXTAREA','SELECT'].includes(el.tagName)) {
            if (el.id) {
                const lb = document.querySelector('label[for="'+CSS.escape(el.id)+'"]');
                if (lb) return lb.textContent.trim();
            }
            const pl = el.closest('label');
            if (pl) { const c=pl.cloneNode(true); c.querySelectorAll('input,textarea,select').forEach(x=>x.remove()); const t=c.textContent.trim(); if(t) return t; }
            if (el.placeholder) return el.placeholder;
            if (el.name) return el.name;
        }
        if (el.tagName==='IMG') return el.alt||el.title||'';
        if (['A','BUTTON','SUMMARY'].includes(el.tagName)) {
            const t = el.textContent.trim();
            return t.length<=80 ? t : t.substring(0,77)+'...';
        }
        if (el.title) return el.title;
        const dt = Array.from(el.childNodes).filter(n=>n.nodeType===3).map(n=>n.textContent.trim()).join(' ').trim();
        return (dt||el.textContent?.trim()||'').substring(0,80);
    }

    const MAX = %d;

    function walk(el, depth) {
        if (nextRef > MAX) return;
        if (SKIP.has(el.tagName)) return;
        if (el.getAttribute('aria-hidden')==='true') return;
        if (!vis(el)) return;

        const isInter = inter(el);
        const r = role(el);
        const n = name(el);

        if (r === 'hidden') {
            // skip hidden inputs
        } else if (isInter && n) {
            const ref = nextRef++;
            el.setAttribute('data-ocref', String(ref));
            let val = '';
            if (el.tagName==='INPUT' && (el.type||'text').toLowerCase()!=='password') val=el.value||'';
            if (el.tagName==='TEXTAREA') val=el.value||'';
            if (el.tagName==='SELECT' && el.selectedIndex>=0) val=el.options[el.selectedIndex]?.text||'';
            results.push({
                ref, role:r||'interactive', name:n,
                value:val.substring(0,50),
                depth:Math.min(depth,6),
                checked:!!el.checked, disabled:!!el.disabled,
                href:(el.tagName==='A'&&el.href)?el.href:''
            });
        } else if (r==='heading') {
            results.push({
                ref:null, role:'heading', name:n.substring(0,100),
                value:'', depth:Math.min(depth,6),
                level:parseInt(el.tagName[1])
            });
        }

        for (const child of el.children) walk(child, depth+1);
    }

    walk(document.body, 0);
    return { title:document.title, url:location.href, elements:results };
})()
""" % _MAX_ELEMENTS


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

@dataclass
class _Tab:
    id: int
    page: Any  # playwright Page
    title: str = ""
    url: str = ""


# --------------------------------------------------------------------------- #
# BrowserManager
# --------------------------------------------------------------------------- #

class BrowserManager:
    """Manages a Playwright browser for agent web interaction.

    * Single browser context → cookies/sessions shared across tabs.
    * Ref-number system: ``snapshot()`` assigns ``[N]`` to interactive
      elements; ``click(ref=N)`` / ``type(ref=N, text=...)`` targets them.
    * SSRF protection: only ``http(s)://`` URLs allowed.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30_000,
        viewport: tuple[int, int] = (1280, 720),
    ) -> None:
        self._headless = headless
        self._timeout = timeout_ms
        self._viewport = {"width": viewport[0], "height": viewport[1]}
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._tabs: dict[int, _Tab] = {}
        self._active_tab: int = -1
        self._next_id: int = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def _ensure(self) -> None:
        """Launch browser on first use (lazy init)."""
        if self._browser is not None:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "playwright is not installed. "
                "Run: pip install playwright && playwright install chromium"
            )
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        self._context = await self._browser.new_context(
            viewport=self._viewport,
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        self._context.set_default_timeout(self._timeout)
        await self._open_tab()

    async def close(self) -> None:
        """Close browser and release all resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        self._browser = None
        self._context = None
        self._pw = None
        self._tabs.clear()
        self._active_tab = -1

    # ------------------------------------------------------------------ #
    # Tab helpers
    # ------------------------------------------------------------------ #

    async def _open_tab(self, url: str | None = None) -> int:
        page = await self._context.new_page()
        tid = self._next_id
        self._next_id += 1
        self._tabs[tid] = _Tab(id=tid, page=page)
        self._active_tab = tid
        if url:
            await page.goto(url, wait_until="domcontentloaded")
            self._tabs[tid].title = await page.title()
            self._tabs[tid].url = page.url
        return tid

    @property
    def _page(self) -> Any:
        if self._active_tab in self._tabs:
            return self._tabs[self._active_tab].page
        return None

    async def _sync_tab_info(self) -> None:
        """Update active tab's title/url from the page."""
        if self._active_tab in self._tabs and self._page:
            try:
                self._tabs[self._active_tab].title = await self._page.title()
                self._tabs[self._active_tab].url = self._page.url
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Action dispatcher
    # ------------------------------------------------------------------ #

    async def execute(self, action: str, **kw: Any) -> str:
        """Execute a browser action by name. Returns result text."""
        await self._ensure()

        method = getattr(self, f"_act_{action}", None)
        if method is None:
            names = sorted(n[5:] for n in dir(self) if n.startswith("_act_"))
            return f"Unknown action '{action}'. Available: {', '.join(names)}"
        try:
            return await method(**kw)
        except Exception as e:
            logger.debug("Browser action '%s' failed", action, exc_info=True)
            return f"Error ({action}): {e}"

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    async def _act_navigate(self, url: str = "", **_: Any) -> str:
        """Navigate to a URL."""
        if not url:
            return "Error: url is required."
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Error: Only HTTP(S) URLs allowed, got '{parsed.scheme}'."
        await self._page.goto(url, wait_until="domcontentloaded")
        await self._sync_tab_info()
        tab = self._tabs[self._active_tab]
        return f"Navigated to: {tab.title}\nURL: {tab.url}"

    async def _act_snapshot(self, **_: Any) -> str:
        """Get AI-readable page structure with numbered refs."""
        data = await self._page.evaluate(_SNAPSHOT_JS)
        return _format_snapshot(data)

    async def _act_click(self, ref: int = 0, **_: Any) -> str:
        """Click element by ref number."""
        if not ref:
            return "Error: ref is required."
        loc = self._page.locator(f'[data-ocref="{ref}"]')
        await loc.click()
        await self._page.wait_for_load_state("domcontentloaded")
        await self._sync_tab_info()
        tab = self._tabs[self._active_tab]
        return f"Clicked [ref={ref}]. Page: {tab.title} | {tab.url}"

    async def _act_type(
        self, ref: int = 0, text: str = "", clear: bool = True, **_: Any
    ) -> str:
        """Type text into an element."""
        if not ref:
            return "Error: ref is required."
        loc = self._page.locator(f'[data-ocref="{ref}"]')
        if clear:
            await loc.fill(text)
        else:
            await loc.type(text)
        return f"Typed into [ref={ref}]: '{text[:60]}'"

    async def _act_screenshot(self, full_page: bool = False, **_: Any) -> str:
        """Take a screenshot and save to a temp file."""
        path = tempfile.mktemp(suffix=".png", prefix="browser_")
        await self._page.screenshot(path=path, full_page=full_page)
        await self._sync_tab_info()
        tab = self._tabs[self._active_tab]
        return (
            f"Screenshot saved: {path}\n"
            f"Page: {tab.title}\n"
            f"URL: {tab.url}\n"
            f"Tip: Use the 'read' tool on the image path to view it."
        )

    async def _act_scroll(
        self, direction: str = "down", amount: int = 500, **_: Any
    ) -> str:
        """Scroll the page."""
        delta = amount if direction == "down" else -amount
        await self._page.mouse.wheel(0, delta)
        await self._page.wait_for_timeout(300)
        return f"Scrolled {direction} by {amount}px."

    async def _act_fill_form(self, fields: dict | None = None, **_: Any) -> str:
        """Fill multiple form fields at once. fields = {"ref": "value", ...}"""
        if not fields:
            return 'Error: fields required, e.g. {"3": "John", "5": "john@email.com"}'
        filled = []
        for ref_str, value in fields.items():
            loc = self._page.locator(f'[data-ocref="{ref_str}"]')
            await loc.fill(str(value))
            filled.append(f"  [{ref_str}] = '{str(value)[:40]}'")
        return f"Filled {len(filled)} fields:\n" + "\n".join(filled)

    async def _act_select(self, ref: int = 0, value: str = "", **_: Any) -> str:
        """Select a dropdown option."""
        if not ref:
            return "Error: ref is required."
        loc = self._page.locator(f'[data-ocref="{ref}"]')
        await loc.select_option(label=value)
        return f"Selected '{value}' in [ref={ref}]."

    async def _act_press_key(self, key: str = "", **_: Any) -> str:
        """Press a keyboard key (Enter, Tab, Escape, ArrowDown, etc.)."""
        if not key:
            return "Error: key is required."
        await self._page.keyboard.press(key)
        return f"Pressed key: {key}"

    async def _act_hover(self, ref: int = 0, **_: Any) -> str:
        """Hover over an element."""
        if not ref:
            return "Error: ref is required."
        loc = self._page.locator(f'[data-ocref="{ref}"]')
        await loc.hover()
        return f"Hovering over [ref={ref}]."

    async def _act_back(self, **_: Any) -> str:
        """Navigate back in browser history."""
        await self._page.go_back(wait_until="domcontentloaded")
        await self._sync_tab_info()
        tab = self._tabs[self._active_tab]
        return f"Navigated back. Page: {tab.title} | {tab.url}"

    async def _act_forward(self, **_: Any) -> str:
        """Navigate forward in browser history."""
        await self._page.go_forward(wait_until="domcontentloaded")
        await self._sync_tab_info()
        tab = self._tabs[self._active_tab]
        return f"Navigated forward. Page: {tab.title} | {tab.url}"

    async def _act_wait(
        self,
        wait_for: str = "load",
        timeout: int = 10_000,
        **_: Any,
    ) -> str:
        """Wait for a page condition.

        wait_for values:
        - "load" / "domcontentloaded" / "networkidle" → page load states
        - any CSS selector string → wait for element to appear
        """
        if wait_for in ("load", "domcontentloaded", "networkidle"):
            await self._page.wait_for_load_state(wait_for, timeout=timeout)
            return f"Page reached '{wait_for}' state."
        # Treat as CSS selector
        await self._page.wait_for_selector(wait_for, timeout=timeout)
        return f"Element '{wait_for}' appeared."

    async def _act_tab_list(self, **_: Any) -> str:
        """List all open tabs."""
        if not self._tabs:
            return "No tabs open."
        lines = []
        for tid, tab in self._tabs.items():
            try:
                tab.title = await tab.page.title()
                tab.url = tab.page.url
            except Exception:
                pass
            marker = " ← active" if tid == self._active_tab else ""
            lines.append(f"  Tab {tid}: {tab.title or '(blank)'} | {tab.url or 'about:blank'}{marker}")
        return f"{len(self._tabs)} tabs:\n" + "\n".join(lines)

    async def _act_new_tab(self, url: str = "", **_: Any) -> str:
        """Open a new tab, optionally navigating to a URL."""
        if url:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return f"Error: Only HTTP(S) URLs allowed, got '{parsed.scheme}'."
        tid = await self._open_tab(url or None)
        tab = self._tabs[tid]
        return f"Opened Tab {tid}. {tab.title or '(blank)'} | {tab.url or 'about:blank'}"

    async def _act_close_tab(self, tab_id: int = -1, **_: Any) -> str:
        """Close a tab by ID."""
        if tab_id < 0:
            tab_id = self._active_tab
        tab = self._tabs.pop(tab_id, None)
        if tab is None:
            return f"Error: Tab {tab_id} not found."
        await tab.page.close()
        if tab_id == self._active_tab:
            self._active_tab = next(iter(self._tabs), -1)
        remaining = len(self._tabs)
        if remaining == 0:
            await self._open_tab()
            return "Closed tab. Opened a fresh blank tab."
        return f"Closed Tab {tab_id}. {remaining} tabs remaining."

    async def _act_focus_tab(self, tab_id: int = 0, **_: Any) -> str:
        """Switch focus to a different tab."""
        if tab_id not in self._tabs:
            return f"Error: Tab {tab_id} not found. Use tab_list to see available tabs."
        self._active_tab = tab_id
        await self._tabs[tab_id].page.bring_to_front()
        await self._sync_tab_info()
        tab = self._tabs[tab_id]
        return f"Focused Tab {tab_id}: {tab.title} | {tab.url}"

    async def _act_evaluate(self, expression: str = "", **_: Any) -> str:
        """Evaluate JavaScript in the page context."""
        if not expression:
            return "Error: expression is required."
        result = await self._page.evaluate(expression)
        text = str(result)
        if len(text) > 4000:
            text = text[:4000] + "\n... [truncated]"
        return f"JS result:\n{text}"

    async def _act_close(self, **_: Any) -> str:
        """Close the browser entirely."""
        await self.close()
        return "Browser closed."


# --------------------------------------------------------------------------- #
# Snapshot formatting
# --------------------------------------------------------------------------- #

def _format_snapshot(data: dict) -> str:
    """Format raw snapshot data into an AI-readable text representation."""
    title = data.get("title", "Untitled")
    url = data.get("url", "")
    elements = data.get("elements", [])

    lines = [
        f"Page: {title}",
        f"URL: {url}",
        "",
    ]

    for el in elements:
        depth = el.get("depth", 0)
        indent = "  " * min(depth, 4)
        ref = el.get("ref")
        role = el.get("role", "")
        name = el.get("name", "")
        value = el.get("value", "")

        prefix = f"[{ref}] " if ref else ""
        parts = [f"{indent}{prefix}{role}"]
        if name:
            parts.append(f'"{name}"')
        if value:
            parts.append(f'value="{value}"')
        if el.get("checked"):
            parts.append("[checked]")
        if el.get("disabled"):
            parts.append("[disabled]")
        if el.get("href") and role == "link":
            href = el["href"]
            if len(href) > 80:
                href = href[:77] + "..."
            parts.append(f"→ {href}")

        lines.append(" ".join(parts))

    ref_count = sum(1 for el in elements if el.get("ref"))
    lines.append("")
    lines.append(f"Interactive elements: {ref_count}")
    if ref_count == _MAX_ELEMENTS:
        lines.append(f"(capped at {_MAX_ELEMENTS} — scroll down for more)")
    lines.append("Tip: Use click/type with ref numbers. Take a new snapshot after page changes.")

    return "\n".join(lines)
