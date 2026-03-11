"""Built-in tool: control a web browser via Playwright.

The real executor is wired up in Agent.__init__() since it needs
a persistent BrowserManager instance across tool calls.
"""

from __future__ import annotations

from openclaw.agent.types import ToolDefinition, ToolParameter

DEFINITION = ToolDefinition(
    name="browser",
    description=(
        "Control a web browser to navigate pages, fill forms, click buttons, "
        "and interact with websites. The browser persists across calls.\n"
        "\n"
        "Workflow: navigate → snapshot (see [ref] numbers) → click/type by ref → snapshot again.\n"
        "\n"
        "Actions:\n"
        "  navigate  - Go to URL. Params: url\n"
        "  snapshot  - Get page elements with [ref] numbers for interaction\n"
        "  click     - Click element. Params: ref\n"
        "  type      - Type into field. Params: ref, text, clear (default true)\n"
        "  fill_form - Fill multiple fields. Params: fields ({\"ref\": \"value\", ...})\n"
        "  select    - Select dropdown option. Params: ref, value\n"
        "  press_key - Press key (Enter, Tab, Escape, ArrowDown...). Params: key\n"
        "  screenshot - Take screenshot. Params: full_page\n"
        "  scroll    - Scroll page. Params: direction (up/down), amount (pixels)\n"
        "  hover     - Hover element. Params: ref\n"
        "  back / forward - Navigate history\n"
        "  wait      - Wait for condition. Params: wait_for (load/networkidle/CSS selector), timeout\n"
        "  tab_list  - List open tabs\n"
        "  new_tab   - Open tab. Params: url\n"
        "  close_tab - Close tab. Params: tab_id\n"
        "  focus_tab - Switch tab. Params: tab_id\n"
        "  evaluate  - Run JavaScript. Params: expression\n"
        "  close     - Close browser"
    ),
    parameters=[
        ToolParameter(
            name="action",
            description="Action to perform (see list above)",
        ),
        ToolParameter(
            name="url",
            description="URL for navigate / new_tab",
            required=False,
        ),
        ToolParameter(
            name="ref",
            type="integer",
            description="Element reference number from snapshot (for click, type, select, hover)",
            required=False,
        ),
        ToolParameter(
            name="text",
            description="Text to type (for type action)",
            required=False,
        ),
        ToolParameter(
            name="expression",
            description="JavaScript expression (for evaluate action)",
            required=False,
        ),
        ToolParameter(
            name="fields",
            type="object",
            description='Form fields as {"ref": "value", ...} (for fill_form)',
            required=False,
        ),
        ToolParameter(
            name="value",
            description="Option text (for select action)",
            required=False,
        ),
        ToolParameter(
            name="key",
            description="Key name: Enter, Tab, Escape, ArrowDown, etc. (for press_key)",
            required=False,
        ),
        ToolParameter(
            name="direction",
            description="Scroll direction: up or down (default down)",
            required=False,
        ),
        ToolParameter(
            name="amount",
            type="integer",
            description="Scroll amount in pixels (default 500)",
            required=False,
        ),
        ToolParameter(
            name="tab_id",
            type="integer",
            description="Tab ID (for focus_tab, close_tab)",
            required=False,
        ),
        ToolParameter(
            name="wait_for",
            description="Wait condition: load, networkidle, or CSS selector (for wait)",
            required=False,
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout in ms (for wait, default 10000)",
            required=False,
        ),
        ToolParameter(
            name="full_page",
            type="boolean",
            description="Capture full page screenshot (default false)",
            required=False,
        ),
        ToolParameter(
            name="clear",
            type="boolean",
            description="Clear field before typing (default true)",
            required=False,
        ),
    ],
)
