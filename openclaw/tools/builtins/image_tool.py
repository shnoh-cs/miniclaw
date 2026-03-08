"""Built-in tool: image analysis using vision-capable models.

이미지를 base64로 인코딩하여 멀티모달 LLM에 전송하고,
분석 결과를 텍스트로 반환한다.

지원 방식:
  1. 로컬 파일: base64 인코딩 후 data URL로 변환
  2. URL: 그대로 전달 (모델이 직접 fetch)

OpenAI 호환 API의 멀티모달 메시지 형식:
  {"role": "user", "content": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="image",
    description="Analyze an image file or URL using a vision model. Returns a text description.",
    parameters=[
        ToolParameter(name="path", description="File path or URL of the image"),
        ToolParameter(
            name="prompt",
            description="What to analyze about the image (default: general description)",
            required=False,
        ),
    ],
)

# 지원하는 이미지 확장자
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}


async def execute(
    args: dict[str, Any], workspace: str = "", provider: Any = None, model: str = ""
) -> ToolResult:
    """이미지를 분석한다.

    provider가 전달되면 실제 비전 모델을 사용하고,
    없으면 이미지 메타데이터만 반환한다.
    """
    image_path = args.get("path", "")
    prompt = args.get("prompt", "이 이미지를 자세히 설명해주세요.")

    if not image_path:
        return ToolResult(tool_use_id="", content="Error: path is required", is_error=True)

    # URL인 경우
    if image_path.startswith(("http://", "https://")):
        image_url = image_path
        if provider:
            return await _analyze_with_vision(provider, model, prompt, image_url=image_url)
        return ToolResult(
            tool_use_id="",
            content=f"Image URL: {image_path}\n(Vision model not configured. Set a vision-capable model to analyze images.)",
        )

    # 로컬 파일인 경우
    path = Path(image_path)
    if not path.is_absolute() and workspace:
        path = Path(workspace) / path

    if not path.exists():
        return ToolResult(tool_use_id="", content=f"Error: File not found: {path}", is_error=True)

    ext = path.suffix.lower()
    if ext not in _IMAGE_EXTENSIONS:
        return ToolResult(
            tool_use_id="",
            content=f"Error: Unsupported image format '{ext}'. Supported: {', '.join(sorted(_IMAGE_EXTENSIONS))}",
            is_error=True,
        )

    # 파일 크기 확인 (20MB 제한)
    file_size = path.stat().st_size
    if file_size > 20 * 1024 * 1024:
        return ToolResult(
            tool_use_id="",
            content=f"Error: Image too large ({file_size / 1024 / 1024:.1f}MB). Max: 20MB",
            is_error=True,
        )

    # base64 인코딩
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    media_type = mimetypes.guess_type(str(path))[0] or "image/png"
    data_url = f"data:{media_type};base64,{image_data}"

    if provider:
        return await _analyze_with_vision(provider, model, prompt, image_url=data_url)

    # provider 없으면 메타데이터만 반환
    return ToolResult(
        tool_use_id="",
        content=(
            f"Image: {path.name}\n"
            f"Size: {file_size / 1024:.1f}KB\n"
            f"Format: {ext}\n"
            f"(Vision model not configured. Set a vision-capable model to analyze images.)"
        ),
    )


async def _analyze_with_vision(
    provider: Any, model: str, prompt: str, *, image_url: str
) -> ToolResult:
    """비전 모델에 이미지를 전송하고 분석 결과를 받는다."""
    try:
        # OpenAI 호환 멀티모달 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        response = await provider.client.chat.completions.create(
            model=model or provider.config.models.default,
            messages=messages,
            max_tokens=1024,
        )

        content = response.choices[0].message.content or ""
        if not content.strip():
            content = "(Vision model returned empty response)"

        return ToolResult(tool_use_id="", content=content)

    except Exception as e:
        return ToolResult(
            tool_use_id="",
            content=f"Error analyzing image: {e}",
            is_error=True,
        )
