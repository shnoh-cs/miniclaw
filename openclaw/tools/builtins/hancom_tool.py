"""Built-in tool: read Hancom Office documents (HWP, HWPX, Show, Cell)."""

from __future__ import annotations

import io
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

# ---------------------------------------------------------------------------
# 지원 확장자 목록
#   .hwp   — HWP5 바이너리 (OLE2 compound document)
#   .hwpx  — 한/글 XML (ZIP + XML, OOXML과 유사한 구조)
#   .show  — 한컴 프레젠테이션 (한쇼, ZIP + XML)
#   .cell  — 한컴 스프레드시트 (한셀, ZIP + XML)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".hwp", ".hwpx", ".show", ".cell"}

DEFINITION = ToolDefinition(
    name="hancom",
    description=(
        "Read and extract text from Hancom Office files "
        "(HWP, HWPX, Show, Cell)."
    ),
    parameters=[
        ToolParameter(
            name="file_path",
            description="Path to the Hancom Office file (.hwp, .hwpx, .show, .cell)",
        ),
        ToolParameter(
            name="max_chars",
            description="Maximum characters to extract (default: 50000)",
            type="integer",
            required=False,
        ),
    ],
)


async def execute(args: dict[str, Any], workspace: str = "") -> ToolResult:
    file_path = args.get("file_path", "")
    max_chars = int(args.get("max_chars", 50000))

    if not file_path:
        return ToolResult(tool_use_id="", content="Error: file_path is required", is_error=True)

    path = Path(file_path)
    if not path.is_absolute() and workspace:
        path = Path(workspace) / path

    if not path.exists():
        return ToolResult(tool_use_id="", content=f"Error: File not found: {path}", is_error=True)

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ToolResult(
            tool_use_id="",
            content=f"Error: Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            is_error=True,
        )

    try:
        if ext == ".hwp":
            text = _extract_hwp(path)
        elif ext == ".hwpx":
            text = _extract_hwpx(path)
        elif ext == ".show":
            text = _extract_show(path)
        elif ext == ".cell":
            text = _extract_cell(path)
        else:
            text = ""
    except Exception as e:
        return ToolResult(tool_use_id="", content=f"Error reading {ext} file: {e}", is_error=True)

    if not text.strip():
        text = f"(No text content extracted from {ext} file)"

    if len(text) > max_chars:
        half = max_chars // 2
        text = (
            text[:half]
            + f"\n\n... [truncated: {len(text)} chars total, showing first and last {half} chars] ...\n\n"
            + text[-half:]
        )

    return ToolResult(tool_use_id="", content=text)


# ===========================================================================
# HWP5 바이너리 파싱 (.hwp)
#
# HWP5 파일은 OLE2 (Compound Binary File) 형식.
# 구조:
#   FileHeader        — 파일 버전, 압축 여부 등 메타데이터
#   BodyText/Section0 — 본문 첫 번째 섹션 (zlib 압축)
#   BodyText/Section1 — 본문 두 번째 섹션 ...
#
# 각 Section 스트림은 zlib으로 압축되어 있고,
# 압축 해제 후 HWP 바이너리 레코드 형식으로 파싱해야 한다.
#
# 레코드 구조: [4바이트 헤더] + [데이터]
#   헤더 = TagID(10bit) | Level(10bit) | Size(12bit)
#   Size가 0xFFF이면 다음 4바이트가 실제 크기 (확장 크기)
#
# TagID == 67 (HWPTAG_PARA_TEXT) 인 레코드가 실제 텍스트를 담고 있다.
# 텍스트는 UTF-16LE로 인코딩되어 있으며,
# 제어 문자(0x0000~0x001F)는 특수 기능(표, 그림 등)을 나타낸다.
# ===========================================================================

# HWP 바이너리 레코드에서 텍스트를 담는 태그 ID
_HWPTAG_PARA_TEXT = 67


def _extract_hwp(path: Path) -> str:
    """HWP5 바이너리 파일에서 텍스트를 추출한다."""
    try:
        import olefile
    except ImportError:
        raise RuntimeError("olefile is required for HWP parsing. Install with: pip install olefile")

    if not olefile.isOleFile(str(path)):
        raise ValueError(f"Not a valid HWP (OLE2) file: {path}")

    ole = olefile.OleFileIO(str(path))

    try:
        # FileHeader에서 압축 여부 확인
        header_stream = ole.openstream("FileHeader")
        header_data = header_stream.read()
        # offset 36~39: 속성 플래그 (little-endian uint32)
        # bit 0: 압축 여부
        is_compressed = False
        if len(header_data) >= 40:
            flags = struct.unpack_from("<I", header_data, 36)[0]
            is_compressed = bool(flags & 0x01)

        # BodyText/SectionN 스트림들을 순서대로 읽기
        sections = []
        for entry in ole.listdir():
            # entry는 리스트: ['BodyText', 'Section0'] 등
            if len(entry) == 2 and entry[0] == "BodyText":
                sections.append(entry)

        # Section 번호 순 정렬
        sections.sort(key=lambda e: int(re.search(r"\d+", e[1]).group()) if re.search(r"\d+", e[1]) else 0)

        all_text: list[str] = []

        for section_entry in sections:
            stream_path = "/".join(section_entry)
            raw = ole.openstream(stream_path).read()

            # 압축 해제
            if is_compressed:
                try:
                    raw = zlib.decompress(raw, -15)
                except zlib.error:
                    try:
                        raw = zlib.decompress(raw)
                    except zlib.error:
                        continue

            # 바이너리 레코드 파싱
            paragraphs = _parse_hwp_records(raw)
            all_text.extend(paragraphs)

        return "\n".join(all_text)

    finally:
        ole.close()


def _parse_hwp_records(data: bytes) -> list[str]:
    """HWP 바이너리 레코드 스트림에서 PARA_TEXT 레코드를 추출한다."""
    paragraphs: list[str] = []
    offset = 0

    while offset + 4 <= len(data):
        # 4바이트 레코드 헤더 읽기
        header = struct.unpack_from("<I", data, offset)[0]
        tag_id = header & 0x3FF           # 하위 10비트: 태그 ID
        size = (header >> 20) & 0xFFF     # 상위 12비트: 데이터 크기
        offset += 4

        # 확장 크기: size == 0xFFF이면 다음 4바이트가 실제 크기
        if size == 0xFFF:
            if offset + 4 > len(data):
                break
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4

        if offset + size > len(data):
            break

        # PARA_TEXT 레코드에서 텍스트 추출
        if tag_id == _HWPTAG_PARA_TEXT:
            text = _decode_hwp_para_text(data[offset : offset + size])
            if text.strip():
                paragraphs.append(text)

        offset += size

    return paragraphs


def _decode_hwp_para_text(data: bytes) -> str:
    """HWP PARA_TEXT 레코드의 바이트를 텍스트로 디코딩한다.

    UTF-16LE로 인코딩되어 있으며, 제어 문자(0~31)는
    특수 기능을 나타내므로 건너뛴다.
    일부 제어 문자는 고정 길이의 추가 바이트를 소비한다.
    """
    chars: list[str] = []
    i = 0

    # 제어 문자별 추가 바이트 소비량 (UTF-16LE 단위 = 2바이트씩)
    # 이 제어 문자들은 자신(2바이트) 이후 추가 바이트를 소비한다
    inline_controls = {
        0x09: 0,    # 탭
        0x0A: 0,    # 줄바꿈
        0x0D: 0,    # 단락 끝
        0x18: 0,    # 하이픈
    }
    # 확장 제어 문자: 자기 자신(2바이트) + 추가 14바이트(7 UTF-16LE chars) = 총 16바이트
    extended_controls = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0B, 0x0C}

    while i + 1 < len(data):
        code = struct.unpack_from("<H", data, i)[0]

        if code == 0x0D:
            chars.append("\n")
            i += 2
        elif code == 0x09:
            chars.append("\t")
            i += 2
        elif code in inline_controls:
            i += 2
        elif code in extended_controls:
            # 확장 제어: 2바이트(자신) + 14바이트(추가) = 16바이트 건너뛰기
            i += 16
        elif code < 0x20:
            # 기타 제어 문자: 건너뛰기
            i += 2
        else:
            chars.append(chr(code))
            i += 2

    return "".join(chars)


# ===========================================================================
# HWPX 파싱 (.hwpx)
#
# HWPX는 ZIP 아카이브 안에 XML 파일들이 있는 구조.
# 텍스트는 Contents/SectionN.xml 파일에 있다.
# XML 네임스페이스: http://www.hancom.co.kr/hwpml/2011/paragraph
#   <hp:p> — 단락
#     <hp:run> — 텍스트 런
#       <hp:t> — 실제 텍스트
# ===========================================================================

# HWPX XML 네임스페이스
_HWPX_NS = {
    "hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
    "hs": "http://www.hancom.co.kr/hwpml/2011/section",
    "hc": "http://www.hancom.co.kr/hwpml/2011/core",
}


def _extract_hwpx(path: Path) -> str:
    """HWPX (ZIP+XML) 파일에서 텍스트를 추출한다."""
    with zipfile.ZipFile(str(path), "r") as zf:
        # Contents/SectionN.xml 파일 찾기
        section_files = sorted(
            [n for n in zf.namelist() if re.match(r"Contents/[Ss]ection\d+\.xml", n)]
        )

        if not section_files:
            # 다른 가능한 경로 시도
            section_files = sorted(
                [n for n in zf.namelist() if n.endswith(".xml") and "section" in n.lower()]
            )

        all_text: list[str] = []

        for sf in section_files:
            xml_data = zf.read(sf)
            paragraphs = _parse_hwpx_xml(xml_data)
            all_text.extend(paragraphs)

        return "\n".join(all_text)


def _parse_hwpx_xml(xml_data: bytes) -> list[str]:
    """HWPX Section XML에서 텍스트를 추출한다."""
    paragraphs: list[str] = []

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return paragraphs

    # <hp:p> 또는 <p> 태그 찾기 (네임스페이스 유무 모두 처리)
    p_tags = root.iter()
    for elem in p_tags:
        tag = _strip_ns(elem.tag)
        if tag == "t":
            text = elem.text or ""
            if text.strip():
                paragraphs.append(text)

    # 중복 제거 없이 단락 단위로 결합
    if not paragraphs:
        # fallback: 모든 텍스트 노드 추출
        paragraphs = _extract_all_text(root)

    return paragraphs


# ===========================================================================
# Show 파싱 (.show)
#
# 한쇼 파일도 ZIP+XML 구조.
# 슬라이드 텍스트는 Contents/SlideN.xml 에 있다.
# ===========================================================================


def _extract_show(path: Path) -> str:
    """한쇼(.show) 파일에서 텍스트를 추출한다."""
    with zipfile.ZipFile(str(path), "r") as zf:
        slide_files = sorted(
            [n for n in zf.namelist() if re.match(r"Contents/[Ss]lide\d+\.xml", n)]
        )

        if not slide_files:
            slide_files = sorted(
                [n for n in zf.namelist() if n.endswith(".xml") and "slide" in n.lower()]
            )

        all_text: list[str] = []

        for i, sf in enumerate(slide_files):
            xml_data = zf.read(sf)
            paragraphs = _parse_hwpx_xml(xml_data)  # 같은 XML 구조 사용
            if paragraphs:
                all_text.append(f"--- Slide {i + 1} ---")
                all_text.extend(paragraphs)

        if not all_text:
            # fallback: 모든 XML에서 텍스트 추출
            all_text = _extract_text_from_all_xmls(zf)

        return "\n".join(all_text)


# ===========================================================================
# Cell 파싱 (.cell)
#
# 한셀 파일도 ZIP+XML 구조.
# 시트 데이터는 Contents/SheetN.xml 에 있다.
# ===========================================================================


def _extract_cell(path: Path) -> str:
    """한셀(.cell) 파일에서 텍스트를 추출한다."""
    with zipfile.ZipFile(str(path), "r") as zf:
        sheet_files = sorted(
            [n for n in zf.namelist() if re.match(r"Contents/[Ss]heet\d+\.xml", n)]
        )

        if not sheet_files:
            sheet_files = sorted(
                [n for n in zf.namelist() if n.endswith(".xml") and "sheet" in n.lower()]
            )

        all_text: list[str] = []

        for i, sf in enumerate(sheet_files):
            xml_data = zf.read(sf)
            rows = _parse_cell_xml(xml_data)
            if rows:
                all_text.append(f"--- Sheet {i + 1} ---")
                all_text.extend(rows)

        if not all_text:
            all_text = _extract_text_from_all_xmls(zf)

        return "\n".join(all_text)


def _parse_cell_xml(xml_data: bytes) -> list[str]:
    """한셀 Sheet XML에서 셀 데이터를 추출한다."""
    rows: list[str] = []

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return rows

    # 셀 데이터에서 텍스트 추출
    current_row: list[str] = []

    for elem in root.iter():
        tag = _strip_ns(elem.tag)

        if tag == "t":
            text = elem.text or ""
            if text.strip():
                current_row.append(text.strip())
        elif tag in ("row", "tr", "tableRow"):
            if current_row:
                rows.append("\t".join(current_row))
                current_row = []

    if current_row:
        rows.append("\t".join(current_row))

    if not rows:
        rows = _extract_all_text(root)

    return rows


# ===========================================================================
# 공통 유틸리티
# ===========================================================================


def _strip_ns(tag: str) -> str:
    """XML 태그에서 네임스페이스를 제거한다.

    예: '{http://www.hancom.co.kr/hwpml/2011/paragraph}t' -> 't'
    """
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _extract_all_text(root: ET.Element) -> list[str]:
    """XML 트리에서 모든 텍스트 노드를 추출한다 (fallback용)."""
    texts: list[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
        if elem.tail and elem.tail.strip():
            texts.append(elem.tail.strip())
    return texts


def _extract_text_from_all_xmls(zf: zipfile.ZipFile) -> list[str]:
    """ZIP 내 모든 XML 파일에서 텍스트를 추출한다 (최후의 fallback)."""
    all_text: list[str] = []
    for name in sorted(zf.namelist()):
        if name.endswith(".xml") and "Contents/" in name:
            try:
                xml_data = zf.read(name)
                root = ET.fromstring(xml_data)
                texts = _extract_all_text(root)
                if texts:
                    all_text.append(f"--- {name} ---")
                    all_text.extend(texts)
            except (ET.ParseError, KeyError):
                continue
    return all_text
