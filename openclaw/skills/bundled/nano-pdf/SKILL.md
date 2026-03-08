---
name: nano-pdf
description: "PDF 편집 도구. nano-pdf CLI로 PDF 페이지에 자연어 명령을 적용하여 텍스트 수정, 페이지 추출, 병합 등을 수행합니다."
user-invocable: true
metadata:
  openclaw:
    requires:
      bins: ["nano-pdf"]
---

# nano-pdf - PDF 편집 스킬

`nano-pdf` CLI를 사용하여 PDF 파일을 편집합니다.
이 스킬은 Bash 도구를 통해 `nano-pdf` 명령을 실행합니다.

## 설치

```bash
pip install nano-pdf
# 또는
uv tool install nano-pdf
```

격리망 환경에서는 오프라인으로 wheel을 설치하세요:
```bash
pip install nano-pdf-*.whl --no-index --find-links /path/to/wheels/
```

## 사용법

### 페이지 편집 (자연어 명령)

```bash
nano-pdf edit report.pdf 1 "제목을 '2024년 3분기 실적'으로 변경하고 부제 오타를 수정"
```

### 페이지 추출

```bash
nano-pdf extract input.pdf 1-5 output.pdf
```

### PDF 병합

```bash
nano-pdf merge file1.pdf file2.pdf file3.pdf -o merged.pdf
```

### 페이지 삭제

```bash
nano-pdf delete report.pdf 3 -o report_trimmed.pdf
```

### PDF 정보 조회

```bash
nano-pdf info document.pdf
```

## 주의사항

- 페이지 번호는 1부터 시작합니다 (일부 버전은 0부터). 결과가 어긋나면 반대 번호로 재시도하세요.
- 편집 후 반드시 결과 PDF를 확인하세요 (Read 도구의 pdf 기능 또는 `nano-pdf info` 사용).
- 대용량 PDF(100MB+)는 처리 시간이 길어질 수 있습니다.
- 이 스킬은 기존 내장 `pdf` 도구(읽기 전용)와 보완적입니다. 읽기는 `pdf` 도구, 편집은 `nano-pdf`를 사용하세요.
