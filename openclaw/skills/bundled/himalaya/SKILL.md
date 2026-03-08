---
name: himalaya
description: "CLI 이메일 클라이언트. IMAP/SMTP를 통해 이메일 목록 조회, 읽기, 작성, 답장, 전달, 검색, 정리를 터미널에서 수행합니다."
user-invocable: true
metadata:
  openclaw:
    requires:
      bins: ["himalaya"]
---

# Himalaya - CLI 이메일 스킬

`himalaya` CLI로 IMAP/SMTP 이메일을 관리합니다.
격리망 내부에 메일 서버가 있으면 에이전트가 이메일을 읽고 보낼 수 있습니다.

## 설치

```bash
# macOS
brew install himalaya

# Linux (바이너리 다운로드)
curl -LO https://github.com/pimalaya/himalaya/releases/latest/download/himalaya-x86_64-linux.tar.gz
tar xzf himalaya-x86_64-linux.tar.gz
mv himalaya /usr/local/bin/

# 격리망: 바이너리를 수동 복사
scp himalaya user@airgapped-host:/usr/local/bin/
```

## 설정

`~/.config/himalaya/config.toml` 생성:

```toml
[accounts.work]
email = "agent@company.internal"
display-name = "AI Agent"
default = true

backend.type = "imap"
backend.host = "mail.company.internal"
backend.port = 993
backend.encryption.type = "tls"
backend.login = "agent@company.internal"
backend.auth.type = "password"
backend.auth.cmd = "cat /etc/openclaw/mail-password"

message.send.backend.type = "smtp"
message.send.backend.host = "smtp.company.internal"
message.send.backend.port = 587
message.send.backend.encryption.type = "start-tls"
message.send.backend.login = "agent@company.internal"
message.send.backend.auth.type = "password"
message.send.backend.auth.cmd = "cat /etc/openclaw/mail-password"
```

대화형 설정 마법사:
```bash
himalaya account configure
```

## 이메일 조회

### 폴더 목록
```bash
himalaya folder list
```

### 받은편지함 목록
```bash
himalaya envelope list
```

### 특정 폴더
```bash
himalaya envelope list --folder "Sent"
```

### 페이지네이션
```bash
himalaya envelope list --page 1 --page-size 20
```

### 검색
```bash
himalaya envelope list from admin@company.internal subject "서버 점검"
```

## 이메일 읽기

```bash
# ID로 읽기 (평문)
himalaya message read 42

# 원본 MIME 내보내기
himalaya message export 42 --full
```

## 이메일 작성

### 템플릿으로 바로 발송
```bash
cat << 'EOF' | himalaya template send
From: agent@company.internal
To: admin@company.internal
Subject: 일일 보고서

안녕하세요,

오늘의 에이전트 운용 보고서입니다.

- 처리된 작업: 15건
- 오류: 0건
- 평균 응답 시간: 2.3초

감사합니다.
EOF
```

### 헤더 플래그로 발송
```bash
himalaya message write \
  -H "To:admin@company.internal" \
  -H "Subject:알림" \
  "서버 디스크 사용량이 90%를 초과했습니다."
```

## 답장/전달

```bash
# 답장
himalaya message reply 42

# 전체 답장
himalaya message reply 42 --all

# 전달
himalaya message forward 42
```

## 이메일 관리

```bash
# 폴더 이동
himalaya message move 42 "Archive"

# 복사
himalaya message copy 42 "Important"

# 삭제
himalaya message delete 42

# 읽음 표시
himalaya flag add 42 --flag seen

# 안읽음 표시
himalaya flag remove 42 --flag seen
```

## 첨부파일

```bash
# 첨부파일 다운로드
himalaya attachment download 42

# 특정 디렉토리에 저장
himalaya attachment download 42 --dir ~/Downloads
```

## 여러 계정 사용

```bash
# 계정 목록
himalaya account list

# 특정 계정으로 조회
himalaya --account personal envelope list
```

## JSON 출력 (파싱용)

```bash
himalaya envelope list --output json
```

## 디버깅

```bash
RUST_LOG=debug himalaya envelope list
```

## 격리망 활용 팁

- 에이전트가 작업 결과를 이메일로 자동 보고하는 용도로 유용합니다.
- 수신 이메일을 읽어 작업 지시를 받는 워크플로도 가능합니다.
- 비밀번호는 파일이나 시스템 키링으로 안전하게 관리하세요.
- 메일 서버의 TLS 인증서가 내부 CA 발급이면 환경변수 설정이 필요할 수 있습니다.
