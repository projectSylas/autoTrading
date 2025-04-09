#!/bin/bash

# 스크립트 실행 중 오류 발생 시 즉시 중단
set -e

echo "프로젝트 디렉토리 구조 변경을 시작합니다..."

# 1. 필요한 디렉토리 생성 (이미 있어도 오류 없이 넘어감)
echo "디렉토리 생성 중..."
mkdir -p src/core src/challenge src/analysis src/backtest src/utils src/config
mkdir -p scripts logs data notebooks tests

# 2. 파일 이동 (파일 존재 여부 확인 없이 실행되므로 주의!)
#    만약 파일 이름이 다르거나 이미 이동했다면 이 부분에서 오류가 발생할 수 있습니다.
echo "파일 이동 중..."
# src/ 하위 모듈들
mv core_portfolio.py src/core/portfolio.py
mv challenge_trading.py src/challenge/strategy.py
mv config.py src/config/settings.py
mv strategy_utils.py src/utils/common.py # 필요시 common.py / indicators.py 등으로 분리 가능
mv notifier.py src/utils/notifier.py
mv sentiment_analysis.py src/analysis/sentiment.py
mv volatility_alert.py src/analysis/volatility.py
mv backtest_strategy.py src/backtest/strategy.py
mv backtest_runner.py src/backtest/runner.py
mv main.py src/main.py

# scripts/ 하위 스크립트
mv auto_report.py scripts/auto_report.py

# logs/ 로 로그 파일 이동 (기존 로그 파일이 있다면)
# 만약 로그 파일 이름 패턴이 다르다면 수정 필요
echo "로그 파일 이동 중 (파일이 없으면 경고 발생)..."
mv log_core.csv logs/core.csv 2>/dev/null || echo "  - log_core.csv 없음"
mv log_challenge.csv logs/challenge.csv 2>/dev/null || echo "  - log_challenge.csv 없음"
mv log_backtest.csv logs/backtest.csv 2>/dev/null || echo "  - log_backtest.csv 없음"
mv log_sentiment.csv logs/sentiment.csv 2>/dev/null || echo "  - log_sentiment.csv 없음"
mv log_volatility.csv logs/volatility.csv 2>/dev/null || echo "  - log_volatility.csv 없음"
# 다른 로그 파일이 있다면 위 패턴에 맞춰 추가

# 3. __init__.py 파일 생성 (이미 있어도 덮어쓰지 않음)
echo "__init__.py 파일 생성 중..."
touch src/__init__.py
touch src/core/__init__.py
touch src/challenge/__init__.py
touch src/analysis/__init__.py
touch src/backtest/__init__.py
touch src/utils/__init__.py
touch src/config/__init__.py

echo "프로젝트 디렉토리 구조 변경 완료!"
echo "주의: 각 Python 파일 내부의 'import' 구문을 새로운 경로에 맞게 수정해야 합니다."