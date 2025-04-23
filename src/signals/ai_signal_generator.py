import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging

# --- 로거 설정 ---
logger = logging.getLogger(__name__) # 적절한 로거 이름 사용 권장

# --- 모델 아키텍처 정의 (가정) ---
class GRUEntryClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=2):
        super(GRUEntryClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # GRU 레이어: batch_first=True 설정으로 (batch, seq_len, input_dim) 형태의 입력 처리
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        # 최종 분류를 위한 Fully Connected 레이어
        self.fc = nn.Linear(hidden_dim, output_dim)
        # 소프트맥스 활성화 함수 (로그 확률이 아닌 실제 확률 반환)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 초기 은닉 상태를 0으로 설정 (batch_size는 x.size(0)에서 동적으로 가져옴)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # GRU 순전파
        out, _ = self.gru(x, h0)
        # 시퀀스의 마지막 타임 스텝의 출력만 사용
        out = self.fc(out[:, -1, :])
        # 소프트맥스 적용
        out = self.softmax(out)
        return out

# --- AI 진입 시그널 생성 함수 ---
def generate_ai_entry_signals(price_df: pd.DataFrame,
                               model_path: str = "models/buy_signal_gru.pth",
                               threshold: float = 0.8,
                               window_size: int = 48,
                               device: str = "cpu") -> pd.Series:
    """
    PyTorch GRU 모델을 사용하여 AI 기반 진입 시그널을 생성합니다.

    Args:
        price_df (pd.DataFrame): OHLCV 데이터 (인덱스는 타임스탬프).
        model_path (str): 학습된 모델 파일 경로.
        threshold (float): 진입 시그널(클래스 1) 확률 임계값.
        window_size (int): 모델 입력 시퀀스 길이 (슬라이딩 윈도우 크기).
        device (str): 추론에 사용할 디바이스 ('cpu' 또는 'cuda').

    Returns:
        pd.Series: 진입 시그널 (1: 진입, 0: 보류), price_df와 동일한 인덱스.

    Raises:
        FileNotFoundError: 모델 파일을 찾을 수 없는 경우.
        ValueError: price_df 컬럼이 부족하거나 형식이 맞지 않을 경우.
    """
    logger.info(f"AI 진입 시그널 생성을 시작합니다. 모델: {model_path}, 임계값: {threshold}, 윈도우: {window_size}")

    # 입력 데이터 검증
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in price_df.columns for col in required_columns):
        raise ValueError(f"입력 DataFrame에는 다음 컬럼이 필요합니다: {required_columns}")
    
    # 필요한 컬럼만 선택 및 결측치 확인/처리 (여기서는 간단히 ffill 후 dropna)
    data = price_df[required_columns].ffill().dropna()
    if len(data) < window_size:
        logger.warning(f"데이터 길이({len(data)})가 윈도우 크기({window_size})보다 작아 시그널을 생성할 수 없습니다.")
        return pd.Series(0, index=price_df.index) # 전체 0 반환

    # 모델 파일 존재 확인
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"지정된 경로에 모델 파일이 없습니다: {model_path}")

    # 모델 로드
    try:
        # input_dim은 데이터 컬럼 수(5)와 일치해야 함
        model = GRUEntryClassifier(input_dim=len(required_columns), hidden_dim=64, num_layers=2, output_dim=2) 
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # 평가 모드 설정
        logger.info(f"모델 로드 완료: {model_path}")
    except Exception as e:
        logger.error(f"모델 로딩 중 오류 발생: {e}", exc_info=True)
        raise

    # 예측 결과를 저장할 리스트
    predictions = []

    # 슬라이딩 윈도우 및 추론 (torch.no_grad() 사용)
    with torch.no_grad():
        for i in range(len(data) - window_size + 1):
            window = data.iloc[i : i + window_size].values # numpy 배열
            
            # 데이터를 텐서로 변환 및 차원 변경: (seq_len, input_dim) -> (1, seq_len, input_dim)
            input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

            # 모델 추론
            output = model(input_tensor) # (1, output_dim) 형태의 출력
            
            # 클래스 1 (진입)의 확률 추출
            prob_class_1 = output[0, 1].item() # .item()으로 스칼라 값 추출
            
            # 임계값과 비교하여 시그널 결정
            signal = 1 if prob_class_1 >= threshold else 0
            predictions.append(signal)

    # 결과 Series 생성
    # 예측값은 window_size - 1 만큼 뒤에서 시작하므로 앞부분은 0으로 채움
    num_predictions = len(predictions)
    padding = [0] * (len(price_df) - num_predictions)
    final_signals = padding + predictions

    signal_series = pd.Series(final_signals, index=price_df.index, name='AI_Entry_Signal')

    logger.info(f"AI 진입 시그널 생성 완료. 총 {signal_series.sum()}개의 진입 시그널 생성됨.")
    # 생성된 시그널 분포 로그 추가 요청됨 (아래 backtest_runner_vbt.py에서 출력)
    # logger.info(f"생성된 AI 진입 시그널 분포:\n{signal_series.value_counts()}") # 주석 처리된 라인 수정
    return signal_series


# --- AI 퇴장 시그널 생성 함수 (Stub) ---
def generate_ai_exit_signals(price_df: pd.DataFrame,
                              model_path: str = "models/exit_signal_gru.pth", # 다른 모델 경로 가정
                              threshold: float = 0.8,
                              window_size: int = 48,
                              device: str = "cpu") -> pd.Series:
    """
    PyTorch GRU 모델을 사용하여 AI 기반 퇴장 시그널을 생성합니다. (구현 예정)

    Args:
        price_df (pd.DataFrame): OHLCV 데이터.
        model_path (str): 학습된 퇴장 시그널 모델 파일 경로.
        threshold (float): 퇴장 시그널(클래스 1) 확률 임계값.
        window_size (int): 모델 입력 시퀀스 길이.
        device (str): 추론에 사용할 디바이스.

    Returns:
        pd.Series: 퇴장 시그널 (1: 퇴장, 0: 유지), price_df와 동일한 인덱스. (현재는 모두 0 반환)
    """
    logger.warning("generate_ai_exit_signals 함수는 아직 구현되지 않았습니다. 기본값(0)을 반환합니다.")
    # 실제 구현 시 generate_ai_entry_signals와 유사한 로직 사용
    return pd.Series(0, index=price_df.index, name='AI_Exit_Signal') 