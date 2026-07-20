# Stock Price Prediction with LSTM & PyTorch

## Files Included

1. **stock_lstm_prediction.ipynb** - Jupyter Notebook (권장)
2. **stock_lstm_prediction.py** - 독립형 Python 스크립트

---

## Requirements

```bash
pip install torch pandas numpy scikit-learn matplotlib yfinance
```

---

## Usage

### Option 1: Jupyter Notebook (권장)
Jupyter Lab에서 `stock_lstm_prediction.ipynb` 열고 셀을 차례대로 실행합니다.

```bash
jupyter lab stock_lstm_prediction.ipynb
```

### Option 2: Python Script
```bash
python stock_lstm_prediction.py
```

---

## 주요 파라미터 조정

**`stock_lstm_prediction.ipynb` Step 8에서:**

```python
TICKER = 'AAPL'        # 다른 주식: 'MSFT', 'GOOGL', '^GSPC' 등
SEQ_LENGTH = 30        # 과거 날짜 수 (더 크면 더 긴 기억)
EPOCHS = 100           # 반복 학습 횟수
BATCH_SIZE = 32        # 한 번에 처리할 샘플 수
```

---

## Output

1. **4-panel visualization**
   - 학습/테스트 Loss 곡선
   - 학습 데이터 예측 vs 실제
   - 테스트 데이터 예측 vs 실제
   - 에러 분포

2. **Performance Metrics**
   - Train RMSE (Root Mean Squared Error)
   - Test RMSE
   - Direction Accuracy (방향 정확도)
   - Average Absolute Error

3. **Next Day Prediction**
   - 다음날 예상 주가
   - 예상 변화액 및 %

---

## 중요한 한계

### 모델이 잘 작동하지 않는 이유:

1. **Lag (지연 효과)**
   - 모델은 과거 추세를 따라가기만 함
   - 급격한 변화에 반응이 느림

2. **외부 정보 부재**
   - 뉴스, 수익 발표, 금리 변화 무시
   - 오직 과거 가격만 사용

3. **시장 효율성**
   - 만약 이 패턴이 수익성 있다면 이미 시장에 반영됨
   - 과거 데이터의 패턴은 미래에 반복되지 않음

4. **과적합 위험**
   - LSTM은 표현력이 강해서 노이즈까지 학습할 수 있음

---

## 개선 방법 (과제)

더 좋은 결과를 원한다면:

1. **특성 추가**
   ```python
   - 거래량 (Volume)
   - 변동성 (Volatility)
   - 기술적 지표 (RSI, MACD, Bollinger Bands)
   - 경제 지표 (금리, 환율, 시장 지수)
   ```

2. **모델 아키텍처 개선**
   - Attention 메커니즘 추가
   - Bidirectional LSTM 사용
   - Ensemble 모델 (여러 모델 결합)

3. **학습 전략**
   - Dropout & L2 정규화로 과적합 방지
   - Early stopping 사용
   - 다른 loss function 시도 (MAE, Huber Loss)

4. **평가 방법**
   - Walk-forward validation (슬라이딩 윈도우)
   - 실제 수익성 검증 (Backtesting)
   - 간단한 베이스라인과 비교 (Moving Average, Random Walk)

---

## Learning Resources

- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Time Series Forecasting with Deep Learning](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)
- [Stock Price Prediction: Challenges and Opportunities](https://arxiv.org/abs/1909.12227)

---

## Notes

- 이 모델은 **교육 목적**으로 제작되었습니다
- **실제 거래에 사용하면 안 됩니다**
- 금융 조언이 아닙니다
- 손실 가능성이 있습니다

---

## Troubleshooting

### GPU 사용 안 됨
```python
# Jupyter에서 확인:
import torch
print(torch.cuda.is_available())  # False면 GPU 없음
# CPU로 자동 변환되니 계속 실행 가능 (다만 느림)
```

### Out of Memory
```python
BATCH_SIZE = 16  # 32에서 16으로 줄임
SEQ_LENGTH = 15  # 30에서 15로 줄임
```

### yfinance 다운로드 오류
```python
# 인터넷 연결 확인
# 혹은 로컬 CSV 파일 사용
raw_data = pd.read_csv('your_stock_data.csv')['Close'].values.reshape(-1, 1)
```

---
