import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================
# 1. 데이터 준비
# ============================================
def load_stock_data(ticker='AAPL', days=500):
    """Yahoo Finance에서 주식 데이터 로드"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data['Close'].values.reshape(-1, 1)

def create_sequences(data, seq_length=30):
    """시퀀스 데이터 생성"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def prepare_data(ticker='AAPL', seq_length=30, test_split=0.2):
    """데이터 로드, 정규화, 분할"""
    # 데이터 로드
    raw_data = load_stock_data(ticker, days=500)
    
    # 정규화
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_data)
    
    # 시퀀스 생성
    X, y = create_sequences(scaled_data, seq_length)
    
    # 학습/테스트 분할 (시간 순서 유지)
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 텐서로 변환
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, y_train, X_test, y_test, scaler

# ============================================
# 2. LSTM 모델 정의
# ============================================
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

# ============================================
# 3. 모델 학습
# ============================================
def train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32, learning_rate=0.001):
    """모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 초기화
    model = StockPriceLSTM(hidden_size=50, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # 학습 루프
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= (len(X_train) // batch_size)
        train_losses.append(train_loss)
        
        # 평가
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test.to(device))
            test_loss = criterion(test_pred, y_test.to(device))
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    return model, train_losses, test_losses, device

# ============================================
# 4. 평가 및 예측
# ============================================
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, device):
    """모델 평가"""
    model.eval()
    
    with torch.no_grad():
        # 학습 데이터 예측
        train_pred = model(X_train.to(device)).cpu().numpy()
        train_actual = y_train.numpy()
        
        # 테스트 데이터 예측
        test_pred = model(X_test.to(device)).cpu().numpy()
        test_actual = y_test.numpy()
    
    # RMSE 계산
    train_rmse = np.sqrt(np.mean((train_pred - train_actual) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - test_actual) ** 2))
    
    # 역정규화
    train_pred_real = scaler.inverse_transform(train_pred)
    train_actual_real = scaler.inverse_transform(train_actual)
    test_pred_real = scaler.inverse_transform(test_pred)
    test_actual_real = scaler.inverse_transform(test_actual)
    
    # 실제 값 기준 RMSE
    train_rmse_real = np.sqrt(np.mean((train_pred_real - train_actual_real) ** 2))
    test_rmse_real = np.sqrt(np.mean((test_pred_real - test_actual_real) ** 2))
    
    print(f"\n===== Model Evaluation =====")
    print(f"Train RMSE (Normalized): {train_rmse:.6f}")
    print(f"Test RMSE (Normalized): {test_rmse:.6f}")
    print(f"Train RMSE (Real Price): ${train_rmse_real:.2f}")
    print(f"Test RMSE (Real Price): ${test_rmse_real:.2f}")
    
    return train_pred_real, train_actual_real, test_pred_real, test_actual_real

# ============================================
# 5. 시각화
# ============================================
def plot_results(train_losses, test_losses, train_pred, train_actual, test_pred, test_actual):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss 곡선
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(test_losses, label='Test Loss')
    axes[0, 0].set_title('Model Loss Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 학습 데이터 예측 vs 실제
    axes[0, 1].plot(train_actual, label='Actual Price', linewidth=2)
    axes[0, 1].plot(train_pred, label='Predicted Price', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Training Data: Predicted vs Actual')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 테스트 데이터 예측 vs 실제
    axes[1, 0].plot(test_actual, label='Actual Price', linewidth=2)
    axes[1, 0].plot(test_pred, label='Predicted Price', linewidth=2, alpha=0.7)
    axes[1, 0].set_title('Test Data: Predicted vs Actual')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 에러 분포
    train_error = train_actual - train_pred
    test_error = test_actual - test_pred
    axes[1, 1].hist(train_error.flatten(), bins=30, alpha=0.6, label='Train Error')
    axes[1, 1].hist(test_error.flatten(), bins=30, alpha=0.6, label='Test Error')
    axes[1, 1].set_title('Prediction Error Distribution')
    axes[1, 1].set_xlabel('Error ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/stock_prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to: /mnt/user-data/outputs/stock_prediction_results.png")

# ============================================
# 메인 실행
# ============================================
if __name__ == "__main__":
    print("Starting Stock Price Prediction with LSTM...")
    print("=" * 50)
    
    # 파라미터
    TICKER = 'AAPL'  # 변경 가능: 'MSFT', 'GOOGL', '^GSPC' 등
    SEQ_LENGTH = 30
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # 데이터 준비
    print(f"\n1. Loading data for {TICKER}...")
    X_train, y_train, X_test, y_test, scaler = prepare_data(TICKER, SEQ_LENGTH)
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    
    # 모델 학습
    print(f"\n2. Training LSTM model...")
    model, train_losses, test_losses, device = train_model(
        X_train, y_train, X_test, y_test, 
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    # 평가
    print(f"\n3. Evaluating model...")
    train_pred, train_actual, test_pred, test_actual = evaluate_model(
        model, X_train, y_train, X_test, y_test, scaler, device
    )
    
    # 시각화
    print(f"\n4. Generating plots...")
    plot_results(train_losses, test_losses, train_pred, train_actual, test_pred, test_actual)
    
    print("\n" + "=" * 50)
    print("Completed!")
