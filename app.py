from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
import numpy as np

app = Flask(__name__)

# 모델 로드
rf_model = pickle.load(open('rf_model.pkl', 'rb'))  # Random Forest
xgb_model = XGBClassifier()
xgb_model.load_model('xgb_model.json')  # XGBoost
mlp_model = load_model('mlp_model.h5')  # MLP
mlp_model_1 = load_model('mlp_model_1.h5')  # Ensemble MLP Model 1
mlp_model_2 = load_model('mlp_model_2.h5')  # Ensemble MLP Model 2
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Scaler

# 입력 데이터 전처리 함수
def preprocess_input(data, scaler):
    processed_data = np.array([
        float(data['Age']),
        float(data['Fare']),
        1 if data['Sex'] == 'female' else 0,  # 성별: male → 1, female → 0
        int(data['Pclass']),
        int(data['Family_Size']),
        1 if int(data['Family_Size']) == 1 else 0  # Is_Alone 계산
    ]).reshape(1, -1)
    return scaler.transform(processed_data)

# 앙상블 MLP 예측 함수
def ensemble_predict(models, X, weights):
    predictions = [model.predict(X)[0][0] for model in models]
    weighted_prediction = sum(w * pred for w, pred in zip(weights, predictions))
    return weighted_prediction

# 생존 확률 계산 함수
def predict_survival(data):
    # Random Forest, XGBoost, MLP 모델의 예측
    rf_prob = rf_model.predict_proba(data)[0][1]
    xgb_prob = xgb_model.predict_proba(data)[0][1]
    mlp_prob = mlp_model.predict(data)[0][0]

    # 앙상블 MLP 예측
    ensemble_prob = ensemble_predict([mlp_model_1, mlp_model_2], data, weights=[0.5, 0.5])

    # 결과 반환
    return {
        "Random Forest": rf_prob,
        "XGBoost": xgb_prob,
        "MLP": mlp_prob,
        "Ensemble MLP": ensemble_prob,
        "Average Probability": (rf_prob + xgb_prob + mlp_prob + ensemble_prob) / 4,
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 사용자 입력 데이터 처리
        data = {
            "Age": float(request.form.get("Age")),
            "Fare": float(request.form.get("Fare")),
            "Sex": request.form.get("Sex"),
            "Pclass": int(request.form.get("Pclass")),
            "Family_Size": int(request.form.get("Family_Size")),
        }

        # 입력 데이터 전처리
        processed_data = preprocess_input(data, scaler)

        # 예측 실행
        predictions = predict_survival(processed_data)

        # 결과 렌더링
        return render_template("index.html", predictions=predictions, data=data)

    return render_template("index.html", predictions=None)

if __name__ == "__main__":
    app.run(debug=True)
