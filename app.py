from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)

# 모델 로드
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
xgb_model = XGBClassifier()
xgb_model.load_model('xgb_model.json')
mlp_model = load_model('mlp_model.h5')
mlp_model_1 = load_model('mlp_model_1.h5')  # 앙상블 모델 1
mlp_model_2 = load_model('mlp_model_2.h5')  # 앙상블 모델 2
scaler = pickle.load(open('scaler.pkl', 'rb'))  # 스케일러 저장 필요

# 특성 처리 함수
def preprocess_input(data):
    # data: 입력된 사용자 값 (dict 형태)
    processed_data = np.array([
        float(data['Age']),
        float(data['Fare']),
        1 if data['Sex'] == 'male' else 0,
        int(data['Pclass']),
        int(data['Family_Size']),
        int(data['Is_Alone'])
    ]).reshape(1, -1)
    return scaler.transform(processed_data)

# 앙상블 MLP 생존 확률 계산 함수
def ensemble_mlp_predict(models, processed_data, weights):
    predictions = [model.predict(processed_data)[0][0] for model in models]
    weighted_prediction = sum(w * p for w, p in zip(weights, predictions))
    return weighted_prediction

# 생존 확률 계산 함수
def predict_survival(data):
    processed_data = preprocess_input(data)

    # 각 모델 예측
    rf_prob = rf_model.predict_proba(processed_data)[0][1]
    xgb_prob = xgb_model.predict_proba(processed_data)[0][1]
    mlp_prob = mlp_model.predict(processed_data)[0][0]

    # 앙상블 MLP 예측
    ensemble_prob = ensemble_mlp_predict(
        [mlp_model_1, mlp_model_2],
        processed_data,
        weights=[0.5, 0.5]
    )

    # 평균 생존 확률 계산
    avg_prob = (rf_prob + xgb_prob + mlp_prob + ensemble_prob) / 4
    return {
        "Random Forest": rf_prob,
        "XGBoost": xgb_prob,
        "MLP": mlp_prob,
        "Ensemble MLP": ensemble_prob,
        "Average Probability": avg_prob
    }

# 웹 인터페이스
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 사용자 입력 처리
        data = {
            "Age": request.form.get("Age"),
            "Fare": request.form.get("Fare"),
            "Sex": request.form.get("Sex"),
            "Pclass": request.form.get("Pclass"),
            "Family_Size": request.form.get("Family_Size"),
            "Is_Alone": request.form.get("Is_Alone")
        }
        predictions = predict_survival(data)
        return render_template("index.html", predictions=predictions, data=data)
    return render_template("index.html", predictions=None)

if __name__ == "__main__":
    app.run(debug=True)
