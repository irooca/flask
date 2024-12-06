from flask import Flask, render_template, request
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 모델 로드
models = {
    "Logistic Regression": joblib.load('logistic_model.pkl'),
    "Random Forest": joblib.load('random_forest_model.pkl'),
    "Gradient Boosting": joblib.load('gradient_boosting_model.pkl'),
    "XGBoost": joblib.load('xgboost_model.pkl'),
    "KNN": joblib.load('knn_model.pkl')
}

# MLP 모델 로드
mlp_model = load_model('mlp_model.h5')

def preprocess_input(age, fare, sex, pclass, family_size):
    # Family_Size에서 Is_Alone 계산
    is_alone = 1 if family_size == 1 else 0
    # 필요한 피처를 리스트로 반환
    return [[age, fare, sex, pclass, family_size, is_alone]]

# 홈 페이지
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 입력값 받기
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        sex = int(request.form['sex'])  # 0=Male, 1=Female
        pclass = int(request.form['pclass'])
        family_size = int(request.form['family_size'])

        # 전처리
        processed_data = preprocess_input(age, fare, sex, pclass, family_size)

        # 결과 저장
        predictions = []

        # 머신러닝 모델 예측
        for model_name, model in models.items():
            prob = model.predict_proba(processed_data)[:, 1][0]
            predictions.append((model_name, round(prob * 100, 2)))

        # 딥러닝 모델 예측
        mlp_prob = mlp_model.predict(processed_data)[0][0]
        predictions.append(("MLP (Deep Learning)", round(mlp_prob * 100, 2)))

        # 결과 반환
        return render_template(
            'result.html', 
            predictions=predictions, 
            input_data={
                'Age': age,
                'Fare': fare,
                'Sex': sex,
                'Pclass': pclass,
                'Family_Size': family_size
            }
        )
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred. Please check the server logs.", 500

if __name__ == '__main__':
    app.run(debug=True)
