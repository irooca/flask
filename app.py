from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

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
from tensorflow.keras.models import load_model
mlp_model = load_model('mlp_model.h5')

def preprocess_input(data):
    # Family_Size에서 Is_Alone 계산
    data['Is_Alone'] = (data['Family_Size'] == 1).astype(int)
    # 필요한 피처만 반환
    features = ['Age', 'Fare', 'Sex', 'Pclass', 'Family_Size', 'Is_Alone']
    return data[features]

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

        # 입력 데이터를 DataFrame으로 변환
        input_data = pd.DataFrame([{
            'Age': age,
            'Fare': fare,
            'Sex': sex,
            'Pclass': pclass,
            'Family_Size': family_size
        }])

        # 전처리
        processed_data = preprocess_input(input_data)

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
