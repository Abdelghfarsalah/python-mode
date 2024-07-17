# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:47:59 2024

@author: Options
"""

from flask import Flask, request, jsonify
import pandas as pd
import joblib
df = pd.read_csv("dataSet/student-scores.csv")

# تحميل النموذج المحفوظ
model = joblib.load('linear_regression_model.pkl')



# إنشاء تطبيق Flask
app = Flask(__name__)

@app.route('/')
def home():
    model_info = {
        'coef': list(model.coef_),  # على سبيل المثال
        'intercept': model.intercept_
    }
    
    # إعادة النتيجة في صيغة JSON
    return jsonify(model_info)

@app.route('/predict', methods=['Post'])
def predict():
    # الحصول على البيانات من طلب POST
    data = request.json
    df = pd.DataFrame(data, index=[0])
    
    # تحويل البيانات إلى نفس الشكل المستخدم في التدريب
    df['gender'] = df['gender'].replace({'male': 0, 'female': 1})
    df['part_time_job'] = df['part_time_job'].astype(int)
    df['extracurricular_activities'] = df['extracurricular_activities'].astype(int)
    
    # استخدام النموذج للتنبؤ
    prediction = model.predict(df)
    print(prediction)
    print("dlskf")
    # إعادة النتيجة في صيغة JSON
    return "jsonify({'prediction': prediction[0]})"
@app.route('/talk',methods=['Post'])
def talk():
    return "اي ي قلب المودل"
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


@app.route('/predict', methods=['GET'])
def predict_get():
    # الحصول على البيانات من معلمات URL
    gender = request.args.get('gender')
    part_time_job = int(request.args.get('part_time_job'))
    extracurricular_activities = int(request.args.get('extracurricular_activities'))
    
    # تحويل القيم إلى نفس الشكل المستخدم في التدريب
    gender = 0 if gender == 'male' else 1
    
    # تجهيز البيانات في DataFrame
    df = pd.DataFrame({
        'gender': [gender],
        'part_time_job': [part_time_job],
        'extracurricular_activities': [extracurricular_activities]
    })
    
    # استخدام النموذج للتنبؤ
    prediction = model.predict(df)
    
    # إعادة النتيجة في صيغة JSON
    return jsonify({'prediction': prediction[0]})



