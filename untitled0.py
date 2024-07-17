
import pandas as pd
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("student-scores.csv")

print(df.head())

df['TOTAL_MARK'] = (df['math_score'] + df['history_score'] + df['physics_score'] + df['chemistry_score'] +
                    df['biology_score'] + df['english_score'] + df['geography_score']) / 7


#مش مهمين
df = df.drop(columns=['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score'])


# تحويل القيم النصية إلى رقمية
df['gender'] = df['gender'].replace({'male': 0, 'female': 1})
df['part_time_job'] = df['part_time_job'].astype(int)
df['extracurricular_activities'] = df['extracurricular_activities'].astype(int)

# استخدام LabelEncoder لتحويل جميع الأعمدة النصية إلى أرقام
labelencoder = LabelEncoder()# كان جايلي ايرور والشات اللي ضافها عشان يحول النص الي ارقام
for column in df.select_dtypes(include=['object']).columns:
    df[column] = labelencoder.fit_transform(df[column])

print(df)

# تحضير البيانات للنموذج
X = df.drop(columns='TOTAL_MARK')#التدريب
y = df['TOTAL_MARK']#النواتج

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# بناء وتدريب النموذج
model = LinearRegression()
model.fit(X_train, y_train)

# التنبؤ باستخدام مجموعة الاختبار
y_pred = model.predict(X_test)

# تقييم النموذج
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')



def pre ():
    return "kdlfjla"



#هنا بقا حاجه كدا زي بصدر النموزج بتاعي واحفظه ف ملف ي غالي
import joblib

# حفظ النموذج إلى ملف
joblib.dump(model, 'linear_regression_model.pkl')
