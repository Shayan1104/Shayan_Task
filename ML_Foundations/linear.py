import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('student_exam_data.csv')


X = df[['Hours Studied ']]
y = df['Exam Score']


X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size = 0.2,
random_state = 42)

model = LinearRegression()
