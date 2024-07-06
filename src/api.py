import utils
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
from preprocessing import ohe_transform
import json

app = FastAPI()

model = utils.deserialize_data("models/random_forest_classifier.pkl")

f = open('data/processed/random_forest_threshold.json')

threshold = json.load(f)

ohe_home_ownership = utils.deserialize_data('models/ohe_home_ownership.pkl')
ohe_loan_intent = utils.deserialize_data('models/ohe_loan_intent.pkl')
ohe_loan_grade = utils.deserialize_data('models/ohe_loan_grade.pkl')
ohe_default_on_file = utils.deserialize_data('models/ohe_default_on_file.pkl')

class Item(BaseModel):
  person_age: int
  person_income: float
  person_home_ownership: str
  person_emp_length: float
  loan_intent: str
  loan_grade: str
  loan_amnt: float
  loan_int_rate: float
  loan_percent_income: float
  cb_person_default_on_file: str
  cb_person_cred_hist_length: int

@app.post("/pred")
async def predict(item: Item):
  try:
    df = pd.DataFrame([item.dict()])

    print(df.head())

    X_predict = preprocessing(df)

    prediction = predict(X_predict)
    return {
      "status": "succes",
      "result": prediction
    }
  except ValidationError as e:
    return {
      "status": "error",
      "message": "validation error",
      "details": e.errors()
    }

def preprocessing(X_predict):
  X_predict = ohe_transform(dataset=X_predict, subset="person_home_ownership", prefix="home_ownership", ohe=ohe_home_ownership)
  X_predict = ohe_transform(dataset=X_predict, subset="loan_intent", prefix="loan_intent", ohe=ohe_loan_intent )
  X_predict = ohe_transform(dataset=X_predict, subset="loan_grade", prefix="loan_grade", ohe=ohe_loan_grade )
  X_predict = ohe_transform(dataset=X_predict, subset="cb_person_default_on_file", prefix="default_onfile", ohe=ohe_default_on_file)

  return X_predict

def predict(X_predict):
  y_proba = model.predict_proba(X_predict)[:, 1]
  if (y_proba >= threshold['threshold']):
    return 'Non-Performing Loan'
  
  return 'Performing Loan'