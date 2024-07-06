import streamlit as st
import requests
from datetime import datetime

current_date = datetime.now().strftime('%d %B %Y')

st.set_page_config(page_title="Credit Risk Analyzer App")
st.title('Credit Risk Analyzer App')
st.markdown(f'*Created by Nanda Fadhil Azman | Batch May 2024* | Last deployed {current_date}')
st.divider()

with st.form(key='analysis-form'):
  person_age = st.number_input('Person Age', min_value=0)
  person_income = st.number_input('Person Income', min_value=0)
  person_home_ownership = st.selectbox('Person Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
  person_emp_length = st.number_input('Person Employment Length (years)', min_value=0)
  loan_intent = st.selectbox('Loan Intent', ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'])
  loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
  loan_amnt = st.number_input('Loan Amount', min_value=0)
  loan_int_rate = st.number_input('Loan Interest Rate', min_value=0.0, format="%.2f")
  loan_percent_income = st.number_input('Loan Percent Income', min_value=0.0, format="%.2f")
  cb_person_default_on_file = st.selectbox('CB Person Default on File', ['Y', 'N'])
  cb_person_cred_hist_length = st.number_input('CB Person Credit History Length', min_value=0)

  submit_button = st.form_submit_button('Predict')

  if submit_button:
    st.balloons()

    data = {
      "person_age": person_age,
      "person_income": person_income,
      "person_home_ownership": person_home_ownership,
      "person_emp_length": person_emp_length,
      "loan_intent": loan_intent,
      "loan_grade": loan_grade,
      "loan_amnt": loan_amnt,
      "loan_int_rate": loan_int_rate,
      "loan_percent_income": loan_percent_income,
      "cb_person_default_on_file": cb_person_default_on_file,
      "cb_person_cred_hist_length": cb_person_cred_hist_length
    }

    with st.spinner('Wait for it...'):
        response = requests.post('http://localhost:8000/pred', json=data)
        result = response.json()
        
        
    # if success
    if response.status_code == 200:
        st.success(result['result'])
        st.balloons()
    else:
        st.error(result['detail_error'])