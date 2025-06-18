import joblib
import numpy as np
import pandas as pd
import pdfkit
import tempfile
import os

# Path to the saved model and its components
MODEL_PATH = 'artifacts/model_data.joblib'

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']

def prepare_input(age, income, loan_amount, loan_tenure_months,
                  avg_dpd_per_delinquency, delinquency_ratio,
                  credit_utilization_ratio, num_open_accounts,
                  residence_type, loan_purpose, loan_type):
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # additional dummy fields
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df[features]
    return df

def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_
    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score[0])
    return default_probability.flatten()[0], int(credit_score[0]), rating

def predict(age, income, loan_amount, loan_tenure_months,
            avg_dpd_per_delinquency, delinquency_ratio,
            credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):

    input_df = prepare_input(age, income, loan_amount, loan_tenure_months,
                             avg_dpd_per_delinquency, delinquency_ratio,
                             credit_utilization_ratio, num_open_accounts,
                             residence_type, loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)
    return probability, credit_score, rating

def generate_pdf(age, income, loan_amount, loan_tenure_months,
                 avg_dpd_per_delinquency, delinquency_ratio,
                 credit_utilization_ratio, num_open_accounts,
                 residence_type, loan_purpose, loan_type,
                 probability, credit_score, rating):

    html_content = f"""
    <h2>Credit Risk Assessment Report</h2>
    <p><strong>Age:</strong> {age}</p>
    <p><strong>Income:</strong> {income}</p>
    <p><strong>Loan Amount:</strong> {loan_amount}</p>
    <p><strong>Loan Tenure (months):</strong> {loan_tenure_months}</p>
    <p><strong>Average DPD:</strong> {avg_dpd_per_delinquency}</p>
    <p><strong>Delinquency Ratio:</strong> {delinquency_ratio}</p>
    <p><strong>Credit Utilization Ratio:</strong> {credit_utilization_ratio}</p>
    <p><strong>Open Loan Accounts:</strong> {num_open_accounts}</p>
    <p><strong>Residence Type:</strong> {residence_type}</p>
    <p><strong>Loan Purpose:</strong> {loan_purpose}</p>
    <p><strong>Loan Type:</strong> {loan_type}</p>
    <hr>
    <p><strong>Default Probability:</strong> {probability:.2%}</p>
    <p><strong>Credit Score:</strong> {credit_score}</p>
    <p><strong>Rating:</strong> {rating}</p>
    """

    config = pdfkit.configuration(wkhtmltopdf=r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdfkit.from_string(html_content, tmpfile.name, configuration=config)
        with open(tmpfile.name, "rb") as file:
            return file.read()
