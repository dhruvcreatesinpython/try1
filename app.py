
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import secrets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io

# Page configuration
st.set_page_config(
    page_title="Hospital Management System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SECURITY SYSTEM ====================
class HospitalSecurity:
    def __init__(self):
        pass
    
    def hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        return hashlib.sha256((password + salt).encode('utf-8')).hexdigest() + ':' + salt
    
    def verify_password(self, password: str, hashed: str) -> bool:
        if ':' in hashed:
            stored_hash, salt = hashed.split(':')
            computed_hash = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
            return stored_hash == computed_hash
        return False

security = HospitalSecurity()
ADMIN_USERS = {
    "admin": {
        "password_hash": security.hash_password("Admin123!"),
        "role": "super_admin",
        "name": "System Administrator"
    },
    "doctor": {
        "password_hash": security.hash_password("Admin123!"),
        "role": "medical_director",
        "name": "Head Doctor"
    }
}

# ==================== DIABETES AI ENGINE ====================
class DiabetesAI:
    def __init__(self):
        self.model = None
        self.features = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
        self._load_and_train_model()

    def _load_and_train_model(self):
        try:
            csv_content = """ID,No_Pation,Gender,AGE,Urea,Cr,HbA1c,Chol,TG,HDL,LDL,VLDL,BMI,CLASS
502,17975,F,50,4.7,46,4.9,4.2,0.9,2.4,1.4,0.5,24,N
735,34221,M,26,4.5,62,4.9,3.7,1.4,1.1,2.1,0.6,23,N
420,47975,F,50,4.7,46,4.9,4.2,0.9,2.4,1.4,0.5,24,N
680,87656,F,50,4.7,46,4.9,4.2,0.9,2.4,1.4,0.5,24,N
504,34223,M,33,7.1,46,4.9,4.9,1,0.8,2,0.4,21,N
634,34224,F,45,2.3,24,4,2.9,1,1,1.5,0.4,21,N
721,34225,F,50,2,50,4,3.6,1.3,0.9,2.1,0.6,24,N
421,34227,M,48,4.7,47,4,2.9,0.8,0.9,1.6,0.4,24,N
670,34229,M,43,2.6,67,4,3.8,0.9,2.4,3.7,1,21,N
759,34230,F,32,3.6,28,4,3.8,2,2.4,3.8,1,24,N
636,34231,F,31,4.4,55,4.2,3.6,0.7,1.7,1.6,0.3,23,N
788,34232,F,33,3.3,53,4,4,1.1,0.9,2.7,1,21,N
82,46815,F,30,3,42,4.1,4.9,1.3,1.2,3.2,0.5,22,N
132,34234,F,45,4.6,54,5.1,4.2,1.7,1.2,2.2,0.8,23,N
402,34235,F,50,3.5,39,4,4,1.5,1.2,2.2,0.7,24,N
566,34236,M,50,5.5,74,5,3.6,1.1,1,2.1,0.5,21,N
596,34237,F,50,5.9,53,5.4,5.3,0.8,1.1,4.1,0.3,21,N
676,87654,F,30,3,42,4.1,4.9,1.3,1.2,3.2,0.5,22,N
729,34238,F,49,2.2,28,4.1,5,1.3,1.2,3.3,0.6,24,N
742,34239,F,49,3.8,55,4,4.4,0.9,1,1.3,0.4,23,N
472,463,M,34,3.9,81,6,6.2,3.9,0.8,1.9,1.8,23,P
85,46300,M,34,3.9,81,6,6.2,3.9,0.8,3.8,1.8,23,P
710,87671,M,34,3.9,81,6,6.2,3.9,0.8,3.8,1.8,23,P
429,48036,M,31,3.4,55,5.7,4.9,1.6,1,3.2,0.7,24,P
702,87667,M,31,3.4,55,5.7,4.9,1.6,1,3.2,0.7,24,P
4,34301,F,43,2.1,55,5.7,4.7,5.3,0.9,1.7,2.4,25,P
189,30383,M,42,5.4,53,5.8,5.9,3.7,1.3,3.1,1.7,23,P
201,45573,M,47,4.1,87,6.2,3.7,1.8,1,2,0.8,23,P
285,47069,M,50,4.3,59,6.1,4,3,1,1.8,1.3,24,P
393,47496,M,49,5,74,6.2,2,0.8,0.6,1,0.4,25,P
468,298,M,50,4.7,53,6.1,4.2,2.2,0.8,2.5,0.9,25,P
492,13281,M,49,3.5,59,6,4,2.1,1.4,1.9,0.9,24,P
496,56826,F,49,3.3,44,6,5.6,1.9,0.75,1.35,0.8,21,P
498,488496,M,50,5,74,6.2,2,0.8,0.6,1,0.4,24,P
684,87658,F,49,3.3,44,6,5.6,1.9,0.75,1.35,0.8,23,P
700,87666,M,42,5.4,53,5.8,5.9,3.7,1.3,3.1,1.7,23,P
716,87674,M,50,4.3,59,6.1,4,3,1,1.8,1.3,24,P
12,23975,M,31,3,60,12.3,4.1,2.2,0.7,2.4,15.4,37.2,Y
18,23977,M,30,7.1,81,6.7,4.1,1.1,1.2,2.4,8.1,27.4,Y
24,23979,M,45,4.1,63,10.2,4.8,1.3,0.9,3.3,9.5,34.3,Y
675,33656789,M,45,4.1,63,10.2,4.8,1.3,0.9,3.3,9.5,34.3,Y
39,23984,M,45,5.3,77,11.2,3.9,1.5,1.3,2,10.4,29.5,Y
474,67036,M,31,3.4,55,6.5,4.9,1.6,1,3.2,0.7,24,Y
648,745326,M,45,5.3,77,11.2,3.9,1.5,1.3,2,10.4,29.5,Y
48,23987,M,30,5,80,6.9,4.5,1.8,1.2,2.6,12.7,34.6,Y
656,768585,M,30,5,80,6.9,4.5,1.8,1.2,2.6,12.7,34.6,Y
57,23990,M,35,4.8,64,7.7,3.7,1,1.2,2,7.2,27.3,Y
658,97643,M,35,4.8,64,7.7,3.7,1,1.2,2,7.2,27.3,Y
69,23994,M,45,4.8,82,7.2,4.7,1.8,0.8,3.1,12.7,31.2,Y
662,12543,M,45,4.8,82,7.2,4.7,1.8,0.8,3.1,12.7,31.2,Y"""

            df = pd.read_csv(io.StringIO(csv_content))
            df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
            X = df[self.features]
            y = df['CLASS']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.sidebar.success(f"✅ Diabetes model trained!")
            st.sidebar.info(f"Diabetes Model Accuracy: {accuracy:.2f}")
        except Exception as e:
            st.sidebar.error(f"Diabetes model error: {str(e)}")
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_minimal = np.random.rand(10, len(self.features))
            y_minimal = np.random.randint(0, 3, 10)
            self.model.fit(X_minimal, y_minimal)

    def predict_diabetes_risk(self, patient_data):
        try:
            input_features = np.array([[
                patient_data['AGE'], patient_data['Urea'], patient_data['Cr'],
                patient_data['HbA1c'], patient_data['Chol'], patient_data['TG'],
                patient_data['HDL'], patient_data['LDL'], patient_data['VLDL'], patient_data['BMI']
            ]])
            prediction = self.model.predict(input_features)[0]
            probabilities = self.model.predict_proba(input_features)[0]
            risk_levels = {0: "Low", 1: "Medium", 2: "High"}
            conditions = {0: "No Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}
            return {
                'risk_category': risk_levels[prediction],
                'condition': conditions[prediction],
                'probability_no_diabetes': float(probabilities[0]),
                'probability_pre_diabetes': float(probabilities[1]),
                'probability_diabetes': float(probabilities[2]),
                'recommendations': self._get_recommendations(prediction, patient_data),
                'key_factors': self._identify_key_factors(patient_data)
            }
        except Exception as e:
            return {'error': str(e)}

    def _get_recommendations(self, prediction, patient_data):
        if prediction == 0:
            return ["Maintain healthy lifestyle with balanced diet", "Regular physical activity (30 mins daily)", "Annual health checkups recommended"]
        elif prediction == 1:
            return ["Lifestyle modification urgently needed", "Monitor blood sugar levels regularly", "Consult endocrinologist for prevention plan", "Weight management and diet control"]
        else:
            return ["Immediate medical consultation required", "Regular glucose monitoring essential", "Strict diet control and medication adherence", "Regular follow-ups with diabetes specialist"]

    def _identify_key_factors(self, patient_data):
        factors = []
        if patient_data['HbA1c'] >= 6.5:
            factors.append("High HbA1c level")
        elif patient_data['HbA1c'] >= 5.7:
            factors.append("Elevated HbA1c (pre-diabetic range)")
        if patient_data['BMI'] >= 30:
            factors.append("Obesity (High BMI)")
        elif patient_data['BMI'] >= 25:
            factors.append("Overweight")
        if patient_data['AGE'] > 45:
            factors.append("Age-related risk")
        if patient_data['TG'] > 2.0:
            factors.append("High Triglycerides")
        return factors if factors else ["All parameters within normal range"]


# ==================== HYPERTENSION AI ENGINE ====================
class HypertensionAI:
    """
    Predicts hypertension stage using clinically relevant parameters.
    Classes:
        0 = Normal         (SBP < 120 and DBP < 80)
        1 = Elevated       (SBP 120-129 and DBP < 80)
        2 = Stage 1 HTN    (SBP 130-139 or DBP 80-89)
        3 = Stage 2 HTN    (SBP >= 140 or DBP >= 90)
    """
    def __init__(self):
        self.model = None
        self.features = ['AGE', 'BMI', 'SBP', 'DBP', 'Chol', 'TG', 'HDL', 'LDL',
                         'Glucose', 'Smoking', 'PhysicalActivity', 'FamilyHistory']
        self._generate_and_train_model()

    def _generate_and_train_model(self):
        """
        Generate a realistic synthetic hypertension dataset and train a classifier.
        In production, replace this with a real labelled clinical dataset.
        """
        try:
            np.random.seed(42)
            n = 600

            age = np.random.randint(18, 80, n)
            bmi = np.round(np.random.normal(27, 5, n).clip(16, 50), 1)
            smoking = np.random.randint(0, 2, n)
            physical_activity = np.random.randint(0, 2, n)  # 1 = active
            family_history = np.random.randint(0, 2, n)
            chol = np.round(np.random.normal(4.8, 1.0, n).clip(2, 9), 1)
            tg = np.round(np.random.normal(1.5, 0.6, n).clip(0.3, 5), 1)
            hdl = np.round(np.random.normal(1.3, 0.3, n).clip(0.5, 3), 1)
            ldl = np.round(np.random.normal(2.8, 0.8, n).clip(0.8, 6), 1)
            glucose = np.round(np.random.normal(5.5, 1.2, n).clip(3, 14), 1)

            # SBP influenced by age, BMI, smoking, family history
            sbp_base = (
                110
                + age * 0.5
                + (bmi - 25) * 0.8
                + smoking * 6
                + family_history * 5
                - physical_activity * 4
                + np.random.normal(0, 8, n)
            )
            sbp = np.round(sbp_base.clip(90, 200)).astype(int)

            dbp_base = (
                70
                + age * 0.2
                + (bmi - 25) * 0.4
                + smoking * 3
                + family_history * 3
                - physical_activity * 2
                + np.random.normal(0, 5, n)
            )
            dbp = np.round(dbp_base.clip(50, 130)).astype(int)

            # Labels based on ACC/AHA guidelines
            labels = []
            for s, d in zip(sbp, dbp):
                if s < 120 and d < 80:
                    labels.append(0)  # Normal
                elif s < 130 and d < 80:
                    labels.append(1)  # Elevated
                elif s < 140 or d < 90:
                    labels.append(2)  # Stage 1
                else:
                    labels.append(3)  # Stage 2

            X = np.column_stack([age, bmi, sbp, dbp, chol, tg, hdl, ldl, glucose, smoking, physical_activity, family_history])
            y = np.array(labels)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            self.model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=12)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.sidebar.success(f"✅ Hypertension model trained!")
            st.sidebar.info(f"HTN Model Accuracy: {accuracy:.2f}")

        except Exception as e:
            st.sidebar.error(f"HTN model error: {str(e)}")
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_minimal = np.random.rand(20, len(self.features))
            y_minimal = np.random.randint(0, 4, 20)
            self.model.fit(X_minimal, y_minimal)

    def predict_hypertension_risk(self, patient_data):
        try:
            input_features = np.array([[
                patient_data['AGE'],
                patient_data['BMI'],
                patient_data['SBP'],
                patient_data['DBP'],
                patient_data['Chol'],
                patient_data['TG'],
                patient_data['HDL'],
                patient_data['LDL'],
                patient_data['Glucose'],
                patient_data['Smoking'],
                patient_data['PhysicalActivity'],
                patient_data['FamilyHistory']
            ]])

            prediction = self.model.predict(input_features)[0]
            probabilities = self.model.predict_proba(input_features)[0]

            # Pad probabilities to always have 4 classes
            full_probs = np.zeros(4)
            for i, cls in enumerate(self.model.classes_):
                full_probs[int(cls)] = probabilities[i]

            stages = {0: "Normal", 1: "Elevated", 2: "Stage 1 Hypertension", 3: "Stage 2 Hypertension"}
            risk_levels = {0: "Low", 1: "Low-Medium", 2: "Medium", 3: "High"}
            colors = {0: "green", 1: "orange", 2: "red", 3: "darkred"}

            return {
                'stage': stages[prediction],
                'risk_category': risk_levels[prediction],
                'risk_color': colors[prediction],
                'prob_normal': float(full_probs[0]),
                'prob_elevated': float(full_probs[1]),
                'prob_stage1': float(full_probs[2]),
                'prob_stage2': float(full_probs[3]),
                'recommendations': self._get_recommendations(prediction, patient_data),
                'key_factors': self._identify_key_factors(patient_data)
            }
        except Exception as e:
            return {'error': str(e)}

    def _get_recommendations(self, prediction, patient_data):
        base = []
        if prediction == 0:
            base = [
                "Blood pressure is within healthy range — keep up current lifestyle",
                "Annual BP monitoring recommended",
                "Maintain physical activity and low-sodium diet"
            ]
        elif prediction == 1:
            base = [
                "Begin lifestyle modifications to prevent progression",
                "Reduce sodium intake to < 2,300 mg/day",
                "Increase aerobic activity (150 min/week)",
                "Monitor blood pressure every 3-6 months"
            ]
        elif prediction == 2:
            base = [
                "Consult a physician — medication may be needed",
                "DASH diet strongly recommended",
                "Weight reduction if BMI > 25",
                "Home blood pressure monitoring daily",
                "Limit alcohol and caffeine"
            ]
        else:
            base = [
                "Immediate medical consultation required",
                "Antihypertensive medication likely required",
                "Strict sodium restriction (< 1,500 mg/day)",
                "Assess for target organ damage (kidney, heart, eyes)",
                "Follow up in 1 week or sooner"
            ]
        if patient_data.get('Smoking', 0) == 1:
            base.append("Smoking cessation strongly advised — significant cardiovascular risk")
        return base

    def _identify_key_factors(self, patient_data):
        factors = []
        if patient_data['SBP'] >= 140:
            factors.append(f"High systolic BP ({patient_data['SBP']} mmHg)")
        elif patient_data['SBP'] >= 130:
            factors.append(f"Elevated systolic BP ({patient_data['SBP']} mmHg)")
        if patient_data['DBP'] >= 90:
            factors.append(f"High diastolic BP ({patient_data['DBP']} mmHg)")
        elif patient_data['DBP'] >= 80:
            factors.append(f"Elevated diastolic BP ({patient_data['DBP']} mmHg)")
        if patient_data['BMI'] >= 30:
            factors.append("Obesity (BMI ≥ 30)")
        elif patient_data['BMI'] >= 25:
            factors.append("Overweight (BMI 25-29.9)")
        if patient_data.get('Smoking', 0) == 1:
            factors.append("Active smoker")
        if patient_data.get('FamilyHistory', 0) == 1:
            factors.append("Family history of hypertension")
        if patient_data.get('PhysicalActivity', 1) == 0:
            factors.append("Sedentary lifestyle")
        if patient_data['AGE'] > 55:
            factors.append("Age-related cardiovascular risk")
        if patient_data['Chol'] > 6.2:
            factors.append("High total cholesterol")
        return factors if factors else ["No major risk factors identified"]


# ==================== DATA STORAGE ====================
class HospitalData:
    def __init__(self):
        self.patients = []
        self._create_sample_data()

    def _create_sample_data(self):
        sample_patients = [
            {
                'patient_id': 'PAT0001', 'name': 'John Smith', 'age': 45, 'gender': 'Male',
                'Urea': 4.7, 'Cr': 46, 'HbA1c': 4.9, 'Chol': 4.2, 'TG': 0.9,
                'HDL': 2.4, 'LDL': 1.4, 'VLDL': 0.5, 'BMI': 24,
                'SBP': 118, 'DBP': 76, 'Glucose': 5.1, 'Smoking': 0,
                'PhysicalActivity': 1, 'FamilyHistory': 0,
                'admission_reason': 'Routine Diabetes Screening',
                'admission_date': '2024-01-10', 'status': 'Admitted'
            },
            {
                'patient_id': 'PAT0002', 'name': 'Jane Doe', 'age': 68, 'gender': 'Female',
                'Urea': 7.5, 'Cr': 82, 'HbA1c': 7.2, 'Chol': 5.6, 'TG': 2.1,
                'HDL': 1.1, 'LDL': 3.2, 'VLDL': 1.0, 'BMI': 31,
                'SBP': 148, 'DBP': 94, 'Glucose': 8.2, 'Smoking': 0,
                'PhysicalActivity': 0, 'FamilyHistory': 1,
                'admission_reason': 'Diabetes & Hypertension Management',
                'admission_date': '2024-01-12', 'status': 'Admitted'
            },
            {
                'patient_id': 'PAT0003', 'name': 'Robert Chang', 'age': 52, 'gender': 'Male',
                'Urea': 5.1, 'Cr': 70, 'HbA1c': 5.4, 'Chol': 6.1, 'TG': 1.8,
                'HDL': 1.0, 'LDL': 3.8, 'VLDL': 0.8, 'BMI': 27,
                'SBP': 134, 'DBP': 86, 'Glucose': 5.8, 'Smoking': 1,
                'PhysicalActivity': 0, 'FamilyHistory': 1,
                'admission_reason': 'Hypertension Monitoring',
                'admission_date': '2024-01-15', 'status': 'Admitted'
            }
        ]

        for patient in sample_patients:
            diabetes_input = {k: patient[k] for k in ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']}
            diabetes_input['AGE'] = patient['age']
            patient['diabetes_assessment'] = diabetes_ai.predict_diabetes_risk(diabetes_input)

            htn_input = {
                'AGE': patient['age'], 'BMI': patient['BMI'],
                'SBP': patient['SBP'], 'DBP': patient['DBP'],
                'Chol': patient['Chol'], 'TG': patient['TG'],
                'HDL': patient['HDL'], 'LDL': patient['LDL'],
                'Glucose': patient['Glucose'], 'Smoking': patient['Smoking'],
                'PhysicalActivity': patient['PhysicalActivity'],
                'FamilyHistory': patient['FamilyHistory']
            }
            patient['htn_assessment'] = hypertension_ai.predict_hypertension_risk(htn_input)

        self.patients = sample_patients

    def add_patient(self, patient_data):
        patient_id = f"PAT{len(self.patients) + 1:04d}"
        patient_data['patient_id'] = patient_id
        patient_data['admission_date'] = datetime.now().strftime("%Y-%m-%d")
        patient_data['status'] = 'Admitted'

        diabetes_input = {k: patient_data[k] for k in ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']}
        diabetes_input['AGE'] = patient_data['age']
        patient_data['diabetes_assessment'] = diabetes_ai.predict_diabetes_risk(diabetes_input)

        htn_input = {
            'AGE': patient_data['age'], 'BMI': patient_data['BMI'],
            'SBP': patient_data['SBP'], 'DBP': patient_data['DBP'],
            'Chol': patient_data['Chol'], 'TG': patient_data['TG'],
            'HDL': patient_data['HDL'], 'LDL': patient_data['LDL'],
            'Glucose': patient_data['Glucose'], 'Smoking': patient_data['Smoking'],
            'PhysicalActivity': patient_data['PhysicalActivity'],
            'FamilyHistory': patient_data['FamilyHistory']
        }
        patient_data['htn_assessment'] = hypertension_ai.predict_hypertension_risk(htn_input)

        self.patients.append(patient_data)
        return patient_id

    def get_patient_stats(self):
        total = len(self.patients)
        admitted = len([p for p in self.patients if p['status'] == 'Admitted'])
        diabetic = len([p for p in self.patients if p.get('diabetes_assessment', {}).get('condition') == 'Diabetes'])
        hypertensive = len([p for p in self.patients if p.get('htn_assessment', {}).get('stage', '') in ['Stage 1 Hypertension', 'Stage 2 Hypertension']])
        comorbid = len([p for p in self.patients if
                        p.get('diabetes_assessment', {}).get('condition') == 'Diabetes' and
                        p.get('htn_assessment', {}).get('stage', '') in ['Stage 1 Hypertension', 'Stage 2 Hypertension']])
        return {
            'total_patients': total,
            'admitted': admitted,
            'diabetic_patients': diabetic,
            'hypertensive_patients': hypertensive,
            'comorbid_patients': comorbid
        }


# ---- Initialize AI models BEFORE HospitalData ----
diabetes_ai = DiabetesAI()
hypertension_ai = HypertensionAI()
hospital_data = HospitalData()


# ==================== STREAMLIT PAGES ====================
def login_section():
    st.sidebar.title("🔐 Hospital Login")
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username in ADMIN_USERS:
                if security.verify_password(password, ADMIN_USERS[username]["password_hash"]):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = ADMIN_USERS[username]["role"]
                    st.session_state.name = ADMIN_USERS[username]["name"]
                    st.sidebar.success(f"Welcome {ADMIN_USERS[username]['name']}!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid password!")
            else:
                st.sidebar.error("User not found!")
        st.sidebar.markdown("---")
        st.sidebar.info("**Demo Credentials:**")
        st.sidebar.info("👨‍💼 admin / Admin123!")
        st.sidebar.info("👨‍⚕️ doctor / Admin123!")
    else:
        st.sidebar.success(f"Logged in as: {st.session_state.name}")
        st.sidebar.info(f"Role: {st.session_state.role}")
        if st.sidebar.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def main_dashboard():
    st.title("🏥 Hospital Management System")
    st.subheader(f"Welcome, {st.session_state.name}!")

    stats = hospital_data.get_patient_stats()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Patients", stats['total_patients'])
    col2.metric("Currently Admitted", stats['admitted'])
    col3.metric("Diabetic Patients", stats['diabetic_patients'])
    col4.metric("Hypertensive Patients", stats['hypertensive_patients'])
    col5.metric("Comorbid (Both)", stats['comorbid_patients'])

    st.subheader("👥 Recent Patient Admissions")
    if hospital_data.patients:
        display_data = []
        for p in hospital_data.patients:
            display_data.append({
                'ID': p['patient_id'],
                'Name': p['name'],
                'Age': p['age'],
                'Status': p['status'],
                'Diabetes Status': p.get('diabetes_assessment', {}).get('condition', 'N/A'),
                'Diabetes Risk': p.get('diabetes_assessment', {}).get('risk_category', 'N/A'),
                'HTN Stage': p.get('htn_assessment', {}).get('stage', 'N/A'),
                'HTN Risk': p.get('htn_assessment', {}).get('risk_category', 'N/A'),
                'Admission Date': p['admission_date']
            })
        st.dataframe(display_data, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Diabetes Status Distribution")
        status_counts = {}
        for p in hospital_data.patients:
            s = p.get('diabetes_assessment', {}).get('condition', 'Unknown')
            status_counts[s] = status_counts.get(s, 0) + 1
        if status_counts:
            fig = px.pie(
                pd.DataFrame({'Status': list(status_counts.keys()), 'Count': list(status_counts.values())}),
                values='Count', names='Status',
                title="Diabetes Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Hypertension Stage Distribution")
        htn_counts = {}
        for p in hospital_data.patients:
            s = p.get('htn_assessment', {}).get('stage', 'Unknown')
            htn_counts[s] = htn_counts.get(s, 0) + 1
        if htn_counts:
            fig = px.pie(
                pd.DataFrame({'Stage': list(htn_counts.keys()), 'Count': list(htn_counts.values())}),
                values='Count', names='Stage',
                title="Hypertension Stage Distribution",
                color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
            )
            st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 HbA1c Level Distribution")
        hba1c_levels = [p['HbA1c'] for p in hospital_data.patients]
        fig = px.histogram(x=hba1c_levels, nbins=15,
                           title="Patient HbA1c Distribution",
                           labels={'x': 'HbA1c Level', 'y': 'Count'},
                           color_discrete_sequence=['#FF6B6B'])
        fig.add_vline(x=5.7, line_dash="dash", line_color="orange", annotation_text="Pre-diabetes")
        fig.add_vline(x=6.5, line_dash="dash", line_color="red", annotation_text="Diabetes")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Systolic BP Distribution")
        sbp_values = [p.get('SBP', 0) for p in hospital_data.patients if p.get('SBP')]
        if sbp_values:
            fig = px.histogram(x=sbp_values, nbins=15,
                               title="Patient Systolic BP Distribution",
                               labels={'x': 'SBP (mmHg)', 'y': 'Count'},
                               color_discrete_sequence=['#3498DB'])
            fig.add_vline(x=120, line_dash="dash", line_color="orange", annotation_text="Elevated")
            fig.add_vline(x=130, line_dash="dash", line_color="red", annotation_text="Stage 1")
            fig.add_vline(x=140, line_dash="dash", line_color="darkred", annotation_text="Stage 2")
            st.plotly_chart(fig, use_container_width=True)


def patient_management():
    st.title("👥 Patient Management")
    tab1, tab2 = st.tabs(["Add New Patient", "View All Patients"])

    with tab1:
        st.subheader("➕ Register New Patient")
        with st.form("patient_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🧍 Demographics**")
                name = st.text_input("Full Name *")
                age = st.number_input("Age *", min_value=0, max_value=120, value=45)
                gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
                admission_reason = st.selectbox("Admission Reason *",
                    ["Routine Diabetes Screening", "Diabetes Management",
                     "Pre-diabetes Consultation", "Hypertension Monitoring",
                     "Diabetes & Hypertension Management", "General Checkup", "Other"])
                bmi = st.slider("BMI *", 15.0, 50.0, 25.0, 0.1)

                st.markdown("**🩸 Diabetes Parameters**")
                hba1c = st.slider("HbA1c Level *", 3.0, 15.0, 5.5, 0.1)
                urea = st.slider("Urea Level *", 1.0, 20.0, 4.5, 0.1)
                cr = st.slider("Creatinine (Cr) *", 10, 200, 60)
                chol = st.slider("Cholesterol (Chol) *", 2.0, 10.0, 4.5, 0.1)
                tg = st.slider("Triglycerides (TG) *", 0.5, 5.0, 1.5, 0.1)
                hdl = st.slider("HDL Cholesterol *", 0.3, 3.0, 1.2, 0.1)
                ldl = st.slider("LDL Cholesterol *", 1.0, 6.0, 2.5, 0.1)
                vldl = st.slider("VLDL Cholesterol *", 0.1, 3.0, 0.8, 0.1)

            with col2:
                st.markdown("**❤️ Hypertension Parameters**")
                sbp = st.slider("Systolic BP (SBP) *", 80, 220, 120)
                dbp = st.slider("Diastolic BP (DBP) *", 50, 130, 80)
                glucose = st.slider("Fasting Glucose (mmol/L) *", 3.0, 15.0, 5.5, 0.1)

                st.markdown("**🚨 Lifestyle & Risk Factors**")
                smoking = st.selectbox("Smoking Status *", ["Non-Smoker", "Smoker"])
                physical_activity = st.selectbox("Physical Activity *", ["Active", "Sedentary"])
                family_history = st.selectbox("Family History of Hypertension *", ["No", "Yes"])

                # BP reference guide
                st.markdown("---")
                st.markdown("**📖 BP Reference (ACC/AHA)**")
                st.markdown("""
                | Category | SBP | DBP |
                |---|---|---|
                | Normal | < 120 | < 80 |
                | Elevated | 120-129 | < 80 |
                | Stage 1 | 130-139 | 80-89 |
                | Stage 2 | ≥ 140 | ≥ 90 |
                """)

            if st.form_submit_button("🚀 Register Patient & Run AI Assessment"):
                if name and age:
                    patient_data = {
                        'name': name, 'age': age, 'gender': gender,
                        'Urea': urea, 'Cr': cr, 'HbA1c': hba1c,
                        'Chol': chol, 'TG': tg, 'HDL': hdl, 'LDL': ldl, 'VLDL': vldl,
                        'BMI': bmi,
                        'SBP': sbp, 'DBP': dbp, 'Glucose': glucose,
                        'Smoking': 1 if smoking == "Smoker" else 0,
                        'PhysicalActivity': 1 if physical_activity == "Active" else 0,
                        'FamilyHistory': 1 if family_history == "Yes" else 0,
                        'admission_reason': admission_reason
                    }
                    patient_id = hospital_data.add_patient(patient_data)
                    st.success(f"✅ Patient **{name}** registered! ID: **{patient_id}**")
                    st.balloons()

                    # ---- Diabetes Assessment ----
                    st.subheader("🤖 AI Diabetes Assessment")
                    da = patient_data['diabetes_assessment']
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Diabetes Status", da['condition'])
                    c1.metric("Risk Category", da['risk_category'])
                    c2.metric("No Diabetes Prob.", f"{da['probability_no_diabetes']*100:.1f}%")
                    c2.metric("Pre-Diabetes Prob.", f"{da['probability_pre_diabetes']*100:.1f}%")
                    c3.metric("Diabetes Prob.", f"{da['probability_diabetes']*100:.1f}%")
                    if da['key_factors']:
                        st.warning(f"🚨 Diabetes Risk Factors: {', '.join(da['key_factors'])}")
                    st.info("💡 Diabetes Recommendations: " + " | ".join(da['recommendations']))

                    st.markdown("---")

                    # ---- Hypertension Assessment ----
                    st.subheader("❤️ AI Hypertension Assessment")
                    ha = patient_data['htn_assessment']
                    c1, c2, c3 = st.columns(3)
                    c1.metric("HTN Stage", ha['stage'])
                    c1.metric("Risk Category", ha['risk_category'])
                    c2.metric("Normal Prob.", f"{ha['prob_normal']*100:.1f}%")
                    c2.metric("Elevated Prob.", f"{ha['prob_elevated']*100:.1f}%")
                    c3.metric("Stage 1 HTN Prob.", f"{ha['prob_stage1']*100:.1f}%")
                    c3.metric("Stage 2 HTN Prob.", f"{ha['prob_stage2']*100:.1f}%")
                    if ha['key_factors']:
                        st.warning(f"🚨 HTN Risk Factors: {', '.join(ha['key_factors'])}")
                    st.info("💡 HTN Recommendations: " + " | ".join(ha['recommendations']))
                else:
                    st.error("Please fill in all required fields.")

    with tab2:
        st.subheader("📋 All Patients")
        if hospital_data.patients:
            c1, c2, c3 = st.columns(3)
            with c1:
                search_name = st.text_input("Search by Name")
            with c2:
                diabetes_filter = st.selectbox("Filter by Diabetes Status", ["All", "No Diabetes", "Pre-Diabetes", "Diabetes"])
            with c3:
                htn_filter = st.selectbox("Filter by HTN Stage", ["All", "Normal", "Elevated", "Stage 1 Hypertension", "Stage 2 Hypertension"])

            filtered = hospital_data.patients
            if search_name:
                filtered = [p for p in filtered if search_name.lower() in p['name'].lower()]
            if diabetes_filter != "All":
                filtered = [p for p in filtered if p.get('diabetes_assessment', {}).get('condition') == diabetes_filter]
            if htn_filter != "All":
                filtered = [p for p in filtered if p.get('htn_assessment', {}).get('stage') == htn_filter]

            if filtered:
                display_data = [{
                    'ID': p['patient_id'], 'Name': p['name'], 'Age': p['age'], 'Gender': p['gender'],
                    'Diabetes Status': p.get('diabetes_assessment', {}).get('condition', 'N/A'),
                    'Diabetes Risk': p.get('diabetes_assessment', {}).get('risk_category', 'N/A'),
                    'HbA1c': p.get('HbA1c', 'N/A'), 'BMI': p.get('BMI', 'N/A'),
                    'HTN Stage': p.get('htn_assessment', {}).get('stage', 'N/A'),
                    'HTN Risk': p.get('htn_assessment', {}).get('risk_category', 'N/A'),
                    'SBP': p.get('SBP', 'N/A'), 'DBP': p.get('DBP', 'N/A'),
                    'Admission': p.get('admission_date', 'N/A')
                } for p in filtered]
                st.dataframe(display_data, use_container_width=True)

                if st.button("📤 Export to CSV"):
                    csv = pd.DataFrame(display_data).to_csv(index=False)
                    st.download_button("Download CSV", csv,
                                       f"patients_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
            else:
                st.warning("No patients match the filters.")
        else:
            st.info("No patients registered yet.")


def ai_predictions():
    st.title("🤖 AI Predictions")

    tab1, tab2 = st.tabs(["🩺 Diabetes Risk Predictor", "❤️ Hypertension Risk Predictor"])

    # ---- DIABETES TAB ----
    with tab1:
        st.subheader("🎯 Interactive Diabetes Risk Predictor")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 0, 100, 45, key="d_age")
            hba1c = st.slider("HbA1c Level", 3.0, 15.0, 5.5, 0.1, key="d_hba1c")
            urea = st.slider("Urea Level", 1.0, 20.0, 4.5, 0.1, key="d_urea")
            cr = st.slider("Creatinine (Cr)", 10, 200, 60, key="d_cr")
            bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1, key="d_bmi")
        with col2:
            chol = st.slider("Cholesterol", 2.0, 10.0, 4.5, 0.1, key="d_chol")
            tg = st.slider("Triglycerides (TG)", 0.5, 5.0, 1.5, 0.1, key="d_tg")
            hdl = st.slider("HDL Cholesterol", 0.3, 3.0, 1.2, 0.1, key="d_hdl")
            ldl = st.slider("LDL Cholesterol", 1.0, 6.0, 2.5, 0.1, key="d_ldl")
            vldl = st.slider("VLDL Cholesterol", 0.1, 3.0, 0.8, 0.1, key="d_vldl")

        if st.button("🔍 Analyse Diabetes Risk"):
            result = diabetes_ai.predict_diabetes_risk({
                'AGE': age, 'Urea': urea, 'Cr': cr, 'HbA1c': hba1c,
                'Chol': chol, 'TG': tg, 'HDL': hdl, 'LDL': ldl, 'VLDL': vldl, 'BMI': bmi
            })

            c1, c2, c3 = st.columns(3)
            c1.metric("Diabetes Status", result['condition'])
            c2.metric("Risk Category", result['risk_category'])
            c3.metric("Diabetes Probability", f"{result['probability_diabetes']*100:.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number", value=result['probability_diabetes'] * 100,
                title={'text': "Diabetes Risk (%)"},
                domain={'x': [0, 0.45], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"},
                       'steps': [{'range': [0, 33], 'color': "lightgreen"},
                                  {'range': [33, 66], 'color': "yellow"},
                                  {'range': [66, 100], 'color': "red"}]}
            ))
            fig.add_trace(go.Indicator(
                mode="gauge+number", value=hba1c,
                title={'text': "HbA1c Level"},
                domain={'x': [0.55, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [3, 15]}, 'bar': {'color': "darkblue"},
                       'steps': [{'range': [3, 5.7], 'color': "lightgreen"},
                                  {'range': [5.7, 6.5], 'color': "yellow"},
                                  {'range': [6.5, 15], 'color': "red"}],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 6.5}}
            ))
            fig.update_layout(height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if result['key_factors']:
                    st.warning("🚨 Key Factors: " + ", ".join(result['key_factors']))
            with c2:
                st.info("💡 Recommendations:\n" + "\n".join(f"- {r}" for r in result['recommendations']))

    # ---- HYPERTENSION TAB ----
    with tab2:
        st.subheader("🎯 Interactive Hypertension Risk Predictor")
        st.info("Enter patient parameters to get AI-powered hypertension stage assessment based on ACC/AHA guidelines.")

        col1, col2 = st.columns(2)
        with col1:
            h_age = st.slider("Age", 18, 100, 45, key="h_age")
            h_bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1, key="h_bmi")
            h_sbp = st.slider("Systolic BP (SBP) mmHg", 80, 220, 120, key="h_sbp")
            h_dbp = st.slider("Diastolic BP (DBP) mmHg", 50, 130, 80, key="h_dbp")
            h_glucose = st.slider("Fasting Glucose (mmol/L)", 3.0, 15.0, 5.5, 0.1, key="h_glucose")
            h_chol = st.slider("Total Cholesterol (mmol/L)", 2.0, 10.0, 4.5, 0.1, key="h_chol")

        with col2:
            h_tg = st.slider("Triglycerides (TG)", 0.5, 5.0, 1.5, 0.1, key="h_tg")
            h_hdl = st.slider("HDL Cholesterol", 0.3, 3.0, 1.2, 0.1, key="h_hdl")
            h_ldl = st.slider("LDL Cholesterol", 1.0, 6.0, 2.5, 0.1, key="h_ldl")
            h_smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"], key="h_smoking")
            h_activity = st.selectbox("Physical Activity", ["Active", "Sedentary"], key="h_activity")
            h_family = st.selectbox("Family History of Hypertension", ["No", "Yes"], key="h_family")

            # Live BP category indicator
            st.markdown("**Current BP Reading:**")
            if h_sbp < 120 and h_dbp < 80:
                st.success(f"🟢 Normal — {h_sbp}/{h_dbp} mmHg")
            elif h_sbp < 130 and h_dbp < 80:
                st.warning(f"🟡 Elevated — {h_sbp}/{h_dbp} mmHg")
            elif h_sbp < 140 or h_dbp < 90:
                st.error(f"🔴 Stage 1 HTN — {h_sbp}/{h_dbp} mmHg")
            else:
                st.error(f"🔴🔴 Stage 2 HTN — {h_sbp}/{h_dbp} mmHg")

        if st.button("🔍 Analyse Hypertension Risk"):
            result = hypertension_ai.predict_hypertension_risk({
                'AGE': h_age, 'BMI': h_bmi, 'SBP': h_sbp, 'DBP': h_dbp,
                'Chol': h_chol, 'TG': h_tg, 'HDL': h_hdl, 'LDL': h_ldl,
                'Glucose': h_glucose,
                'Smoking': 1 if h_smoking == "Smoker" else 0,
                'PhysicalActivity': 1 if h_activity == "Active" else 0,
                'FamilyHistory': 1 if h_family == "Yes" else 0
            })

            c1, c2, c3 = st.columns(3)
            c1.metric("HTN Stage", result['stage'])
            c2.metric("Risk Category", result['risk_category'])
            c3.metric("Stage 2 HTN Probability", f"{result['prob_stage2']*100:.1f}%")

            # Dual gauges: SBP and model risk score
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number", value=h_sbp,
                title={'text': "Systolic BP (mmHg)"},
                domain={'x': [0, 0.45], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [80, 200]},
                    'bar': {'color': "#3498DB"},
                    'steps': [
                        {'range': [80, 120], 'color': "lightgreen"},
                        {'range': [120, 130], 'color': "#f1c40f"},
                        {'range': [130, 140], 'color': "orange"},
                        {'range': [140, 200], 'color': "red"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 140}
                }
            ))
            fig.add_trace(go.Indicator(
                mode="gauge+number", value=h_dbp,
                title={'text': "Diastolic BP (mmHg)"},
                domain={'x': [0.55, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [50, 130]},
                    'bar': {'color': "#8e44ad"},
                    'steps': [
                        {'range': [50, 80], 'color': "lightgreen"},
                        {'range': [80, 90], 'color': "orange"},
                        {'range': [90, 130], 'color': "red"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig.update_layout(height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Probability bar chart
            st.subheader("📊 Probability Breakdown")
            prob_df = pd.DataFrame({
                'Stage': ['Normal', 'Elevated', 'Stage 1 HTN', 'Stage 2 HTN'],
                'Probability (%)': [
                    result['prob_normal'] * 100,
                    result['prob_elevated'] * 100,
                    result['prob_stage1'] * 100,
                    result['prob_stage2'] * 100
                ]
            })
            fig2 = px.bar(prob_df, x='Stage', y='Probability (%)',
                          color='Stage',
                          color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad'],
                          title="HTN Stage Probability Distribution")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if result['key_factors']:
                    st.warning("🚨 Key Risk Factors:\n" + "\n".join(f"- {f}" for f in result['key_factors']))
            with c2:
                st.info("💡 Recommendations:\n" + "\n".join(f"- {r}" for r in result['recommendations']))


# ==================== MAIN APP ====================
def main():
    login_section()

    if not st.session_state.get('authenticated', False):
        st.title("🏥 Hospital Management System")
        st.markdown("""
        ## Welcome to the Hospital Management System
        
        A specialized healthcare platform with AI-powered **Diabetes** and **Hypertension** risk assessment.
        
        ### 🔐 Please login to continue
        Use the sidebar to login with your credentials.
        
        ### Demo Credentials:
        - **Admin:** `admin` / `Admin123!`
        - **Doctor:** `doctor` / `Admin123!`
        
        ### 🚀 Features:
        - **Diabetes-focused Patient Management** — Complete patient registration with diabetes parameters
        - **Hypertension Management** — ACC/AHA guideline-based staging and risk stratification
        - **AI Risk Assessment** — Predictive analytics for both conditions
        - **Comorbidity Tracking** — Identify patients with both diabetes and hypertension
        - **Real-time Analytics** — Interactive dashboards and reports
        - **Secure Access** — Role-based authentication system

        ### 📊 Clinical Parameters Tracked:
        **Diabetes:** HbA1c, Lipid Profile (Cholesterol, TG, HDL, LDL, VLDL), Kidney Function (Urea, Creatinine), BMI
        
        **Hypertension:** Systolic & Diastolic BP, Glucose, Lipid Profile, BMI, Smoking, Physical Activity, Family History
        """)
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Patient Management", "AI Predictions"])

    if page == "Dashboard":
        main_dashboard()
    elif page == "Patient Management":
        patient_management()
    elif page == "AI Predictions":
        ai_predictions()

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Patients:** {len(hospital_data.patients)}")
    st.sidebar.info(f"**Last Update:** {datetime.now().strftime('%H:%M')}")


if __name__ == "__main__":
    main()
