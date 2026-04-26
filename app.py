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
from sklearn.metrics import accuracy_score, classification_report
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PyPDF2 import PdfReader, PdfWriter
import io

# Page configuration
st.set_page_config(
    page_title="Diabetes Hospital Management System",
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

# Initialize security and users
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

def generate_pdf(patient, password):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Medical Report", styles['Title']))
    content.append(Spacer(1,10))

    # Safe access
    name = patient.get('name', 'N/A')
    age = patient.get('age', 'N/A')

    diabetes = patient.get('diabetes_assessment', {}).get('condition', 'N/A')
    bp = patient.get('bp_assessment', {}).get('category', 'N/A')

    content.append(Paragraph(f"Name: {name}", styles['Normal']))
    content.append(Paragraph(f"Age: {age}", styles['Normal']))
    content.append(Paragraph(f"Diabetes: {diabetes}", styles['Normal']))
    content.append(Paragraph(f"BP: {bp}", styles['Normal']))

    doc.build(content)
    buffer.seek(0)

    # Encrypt PDF
    reader = PdfReader(buffer)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    writer.encrypt(password)

    secured = io.BytesIO()
    writer.write(secured)
    secured.seek(0)

    return secured

# ==================== DIABETES AI ENGINE ====================
class DiabetesAI:
    def __init__(self):
        self.model = None
        self.features = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
        self._load_and_train_model()
    
    def _load_and_train_model(self):
        try:
            # Parse the actual dataset from the provided CSV content
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
            
            # Parse the CSV content
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Map CLASS to numerical values
            df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
            
            # Prepare features and target
            X = df[self.features]
            y = df['CLASS']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.sidebar.success(f"✅ Diabetes model trained successfully!")
            st.sidebar.info(f"Model Accuracy: {accuracy:.2f}")
            
        except Exception as e:
            st.sidebar.error(f"Model training error: {str(e)}")
            # Fallback to a simple model if training fails
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Create minimal training data
            X_minimal = np.random.rand(10, len(self.features))
            y_minimal = np.random.randint(0, 3, 10)
            self.model.fit(X_minimal, y_minimal)
    
    def predict_diabetes_risk(self, patient_data):
        try:
            # Prepare input features in correct order
            input_features = np.array([[
                patient_data['AGE'],
                patient_data['Urea'],
                patient_data['Cr'], 
                patient_data['HbA1c'],
                patient_data['Chol'],
                patient_data['TG'],
                patient_data['HDL'],
                patient_data['LDL'],
                patient_data['VLDL'],
                patient_data['BMI']
            ]])
            
            # Make prediction
            prediction = self.model.predict(input_features)[0]
            probabilities = self.model.predict_proba(input_features)[0]
            
            # Map to meaningful labels
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
        if prediction == 0:  # No diabetes
            return [
                "Maintain healthy lifestyle with balanced diet",
                "Regular physical activity (30 mins daily)",
                "Annual health checkups recommended"
            ]
        elif prediction == 1:  # Pre-diabetes
            return [
                "Lifestyle modification urgently needed",
                "Monitor blood sugar levels regularly", 
                "Consult endocrinologist for prevention plan",
                "Weight management and diet control"
            ]
        else:  # Diabetes
            return [
                "Immediate medical consultation required",
                "Regular glucose monitoring essential",
                "Strict diet control and medication adherence",
                "Regular follow-ups with diabetes specialist"
            ]
    
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

# Initialize AI
diabetes_ai = DiabetesAI()

def generate_pdf(patient, password):
    import io
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Medical Report", styles['Title']))
    content.append(Spacer(1,10))

    content.append(Paragraph(f"Name: {patient['name']}", styles['Normal']))
    content.append(Paragraph(f"Age: {patient['age']}", styles['Normal']))
    content.append(Paragraph(f"Diabetes: {patient['diabetes_assessment']['condition']}", styles['Normal']))
    content.append(Paragraph(f"BP: {patient['bp_assessment']['category']}", styles['Normal']))

    doc.build(content)
    buffer.seek(0)

    # 🔐 Encrypt PDF
    reader = PdfReader(buffer)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    writer.encrypt(password)

    secured = io.BytesIO()
    writer.write(secured)
    secured.seek(0)

    return secured

# ==================== DATA STORAGE ====================
class HospitalData:
    def __init__(self):
        self.patients = []
        self._create_sample_data()
    
    def _create_sample_data(self):
        # Create initial sample patients with diabetes-focused data
        sample_patients = [
            {
                'patient_id': 'PAT0001',
                'name': 'John Smith',
                'age': 45,
                'gender': 'Male',
                'Urea': 4.7,
                'Cr': 46,
                'HbA1c': 4.9,
                'Chol': 4.2,
                'TG': 0.9,
                'HDL': 2.4,
                'LDL': 1.4,
                'VLDL': 0.5,
                'BMI': 24,
                'admission_reason': 'Routine Diabetes Screening',
                'admission_date': '2024-01-10',
                'status': 'Admitted'
            },
            {
                'patient_id': 'PAT0002',
                'name': 'Jane Doe', 
                'age': 68,
                'gender': 'Female',
                'Urea': 7.5,
                'Cr': 82,
                'HbA1c': 7.2,
                'Chol': 5.6,
                'TG': 2.1,
                'HDL': 1.1,
                'LDL': 3.2,
                'VLDL': 1.0,
                'BMI': 31,
                'admission_reason': 'Diabetes Management',
                'admission_date': '2024-01-12',
                'status': 'Admitted'
            }
        ]
        
        for patient in sample_patients:
            # Prepare data for prediction
            prediction_data = {
                'AGE': patient['age'],
                'Urea': patient['Urea'],
                'Cr': patient['Cr'],
                'HbA1c': patient['HbA1c'],
                'Chol': patient['Chol'],
                'TG': patient['TG'],
                'HDL': patient['HDL'],
                'LDL': patient['LDL'],
                'VLDL': patient['VLDL'],
                'BMI': patient['BMI']
            }
            diabetes_assessment = diabetes_ai.predict_diabetes_risk(prediction_data)
            patient['diabetes_assessment'] = diabetes_assessment
        
        self.patients = sample_patients
    
    def add_patient(self, patient_data):
        patient_id = f"PAT{len(self.patients) + 1:04d}"
        patient_data['patient_id'] = patient_id
        patient_data['admission_date'] = datetime.now().strftime("%Y-%m-%d")
        patient_data['status'] = 'Admitted'
        
        # Prepare data for diabetes prediction
        prediction_data = {
            'AGE': patient_data['age'],
            'Urea': patient_data['Urea'],
            'Cr': patient_data['Cr'],
            'HbA1c': patient_data['HbA1c'],
            'Chol': patient_data['Chol'],
            'TG': patient_data['TG'],
            'HDL': patient_data['HDL'],
            'LDL': patient_data['LDL'],
            'VLDL': patient_data['VLDL'],
            'BMI': patient_data['BMI']
        }
        
        diabetes_assessment = diabetes_ai.predict_diabetes_risk(prediction_data)
        patient_data['diabetes_assessment'] = diabetes_assessment
        
        self.patients.append(patient_data)
        return patient_id
    
    def get_patient_stats(self):
        total = len(self.patients)
        admitted = len([p for p in self.patients if p['status'] == 'Admitted'])
        diabetic = len([p for p in self.patients if p['diabetes_assessment']['condition'] == 'Diabetes'])
        
        return {
            'total_patients': total,
            'admitted': admitted,
            'diabetic_patients': diabetic
        }

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
    st.title("🏥 Diabetes Management System")
    st.subheader(f"Welcome, {st.session_state.name}!")
    
    # Key Metrics
    stats = hospital_data.get_patient_stats()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patients", stats['total_patients'])
    with col2:
        st.metric("Currently Admitted", stats['admitted'])
    with col3:
        st.metric("Diabetic Patients", stats['diabetic_patients'])
    
    # Recent Patients
    st.subheader("👥 Recent Patient Admissions")
    if hospital_data.patients:
        display_data = []
        for patient in hospital_data.patients:
            display_data.append({
                'ID': patient['patient_id'],
                'Name': patient['name'],
                'Age': patient['age'],
                'Status': patient['status'],
                'Diabetes Status': patient['diabetes_assessment']['condition'],
                'Risk Level': patient['diabetes_assessment']['risk_category'],
                'Admission Date': patient['admission_date']
            })
        
        st.dataframe(display_data, use_container_width=True)
    else:
        st.info("No patients registered yet.")
    
    # Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Diabetes Status Distribution")
        if hospital_data.patients:
            status_counts = {}
            for patient in hospital_data.patients:
                status = patient['diabetes_assessment']['condition']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                status_df = pd.DataFrame({
                    'Diabetes Status': list(status_counts.keys()),
                    'Count': list(status_counts.values())
                })
                fig = px.pie(status_df, values='Count', names='Diabetes Status', 
                            title="Patient Diabetes Status Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 HbA1c Level Distribution")
        if hospital_data.patients:
            hba1c_levels = [p['HbA1c'] for p in hospital_data.patients]
            fig = px.histogram(x=hba1c_levels, nbins=15, 
                             title="Patient HbA1c Distribution",
                             labels={'x': 'HbA1c Level', 'y': 'Number of Patients'},
                             color_discrete_sequence=['#FF6B6B'])
            fig.add_vline(x=5.7, line_dash="dash", line_color="orange", annotation_text="Pre-diabetes")
            fig.add_vline(x=6.5, line_dash="dash", line_color="red", annotation_text="Diabetes")
            st.plotly_chart(fig, use_container_width=True)

def patient_management():
    st.title("👥 Patient Management")
    
    tab1, tab2 = st.tabs(["Add New Patient", "View All Patients"])
    
    with tab1:
        st.subheader("➕ Register New Patient")
        
        with st.form("patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name *")
                age = st.number_input("Age *", min_value=0, max_value=120, value=45)
                gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
                admission_reason = st.selectbox("Admission Reason *", 
                                              ["Routine Diabetes Screening", "Diabetes Management", 
                                               "Pre-diabetes Consultation", "Other"])
                
                # Diabetes-specific fields
                hba1c = st.slider("HbA1c Level *", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
                chol = st.slider("Cholesterol (Chol) *", min_value=2.0, max_value=10.0, value=4.5, step=0.1)
                tg = st.slider("Triglycerides (TG) *", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
            
            with col2:
                urea = st.slider("Urea Level *", min_value=1.0, max_value=20.0, value=4.5, step=0.1)
                cr = st.slider("Creatinine (Cr) *", min_value=10, max_value=200, value=60)
                hdl = st.slider("HDL Cholesterol *", min_value=0.3, max_value=3.0, value=1.2, step=0.1)
                ldl = st.slider("LDL Cholesterol *", min_value=1.0, max_value=6.0, value=2.5, step=0.1)
                vldl = st.slider("VLDL Cholesterol *", min_value=0.1, max_value=3.0, value=0.8, step=0.1)
                bmi = st.slider("BMI *", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
            
            if st.form_submit_button("🚀 Register Patient"):
                if name and age:
                    patient_data = {
                        'name': name,
                        'age': age,
                        'gender': gender,
                        'Urea': urea,
                        'Cr': cr,
                        'HbA1c': hba1c,
                        'Chol': chol,
                        'TG': tg,
                        'HDL': hdl,
                        'LDL': ldl,
                        'VLDL': vldl,
                        'BMI': bmi,
                        'admission_reason': admission_reason
                    }
                    
                    patient_id = hospital_data.add_patient(patient_data)
                    
                    st.success(f"✅ Patient {name} registered successfully! ID: {patient_id}")
                    st.balloons()
                    
                    # Show diabetes assessment
                    assessment = patient_data['diabetes_assessment']
                    st.subheader("🤖 AI Diabetes Assessment")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Diabetes Status", assessment['condition'])
                        st.metric("Risk Category", assessment['risk_category'])
                    with col2:
                        st.metric("No Diabetes Probability", f"{assessment['probability_no_diabetes']*100:.1f}%")
                        st.metric("Pre-Diabetes Probability", f"{assessment['probability_pre_diabetes']*100:.1f}%")
                    with col3:
                        st.metric("Diabetes Probability", f"{assessment['probability_diabetes']*100:.1f}%")
                    
                    # Key factors
                    if assessment['key_factors']:
                        st.warning(f"🚨 Key Factors: {', '.join(assessment['key_factors'])}")
                    
                    # Recommendations
                    st.info("💡 Recommendations:")
                    for rec in assessment['recommendations']:
                        st.write(f"- {rec}")
                else:
                    st.error("Please fill in all required fields")
    
    with tab2:
        st.subheader("📋 All Patients")
        
        if hospital_data.patients:
            # Search and filters
            col1, col2 = st.columns(2)
            with col1:
                search_name = st.text_input("Search by Name")
            with col2:
                diabetes_filter = st.selectbox("Filter by Diabetes Status", ["All", "No Diabetes", "Pre-Diabetes", "Diabetes"])
            
            # Filter patients
            filtered_patients = hospital_data.patients
            if search_name:
                filtered_patients = [p for p in filtered_patients if search_name.lower() in p['name'].lower()]
            if diabetes_filter != "All":
                filtered_patients = [p for p in filtered_patients if p['diabetes_assessment']['condition'] == diabetes_filter]
            
            if filtered_patients:
                display_data = []
                for patient in filtered_patients:
                    display_data.append({
                        'ID': patient['patient_id'],
                        'Name': patient['name'],
                        'Age': patient['age'],
                        'Gender': patient['gender'],
                        'Status': patient['status'],
                        'Diabetes Status': patient['diabetes_assessment']['condition'],
                        'Risk Level': patient['diabetes_assessment']['risk_category'],
                        'HbA1c': patient['HbA1c'],
                        'BMI': patient['BMI'],
                        'Admission Date': patient['admission_date'],
                        'Condition': patient['admission_reason']
                    })
                
                st.dataframe(display_data, use_container_width=True)
                
                # Export option
                if st.button("📤 Export to CSV"):
                    df = pd.DataFrame(display_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"diabetes_patients_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No patients match the search criteria.")
        else:
            st.info("No patients registered yet.")

def ai_predictions():
    st.title("🤖 Diabetes AI Predictions")
    
    st.subheader("🎯 Interactive Diabetes Risk Predictor")
    st.info("Enter patient parameters to get instant diabetes risk assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 0, 100, 45)
        hba1c = st.slider("HbA1c Level", 3.0, 15.0, 5.5, 0.1)
        urea = st.slider("Urea Level", 1.0, 20.0, 4.5, 0.1)
        cr = st.slider("Creatinine (Cr)", 10, 200, 60)
    
    with col2:
        chol = st.slider("Cholesterol", 2.0, 10.0, 4.5, 0.1)
        tg = st.slider("Triglycerides (TG)", 0.5, 5.0, 1.5, 0.1)
        hdl = st.slider("HDL Cholesterol", 0.3, 3.0, 1.2, 0.1)
        ldl = st.slider("LDL Cholesterol", 1.0, 6.0, 2.5, 0.1)
        vldl = st.slider("VLDL Cholesterol", 0.1, 3.0, 0.8, 0.1)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
    
    if st.button("🔍 Analyze Diabetes Risk"):
        test_patient = {
            'AGE': age,
            'Urea': urea,
            'Cr': cr,
            'HbA1c': hba1c,
            'Chol': chol,
            'TG': tg,
            'HDL': hdl,
            'LDL': ldl,
            'VLDL': vldl,
            'BMI': bmi
        }
        
        prediction = diabetes_ai.predict_diabetes_risk(test_patient)
        
        # Display results
        st.subheader("📊 Diabetes Risk Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_color = "green" if prediction['risk_category'] == "Low" else "orange" if prediction['risk_category'] == "Medium" else "red"
            st.metric("Diabetes Status", prediction['condition'])
        with col2:
            st.metric("Risk Category", prediction['risk_category'])
        with col3:
            st.metric("Diabetes Probability", f"{prediction['probability_diabetes']*100:.1f}%")
        
        # Gauge charts
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = prediction['probability_diabetes'] * 100,
            title = {'text': "Diabetes Risk"},
            domain = {'x': [0, 0.45], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 65
                }
            }
        ))
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number", 
            value = hba1c,
            title = {'text': "HbA1c Level"},
            domain = {'x': [0.55, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [3, 15]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [3, 5.7], 'color': "lightgreen"},
                    {'range': [5.7, 6.5], 'color': "yellow"},
                    {'range': [6.5, 15], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 6.5
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Key factors and recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction['key_factors']:
                st.warning(f"🚨 Key Risk Factors: {', '.join(prediction['key_factors'])}")
        
        with col2:
            st.info("💡 Recommendations:")
            for rec in prediction['recommendations']:
                st.write(f"- {rec}")

# ==================== MAIN APP ====================
# ==================== MAIN APP ====================
def main():
    login_section()
    
    if not st.session_state.get('authenticated', False):
        st.title("🏥 Diabetes Management System")
        st.markdown("""
        ## Welcome to the Diabetes Management System
        
        A specialized healthcare platform with AI-powered diabetes risk assessment using real clinical data.
        
        ### 🔐 Please login to continue
        Use the sidebar to login with your credentials.
        
        ### Demo Credentials:
        - **Admin:** `admin` / `Admin123!`
        - **Doctor:** `doctor` / `Admin123!`
        
        ### 🚀 Features:
        - **Diabetes-focused Patient Management** - Complete patient registration with diabetes parameters
        - **AI Diabetes Risk Assessment** - Predictive analytics using real clinical data
        - **Real-time Analytics** - Interactive diabetes dashboards and reports
        - **Secure Access** - Role-based authentication system
        
        ### 📊 Clinical Parameters Tracked:
        - HbA1c, Blood Glucose Levels
        - Lipid Profile (Cholesterol, Triglycerides, HDL, LDL, VLDL)
        - Kidney Function (Urea, Creatinine)
        - BMI and Anthropometric Data
        """)
        return
    
    # Main navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Patient Management", "AI Predictions"])
    
    if page == "Dashboard":
        main_dashboard()
    elif page == "Patient Management":
        patient_management()
    elif page == "AI Predictions":
        ai_predictions()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Patients:** {len(hospital_data.patients)}")
    st.sidebar.info(f"**Last Update:** {datetime.now().strftime('%H:%M')}")

if __name__ == "__main__":
    main()
