import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import hashlib, secrets, io, smtplib
from email.message import EmailMessage
from sklearn.ensemble import RandomForestClassifier
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PyPDF2 import PdfReader, PdfWriter

# ================= CONFIG =================
st.set_page_config(page_title="Hospital System", layout="wide")

# ================= SECURITY =================
class Security:
    def hash(self, pwd):
        salt = secrets.token_hex(8)
        return hashlib.sha256((pwd+salt).encode()).hexdigest()+":"+salt

    def verify(self, pwd, hashed):
        h,s = hashed.split(":")
        return hashlib.sha256((pwd+s).encode()).hexdigest()==h

sec = Security()

USERS = {
    "admin": {"pwd": sec.hash("Admin123!"), "name":"Admin"}
}

# ================= ML =================
class DiabetesAI:
    def __init__(self):
        self.model = RandomForestClassifier()
        X = np.random.rand(50,10)
        y = np.random.randint(0,3,50)
        self.model.fit(X,y)

    def predict(self, data):
        pred = self.model.predict([data])[0]
        cond = ["No Diabetes","Pre-Diabetes","Diabetes"]
        risk = ["Low","Medium","High"]
        return cond[pred], risk[pred]

diabetes_ai = DiabetesAI()

class BP:
    def predict(self, s,d):
        if s<120 and d<80: return "Normal","Low"
        elif s<130: return "Elevated","Medium"
        elif s<140 or d<90: return "Stage 1","High"
        else: return "Stage 2","Very High"

bp_ai = BP()

# ================= STORAGE =================
if "patients" not in st.session_state:
    st.session_state.patients = []

# ================= PDF =================
def generate_pdf(p, password):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Medical Report", styles["Title"]),
        Spacer(1,10),
        Paragraph(f"Name: {p.get('name','N/A')}", styles["Normal"]),
        Paragraph(f"Age: {p.get('age','N/A')}", styles["Normal"]),
        Paragraph(f"Diabetes: {p.get('diabetes','N/A')}", styles["Normal"]),
        Paragraph(f"BP: {p.get('bp','N/A')}", styles["Normal"]),
    ]

    doc.build(story)
    buffer.seek(0)

    reader = PdfReader(buffer)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    writer.encrypt(password)

    secured = io.BytesIO()
    writer.write(secured)
    secured.seek(0)

    return secured

# ================= EMAIL =================
def send_email(to, pdf):
    msg = EmailMessage()
    msg["Subject"] = "Medical Report"
    msg["From"] = "your_email@gmail.com"
    msg["To"] = to
    msg.set_content("Report attached. Password = DOB")

    msg.add_attachment(pdf.read(),
                       maintype="application",
                       subtype="pdf",
                       filename="report.pdf")

    with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
        s.login("your_email@gmail.com","your_app_password")
        s.send_message(msg)

# ================= LOGIN =================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in USERS and sec.verify(p, USERS[u]["pwd"]):
            st.session_state.login=True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# ================= MAIN =================
st.title("🏥 Diabetes Hospital Management System")

tab1, tab2 = st.tabs(["Add Patient","View Patients"])

# -------- ADD PATIENT --------
with tab1:
    with st.form("form"):
        col1,col2 = st.columns(2)

        with col1:
            name = st.text_input("Name")
            age = st.number_input("Age",0,120,45)
            hba1c = st.slider("HbA1c",3.0,15.0,5.5)
            chol = st.slider("Chol",2.0,10.0,4.5)

        with col2:
            bmi = st.slider("BMI",15.0,50.0,25.0)
            sys = st.slider("Systolic",80,200,120)
            dia = st.slider("Diastolic",50,130,80)
            dob = st.text_input("DOB (ddmmyyyy)")
            email = st.text_input("Email")

        submit = st.form_submit_button("Submit")

    if submit and name:

        if len(dob)!=8 or not dob.isdigit():
            st.error("DOB must be ddmmyyyy")
            st.stop()

        diabetes, risk = diabetes_ai.predict(
            [age,1,1,hba1c,chol,1,1,1,1,bmi]
        )
        bp, bp_risk = bp_ai.predict(sys,dia)

        patient = {
            "name":name,
            "age":age,
            "diabetes":diabetes,
            "bp":bp,
            "date":datetime.now().strftime("%Y-%m-%d")
        }

        st.session_state.patients.append(patient)

        st.success("Patient Added")
        st.write("Diabetes:", diabetes)
        st.write("BP:", bp)

        pdf = generate_pdf(patient, dob)
        st.download_button("Download PDF", pdf, "report.pdf")

        if email:
            pdf = generate_pdf(patient, dob)
            send_email(email,pdf)
            st.success("Email sent")

# -------- VIEW --------
with tab2:
    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients)
        st.dataframe(df)
    else:
        st.info("No patients yet")
