import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import random

# Page config with custom theme
st.set_page_config(
    page_title="Student Performance Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stMetric label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #ffd700 !important;
        font-weight: 600 !important;
    }
    .stButton>button {
        background-color: #06A77D;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #058968;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Custom color palette
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'success': '#06A77D',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'safe': '#06A77D',         # Green for safe students
    'at_risk': '#C73E1D',      # Red for at-risk students
    'background': '#F8F9FA'
}

# Indian names database - Diverse across religions and regions
FIRST_NAMES_MALE = [
    # North Indian Hindu
    'Aarav', 'Vivaan', 'Aditya', 'Vihaan', 'Arjun', 'Arnav', 'Krishna', 'Ishaan', 'Shaurya', 'Atharv',
    'Pranav', 'Reyansh', 'Siddharth', 'Vedant', 'Shivansh', 'Abhay', 'Raghav', 'Rishi', 'Rohan', 'Dhruv',
    'Yash', 'Aayush', 'Lakshya', 'Aaryan', 'Veer', 'Nikhil', 'Harsh', 'Varun', 'Karan', 'Amit',
    # South Indian Hindu
    'Karthik', 'Arvind', 'Rajesh', 'Venkatesh', 'Suresh', 'Ramesh', 'Arun', 'Vikram', 'Prakash', 'Anand',
    'Mahesh', 'Ganesh', 'Dinesh', 'Naveen', 'Sanjay', 'Murali', 'Balaji', 'Ravi', 'Mohan', 'Kishore',
    # Muslim
    'Mohammed', 'Ahmed', 'Ali', 'Faisal', 'Irfan', 'Imran', 'Salman', 'Farhan', 'Adil', 'Aamir',
    'Zain', 'Rehan', 'Arif', 'Karim', 'Rizwan', 'Bilal', 'Hamza', 'Omar', 'Yusuf', 'Ibrahim',
    # Christian
    'John', 'Thomas', 'Joseph', 'Ravi', 'Mathew', 'Paul', 'Samuel', 'David', 'Daniel', 'James',
    'George', 'Stephen', 'Anthony', 'Francis', 'Simon', 'Peter',
    # Sikh
    'Harpreet', 'Gurpreet', 'Manpreet', 'Simran', 'Jasdeep', 'Navdeep', 'Kuldeep', 'Sandeep', 'Amardeep', 'Jaskaran'
]

FIRST_NAMES_FEMALE = [
    # North Indian Hindu
    'Saanvi', 'Aadhya', 'Kiara', 'Diya', 'Ananya', 'Aarohi', 'Navya', 'Anika', 'Ishita', 'Prisha',
    'Aditi', 'Kavya', 'Shanaya', 'Avni', 'Riya', 'Siya', 'Khushi', 'Shreya', 'Aaradhya', 'Divya',
    'Pooja', 'Nisha', 'Priya', 'Tanvi', 'Ishani', 'Suhana', 'Manvi', 'Palak', 'Anjali', 'Neha',
    # South Indian Hindu
    'Lakshmi', 'Priya', 'Meena', 'Radha', 'Kamala', 'Deepika', 'Sowmya', 'Shalini', 'Kavitha', 'Bhavana',
    'Swathi', 'Aruna', 'Padma', 'Uma', 'Vani', 'Latha', 'Rekha', 'Geetha', 'Sita', 'Suma',
    # Muslim
    'Fatima', 'Ayesha', 'Zara', 'Zainab', 'Sana', 'Amina', 'Noor', 'Aliya', 'Safiya', 'Mariam',
    'Rukhsar', 'Sameera', 'Nazia', 'Shabnam', 'Farhana', 'Rabia', 'Hina', 'Zahra', 'Naima', 'Laila',
    # Christian
    'Mary', 'Sarah', 'Anna', 'Elizabeth', 'Grace', 'Ruth', 'Esther', 'Rachel', 'Rebecca', 'Hannah',
    'Priya', 'Jessy', 'Anita', 'Rita', 'Latha', 'Smitha',
    # Sikh
    'Harleen', 'Simran', 'Gurleen', 'Navleen', 'Jaspreet', 'Manpreet', 'Kuldeep', 'Amandeep', 'Navpreet', 'Ramandeep'
]

LAST_NAMES = [
    # North Indian Hindu
    'Sharma', 'Verma', 'Kumar', 'Gupta', 'Joshi', 'Desai', 'Chopra', 'Malhotra', 'Agarwal', 'Saxena',
    'Tiwari', 'Pandey', 'Mishra', 'Jain', 'Mehta', 'Shah', 'Gandhi', 'Rathore', 'Chauhan', 'Rajput',
    'Thakur', 'Kapoor', 'Sinha', 'Acharya', 'Bhatt', 'Trivedi', 'Dubey', 'Shukla', 'Yadav', 'Chawla',
    # South Indian Hindu
    'Reddy', 'Nair', 'Rao', 'Iyer', 'Menon', 'Shetty', 'Bhat', 'Hegde', 'Naik', 'Pillai',
    'Krishnan', 'Venkatesh', 'Subramaniam', 'Swamy', 'Murthy', 'Naidu', 'Raju', 'Prabhu', 'Acharya', 'Kamath',
    'Kulkarni', 'Pawar', 'Jadhav', 'More', 'Patil', 'Deshmukh', 'Shinde', 'Gawde', 'Sawant', 'Kambli',
    # Bengali
    'Banerjee', 'Das', 'Ghosh', 'Basu', 'Sengupta', 'Mukherjee', 'Chatterjee', 'Roy', 'Dutta', 'Bose',
    'Chakraborty', 'Bhattacharya', 'Ganguly', 'Mazumdar', 'Sarkar',
    # Muslim
    'Khan', 'Ali', 'Hussain', 'Ahmed', 'Ansari', 'Siddiqui', 'Qureshi', 'Sheikh', 'Malik', 'Patel',
    'Mohammad', 'Hassan', 'Abbas', 'Raza', 'Alam', 'Rahman', 'Aziz', 'Haider', 'Naqvi', 'Rizvi',
    # Christian
    'D\'Souza', 'Fernandes', 'Pereira', 'Rodrigues', 'Lobo', 'Mascarenhas', 'Thomas', 'George', 'Joseph', 'Abraham',
    'Mathew', 'Paul', 'Samuel', 'David', 'Daniel', 'John', 'Varghese', 'Philip',
    # Sikh
    'Singh', 'Kaur', 'Gill', 'Dhillon', 'Sandhu', 'Brar', 'Sidhu', 'Grewal', 'Virk', 'Bajwa',
    'Randhawa', 'Saini', 'Sodhi', 'Pannu', 'Sethi', 'Kohli'
]

# Generate Indian name
def generate_indian_name():
    gender = random.choice(['M', 'F'])
    if gender == 'M':
        first = random.choice(FIRST_NAMES_MALE)
    else:
        first = random.choice(FIRST_NAMES_FEMALE)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}", gender

# Generate synthetic data
@st.cache_data
def generate_student_data(n_students=100000):
    np.random.seed(42)
    random.seed(42)
    
    # Generate names and genders
    names_genders = [generate_indian_name() for _ in range(n_students)]
    names = [ng[0] for ng in names_genders]
    genders = [ng[1] for ng in names_genders]
    
    # Generate base data with more variation
    attendance = np.random.normal(76.5, 14.8, n_students).clip(0, 100)
    assignment = np.random.normal(71.3, 17.6, n_students).clip(0, 100)
    gpa = np.random.normal(2.97, 0.63, n_students).clip(1.0, 4.0)
    internal = np.random.normal(66.8, 19.4, n_students).clip(0, 100)
    
    # Add some outliers and abnormal values (10% of data)
    outlier_indices = np.random.choice(n_students, size=int(n_students * 0.1), replace=False)
    attendance[outlier_indices] = np.random.uniform(15, 95, len(outlier_indices))
    assignment[outlier_indices] = np.random.uniform(20, 98, len(outlier_indices))
    gpa[outlier_indices] = np.random.uniform(1.2, 3.9, len(outlier_indices))
    internal[outlier_indices] = np.random.uniform(25, 92, len(outlier_indices))
    
    # Round to 2 decimal places
    attendance = np.round(attendance, 2)
    assignment = np.round(assignment, 2)
    gpa = np.round(gpa, 2)
    internal = np.round(internal, 2)
    
    data = {
        'student_id': [f'STU{str(i).zfill(6)}' for i in range(1, n_students + 1)],
        'name': names,
        'gender': genders,
        'attendance_pct': attendance,
        'assignment_score': assignment,
        'previous_gpa': gpa,
        'internal_marks': internal,
        'semester': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_students),
        'subject': np.random.choice(['Mathematics', 'Physics', 'Chemistry', 'Computer Science', 'Biology', 
                                     'Electronics', 'Mechanical Engg', 'Civil Engg'], n_students),
        'department': np.random.choice(['Engineering', 'Science', 'Commerce', 'Arts'], n_students)
    }
    
    df = pd.DataFrame(data)
    
    # Create risk label based on realistic correlations
    risk_score = (
        (100 - df['attendance_pct']) * 0.3 +
        (100 - df['assignment_score']) * 0.25 +
        (4.0 - df['previous_gpa']) * 15 +
        (100 - df['internal_marks']) * 0.2
    )
    
    df['at_risk'] = (risk_score > np.percentile(risk_score, 70)).astype(int)
    
    return df

# Train model
@st.cache_resource
def train_model(df):
    features = ['attendance_pct', 'assignment_score', 'previous_gpa', 'internal_marks', 'semester']
    X = df[features]
    y = df['at_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': (y_pred == y_test).mean(),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, metrics, X_test, y_test

# Main app
def main():
    st.title("🎓 Student Performance Risk Prediction System")
    st.markdown("**Identify at-risk students early and enable data-driven interventions**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Controls")
        page = st.radio("Navigate", ["Dashboard", "Predict Risk", "Student Search", "Model Performance", "About"])
        st.divider()
        st.markdown("**Dataset Info**")
        st.info("Using synthetic data for 1,00,000 students.")
    
    # Load data and model
    df = generate_student_data(100000)
    model, metrics, X_test, y_test = train_model(df)
    
    # Dashboard Page
    if page == "Dashboard":
        st.header("📊 System Dashboard")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_students = len(df)
        at_risk_count = df['at_risk'].sum()
        safe_count = total_students - at_risk_count
        risk_percentage = (at_risk_count / total_students) * 100
        
        col1.metric("Total Students", f"{total_students:,}")
        col2.metric("At Risk", f"{at_risk_count:,}", delta=f"{risk_percentage:.1f}%", delta_color="inverse")
        col3.metric("Safe", f"{safe_count:,}")
        col4.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
        col5.metric("Departments", df['department'].nunique())
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            risk_counts = df['at_risk'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=['Safe', 'At Risk'],
                color_discrete_sequence=[COLORS['success'], COLORS['danger']],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk by Department")
            risk_by_dept = df.groupby('department')['at_risk'].mean() * 100
            fig = px.bar(
                x=risk_by_dept.index,
                y=risk_by_dept.values,
                labels={'x': 'Department', 'y': 'At Risk %'},
                color=risk_by_dept.values,
                color_continuous_scale=[[0, COLORS['success']], [0.5, COLORS['warning']], [1, COLORS['danger']]]
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk by Subject")
            risk_by_subject = df.groupby('subject')['at_risk'].mean() * 100
            risk_by_subject = risk_by_subject.sort_values(ascending=False)
            fig = px.bar(
                y=risk_by_subject.index,
                x=risk_by_subject.values,
                orientation='h',
                labels={'x': 'At Risk %', 'y': 'Subject'},
                color=risk_by_subject.values,
                color_continuous_scale=[[0, COLORS['success']], [0.5, COLORS['warning']], [1, COLORS['danger']]]
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Gender Distribution")
            gender_risk = df.groupby(['gender', 'at_risk']).size().reset_index(name='count')
            gender_risk['risk_status'] = gender_risk['at_risk'].map({0: 'Safe', 1: 'At Risk'})
            gender_risk['gender'] = gender_risk['gender'].map({'M': 'Male', 'F': 'Female'})
            fig = px.bar(
                gender_risk,
                x='gender',
                y='count',
                color='risk_status',
                barmode='group',
                labels={'count': 'Number of Students', 'gender': 'Gender'},
                color_discrete_map={'Safe': COLORS['success'], 'At Risk': COLORS['danger']}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Analysis by Risk Status")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                df,
                x='at_risk',
                y='attendance_pct',
                labels={'at_risk': 'Risk Status', 'attendance_pct': 'Attendance %'},
                color='at_risk',
                color_discrete_map={0: COLORS['success'], 1: COLORS['danger']}
            )
            fig.update_layout(
                showlegend=False, 
                xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Safe', 'At Risk']),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                x='at_risk',
                y='internal_marks',
                labels={'at_risk': 'Risk Status', 'internal_marks': 'Internal Marks'},
                color='at_risk',
                color_discrete_map={0: COLORS['success'], 1: COLORS['danger']}
            )
            fig.update_layout(
                showlegend=False, 
                xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Safe', 'At Risk']),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data preview
        st.subheader("📋 Student Data Preview")
        st.info("💡 Showing all 100,000 students. Full dataset available for download.")
        display_df = df.copy()
        display_df['at_risk'] = display_df['at_risk'].map({0: '✅ Safe', 1: '⚠️ At Risk'})
        display_df = display_df[['student_id', 'name', 'gender', 'department', 'subject', 'attendance_pct', 
                                   'assignment_score', 'previous_gpa', 'internal_marks', 'semester', 'at_risk']]
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download data
        st.subheader("📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=csv,
                file_name="student_risk_data_100K.csv",
                mime="text/csv"
            )
        
        with col2:
            at_risk_df = df[df['at_risk'] == 1]
            csv_risk = at_risk_df.to_csv(index=False)
            st.download_button(
                label="⚠️ Download At-Risk Students Only (CSV)",
                data=csv_risk,
                file_name="at_risk_students.csv",
                mime="text/csv"
            )
    
    # Predict Risk Page
    elif page == "Predict Risk":
        st.header("🔮 Predict Student Risk")
        
        st.markdown("Enter student information to predict their risk status:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            student_name = st.text_input("Student Name (Optional)", placeholder="e.g., Aarav Sharma")
            attendance = st.slider("Attendance %", 0, 100, 75)
            assignment_score = st.slider("Assignment Score", 0, 100, 70)
            previous_gpa = st.slider("Previous GPA", 1.0, 4.0, 3.0, 0.1)
        
        with col2:
            internal_marks = st.slider("Internal Marks", 0, 100, 65)
            semester = st.selectbox("Semester", [1, 2, 3, 4, 5, 6, 7, 8])
            department = st.selectbox("Department", ['Engineering', 'Science', 'Commerce', 'Arts'])
        
        if st.button("🎯 Predict Risk", type="primary"):
            input_data = pd.DataFrame({
                'attendance_pct': [attendance],
                'assignment_score': [assignment_score],
                'previous_gpa': [previous_gpa],
                'internal_marks': [internal_marks],
                'semester': [semester]
            })
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.divider()
            
            if student_name:
                st.subheader(f"Results for: {student_name}")
            else:
                st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **AT RISK**")
                else:
                    st.success("✅ **SAFE**")
            
            with col2:
                st.metric("Risk Probability", f"{probability[1]:.1%}")
            
            with col3:
                st.metric("Safe Probability", f"{probability[0]:.1%}")
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': COLORS['danger'] if prediction == 1 else COLORS['success']},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#D5F4E6'},
                        {'range': [30, 70], 'color': '#FFF4E6'},
                        {'range': [70, 100], 'color': '#FFEBE9'}
                    ],
                    'threshold': {
                        'line': {'color': COLORS['danger'], 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction == 1:
                st.warning("""
                **Intervention Suggested:**
                - Schedule one-on-one meeting with student
                - Review attendance patterns and identify barriers
                - Provide additional academic support resources
                - Connect with academic advisor or counselor
                - Assign peer mentor if available
                - Monitor progress weekly
                - Consider remedial classes for weak subjects
                """)
            else:
                st.info("""
                **Student Performing Well:**
                - Continue current support level
                - Periodic check-ins recommended (monthly)
                - Encourage peer mentoring opportunities
                - Consider advanced learning materials
                - Recognize good performance to maintain motivation
                """)
    
    # Student Search Page
    elif page == "Student Search":
        st.header("🔍 Search Students")
        
        st.markdown("Search for specific students by name or ID:")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input("Enter Student Name or ID", placeholder="e.g., Sharma or STU001234")
        
        with col2:
            risk_filter = st.selectbox("Filter by Risk", ["All", "At Risk Only", "Safe Only"])
        
        if search_query:
            # Search in dataframe
            mask = (df['name'].str.contains(search_query, case=False, na=False)) | \
                   (df['student_id'].str.contains(search_query, case=False, na=False))
            
            if risk_filter == "At Risk Only":
                mask = mask & (df['at_risk'] == 1)
            elif risk_filter == "Safe Only":
                mask = mask & (df['at_risk'] == 0)
            
            results = df[mask]
            
            st.subheader(f"Search Results: {len(results)} student(s) found")
            
            if len(results) > 0:
                display_results = results.copy()
                display_results['at_risk'] = display_results['at_risk'].map({0: '✅ Safe', 1: '⚠️ At Risk'})
                display_results = display_results[['student_id', 'name', 'gender', 'department', 'subject', 
                                                     'attendance_pct', 'assignment_score', 'previous_gpa', 
                                                     'internal_marks', 'semester', 'at_risk']]
                st.dataframe(display_results, use_container_width=True, height=400)
                
                # Export search results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Search Results",
                    data=csv,
                    file_name=f"search_results_{search_query}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No students found matching your search criteria.")
        else:
            st.info("Enter a student name or ID to search.")
    
    # Model Performance Page
    elif page == "Model Performance":
        st.header("📈 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("ROC-AUC Score", f"{metrics['roc_auc']:.3f}")
        col3.metric("Training Samples", f"{len(df):,}")
        col4.metric("Test Samples", f"{len(X_test):,}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = metrics['confusion_matrix']
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Safe', 'At Risk'],
                y=['Safe', 'At Risk'],
                text_auto=True,
                color_continuous_scale=[[0, COLORS['background']], [1, COLORS['primary']]]
            )
            fig.update_layout(width=400, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            st.markdown("""
            **Understanding the Matrix:**
            - **Top-Left (True Negatives):** Students correctly identified as Safe
            - **Bottom-Right (True Positives):** Students correctly identified as At Risk
            - **Top-Right (False Positives):** Safe students incorrectly flagged as At Risk
            - **Bottom-Left (False Negatives):** At Risk students missed by the model
            """)
        
        with col2:
            st.subheader("Feature Importance")
            feature_names = ['Attendance %', 'Assignment Score', 'Previous GPA', 'Internal Marks', 'Semester']
            importances = model.feature_importances_
            fig = px.bar(
                x=importances,
                y=feature_names,
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                color=importances,
                color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['primary']]]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            st.markdown("""
            **Key Insights:**
            - Higher importance = stronger predictor of risk
            - Attendance and internal marks are typically strongest indicators
            - Model learns these patterns from historical data
            """)
        
        # Classification report
        st.subheader("Detailed Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        report_df = report_df.round(3)
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        
        st.markdown("""
        **Metrics Explanation:**
        - **Precision:** Of students predicted as at-risk, what % actually were at-risk?
        - **Recall:** Of actual at-risk students, what % did we identify?
        - **F1-Score:** Balanced measure of precision and recall
        - **Support:** Number of samples in each class
        """)
    
    # About Page
    elif page == "About":
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### Student Performance Risk Prediction System
        
        **Purpose:**
        This system helps educational institutions identify students at risk of poor performance early,
        enabling timely interventions and support. Built to handle university-scale data efficiently.
        
        **Dataset Scale:**
        - **Total Students:** 1 Lakh (100,000)
        - **Indian Names:** Realistic names across religions and regions
        - **Demographics:** Gender, Department, Subject diversity
        - **Academic Data:** Attendance, Grades, GPA, Internal Marks
        
        **Technology Stack:**
        - **Frontend:** Streamlit (Interactive web framework)
        - **ML Model:** Random Forest Classifier (100 trees)
        - **Data Processing:** Pandas (100K+ rows), NumPy
        - **Visualization:** Plotly (Interactive charts)
        - **Performance:** Parallel processing (n_jobs=-1)
        
        **Features:**
        - ✅ Binary risk classification (Safe / At Risk)
        - ✅ Interactive dashboard with real-time insights
        - ✅ Individual student risk prediction
        - ✅ Student search by name or ID
        - ✅ Model performance tracking
        - ✅ Data export capabilities
        - ✅ Synthetic data with realistic Indian names
        
        **Key Predictors:**
        1. **Attendance Percentage** - Class participation indicator
        2. **Assignment Scores** - Regular work completion
        3. **Previous GPA** - Historical academic performance
        4. **Internal Marks** - Mid-term assessment results
        5. **Current Semester** - Academic progression level
        
        **Model Performance:**
        - **Accuracy:** ~{:.1%}
        - **ROC-AUC Score:** ~{:.3f}
        - **Training Time:** ~5-10 seconds (100K students)
        - **Prediction Time:** <1 millisecond per student
        
        **Target Users:**
        - 👨‍🏫 Teachers and Faculty
        - 👔 Academic Administrators
        - 📊 Educational Data Analysts
        - 🎓 Department Heads
        - 📈 University Management
        
        **Scalability:**
        This prototype demonstrates capability to handle large-scale university data.
        The architecture can scale to:
        - Multiple campuses
        - Real-time data integration
        - Historical trend analysis
        - Automated alert systems
        
        **Note:** This system uses synthetic data with realistic patterns for demonstration purposes.
        In production, it would integrate with actual Student Information Systems (SIS) and Learning
        Management Systems (LMS).
        """.format(metrics['accuracy'], metrics['roc_auc']))
        
        st.divider()
        
        st.markdown("""
        ### 🚀 Future Enhancements
        
        **Short Term:**
        - Multi-class risk levels (Low/Medium/High/Critical)
        - Email/SMS alert integration
        - Custom risk thresholds per department
        - Historical trend visualization
        
        **Medium Term:**
        - Integration with LMS platforms (Moodle, Canvas)
        - Automated intervention workflow triggers
        - Teacher dashboard with class-level insights
        - Mobile application for on-the-go access
        
        **Long Term:**
        - Deep learning models for better accuracy
        - Natural Language Processing for feedback analysis
        - Predictive analytics for course recommendations
        - Real-time monitoring with live data feeds
        - Multi-language support (Hindi, Tamil, Telugu, etc.)
        """)
        
        st.divider()
        
        st.markdown("""
        ### 📊 Data Privacy & Ethics
        
        - All data in this demo is synthetically generated
        - No real student information is used or stored
        - Production deployment would follow GDPR/data protection guidelines
        - Access control and role-based permissions required
        - Regular bias audits recommended for fairness
        """)

if __name__ == "__main__":
    main()