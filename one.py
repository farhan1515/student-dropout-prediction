import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ucimlrepo import fetch_ucirepo

# Page configuration
st.set_page_config(
    page_title="Student Dropout Prediction Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess the student data"""
    predict_students_dropout_and_academic_success = fetch_ucirepo(
        name="Predict Students' Dropout and Academic Success"
    )
    student_data = predict_students_dropout_and_academic_success.data.original.copy()
    student_data.rename(columns={'Nacionality': 'Nationality'}, inplace=True)
    return student_data

# Load data
try:
    data = load_data()
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and Introduction
st.markdown('<h1 class="main-header">🎓 Student Dropout Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This dashboard visualizes key factors affecting student dropout rates based on exploratory data analysis.
Focus areas: **Financial Factors**, **Demographics**, **Academic Performance**, and **Course-specific trends**.
""")

# Sidebar for filters
st.sidebar.header("📊 Dashboard Controls")
visualization_option = st.sidebar.selectbox(
    "Select Visualization Category:",
    ["Overview", "Financial Factors", "Demographics", "Course Analysis", "Academic Performance", "Model Comparison"]
)

# Main content based on selection
if visualization_option == "Overview":
    st.markdown('<h2 class="sub-header">📈 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", f"{len(data):,}")
    with col2:
        st.metric("Total Features", data.shape[1])
    with col3:
        dropout_rate = (data['Target'] == 'Dropout').sum() / len(data) * 100
        st.metric("Overall Dropout Rate", f"{dropout_rate:.1f}%")
    
    st.markdown("---")
    
    # Target Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Student Outcomes")
        target_counts = data["Target"].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(target_counts.index, target_counts.values, color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_xlabel("Outcomes", fontsize=12)
        ax.set_ylabel("Number of Students", fontsize=12)
        ax.set_title("Distribution of Student Outcomes", fontsize=14)
        
        for i, (idx, count) in enumerate(target_counts.items()):
            ax.text(i, count + 50, str(count), ha="center", va="bottom", fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Outcome Proportions")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        wedges, texts, autotexts = ax.pie(
            target_counts.values,
            labels=target_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        ax.set_title("Distribution of Student Outcomes", fontsize=14)
        st.pyplot(fig)
        plt.close()

elif visualization_option == "Financial Factors":
    st.markdown('<h2 class="sub-header">💰 Financial Factors Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    **Key Finding**: Financial factors are the **strongest predictors** of student dropout.
    Students with debt have ~70% dropout rate vs majority graduation for debt-free students.
    """)
    
    # Debt Status Analysis
    st.subheader("1️⃣ Debt Status Impact")
    debt_target_counts = data.groupby(["Debtor", "Target"])["Target"].count()
    
    col1, col2 = st.columns(2)
    
    with col1:
        no_debt_data = debt_target_counts.loc[0]
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#2ecc71", "#e74c3c", "#3498db"]
        explode = (0.05, 0, 0)
        wedges, texts, autotexts = ax.pie(
            no_debt_data.values,
            labels=no_debt_data.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode
        )
        ax.set_title("No Debt Students", fontsize=14, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        debt_data = debt_target_counts.loc[1]
        fig, ax = plt.subplots(figsize=(7, 6))
        wedges, texts, autotexts = ax.pie(
            debt_data.values,
            labels=debt_data.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode
        )
        ax.set_title("Students With Debt", fontsize=14, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Scholarship Status
    st.subheader("2️⃣ Scholarship Status Impact")
    scholarship_target_counts = data.groupby(["Scholarship holder", "Target"])["Target"].count()
    
    col1, col2 = st.columns(2)
    
    with col1:
        no_scholarship_data = scholarship_target_counts.loc[0]
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#9b59b6", "#f39c12", "#16a085"]
        explode = (0.05, 0, 0)
        wedges, texts, autotexts = ax.pie(
            no_scholarship_data.values,
            labels=no_scholarship_data.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode
        )
        ax.set_title("No Scholarship", fontsize=14, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        scholarship_data = scholarship_target_counts.loc[1]
        fig, ax = plt.subplots(figsize=(7, 6))
        wedges, texts, autotexts = ax.pie(
            scholarship_data.values,
            labels=scholarship_data.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode
        )
        ax.set_title("With Scholarship", fontsize=14, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Tuition Payment Status
    st.subheader("3️⃣ Tuition Payment Status")
    tuition_target_counts = data.groupby(["Tuition fees up to date", "Target"])["Target"].count()
    
    col1, col2 = st.columns(2)
    
    with col1:
        tuition_not_paid_data = tuition_target_counts.loc[0]
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        explode = (0.05, 0, 0)
        wedges, texts, autotexts = ax.pie(
            tuition_not_paid_data.values,
            labels=tuition_not_paid_data.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode
        )
        ax.set_title("Tuition Not Paid", fontsize=14, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight=('bold')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        tuition_paid_data = tuition_target_counts.loc[1]
        fig, ax = plt.subplots(figsize=(7, 6))
        wedges, texts, autotexts = ax.pie(
            tuition_paid_data.values,
            labels=tuition_paid_data.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode
        )
        ax.set_title("Tuition Paid", fontsize=14, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        st.pyplot(fig)
        plt.close()
    
    st.info("💡 **Recommendation**: Implement emergency financial aid programs and flexible payment plans to reduce dropout rates.")

elif visualization_option == "Demographics":
    st.markdown('<h2 class="sub-header">👥 Demographic Analysis</h2>', unsafe_allow_html=True)
    
    # Age Distribution
    st.subheader("Age at Enrollment Distribution")
    
    dropout_data = data[data["Target"] == "Dropout"]
    graduate_data = data[data["Target"] == "Graduate"]
    enrolled_data = data[data["Target"] == "Enrolled"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(dropout_data["Age at enrollment"], bins=50, alpha=0.7, label="Dropout", color="red")
        ax.hist(graduate_data["Age at enrollment"], bins=50, alpha=0.6, label="Graduate", color="green")
        ax.hist(enrolled_data["Age at enrollment"], bins=50, alpha=0.5, label="Enrolled", color="blue")
        ax.set_xlabel("Age at Enrollment", fontsize=12)
        ax.set_ylabel("Number of Students", fontsize=12)
        ax.set_title("Distribution of Age at Enrollment by Outcome", fontsize=14)
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(
            data=data,
            x="Age at enrollment",
            hue="Target",
            fill=True,
            alpha=0.5,
            palette={"Dropout": "red", "Graduate": "green", "Enrolled": "blue"},
            ax=ax
        )
        ax.set_xlabel("Age at Enrollment", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Probability Distribution of Age at Enrollment", fontsize=14)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Gender Analysis
    st.subheader("Gender Distribution by Outcome")
    data_gender = data.copy()
    data_gender["Gender"] = data_gender["Gender"].replace({0: "Female", 1: "Male"})
    gender_target_counts = pd.crosstab(data_gender["Gender"], data_gender["Target"], normalize="index") * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    gender_target_counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_xlabel("Gender", fontsize=12)
    ax.set_ylabel("Percentage of Students", fontsize=12)
    ax.set_title("Gender Distribution by Student Outcome", fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Target", title_fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    for p in ax.patches:
        percentage = f"{p.get_height():.1f}%"
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2.0, p.get_height() / 2.0),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight='bold'
        )
    
    st.pyplot(fig)
    plt.close()
    
    st.info("💡 **Key Insight**: Younger students (18-22) have significantly higher graduation rates. Age is a significant predictor.")

elif visualization_option == "Course Analysis":
    st.markdown('<h2 class="sub-header">📚 Course-Specific Analysis</h2>', unsafe_allow_html=True)
    
    courses_map = {
        33: 'Biofuel Production Technologies',
        171: 'Animation and Multimedia Design',
        8014: 'Social Service (evening)',
        9003: 'Agronomy',
        9070: 'Communication Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering',
        9130: 'Equinculture',
        9147: 'Management',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising and Marketing Management',
        9773: 'Journalism and Communication',
        9853: 'Basic Education',
        9991: 'Management (evening)'
    }
    
    data_courses = data.copy()
    data_courses["Course Name"] = data_courses["Course"].map(courses_map)
    course_target_counts = pd.crosstab(
        data_courses["Course Name"],
        data_courses["Target"],
        normalize="index"
    ) * 100
    
    dropout_percentages = course_target_counts["Dropout"].sort_values(ascending=True)
    
    st.subheader("Dropout Rates by Course")
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(dropout_percentages.index, dropout_percentages.values, color="#184ABB")
    ax.set_xlabel("Percentage of Students (%)", fontsize=12)
    ax.set_ylabel("Course Name", fontsize=12)
    ax.set_title("Dropout Rates by Course", fontsize=14, fontweight='bold')
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
    
    for i, v in enumerate(dropout_percentages.values):
        ax.text(v / 2, i, f"{v:.1f}%", color="white", fontweight="bold", va="center", ha="center")
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Highlight best and worst courses
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Lowest Dropout Rate**: {dropout_percentages.index[0]} ({dropout_percentages.values[0]:.1f}%)")
    with col2:
        st.error(f"**Highest Dropout Rate**: {dropout_percentages.index[-1]} ({dropout_percentages.values[-1]:.1f}%)")
    
    st.warning("⚠️ **Pattern Observed**: Evening attendance courses consistently show higher dropout rates.")

elif visualization_option == "Academic Performance":
    st.markdown('<h2 class="sub-header">📊 Academic Performance Indicators</h2>', unsafe_allow_html=True)
    
    # Correlation Matrix
    st.subheader("Correlation Matrix of Numerical Features")
    numerical_columns = [
        "Admission grade",
        "Age at enrollment",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]
    data_numeric = data[numerical_columns]
    correlation_matrix = data_numeric.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        center=0
    )
    ax.set_title("Correlation Matrix of Numerical Features", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Admission Grade Distribution by Course (Sample)
    st.subheader("Admission Grade Distributions (Select Courses)")
    
    courses_map = {
        9500: 'Nursing',
        9147: 'Management',
        9773: 'Journalism and Communication',
        8014: 'Social Service (evening)'
    }
    
    data_courses = data.copy()
    data_courses["Course Name"] = data_courses["Course"].map(courses_map)
    data_sample = data_courses[data_courses["Course Name"].notna()]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (course_name, course_data) in enumerate(data_sample.groupby("Course Name")):
        if idx < 4:
            axes[idx].hist(course_data["Admission grade"], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[idx].set_title(f"{course_name}", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Admission Grade", fontsize=10)
            axes[idx].set_ylabel("Frequency", fontsize=10)
            axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

elif visualization_option == "Model Comparison":
    st.markdown('<h2 class="sub-header">🤖 Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Based on the machine learning models trained on this dataset, here's a comparison of their performance.
    **Focus Metric**: Recall (ability to identify dropout students)
    """)
    
    # Model results (from your document)
    model_results = {
        'Model': ['KNN', 'Logistic Regression', 'SVM', 'Decision Tree'],
        'Test_Accuracy': [0.8780, 0.9097, 0.3819, 0.8685],
        'Test_Precision': [0.8775, 0.8783, 0.3707, 0.8304],
        'Test_Recall': [0.7749, 0.8745, 0.9870, 0.8052],
        'Test_F1': [0.8230, 0.8764, 0.5390, 0.8176]
    }
    
    results_df = pd.DataFrame(model_results)
    
    # Display metrics table
    st.subheader("Model Performance Metrics")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Test_Accuracy', 'Test_Recall', 'Test_F1'], color='lightgreen'))
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Test Recall Comparison")
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(results_df['Model'], results_df['Test_Recall'], color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax.set_ylabel('Recall Score', fontsize=12)
        ax.set_title('Test Recall by Model (Primary Metric)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(results_df['Test_Recall']):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Test Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(results_df['Model'], results_df['Test_Accuracy'], color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax.set_ylabel('Accuracy Score', fontsize=12)
        ax.set_title('Test Accuracy by Model', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(results_df['Test_Accuracy']):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    # All metrics comparison
    st.subheader("All Metrics Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.2
    
    ax.bar(x - 1.5*width, results_df['Test_Accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, results_df['Test_Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, results_df['Test_Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, results_df['Test_F1'], width, label='F1 Score', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comprehensive Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Best model summary
    best_recall_model = results_df.loc[results_df['Test_Recall'].idxmax(), 'Model']
    best_accuracy_model = results_df.loc[results_df['Test_Accuracy'].idxmax(), 'Model']
    best_f1_model = results_df.loc[results_df['Test_F1'].idxmax(), 'Model']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"**Best Recall**: {best_recall_model}")
    with col2:
        st.info(f"**Best Accuracy**: {best_accuracy_model}")
    with col3:
        st.warning(f"**Best F1 Score**: {best_f1_model}")
    
    st.info("💡 **Recommendation**: Logistic Regression offers the best balance between all metrics for production deployment.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Student Dropout Prediction Dashboard | Data Source: UCI Machine Learning Repository</p>
        <p>Focus: Financial support, early intervention, and targeted counseling can significantly reduce dropout rates.</p>
    </div>
""", unsafe_allow_html=True)