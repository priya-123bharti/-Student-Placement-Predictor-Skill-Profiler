import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Student Placement Predictor & Skill Profiler",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .persona-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 10px;
    }
    .badge-tech { background-color: #312E81; color: #818CF8; border: 1px solid #6366F1; }
    .badge-analytical { background-color: #064E3B; color: #34D399; border: 1px solid #10B981; }
    .badge-leadership { background-color: #78350F; color: #FBBF24; border: 1px solid #F59E0B; }
    .badge-balanced { background-color: #581C87; color: #C084FC; border: 1px solid #A855F7; }
    .badge-growing { background-color: #1E3A8A; color: #60A5FA; border: 1px solid #3B82F6; }
    </style>
""", unsafe_allow_html=True)

def generate_txt_report(
    student_details,
    prediction_status,
    confidence_pct,
    persona_name,
    persona_desc,
    overall_index,
    tech_index,
    analytical_index,
    soft_index,
    practical_index,
    strengths,
    weaknesses,
    roadmap_df
):
    now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        "==========================================================================",
        "          🎓 STUDENT PLACEMENT PREDICTOR & SKILL PROFILE REPORT            ",
        "==========================================================================",
        f"Report Generated On: {now_str}",
        "",
        "--------------------------------------------------------------------------",
        "1. CANDIDATE PROFILE DETAILS",
        "--------------------------------------------------------------------------"
    ]
    for k, v in student_details.items():
        lines.append(f"  • {k:28s}: {v}")
    
    lines.extend([
        "",
        "--------------------------------------------------------------------------",
        "2. PLACEMENT EVALUATION & PERSONA",
        "--------------------------------------------------------------------------",
        f"  • Placement Result           : {prediction_status}",
        f"  • Model Confidence           : {confidence_pct:.1f}%",
        f"  • Persona Classification     : {persona_name}",
        f"  • Persona Description        : {persona_desc}",
        f"  • Overall Skill Index        : {overall_index} / 100",
        "",
        "--------------------------------------------------------------------------",
        "3. SKILL PILLAR INDEX BREAKDOWN",
        "--------------------------------------------------------------------------",
        f"  • Technical Skill Index      : {tech_index} / 100",
        f"  • Analytical Skill Index     : {analytical_index} / 100",
        f"  • Soft Skills Index          : {soft_index} / 100",
        f"  • Practical & Career Index   : {practical_index} / 100",
        "",
        "--------------------------------------------------------------------------",
        "4. KEY SKILL STRENGTHS",
        "--------------------------------------------------------------------------"
    ])
    
    if strengths:
        for s in strengths:
            clean_s = s.replace("💪", "").replace("**", "").strip()
            lines.append(f"  [✓] {clean_s}")
    else:
        lines.append("  • No standout skill strengths identified above benchmark.")
        
    lines.extend([
        "",
        "--------------------------------------------------------------------------",
        "5. TARGETED RECOMMENDATIONS & SKILL GAPS",
        "--------------------------------------------------------------------------"
    ])
    
    if weaknesses:
        for item, curr, target, rec in weaknesses:
            lines.append(f"  [!] {item} (Current Score: {curr} | Target Benchmark: {target})")
            lines.append(f"      Recommendation: {rec}")
    else:
        lines.append("  [✓] Outstanding Profile! All core skill areas meet or exceed target benchmarks.")
        
    lines.extend([
        "",
        "--------------------------------------------------------------------------",
        "6. RECOMMENDED TARGET BENCHMARK ROADMAP",
        "--------------------------------------------------------------------------",
        roadmap_df.to_string(index=False),
        "",
        "==========================================================================",
        "     🎓 Powered by Student Placement Predictor & Skill Profiler            ",
        "=========================================================================="
    ])
    return "\n".join(lines)

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, features, encoders

try:
    model, scaler, FEATURES, encoders = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first.")
    st.stop()

# Presets configuration
PRESETS = {
    "Custom / Default": {
        "age": 22, "gender": "Male", "cgpa": 7.5, "branch": "CSE", "college_tier": "Tier 2",
        "internships_count": 1, "projects_count": 2, "certifications_count": 1,
        "coding_skill_score": 65.0, "aptitude_score": 60.0, "communication_skill_score": 60.0,
        "logical_reasoning_score": 60.0, "hackathons_participated": 1, "github_repos": 3,
        "linkedin_connections": 200, "mock_interview_score": 65.0, "attendance_percentage": 80.0,
        "backlogs": 0, "extracurricular_score": 50.0, "leadership_score": 50.0,
        "volunteer_experience": "No", "sleep_hours": 7.0, "study_hours_per_day": 4.0
    },
    "💻 Tech Specialist": {
        "age": 21, "gender": "Male", "cgpa": 8.2, "branch": "CSE", "college_tier": "Tier 1",
        "internships_count": 2, "projects_count": 5, "certifications_count": 3,
        "coding_skill_score": 92.0, "aptitude_score": 75.0, "communication_skill_score": 65.0,
        "logical_reasoning_score": 80.0, "hackathons_participated": 4, "github_repos": 12,
        "linkedin_connections": 450, "mock_interview_score": 78.0, "attendance_percentage": 88.0,
        "backlogs": 0, "extracurricular_score": 45.0, "leadership_score": 55.0,
        "volunteer_experience": "No", "sleep_hours": 6.5, "study_hours_per_day": 6.0
    },
    "🧠 Analytical Scholar": {
        "age": 22, "gender": "Female", "cgpa": 9.3, "branch": "IT", "college_tier": "Tier 1",
        "internships_count": 2, "projects_count": 3, "certifications_count": 2,
        "coding_skill_score": 78.0, "aptitude_score": 95.0, "communication_skill_score": 70.0,
        "logical_reasoning_score": 92.0, "hackathons_participated": 2, "github_repos": 5,
        "linkedin_connections": 300, "mock_interview_score": 82.0, "attendance_percentage": 95.0,
        "backlogs": 0, "extracurricular_score": 60.0, "leadership_score": 60.0,
        "volunteer_experience": "Yes", "sleep_hours": 7.5, "study_hours_per_day": 7.0
    },
    "🗣️ Leadership & Comm Star": {
        "age": 22, "gender": "Female", "cgpa": 7.8, "branch": "ECE", "college_tier": "Tier 2",
        "internships_count": 2, "projects_count": 2, "certifications_count": 2,
        "coding_skill_score": 62.0, "aptitude_score": 70.0, "communication_skill_score": 92.0,
        "logical_reasoning_score": 68.0, "hackathons_participated": 2, "github_repos": 3,
        "linkedin_connections": 800, "mock_interview_score": 90.0, "attendance_percentage": 85.0,
        "backlogs": 0, "extracurricular_score": 88.0, "leadership_score": 92.0,
        "volunteer_experience": "Yes", "sleep_hours": 7.0, "study_hours_per_day": 4.5
    },
    "🚀 High-Potential All-Rounder": {
        "age": 22, "gender": "Male", "cgpa": 8.8, "branch": "CSE", "college_tier": "Tier 1",
        "internships_count": 3, "projects_count": 4, "certifications_count": 3,
        "coding_skill_score": 85.0, "aptitude_score": 88.0, "communication_skill_score": 85.0,
        "logical_reasoning_score": 86.0, "hackathons_participated": 3, "github_repos": 8,
        "linkedin_connections": 600, "mock_interview_score": 88.0, "attendance_percentage": 90.0,
        "backlogs": 0, "extracurricular_score": 75.0, "leadership_score": 80.0,
        "volunteer_experience": "Yes", "sleep_hours": 7.0, "study_hours_per_day": 5.5
    }
}

# Placed benchmark averages derived from dataset
BENCHMARKS = {
    "Coding Skill": 70.5,
    "Aptitude": 65.4,
    "Communication": 68.4,
    "Logical Reasoning": 66.4,
    "Mock Interview": 70.5,
    "Academics (CGPA x 10)": 75.1,
    "Projects & GitHub": 65.0
}

# Sidebar - Preset Selector & Header
st.sidebar.title("⚡ Profile Presets")
selected_preset = st.sidebar.selectbox(
    "Choose a Student Preset Profile:",
    list(PRESETS.keys())
)

preset_data = PRESETS[selected_preset]

# Initialize or update session state variables based on preset
if "last_preset" not in st.session_state or st.session_state.last_preset != selected_preset:
    st.session_state.last_preset = selected_preset
    for key, val in preset_data.items():
        st.session_state[f"input_{key}"] = val

st.sidebar.divider()
st.sidebar.info("💡 **Tip**: Select a preset above or adjust student details in the form to analyze placement probability and skill profile.")

# App Header
st.markdown("<h1 class='main-header'>🎓 Student Placement Predictor & Skill Profiler</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analyze student skills, benchmark against placed candidates, identify skill gaps, and predict placement probability.</p>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_predict, tab_profile, tab_recommendations = st.tabs([
    "🔮 Placement Predictor & Form",
    "📊 Skill Profile & Benchmarks",
    "💡 Skill Gap & Recommendations"
])

# ==================== TAB 1: PLACEMENT PREDICTOR FORM ====================
with tab_predict:
    st.subheader("📝 Student Profile Input")
    st.caption("Fill in student details across academic, technical, analytical, and interpersonal dimensions.")
    
    with st.form("placement_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 Personal & Academic")
            age = st.number_input("Age", 18, 30, key="input_age")
            gender = st.selectbox("Gender", encoders["gender"].classes_, key="input_gender")
            cgpa = st.slider("CGPA", 4.0, 10.0, step=0.01, key="input_cgpa")
            branch = st.selectbox("Branch", encoders["branch"].classes_, key="input_branch")
            college_tier = st.selectbox("College Tier", encoders["college_tier"].classes_, key="input_college_tier")
            attendance_percentage = st.slider("Attendance %", 0.0, 100.0, key="input_attendance_percentage")
            backlogs = st.number_input("Active Backlogs", 0, 10, key="input_backlogs")
            sleep_hours = st.slider("Sleep Hours/Day", 0.0, 12.0, key="input_sleep_hours")
            study_hours_per_day = st.slider("Study Hours/Day", 0.0, 12.0, key="input_study_hours_per_day")

        with col2:
            st.markdown("##### 💻 Technical & Practical")
            coding_skill_score = st.slider("Coding Skill Score", 0.0, 100.0, key="input_coding_skill_score")
            projects_count = st.number_input("Projects Completed", 0, 15, key="input_projects_count")
            github_repos = st.number_input("GitHub Repositories", 0, 30, key="input_github_repos")
            certifications_count = st.number_input("Certifications", 0, 10, key="input_certifications_count")
            internships_count = st.number_input("Internships Completed", 0, 10, key="input_internships_count")
            hackathons_participated = st.number_input("Hackathons Participated", 0, 10, key="input_hackathons_participated")

        with col3:
            st.markdown("##### 🧠 Analytical & Soft Skills")
            aptitude_score = st.slider("Aptitude Score", 0.0, 100.0, key="input_aptitude_score")
            logical_reasoning_score = st.slider("Logical Reasoning Score", 0.0, 100.0, key="input_logical_reasoning_score")
            communication_skill_score = st.slider("Communication Skill Score", 0.0, 100.0, key="input_communication_skill_score")
            mock_interview_score = st.slider("Mock Interview Score", 0.0, 100.0, key="input_mock_interview_score")
            leadership_score = st.slider("Leadership Score", 0.0, 100.0, key="input_leadership_score")
            extracurricular_score = st.slider("Extracurricular Score", 0.0, 100.0, key="input_extracurricular_score")
            linkedin_connections = st.number_input("LinkedIn Connections", 0, 2000, key="input_linkedin_connections")
            volunteer_experience = st.selectbox("Volunteer Experience", encoders["volunteer_experience"].classes_, key="input_volunteer_experience")

        submit_btn = st.form_submit_button("🔮 Predict Placement & Analyze Profile", type="primary", use_container_width=True)

# Build current student dataframe
raw_input = {
    "age": age,
    "gender": encoders["gender"].transform([gender])[0],
    "cgpa": cgpa,
    "branch": encoders["branch"].transform([branch])[0],
    "college_tier": encoders["college_tier"].transform([college_tier])[0],
    "internships_count": internships_count,
    "projects_count": projects_count,
    "certifications_count": certifications_count,
    "coding_skill_score": coding_skill_score,
    "aptitude_score": aptitude_score,
    "communication_skill_score": communication_skill_score,
    "logical_reasoning_score": logical_reasoning_score,
    "hackathons_participated": hackathons_participated,
    "github_repos": github_repos,
    "linkedin_connections": linkedin_connections,
    "mock_interview_score": mock_interview_score,
    "attendance_percentage": attendance_percentage,
    "backlogs": backlogs,
    "extracurricular_score": extracurricular_score,
    "leadership_score": leadership_score,
    "volunteer_experience": encoders["volunteer_experience"].transform([volunteer_experience])[0],
    "sleep_hours": sleep_hours,
    "study_hours_per_day": study_hours_per_day,
}

input_df = pd.DataFrame([raw_input])[FEATURES]
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
proba = model.predict_proba(scaled_input)[0][1]

# Display prediction summary under Tab 1
with tab_predict:
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        if prediction == 1:
            st.success(f"### ✅ Likely to be PLACED")
            st.markdown(f"**Confidence:** `{proba*100:.1f}%`")
        else:
            st.error(f"### ❌ Likely NOT Placed")
            st.markdown(f"**Confidence:** `{(1-proba)*100:.1f}%`")
        st.progress(float(proba))
    
    with res_col2:
        st.info(
            "ℹ️ **Placement Insights**: Model is trained on comprehensive placement data. "
            "Switch to the **📊 Skill Profile & Benchmarks** tab to see how your skill scores compare with successful candidates!"
        )

# ==================== SKILL CALCULATIONS & DOMAINS ====================
# Normalize project & repo metrics to 0-100 scale for radar comparison
project_repo_score = min(100.0, (projects_count * 15.0 + github_repos * 5.0 + certifications_count * 10.0))
practical_score = min(100.0, (internships_count * 30.0 + hackathons_participated * 20.0 + attendance_percentage * 0.4))

# Domain Scores
tech_index = round(float(np.mean([coding_skill_score, project_repo_score])), 1)
analytical_index = round(float(np.mean([aptitude_score, logical_reasoning_score, cgpa * 10])), 1)
soft_index = round(float(np.mean([communication_skill_score, mock_interview_score, leadership_score, extracurricular_score])), 1)
practical_index = round(float(practical_score), 1)

overall_skill_index = round(float(np.mean([tech_index, analytical_index, soft_index, practical_index])), 1)

# Classify Skill Persona
if coding_skill_score >= 80 and projects_count >= 3:
    persona_name = "🚀 Tech Maverick"
    persona_desc = "Strong technical proficiency, active GitHub presence, and robust coding fundamentals."
    persona_badge_class = "badge-tech"
elif aptitude_score >= 85 and logical_reasoning_score >= 80:
    persona_name = "🧠 Analytical Specialist"
    persona_desc = "Exceptional problem-solving capabilities, logical reasoning, and academic rigor."
    persona_badge_class = "badge-analytical"
elif communication_skill_score >= 85 and leadership_score >= 75:
    persona_name = "🗣️ Leadership & Comm Star"
    persona_desc = "Outstanding communication, interview readiness, and interpersonal influence."
    persona_badge_class = "badge-leadership"
elif overall_skill_index >= 75:
    persona_name = "🌟 Balanced High-Performer"
    persona_desc = "Well-rounded competence across technical, analytical, soft, and practical domains."
    persona_badge_class = "badge-balanced"
else:
    persona_name = "🌱 Growing Candidate"
    persona_desc = "Steadily developing foundational skills across key domain areas."
    persona_badge_class = "badge-growing"

# ==================== TAB 2: SKILL PROFILE & BENCHMARKS ====================
with tab_profile:
    st.subheader("📊 Skill Profile & Benchmark Comparison")
    
    # Persona & Skill Index Overview
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        st.markdown(f"<div class='persona-badge {persona_badge_class}'>{persona_name}</div>", unsafe_allow_html=True)
        st.caption(persona_desc)
    with p_col2:
        st.metric("Overall Skill Index", f"{overall_skill_index} / 100", delta=f"{overall_skill_index - 70.0:.1f} vs benchmark")
    with p_col3:
        st.metric("Placement Confidence", f"{proba*100:.1f}%", delta="High" if proba >= 0.6 else "Needs Improvement")
    with p_col4:
        st.metric("Academic Standing (CGPA)", f"{cgpa:.2f}", delta=f"{cgpa - 7.5:.2f} vs avg placed")

    st.divider()
    
    chart_col, domain_col = st.columns([1.2, 1])
    
    with chart_col:
        st.markdown("##### 🎯 Radar Chart: Student Profile vs Placed Candidates Benchmark")
        
        categories = [
            "Coding Skill", "Aptitude", "Communication", 
            "Logical Reasoning", "Mock Interview", "Academics (CGPA x 10)", "Projects & Repos"
        ]
        
        student_scores = [
            coding_skill_score, aptitude_score, communication_skill_score,
            logical_reasoning_score, mock_interview_score, cgpa * 10, project_repo_score
        ]
        
        benchmark_scores = [
            BENCHMARKS["Coding Skill"], BENCHMARKS["Aptitude"], BENCHMARKS["Communication"],
            BENCHMARKS["Logical Reasoning"], BENCHMARKS["Mock Interview"], BENCHMARKS["Academics (CGPA x 10)"], BENCHMARKS["Projects & GitHub"]
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=student_scores + [student_scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Current Student',
            line_color='#6366F1',
            fillcolor='rgba(99, 102, 241, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=benchmark_scores + [benchmark_scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Placed Benchmark Avg',
            line_color='#10B981',
            fillcolor='rgba(16, 185, 129, 0.15)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            margin=dict(l=40, r=40, t=30, b=30),
            showlegend=True,
            height=400,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with domain_col:
        st.markdown("##### 🏛️ Skill Pillar Index Breakdown")
        
        st.markdown(f"**💻 Technical Index**: `{tech_index}/100`")
        st.progress(min(1.0, tech_index / 100.0))
        st.caption(f"Coding Score: {coding_skill_score} | Projects: {projects_count} | GitHub Repos: {github_repos}")
        
        st.markdown(f"**🧠 Analytical Index**: `{analytical_index}/100`")
        st.progress(min(1.0, analytical_index / 100.0))
        st.caption(f"Aptitude: {aptitude_score} | Logical: {logical_reasoning_score} | CGPA: {cgpa}")
        
        st.markdown(f"**🗣️ Soft Skills Index**: `{soft_index}/100`")
        st.progress(min(1.0, soft_index / 100.0))
        st.caption(f"Comm: {communication_skill_score} | Mock Interview: {mock_interview_score} | Leadership: {leadership_score}")

        st.markdown(f"**🛠️ Practical & Career Readiness**: `{practical_index}/100`")
        st.progress(min(1.0, practical_index / 100.0))
        st.caption(f"Internships: {internships_count} | Hackathons: {hackathons_participated} | Certifications: {certifications_count}")

# ==================== TAB 3: SKILL GAP & RECOMMENDATIONS ====================
with tab_recommendations:
    st.subheader("💡 Skill Gap Analysis & Actionable Recommendations")
    
    # Identify Strengths & Weaknesses
    strengths = []
    weaknesses = []
    
    if coding_skill_score >= 70:
        strengths.append(f"💪 **Coding Skill ({coding_skill_score}/100)**: Solid programming fundamentals.")
    else:
        weaknesses.append(("Coding Skill", coding_skill_score, 75.0, "Practice DSA problems on LeetCode/HackerRank to boost coding speed and accuracy."))
        
    if aptitude_score >= 68:
        strengths.append(f"💪 **Aptitude Score ({aptitude_score}/100)**: Strong quantitative & analytical foundation.")
    else:
        weaknesses.append(("Aptitude", aptitude_score, 70.0, "Practice daily quantitative aptitude drills & time-bound mock tests."))
        
    if communication_skill_score >= 70:
        strengths.append(f"💪 **Communication ({communication_skill_score}/100)**: Effective verbal & interpersonal expression.")
    else:
        weaknesses.append(("Communication", communication_skill_score, 75.0, "Participate in group discussions, mock interviews, and presentation sessions."))

    if mock_interview_score >= 70:
        strengths.append(f"💪 **Mock Interview ({mock_interview_score}/100)**: Great interview confidence & articulation.")
    else:
        weaknesses.append(("Mock Interview Score", mock_interview_score, 75.0, "Conduct 3-5 technical & behavioral mock interviews with mentors or peers."))

    if projects_count >= 3:
        strengths.append(f"💪 **Project Portfolio ({projects_count} projects)**: Solid practical project experience.")
    else:
        weaknesses.append(("Projects Count", projects_count, 3, "Build at least 1-2 end-to-end projects showcasing full-stack, data engineering, or ML capabilities."))

    if github_repos >= 4:
        strengths.append(f"💪 **GitHub Presence ({github_repos} repos)**: Good code repository showcase.")
    else:
        weaknesses.append(("GitHub Repos", github_repos, 5, "Publish clean, documented project code with README files on GitHub."))

    col_str, col_rec = st.columns(2)
    
    with col_str:
        st.markdown("##### 🌟 Key Skill Strengths")
        if strengths:
            for s in strengths:
                st.success(s)
        else:
            st.info("Focus on boosting core skill scores above 70 to build standout strengths.")
            
    with col_rec:
        st.markdown("##### 🎯 Targeted Skill Recommendations")
        if weaknesses:
            for item, curr, target, rec in weaknesses:
                st.warning(f"**{item}** (Current: `{curr}` ➔ Target: `{target}`)\n\n👉 {rec}")
        else:
            st.success("🎉 Outstanding Profile! All core skill areas meet or exceed target placement benchmarks.")

    st.divider()
    st.markdown("##### 📋 Recommended Target Skill Roadmap")
    
    roadmap_data = pd.DataFrame([
        {"Metric": "Coding Skill Score", "Current": coding_skill_score, "Placed Avg Benchmark": 70.5, "Recommended Target": max(75.0, coding_skill_score)},
        {"Metric": "Aptitude Score", "Current": aptitude_score, "Placed Avg Benchmark": 65.4, "Recommended Target": max(70.0, aptitude_score)},
        {"Metric": "Communication Score", "Current": communication_skill_score, "Placed Avg Benchmark": 68.4, "Recommended Target": max(75.0, communication_skill_score)},
        {"Metric": "Mock Interview Score", "Current": mock_interview_score, "Placed Avg Benchmark": 70.5, "Recommended Target": max(75.0, mock_interview_score)},
        {"Metric": "Projects Count", "Current": projects_count, "Placed Avg Benchmark": 3.1, "Recommended Target": max(3, projects_count)},
        {"Metric": "GitHub Repositories", "Current": github_repos, "Placed Avg Benchmark": 3.5, "Recommended Target": max(5, github_repos)},
        {"Metric": "Internships Completed", "Current": internships_count, "Placed Avg Benchmark": 1.6, "Recommended Target": max(2, internships_count)}
    ])
    
    st.dataframe(roadmap_data, use_container_width=True)

    # Generate Report Data
    student_profile_dict = {
        "Age": age,
        "Gender": gender,
        "Branch": branch,
        "CGPA": cgpa,
        "College Tier": college_tier,
        "Attendance %": f"{attendance_percentage}%",
        "Active Backlogs": backlogs,
        "Coding Skill Score": f"{coding_skill_score}/100",
        "Aptitude Score": f"{aptitude_score}/100",
        "Logical Reasoning": f"{logical_reasoning_score}/100",
        "Communication Score": f"{communication_skill_score}/100",
        "Mock Interview Score": f"{mock_interview_score}/100",
        "Leadership Score": f"{leadership_score}/100",
        "Extracurricular Score": f"{extracurricular_score}/100",
        "Projects Completed": projects_count,
        "GitHub Repositories": github_repos,
        "Certifications": certifications_count,
        "Internships": internships_count,
        "Hackathons": hackathons_participated,
        "LinkedIn Connections": linkedin_connections,
        "Volunteer Experience": volunteer_experience,
    }

    prediction_label = "✅ Likely to be PLACED" if prediction == 1 else "❌ Likely NOT Placed"
    confidence_val = proba * 100 if prediction == 1 else (1 - proba) * 100

    txt_report = generate_txt_report(
        student_profile_dict,
        prediction_label,
        confidence_val,
        persona_name,
        persona_desc,
        overall_skill_index,
        tech_index,
        analytical_index,
        soft_index,
        practical_index,
        strengths,
        weaknesses,
        roadmap_data
    )

    csv_report = roadmap_data.to_csv(index=False)

    st.divider()
    st.markdown("##### 📥 Download Placement & Skill Evaluation Report")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="📄 Download Detailed Evaluation Report (.txt)",
            data=txt_report,
            file_name=f"placement_report_{branch}_{cgpa}.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="📊 Download Target Roadmap (.csv)",
            data=csv_report,
            file_name=f"skill_roadmap_{branch}_{cgpa}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Sidebar Download Section
    st.sidebar.divider()
    st.sidebar.markdown("### 📥 Download Report")
    st.sidebar.download_button(
        label="📄 Download Full Report (.txt)",
        data=txt_report,
        file_name=f"placement_report_{branch}_{cgpa}.txt",
        mime="text/plain",
        key="sb_dl_txt",
        use_container_width=True
    )
    st.sidebar.download_button(
        label="📊 Download Roadmap (.csv)",
        data=csv_report,
        file_name=f"skill_roadmap_{branch}_{cgpa}.csv",
        mime="text/csv",
        key="sb_dl_csv",
        use_container_width=True
    )

st.divider()
st.caption("🎓 Student Placement Predictor & Skill Profiler | Powered by scikit-learn, Plotly, Pandas & Streamlit.")
