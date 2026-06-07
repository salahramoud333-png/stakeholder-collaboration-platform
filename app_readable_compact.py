
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Stakeholder Collaboration Decision-Support Platform
# ============================================================

st.set_page_config(
    page_title="Stakeholder Collaboration Decision-Support Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        background: linear-gradient(135deg, #0f2f57 0%, #165f8f 100%);
        padding: 28px 34px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(15, 47, 87, 0.18);
    }
    .hero h1 {
        font-size: 34px;
        margin-bottom: 8px;
        color: white;
    }
    .hero p {
        font-size: 16px;
        line-height: 1.55;
        margin-bottom: 0;
        color: #eaf4ff;
    }
    .section-card {
        background-color: white;
        padding: 20px 22px;
        border-radius: 16px;
        border: 1px solid #e7edf5;
        box-shadow: 0 4px 18px rgba(12, 31, 56, 0.06);
        margin-bottom: 16px;
    }
    .metric-card {
        background-color: white;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #e7edf5;
        box-shadow: 0 4px 18px rgba(12, 31, 56, 0.06);
        text-align: center;
    }
    .score-big {
        font-size: 42px;
        font-weight: 800;
        color: #0f2f57;
        margin: 0;
    }
    .level-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 700;
        color: white;
        margin-top: 6px;
    }
    .low { background-color: #b42318; }
    .moderate-low { background-color: #f79009; }
    .moderate-high { background-color: #1570ef; }
    .high { background-color: #039855; }
    .rec-card {
        background-color: #ffffff;
        border-left: 6px solid #165f8f;
        padding: 14px 16px;
        border-radius: 12px;
        border-top: 1px solid #e7edf5;
        border-right: 1px solid #e7edf5;
        border-bottom: 1px solid #e7edf5;
        margin-bottom: 10px;
        box-shadow: 0 4px 16px rgba(12, 31, 56, 0.05);
        color: #1f2937 !important;
    }
    .rec-card p, .rec-card div, .rec-card span {
        color: #1f2937 !important;
    }
    .rec-card b {
        color: #0f2f57 !important;
    }
    .compact-rec-card {
        background-color: #ffffff;
        border-left: 5px solid #165f8f;
        padding: 10px 12px;
        border-radius: 10px;
        border-top: 1px solid #e7edf5;
        border-right: 1px solid #e7edf5;
        border-bottom: 1px solid #e7edf5;
        margin-bottom: 8px;
        color: #1f2937 !important;
        box-shadow: 0 2px 10px rgba(12, 31, 56, 0.04);
    }
    .compact-rec-card p, .compact-rec-card div, .compact-rec-card span {
        color: #1f2937 !important;
    }
    .compact-title {
        font-size: 15px;
        font-weight: 800;
        color: #0f2f57 !important;
        margin-bottom: 4px;
    }
    .compact-text {
        font-size: 13px;
        line-height: 1.35;
        margin: 4px 0 0 0;
        color: #1f2937 !important;
    }
    .rec-title {
        font-size: 17px;
        font-weight: 800;
        color: #0f2f57 !important;
        margin-bottom: 6px;
    }
    .tag {
        display: inline-block;
        background-color: #eaf4ff;
        color: #0f2f57 !important;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 5px;
        margin-bottom: 6px;
    }
    .danger-tag {
        background-color: #fff1f3;
        color: #b42318 !important;
    }
    .success-box {
        background-color: #ecfdf3;
        border: 1px solid #abefc6;
        color: #05603a;
        padding: 18px 20px;
        border-radius: 14px;
        margin-bottom: 14px;
    }
    .small-note {
        color: #5d6b82;
        font-size: 13px;
        line-height: 1.45;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0f2f57;
    }
    p, li, label, span, div {
        color: inherit;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Feature definitions
# -----------------------------
ADOPTION_COLS = [
    "BIM_usage_for_coordination",
    "CDE_usage_for_information_sharing",
    "BEP_clarity_and_application",
    "Cloud_based_platform_usage",
    "Organisational_digital_adoption"
]

BARRIER_COLS = [
    "Lack_of_digital_skills",
    "Resistance_to_change",
    "High_cost_of_digital_tools",
    "Lack_of_standardised_procedures",
    "Poor_internet_infrastructure"
]

FEATURE_COLS = ADOPTION_COLS + BARRIER_COLS

DISPLAY_NAMES = {
    "BIM_usage_for_coordination": "BIM Usage for Coordination",
    "CDE_usage_for_information_sharing": "CDE Usage for Information Sharing",
    "BEP_clarity_and_application": "BEP Clarity and Application",
    "Cloud_based_platform_usage": "Cloud-Based Platform Usage",
    "Organisational_digital_adoption": "Organisational Digital Adoption",
    "Lack_of_digital_skills": "Lack of Digital Skills",
    "Resistance_to_change": "Resistance to Change",
    "High_cost_of_digital_tools": "High Cost of Digital Tools",
    "Lack_of_standardised_procedures": "Lack of Standardised Procedures",
    "Poor_internet_infrastructure": "Poor Internet Infrastructure and Technical Limitations"
}

ADOPTION_HELP = {
    "BIM_usage_for_coordination": "1 = no BIM coordination, 5 = systematic BIM coordination with clash detection and multidisciplinary review.",
    "CDE_usage_for_information_sharing": "1 = no shared information environment, 5 = CDE used as the main source of truth for project information.",
    "BEP_clarity_and_application": "1 = no clear BEP, 5 = clear and applied BEP with responsibilities, workflows, and information exchange rules.",
    "Cloud_based_platform_usage": "1 = no cloud collaboration, 5 = cloud platform used consistently for information sharing and coordination.",
    "Organisational_digital_adoption": "1 = very low organisational adoption, 5 = strong leadership support and standardised digital workflows."
}

BARRIER_HELP = {
    "Lack_of_digital_skills": "1 = not a barrier, 5 = severe skills gap affecting digital collaboration.",
    "Resistance_to_change": "1 = not a barrier, 5 = strong resistance to adopting digital workflows.",
    "High_cost_of_digital_tools": "1 = not a barrier, 5 = severe cost barrier affecting implementation.",
    "Lack_of_standardised_procedures": "1 = not a barrier, 5 = severe lack of standard digital procedures.",
    "Poor_internet_infrastructure": "1 = not a barrier, 5 = severe technical/internet limitation."
}

RECOMMENDATION_LIBRARY = {
    "BIM_usage_for_coordination": {
        "category": "Digitalization adoption weakness",
        "why": "Weak BIM coordination increases the likelihood of design clashes, unresolved interdisciplinary conflicts, RFIs, rework, and delays in decision-making between consultants, contractors, and site teams.",
        "action": "Use federated BIM models for architectural, structural, and MEP coordination. Schedule regular model coordination meetings before construction, run clash detection, assign clash ownership, and track issue closure before site execution.",
        "improvement": "Better BIM coordination supports earlier conflict detection, fewer RFIs, clearer design communication, and smoother collaboration between design and construction stakeholders."
    },
    "CDE_usage_for_information_sharing": {
        "category": "Digitalization adoption weakness",
        "why": "Poor information sharing causes stakeholders to work from different document versions, which leads to approval delays, duplicated communication, unclear responsibility, and inconsistent project decisions.",
        "action": "Adopt a Common Data Environment as the single source of truth for drawings, BIM models, RFIs, approvals, submittals, meeting records, and correspondence. Apply clear permissions, version control, document status codes, and approval workflows.",
        "improvement": "A well-managed CDE improves transparency, reduces document confusion, supports faster approvals, and ensures that owners, consultants, contractors, subcontractors, and BIM teams work with current information."
    },
    "BEP_clarity_and_application": {
        "category": "Digitalization adoption weakness",
        "why": "Unclear BEP application creates role ambiguity and weak coordination because stakeholders may not know what information should be delivered, when it should be delivered, and who is responsible for it.",
        "action": "Prepare and enforce a project-specific BIM Execution Plan defining BIM uses, model responsibilities, information exchange milestones, software platforms, naming conventions, LOD requirements, review cycles, and approval procedures.",
        "improvement": "A clear BEP improves accountability, standardises digital collaboration, reduces coordination gaps, and creates a shared workflow for managing BIM information throughout the project."
    },
    "Cloud_based_platform_usage": {
        "category": "Digitalization adoption weakness",
        "why": "Limited cloud collaboration restricts real-time access to project information, especially when project parties are working from different offices, sites, or organisations.",
        "action": "Introduce cloud-based collaboration progressively. Start with document management and model sharing, then expand to RFIs, submittals, issue tracking, approvals, dashboards, and coordination records.",
        "improvement": "Cloud collaboration supports faster communication, improved stakeholder engagement, better information access, and more responsive decision-making across distributed project teams."
    },
    "Organisational_digital_adoption": {
        "category": "Digitalization adoption weakness",
        "why": "Low organisational digital adoption prevents BIM, CDE, BEP, and cloud platforms from becoming consistent project practices. Digital tools become isolated efforts instead of coordinated organisational workflows.",
        "action": "Create a digital transformation plan supported by senior management. Assign BIM/digital champions, standardise workflows, define responsibilities, provide training, and monitor adoption across projects.",
        "improvement": "Stronger organisational digital adoption improves consistency, supports long-term capability building, and helps digitalization become a normal part of project delivery rather than an optional activity."
    },
    "Lack_of_digital_skills": {
        "category": "Significant implementation barrier",
        "why": "A skills gap limits the ability of project teams to use BIM models, manage CDE workflows, apply BEPs, and communicate effectively through digital platforms.",
        "action": "Deliver role-based training for project managers, site engineers, design engineers, BIM coordinators, quantity surveyors, and document controllers. Training should cover BIM coordination, CDE use, BEP implementation, cloud workflows, and digital information management.",
        "improvement": "Improved skills increase confidence, reduce mistakes in digital workflows, and allow stakeholders to use digital tools as part of normal project coordination."
    },
    "Resistance_to_change": {
        "category": "Significant implementation barrier",
        "why": "Resistance to change slows digitalization because teams may continue using informal communication, paper-based records, and fragmented email workflows even when digital tools are available.",
        "action": "Use leadership support, pilot implementation, internal digital champions, and clear communication of benefits. Begin with high-impact workflows such as faster approvals, RFI tracking, document control, and clash resolution.",
        "improvement": "Reducing resistance supports smoother adoption, stronger user engagement, and better alignment between traditional project teams and digital collaboration practices."
    },
    "High_cost_of_digital_tools": {
        "category": "Significant implementation barrier",
        "why": "High software, training, and implementation costs can prevent organisations from adopting BIM coordination tools, CDE platforms, and cloud collaboration systems.",
        "action": "Use phased implementation. Prioritise tools with direct collaboration value first, such as BIM coordination software and CDE platforms. Expand later to advanced dashboards, automation, and integrated reporting after demonstrating value.",
        "improvement": "A phased cost strategy reduces financial pressure, supports gradual capability building, and helps organisations justify digital investment through coordination benefits and reduced rework."
    },
    "Lack_of_standardised_procedures": {
        "category": "Significant implementation barrier",
        "why": "Without standard procedures, each project team may use different naming rules, approval steps, model exchange methods, and responsibilities, causing confusion and inconsistent collaboration.",
        "action": "Develop standardised digital procedures including BEP templates, CDE folder structures, document naming conventions, model exchange protocols, RFI workflows, approval cycles, and responsibility matrices.",
        "improvement": "Standard procedures improve interoperability, reduce communication ambiguity, and make digital collaboration repeatable across projects and organisations."
    },
    "Poor_internet_infrastructure": {
        "category": "Significant implementation barrier",
        "why": "Poor internet reliability, weak hardware capability, or technical limitations reduce the effectiveness of cloud-based BIM and CDE workflows, especially for large model files and real-time access.",
        "action": "Assess technical readiness before implementation. Improve internet connectivity, hardware capacity, cloud access, data backup procedures, cybersecurity arrangements, and technical support for project teams.",
        "improvement": "Better technical readiness supports reliable access to digital information, reduces delays in model sharing, and improves the practical usability of cloud-based collaboration systems."
    }
}

# -----------------------------
# Utility functions
# -----------------------------
def collaboration_level(score: float) -> str:
    if score < 2:
        return "Low"
    elif score < 3:
        return "Moderate-Low"
    elif score < 4:
        return "Moderate-High"
    return "High"

def level_class(level: str) -> str:
    return {
        "Low": "low",
        "Moderate-Low": "moderate-low",
        "Moderate-High": "moderate-high",
        "High": "high"
    }.get(level, "moderate-high")

def load_model():
    model_path = "stakeholder_collaboration_model.pkl"
    features_path = "model_features.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found: stakeholder_collaboration_model.pkl")
        st.stop()

   with open(model_path, "rb") as f:
    model = pickle.load(f)

if os.path.exists(features_path):
    with open(features_path, "rb") as f:
        features = pickle.load(f)
else:
    features = FEATURE_COLS

    return model, features

def compact_action(text: str) -> str:
    # First sentence/compact summary for Word screenshots
    replacements = {
        "Use federated BIM models for architectural, structural, and MEP coordination. Schedule regular model coordination meetings before construction, run clash detection, assign clash ownership, and track issue closure before site execution.": "Use federated BIM models for architectural, structural, and MEP coordination; run clash detection before construction and track issue closure to reduce RFIs and rework.",
        "Adopt a Common Data Environment as the single source of truth for drawings, BIM models, RFIs, approvals, submittals, meeting records, and correspondence. Apply clear permissions, version control, document status codes, and approval workflows.": "Adopt a CDE as the single source of truth for drawings, models, RFIs, approvals, and correspondence, with version control and approval workflows.",
        "Prepare and enforce a project-specific BIM Execution Plan defining BIM uses, model responsibilities, information exchange milestones, software platforms, naming conventions, LOD requirements, review cycles, and approval procedures.": "Prepare and enforce a project-specific BEP defining BIM uses, responsibilities, exchange milestones, naming rules, review cycles, and approval procedures.",
        "Introduce cloud-based collaboration progressively. Start with document management and model sharing, then expand to RFIs, submittals, issue tracking, approvals, dashboards, and coordination records.": "Introduce cloud collaboration progressively, starting with document management and model sharing, then expanding to RFIs, approvals, issue tracking, and dashboards.",
        "Create a digital transformation plan supported by senior management. Assign BIM/digital champions, standardise workflows, define responsibilities, provide training, and monitor adoption across projects.": "Create a management-supported digital transformation plan, assign BIM/digital champions, standardise workflows, provide training, and monitor adoption.",
        "Deliver role-based training for project managers, site engineers, design engineers, BIM coordinators, quantity surveyors, and document controllers. Training should cover BIM coordination, CDE use, BEP implementation, cloud workflows, and digital information management.": "Deliver role-based training covering BIM coordination, CDE use, BEP implementation, cloud workflows, and digital information management for all key project roles.",
        "Use leadership support, pilot implementation, internal digital champions, and clear communication of benefits. Begin with high-impact workflows such as faster approvals, RFI tracking, document control, and clash resolution.": "Use leadership support, pilot implementation, digital champions, and quick-win workflows such as faster approvals, RFI tracking, document control, and clash resolution.",
        "Use phased implementation. Prioritise tools with direct collaboration value first, such as BIM coordination software and CDE platforms. Expand later to advanced dashboards, automation, and integrated reporting after demonstrating value.": "Use phased implementation by prioritising BIM coordination software and CDE platforms first, then expanding to dashboards, automation, and reporting after value is demonstrated.",
        "Develop standardised digital procedures including BEP templates, CDE folder structures, document naming conventions, model exchange protocols, RFI workflows, approval cycles, and responsibility matrices.": "Develop standardised procedures including BEP templates, CDE folder structures, naming conventions, model exchange protocols, RFI workflows, approvals, and responsibility matrices.",
        "Assess technical readiness before implementation. Improve internet connectivity, hardware capacity, cloud access, data backup procedures, cybersecurity arrangements, and technical support for project teams.": "Assess technical readiness and improve internet reliability, hardware capacity, cloud access, backups, cybersecurity, and support before relying on cloud-based workflows."
    }
    return replacements.get(text, text)

def generate_recommendations(input_df: pd.DataFrame) -> list:
    output = []

    for col in ADOPTION_COLS:
        value = int(input_df[col].iloc[0])
        if value < 3:
            item = RECOMMENDATION_LIBRARY[col].copy()
            item.update({
                "variable": DISPLAY_NAMES[col],
                "value": value,
                "rule": "Adoption variable below 3"
            })
            output.append(item)

    for col in BARRIER_COLS:
        value = int(input_df[col].iloc[0])
        if value > 3:
            item = RECOMMENDATION_LIBRARY[col].copy()
            item.update({
                "variable": DISPLAY_NAMES[col],
                "value": value,
                "rule": "Barrier variable above 3"
            })
            output.append(item)

    return output

def build_radar(values: dict):
    labels = [DISPLAY_NAMES[c] for c in FEATURE_COLS]
    raw_values = [values[c] for c in FEATURE_COLS]

    # For barriers, invert the value for visual profile quality:
    # Lower barrier = stronger profile, higher barrier = weaker profile.
    profile_values = []
    for c in FEATURE_COLS:
        if c in ADOPTION_COLS:
            profile_values.append(values[c])
        else:
            profile_values.append(6 - values[c])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    profile_values += profile_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, profile_values, linewidth=2)
    ax.fill(angles, profile_values, alpha=0.18)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Digital Collaboration Profile", pad=24, fontsize=14, fontweight="bold")
    return fig

# -----------------------------
# Load model
# -----------------------------
model, model_features = load_model()

# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Stakeholder Collaboration Decision-Support Platform</h1>
        <p>
        Assess the digital collaboration profile of a construction project using key variables related to BIM coordination,
        Common Data Environment usage, BIM Execution Plan application, cloud-based collaboration, organisational digital adoption,
        and implementation barriers.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Project Input Values")
    st.caption("Set each variable from 1 to 5 based on the project condition.")
    compact_mode = st.toggle("Word screenshot mode", value=True, help="Shows shorter, clearer recommendation cards for snipping and inserting into Word.")

    st.subheader("Digitalization Adoption Variables")
    input_values = {}

    for col in ADOPTION_COLS:
        input_values[col] = st.slider(
            DISPLAY_NAMES[col],
            min_value=1,
            max_value=5,
            value=3,
            help=ADOPTION_HELP[col]
        )

    st.divider()

    st.subheader("Implementation Barrier Variables")
    for col in BARRIER_COLS:
        input_values[col] = st.slider(
            DISPLAY_NAMES[col],
            min_value=1,
            max_value=5,
            value=3,
            help=BARRIER_HELP[col]
        )

# -----------------------------
# Prepare prediction
# -----------------------------
input_df = pd.DataFrame([input_values])

# Ensure correct model feature order
try:
    prediction_input = input_df[model_features]
except Exception:
    prediction_input = input_df[FEATURE_COLS]

predicted_score = float(model.predict(prediction_input)[0])
predicted_score = max(1, min(5, predicted_score))
predicted_level = collaboration_level(predicted_score)

adoption_average = np.mean([input_values[c] for c in ADOPTION_COLS])
barrier_average = np.mean([input_values[c] for c in BARRIER_COLS])
recommendation_items = generate_recommendations(input_df)

# -----------------------------
# Main dashboard
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <p class="small-note">Predicted Collaboration Score</p>
            <p class="score-big">{predicted_score:.2f}</p>
            <p class="small-note">Scale: 1 = weakest, 5 = strongest</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <p class="small-note">Predicted Collaboration Level</p>
            <span class="level-badge {level_class(predicted_level)}">{predicted_level}</span>
            <p class="small-note" style="margin-top:12px;">Generated from the entered project profile</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <p class="small-note">Flagged Improvement Areas</p>
            <p class="score-big">{len(recommendation_items)}</p>
            <p class="small-note">Weak adoption variables and significant barriers</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("")

left, right = st.columns([1.15, 0.85])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Project Digital Collaboration Profile")
    fig = build_radar(input_values)
    st.pyplot(fig, use_container_width=True)
    st.markdown(
        """
        <p class="small-note">
        For adoption variables, higher values indicate stronger implementation. For barrier variables, the radar profile reverses
        the score so that lower barriers appear as stronger conditions.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Profile Summary")

    st.write("**Average Digitalization Adoption:**", f"{adoption_average:.2f} / 5")
    st.progress(adoption_average / 5)

    st.write("**Average Implementation Barrier Severity:**", f"{barrier_average:.2f} / 5")
    st.progress(barrier_average / 5)

    if adoption_average >= 4 and barrier_average <= 2:
        st.success("Strong digital collaboration profile with low implementation barriers.")
    elif adoption_average < 3 and barrier_average > 3:
        st.error("Weak digital collaboration profile with significant implementation barriers.")
    else:
        st.info("Developing digital collaboration profile with selected areas requiring improvement.")

    st.markdown(
        """
        <p class="small-note">
        The platform evaluates the combined effect of digital adoption and implementation barriers to support project-level
        collaboration improvement decisions.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Recommendations
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Targeted Recommendations")

if len(recommendation_items) == 0:
    st.markdown(
        """
        <div class="success-box">
            <b>No major weakness or significant barrier was flagged.</b><br>
            Maintain current digital collaboration practices, continue periodic BIM coordination reviews, monitor CDE usage,
            update the BIM Execution Plan when project conditions change, and continue role-based digital training to sustain performance.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.write(
        "The following recommendations are generated from the entered project values. "
        "Each recommendation is linked to a specific weakness or barrier."
    )

    for i, item in enumerate(recommendation_items, start=1):
        tag_class = "danger-tag" if "barrier" in item["category"].lower() else ""

        if compact_mode:
            st.markdown(
                f"""
                <div class="compact-rec-card">
                    <div class="compact-title">{i}. {item['variable']}</div>
                    <span class="tag {tag_class}">{item['category']}</span>
                    <span class="tag">Value: {item['value']}</span>
                    <p class="compact-text"><b>Recommendation:</b> {compact_action(item['action'])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-title">{i}. {item['variable']}</div>
                    <span class="tag {tag_class}">{item['category']}</span>
                    <span class="tag">Entered value: {item['value']}</span>
                    <span class="tag">Rule: {item['rule']}</span>
                    <p><b>Why this matters:</b><br>{item['why']}</p>
                    <p><b>Recommended action:</b><br>{item['action']}</p>
                    <p><b>Expected improvement:</b><br>{item['improvement']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Detailed input table
# -----------------------------
with st.expander("View entered project values"):
    display_df = pd.DataFrame({
        "Variable": [DISPLAY_NAMES[c] for c in FEATURE_COLS],
        "Category": ["Digitalization Adoption"] * len(ADOPTION_COLS) + ["Implementation Barrier"] * len(BARRIER_COLS),
        "Entered Value": [input_values[c] for c in FEATURE_COLS]
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)
