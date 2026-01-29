import streamlit as st
from openai import OpenAI
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Candidate Suitability Analysis",
    page_icon="📊",
    layout="wide"
)

# --- OpenAI Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Helper Functions ---
def generate_transcript():
    """Generate a formatted transcript of the interview with emotional analysis."""
    if 'messages' not in st.session_state or 'vacancy_text' not in st.session_state:
        return None

    transcript = f"""INTERVIEW TRANSCRIPT
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
JOB VACANCY DETAILS
{'='*80}

{st.session_state.vacancy_text}

{'='*80}
INTERVIEW CONVERSATION
{'='*80}

"""

    for i, msg in enumerate(st.session_state.messages, 1):
        role = "INTERVIEWER (AI)" if msg["role"] == "assistant" else "CANDIDATE"
        transcript += f"\n[Message {i}] {role}:\n"
        transcript += f"{msg['content']}\n"

        # Add emotional analysis if available for candidate responses
        if msg["role"] == "user" and "emotion" in msg:
            emotion_data = msg["emotion"]
            transcript += f"\n[EMOTIONAL ANALYSIS]\n"
            transcript += f"  - Detected Emotion: {emotion_data.get('emotion', 'N/A')}\n"
            transcript += f"  - Confidence Level: {emotion_data.get('confidence', 'N/A')}\n"
            transcript += f"  - Analysis: {emotion_data.get('explanation', 'N/A')}\n"

        transcript += "\n" + "-"*80 + "\n"

    transcript += f"\n{'='*80}\n"
    transcript += f"END OF TRANSCRIPT\n"
    transcript += f"Total Messages: {len(st.session_state.messages)}\n"
    transcript += f"{'='*80}\n"

    return transcript

def analyze_candidate_suitability(transcript):
    """Analyze the interview transcript and generate a suitability report."""
    analysis_prompt = f"""You are an expert HR analyst. Review the following interview transcript and provide a comprehensive suitability assessment for the candidate.

{transcript}

Please provide a detailed analysis in the following format:

## CANDIDATE SUITABILITY REPORT

### Overall Suitability Score
[Rate from 1-10 with justification]

### Strengths
[List 3-5 key strengths demonstrated during the interview]

### Areas of Concern
[List any concerns or weaknesses identified]

### Technical Skills Assessment
[Evaluate technical competencies based on job requirements]

### Soft Skills & Cultural Fit
[Assess communication, enthusiasm, confidence, and alignment with company culture]

### Emotional Intelligence Observations
[Analyze the candidate's emotional patterns throughout the interview, considering the emotional analysis data provided]

### Key Insights from Responses
[Highlight notable answers or patterns in responses]

### Recommendation
[Provide a clear hiring recommendation: Strongly Recommend, Recommend, Maybe, or Do Not Recommend]

### Next Steps
[Suggest follow-up actions or additional assessments if needed]
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content

# --- Main Page UI ---
st.title("📊 Candidate Suitability Analysis")
st.markdown("---")

# Check if interview data exists
if 'messages' not in st.session_state or len(st.session_state.messages) <= 1:
    st.warning("⚠️ No interview data available.")
    st.info("Please conduct an interview first on the main page before generating a suitability report.")
    st.page_link("job_agent_demo.py", label="← Go to Interview Page", icon="🏠")
else:
    # Display interview statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Messages", len(st.session_state.messages))

    with col2:
        candidate_responses = sum(1 for m in st.session_state.messages if m["role"] == "user")
        st.metric("Candidate Responses", candidate_responses)

    with col3:
        responses_with_emotion = sum(1 for m in st.session_state.messages if m["role"] == "user" and "emotion" in m)
        st.metric("Emotional Data Points", responses_with_emotion)

    st.markdown("---")

    # Generate Analysis Section
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Generate AI Analysis Report")
        st.write("Click the button below to generate a comprehensive suitability analysis of the candidate based on the interview transcript and emotional data.")

    with col_right:
        if st.button("🤖 Generate Report", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing interview transcript... This may take 30-60 seconds..."):
                transcript = generate_transcript()
                st.session_state.suitability_report = analyze_candidate_suitability(transcript)
                st.success("✅ Suitability report generated successfully!")
                st.rerun()

    # Display report if available
    if 'suitability_report' in st.session_state and st.session_state.suitability_report:
        st.markdown("---")

        # Report header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">📋 Candidate Suitability Report</h2>
            <p style="color: white; margin: 5px 0; text-align: center; opacity: 0.9;">AI-Generated Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # Report content in a styled container
        st.markdown("""
        <style>
        .report-container {
            background-color: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown(st.session_state.suitability_report)

        # Action buttons
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

        with col_btn1:
            st.download_button(
                label="📥 Download Report (TXT)",
                data=st.session_state.suitability_report,
                file_name=f"suitability_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col_btn2:
            transcript = generate_transcript()
            st.download_button(
                label="📄 Download Transcript",
                data=transcript,
                file_name=f"interview_transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col_btn3:
            if st.button("🔄 Generate New Report", use_container_width=True):
                st.session_state.suitability_report = None
                st.rerun()

    else:
        # Placeholder when no report is generated
        st.info("💡 Generate a report to see the AI analysis here.")

        # Show preview of available data
        with st.expander("📋 Preview Interview Transcript"):
            transcript = generate_transcript()
            st.text_area("Transcript", value=transcript, height=400, disabled=True)

# Footer
st.markdown("---")
st.caption("Powered by OpenAI GPT-4o | Emotional Analysis by GPT-4o-mini")
