import streamlit as st
from openai import OpenAI
from elevenlabs.client import ElevenLabs
import base64
import random
import streamlit.components.v1 as components
from pypdf import PdfReader
from docx import Document
import datetime
import numpy as np
from scipy.io import wavfile
import librosa

# --- Page Configuration ---
st.set_page_config(
    page_title="AI HR Interview Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- OpenAI Setup ---
client = OpenAI(api_key="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")

# --- ElevenLabs Setup ---
elevenlabs_client = ElevenLabs(api_key="sk_c3fda08735a26f68ab62bf5ef7c37aabb4699998df8156a6")

# --- Default Vacancy Data ---
default_vacancy_text = """
Job Title: Software Engineer
Location: Remote
Responsibilities:
- Develop web applications
- Collaborate with the product team
Requirements:
- 3+ years experience in Python
- Knowledge of React
- Strong problem-solving skills
Benefits:
- Flexible hours
- Health insurance
"""

# --- Audio Feature Extraction Function ---
def extract_audio_features(audio_path):
    """Extract acoustic features from audio for emotion analysis."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=16000)

        # Extract features
        # 1. Pitch (fundamental frequency)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # 2. Energy/Loudness
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        # 3. Speaking rate (zero crossing rate as proxy)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # 4. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_mean = np.mean(spectral_centroid)

        # 5. Duration
        duration = librosa.get_duration(y=y, sr=sr)

        features = {
            'pitch_mean': float(pitch_mean),
            'pitch_variation': float(pitch_std),
            'energy_mean': float(energy_mean),
            'energy_variation': float(energy_std),
            'speaking_rate': float(zcr_mean),
            'spectral_brightness': float(spectral_mean),
            'duration': float(duration)
        }

        return features
    except Exception as e:
        st.warning(f"Could not extract audio features: {e}")
        return None

# --- Enhanced Emotion Analysis Function ---
def analyze_emotion_with_audio(text, audio_features=None):
    """Analyze emotional tone using both text content and audio characteristics."""

    # Build prompt with audio context if available
    if audio_features:
        audio_context = f"""
Audio Analysis:
- Pitch (tone): {'High' if audio_features['pitch_mean'] > 200 else 'Medium' if audio_features['pitch_mean'] > 100 else 'Low'} (mean: {audio_features['pitch_mean']:.1f} Hz)
- Pitch Variation: {'Highly varied' if audio_features['pitch_variation'] > 50 else 'Moderate variation' if audio_features['pitch_variation'] > 20 else 'Monotone'}
- Voice Energy: {'Loud/Energetic' if audio_features['energy_mean'] > 0.1 else 'Moderate' if audio_features['energy_mean'] > 0.05 else 'Soft/Quiet'}
- Speaking Rate: {'Fast' if audio_features['speaking_rate'] > 0.15 else 'Moderate' if audio_features['speaking_rate'] > 0.08 else 'Slow'}
- Voice Quality: {'Bright/Clear' if audio_features['spectral_brightness'] > 2000 else 'Warm/Mellow'}
- Duration: {audio_features['duration']:.1f} seconds
"""
    else:
        audio_context = "Audio analysis not available - analyzing text only."

    emotion_prompt = f"""You are an expert in emotional intelligence and voice analysis. Analyze the following interview response considering BOTH the spoken words AND the voice characteristics.

Text Content: "{text}"

{audio_context}

Based on both the text content and vocal characteristics (tone, pitch, energy, speaking rate), provide a comprehensive emotional analysis:

1. Primary emotion (e.g., confident, nervous, enthusiastic, uncertain, calm, anxious, excited, frustrated, stressed, comfortable)
2. Confidence level (High, Medium, Low)
3. Brief explanation considering both what was said and HOW it was said (voice tone, pitch, energy)

Respond in this exact format:
Emotion: [emotion]
Confidence: [level]
Explanation: [brief explanation mentioning both text and vocal cues]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": emotion_prompt}],
        temperature=0.3
    )

    analysis = response.choices[0].message.content

    # Parse the response
    emotion_data = {}
    for line in analysis.split('\n'):
        if line.startswith('Emotion:'):
            emotion_data['emotion'] = line.replace('Emotion:', '').strip()
        elif line.startswith('Confidence:'):
            emotion_data['confidence'] = line.replace('Confidence:', '').strip()
        elif line.startswith('Explanation:'):
            emotion_data['explanation'] = line.replace('Explanation:', '').strip()

    # Add audio features to the emotion data
    if audio_features:
        emotion_data['audio_features'] = audio_features

    return emotion_data

# --- Transcript Generation Function ---
def generate_transcript():
    """Generate a formatted transcript of the interview with emotional analysis."""
    import datetime

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

            # Add audio features if available
            if 'audio_features' in emotion_data:
                audio_feat = emotion_data['audio_features']
                transcript += f"\n[VOCAL CHARACTERISTICS]\n"
                transcript += f"  - Pitch: {audio_feat.get('pitch_mean', 0):.1f} Hz (variation: {audio_feat.get('pitch_variation', 0):.1f})\n"
                transcript += f"  - Energy Level: {audio_feat.get('energy_mean', 0):.3f}\n"
                transcript += f"  - Speaking Rate: {audio_feat.get('speaking_rate', 0):.3f}\n"
                transcript += f"  - Voice Brightness: {audio_feat.get('spectral_brightness', 0):.1f} Hz\n"
                transcript += f"  - Response Duration: {audio_feat.get('duration', 0):.1f} seconds\n"

        transcript += "\n" + "-"*80 + "\n"

    transcript += f"\n{'='*80}\n"
    transcript += f"END OF TRANSCRIPT\n"
    transcript += f"Total Messages: {len(st.session_state.messages)}\n"
    transcript += f"{'='*80}\n"

    return transcript

# --- Candidate Suitability Analysis Function ---
def analyze_candidate_suitability():
    """Analyze the interview transcript and generate a suitability report."""
    transcript = generate_transcript()

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

# --- Initialize Session State ---
if 'vacancy_text' not in st.session_state:
    st.session_state.vacancy_text = default_vacancy_text

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'play_audio' not in st.session_state:
    st.session_state.play_audio = False

if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""

if 'audio_recorded' not in st.session_state:
    st.session_state.audio_recorded = None

if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = None

if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False

if 'suitability_report' not in st.session_state:
    st.session_state.suitability_report = None

if 'conversation_mode' not in st.session_state:
    st.session_state.conversation_mode = False

if 'processing_audio' not in st.session_state:
    st.session_state.processing_audio = False

if 'last_audio_hash' not in st.session_state:
    st.session_state.last_audio_hash = None

# Initialize with starting message if no messages
if not st.session_state.messages:
    initial_text = "Hey I am an AI agent that is going to interview you. Can you introduce yourself?"
    # Generate audio for initial message
    audio = elevenlabs_client.text_to_speech.convert(text=initial_text, voice_id="21m00Tcm4TlvDq8ikWAM")
    audio_bytes = b''.join(audio)
    audio_base64 = base64.b64encode(audio_bytes).decode()
    st.session_state.messages.append({"role": "assistant", "content": initial_text, "audio": audio_base64})
    st.session_state.play_audio = True

# --- Sidebar for Vacancy Input ---
with st.sidebar:
    st.title("Job Vacancy Setup")
    st.write("Set up the job vacancy details for the interview.")
    option = st.selectbox("Choose vacancy input method:", ["Default text", "Type text", "Upload file"])
    
    if option == "Type text":
        st.session_state.vacancy_text = st.text_area("Enter vacancy text:", value=st.session_state.vacancy_text, height=200)
    elif option == "Upload file":
        uploaded_file = st.file_uploader("Upload PDF or DOCX file", type=["pdf", "docx"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                st.session_state.vacancy_text = text
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])
                st.session_state.vacancy_text = text
            st.success("File uploaded and parsed successfully!")
    
    st.subheader("Current Vacancy")
    st.text_area("Vacancy Text:", value=st.session_state.vacancy_text, height=150, disabled=True)

    # Transcript Section
    st.markdown("---")
    st.subheader("Interview Transcript")

    if len(st.session_state.messages) > 1:  # More than just the initial greeting
        st.write(f"Messages recorded: {len(st.session_state.messages)}")

        # Generate and display transcript
        transcript_text = generate_transcript()

        # Download button
        st.download_button(
            label="Download Transcript (TXT)",
            data=transcript_text,
            file_name=f"interview_transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download the full interview transcript with emotional analysis for further AI processing"
        )

        # Preview transcript in expander
        with st.expander("Preview Transcript"):
            st.text_area("Transcript Preview", value=transcript_text, height=300, disabled=True)

        # Link to analysis page
        st.markdown("---")
        st.info("📊 Ready to analyze? Go to the **Suitability Analysis** page from the sidebar!")
    else:
        st.info("Transcript will be available after the interview begins.")

# --- Main Chat Interface ---
st.title("AI HR Interview Assistant")

# Mode toggle
col_toggle1, col_toggle2 = st.columns([3, 1])
with col_toggle1:
    st.write("Engage in an interview for the job vacancy. The AI will ask questions to assess your fit for the role.")
with col_toggle2:
    conversation_mode = st.toggle(
        "💬 Conversation Mode",
        value=st.session_state.conversation_mode,
        help="Enable for natural conversation without manual review steps"
    )
    if conversation_mode != st.session_state.conversation_mode:
        st.session_state.conversation_mode = conversation_mode
        st.rerun()

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if st.session_state.play_audio and i == len(st.session_state.messages) - 1 and message["role"] == "assistant" and "audio" in message:
            audio_id = random.randint(0, 1000000)
            html = f'''
            <audio id="audio_{audio_id}"><source src="data:audio/mp3;base64,{message["audio"]}" type="audio/mp3"></audio>
            <script>
            var audio = document.getElementById('audio_{audio_id}');
            audio.play().catch(function(e) {{
                console.log('Autoplay blocked:', e);
            }});
            </script>
            '''
            components.html(html, height=50)
            st.session_state.play_audio = False

# Voice input section
st.subheader("Voice Response")

if st.session_state.conversation_mode:
    # Conversation Mode: Simple audio input with auto-submit
    audio_input = st.audio_input("🎤 Record your response (auto-submits after transcription)", key="conv_audio")

    if audio_input:
        # Generate hash of audio to detect if it's new
        import hashlib
        audio_input.seek(0)
        audio_hash = hashlib.md5(audio_input.read()).hexdigest()
        audio_input.seek(0)  # Reset for reading again

        # Only process if this is a new audio recording
        if audio_hash != st.session_state.last_audio_hash and not st.session_state.processing_audio:
            # Mark as processing and store hash
            st.session_state.processing_audio = True
            st.session_state.last_audio_hash = audio_hash

            with st.spinner("Processing your response..."):
                # Save audio to temporary file
                audio_input.seek(0)
                audio_bytes = audio_input.read()

                # Create a temporary file for Whisper API
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file_path = tmp_file.name

                # Transcribe with Whisper API
                with open(tmp_file_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )

                transcribed_text = transcript.text

                # Extract audio features for emotion analysis (still done in background)
                audio_features = extract_audio_features(tmp_file_path)
                emotion_data = analyze_emotion_with_audio(transcribed_text, audio_features)

                # Clean up temporary file
                os.unlink(tmp_file_path)

                # Auto-submit: Add user message
                message_data = {"role": "user", "content": transcribed_text}
                message_data["emotion"] = emotion_data
                st.session_state.messages.append(message_data)

                # Prepare emotion context for AI
                emotion = emotion_data.get('emotion', '')
                confidence = emotion_data.get('confidence', '')
                emotion_context = f"\n\nNote: The candidate's emotional state during this response was detected as '{emotion}' with '{confidence}' confidence. Consider this in your assessment and follow-up questions."

                # Prepare messages for API
                system_message = f"You are an HR representative conducting an interview for the following job vacancy. Ask relevant questions to assess the candidate's fit for the role based on the vacancy details. Engage in a natural conversation and ask follow-up questions as appropriate.\n\nJob Vacancy:\n{st.session_state.vacancy_text}"
                api_messages = [{"role": "system", "content": system_message}]

                # Add conversation history
                for i, m in enumerate(st.session_state.messages):
                    msg_content = m["content"]
                    if i == len(st.session_state.messages) - 1 and m["role"] == "user" and "emotion" in m:
                        msg_content += emotion_context
                    api_messages.append({"role": m["role"], "content": msg_content})

                # Get AI response
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=api_messages
                )
                answer = response.choices[0].message.content

                # Generate audio
                audio = elevenlabs_client.text_to_speech.convert(text=answer, voice_id="21m00Tcm4TlvDq8ikWAM")
                audio_bytes = b''.join(audio)
                audio_base64 = base64.b64encode(audio_bytes).decode()

                # Add assistant message with audio
                st.session_state.messages.append({"role": "assistant", "content": answer, "audio": audio_base64})

                # Trigger audio playback
                st.session_state.play_audio = True

                # Reset processing flag for next audio
                st.session_state.processing_audio = False

                # Rerun
                st.rerun()
    elif not audio_input:
        # Reset flags when audio widget is cleared
        if st.session_state.processing_audio:
            st.session_state.processing_audio = False
        if st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = None

else:
    # Review Mode: Manual transcription and editing
    col1, col2 = st.columns([2, 1])

    with col1:
        audio_input = st.audio_input("Record your response")

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Transcribe Audio", disabled=(audio_input is None)):
            if audio_input:
                with st.spinner("Transcribing..."):
                    # Save audio to temporary file
                    audio_input.seek(0)
                    audio_bytes = audio_input.read()

                    # Create a temporary file for Whisper API
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_file_path = tmp_file.name

                    # Transcribe with Whisper API
                    with open(tmp_file_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )

                    st.session_state.transcribed_text = transcript.text

                    # Extract audio features for emotion analysis
                    audio_features = extract_audio_features(tmp_file_path)

                    # Analyze emotion from both text and audio
                    emotion_data = analyze_emotion_with_audio(transcript.text, audio_features)
                    st.session_state.current_emotion = emotion_data

                    # Clean up temporary file
                    import os
                    os.unlink(tmp_file_path)

                    st.success("Audio transcribed successfully!")
                    st.rerun()

# Only show emotion analysis and edit controls in Review Mode
if not st.session_state.conversation_mode:
    # Display emotion analysis if available
    if st.session_state.current_emotion:
        emotion_data = st.session_state.current_emotion

        # Color coding based on confidence
        confidence_colors = {
            'High': '#28a745',  # Green
            'Medium': '#fd7e14',  # Orange
            'Low': '#dc3545'  # Red
        }
        confidence = emotion_data.get('confidence', 'Unknown')
        color = confidence_colors.get(confidence, '#6c757d')  # Gray for unknown

        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #0066cc;">
            <h4 style="margin: 0 0 10px 0; color: #1a1a1a;">Emotional Analysis</h4>
            <p style="margin: 5px 0; color: #2d2d2d;"><strong>Emotion:</strong> {emotion_data.get('emotion', 'Unknown')}</p>
            <p style="margin: 5px 0; color: #2d2d2d;"><strong>Confidence Level:</strong> <span style="color: {color}; font-weight: bold;">{confidence}</span></p>
            <p style="margin: 5px 0; font-style: italic; color: #4a4a4a;">{emotion_data.get('explanation', '')}</p>
        </div>
        """, unsafe_allow_html=True)

    # Editable text area for transcribed or typed response
    st.subheader("Edit Your Response")
    user_response = st.text_area(
        "You can edit the transcribed text or type your response directly:",
        value=st.session_state.transcribed_text,
        height=150,
        placeholder="Type your response here or use the microphone above..."
    )

    # Submit button
    if st.button("Submit Response", type="primary", disabled=(not user_response.strip())):
        if user_response.strip():
            # Add user message with emotion data
            message_data = {"role": "user", "content": user_response}
            if st.session_state.current_emotion:
                message_data["emotion"] = st.session_state.current_emotion
            st.session_state.messages.append(message_data)

            # Clear transcribed text and emotion for next response
            st.session_state.transcribed_text = ""
            emotion_context = ""
            if st.session_state.current_emotion:
                emotion = st.session_state.current_emotion.get('emotion', '')
                confidence = st.session_state.current_emotion.get('confidence', '')
                emotion_context = f"\n\nNote: The candidate's emotional state during this response was detected as '{emotion}' with '{confidence}' confidence. Consider this in your assessment and follow-up questions."
            st.session_state.current_emotion = None

            # Prepare messages for API
            system_message = f"You are an HR representative conducting an interview for the following job vacancy. Ask relevant questions to assess the candidate's fit for the role based on the vacancy details. Engage in a natural conversation and ask follow-up questions as appropriate.\n\nJob Vacancy:\n{st.session_state.vacancy_text}"

            api_messages = [{"role": "system", "content": system_message}]

            # Add conversation history with emotion context in last message
            for i, m in enumerate(st.session_state.messages):
                msg_content = m["content"]
                # Add emotion context to the last user message
                if i == len(st.session_state.messages) - 1 and m["role"] == "user" and "emotion" in m:
                    msg_content += emotion_context
                api_messages.append({"role": m["role"], "content": msg_content})

            # Get AI response
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages
            )
            answer = response.choices[0].message.content

            # Generate audio
            audio = elevenlabs_client.text_to_speech.convert(text=answer, voice_id="21m00Tcm4TlvDq8ikWAM")
            audio_bytes = b''.join(audio)
            audio_base64 = base64.b64encode(audio_bytes).decode()

            # Add assistant message with audio
            st.session_state.messages.append({"role": "assistant", "content": answer, "audio": audio_base64})

            # Trigger audio playback on next rerun
            st.session_state.play_audio = True

            # Rerun to display new messages
            st.rerun()
else:
    # Conversation Mode: Show simple instruction
    st.info("💬 **Conversation Mode Active**: Just speak your response - it will be automatically processed and the AI will reply!")
