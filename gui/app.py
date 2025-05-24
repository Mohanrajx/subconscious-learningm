import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import streamlit as st
import numpy as np
import pandas as pd
from utils.preprocessing import preprocess_eeg
from models.sleep_stage_model import build_model
from rl_agent.agent import SleepRLAgent
from tts.speaker import text_to_speech
import os

# Streamlit page config
st.set_page_config(page_title="Sleep Learning System", layout="centered")
st.title("🧠 Subconscious Learning While Sleeping")

# Section: EEG Upload
st.header("1. Upload EEG Signal")
uploaded = st.file_uploader("Upload a .csv file with 'eeg_signal' column", type=['csv'])

if uploaded:
    try:
        df = pd.read_csv(uploaded)

        # Check for correct column
        if 'eeg_signal' not in df.columns:
            st.error("CSV must contain a column named 'eeg_signal'.")
        else:
            eeg_signal = df['eeg_signal'].values

            if len(eeg_signal) < 100:
                st.error(f"EEG signal must have at least 100 data points. Your file has only {len(eeg_signal)}.")
            else:
                eeg_signal = eeg_signal[:100]
                st.line_chart(eeg_signal)

                # Preprocess EEG
                signal = preprocess_eeg(eeg_signal)
                X = signal.reshape(1, 100, 1)

                # Load (rebuild) model and predict
                model = build_model((100, 1), 3)
                prediction = model.predict(X)
                stage_idx = int(np.argmax(prediction))

                stage_map = {0: 'Light', 1: 'Deep', 2: 'REM'}
                sleep_stage = stage_map[stage_idx]
                st.success(f"🧠 Predicted Sleep Stage: **{sleep_stage}**")

                # RL agent decision
                agent = SleepRLAgent()
                # For demo: force Play if stage is Light
                # action_idx = 0 if sleep_stage == "Light" else 1
                action_idx = agent.get_action(stage_idx)
                action = agent.actions[action_idx]
                st.markdown(f"🤖 RL Agent Decision: **{action}**")

                # Learning Content Input
                if action == "Play":
                    st.header("2. Choose or Enter Learning Content")

                    sample_choices = {
                        "General Knowledge": "The capital of France is Paris. Water freezes at zero degrees Celsius. The chemical symbol for gold is A-U.",
                        "Science Facts": "Photosynthesis occurs in plant leaves. Gravity pulls objects toward Earth. The largest planet in our solar system is Jupiter.",
                        "Vocabulary Words": "Adapt means to change to fit a new situation. Observe means to watch carefully. Efficient means performing well without waste."
                    }

                    selected_topic = st.selectbox("Select Sample Topic (optional)", list(sample_choices.keys()))
                    sample_text = sample_choices[selected_topic]

                    user_text = st.text_area("Or write your own text below:", value=sample_text)

                    if st.button("Generate Audio"):
                        if not user_text.strip():
                            st.warning("Please enter some text to convert to speech.")
                        else:
                            try:
                                if not os.path.exists("data"):
                                    os.makedirs("data")
                                filename = "data/gui_audio.wav"
                                text_to_speech(user_text, filename=filename)
                                with open(filename, 'rb') as audio_file:
                                    st.audio(audio_file.read(), format="audio/wav")

                                # Section: Retention Quiz
                                st.markdown("### 🧪 Retention Quiz")

                                quiz_data = {
                                    "General Knowledge": [
                                        {"question": "What is the capital of France?", "answer": "paris"},
                                        {"question": "At what temperature (in Celsius) does water freeze?", "answer": "0"},
                                        {"question": "What is the chemical symbol for gold?", "answer": "au"}
                                    ],
                                    "Science Facts": [
                                        {"question": "Where does photosynthesis occur?", "answer": "leaves"},
                                        {"question": "What force pulls things to Earth?", "answer": "gravity"},
                                        {"question": "What is the largest planet?", "answer": "jupiter"}
                                    ],
                                    "Vocabulary Words": [
                                        {"question": "What does 'adapt' mean?", "answer": "change"},
                                        {"question": "What does 'observe' mean?", "answer": "watch"},
                                        {"question": "What does 'efficient' mean?", "answer": "performing well"}
                                    ]
                                }

                                questions = quiz_data.get(selected_topic, [])
                                user_answers = []

                                with st.form("quiz_form"):
                                    for i, q in enumerate(questions):
                                        ans = st.text_input(f"Q{i+1}: {q['question']}")
                                        user_answers.append(ans)
                                    submitted = st.form_submit_button("Submit Quiz")

                                if submitted:
                                    score = 0
                                    for i, q in enumerate(questions):
                                        correct = q['answer'].lower()
                                        given = user_answers[i].strip().lower()
                                        if correct in given:
                                            score += 1

                                    total = len(questions)
                                    percent = (score / total) * 100
                                    st.success(f"✅ You scored {score} out of {total} ({percent:.1f}%)")
                                    st.info("🧠 Retention score could be used as a reward in RL training.")
                            except Exception as e:
                                st.error(f"Audio generation or quiz failed: {e}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("⬆️ Please upload an EEG CSV file to get started.")
