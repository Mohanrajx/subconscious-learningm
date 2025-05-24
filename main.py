from utils.preprocessing import preprocess_eeg
from models.sleep_stage_model import build_model
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Simulate EEG data (1000 samples, 100 timesteps)
X = np.random.randn(1000, 100)
X = np.array([preprocess_eeg(x) for x in X])
X = X.reshape((1000, 100, 1))  # LSTM input shape

# Simulate labels: 0 = Light, 1 = Deep, 2 = REM
y = np.random.randint(0, 3, 1000)
y_cat = to_categorical(y, num_classes=3)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build and train model
model = build_model(input_shape=(100,1), num_classes=3)
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
from rl_agent.agent import SleepRLAgent

# Simulate prediction
sample_sleep_stage = np.argmax(y_test[0])  # pretend we're in a certain stage

stage_map = {0: 'Light', 1: 'Deep', 2: 'REM'}
agent = SleepRLAgent()

print("Initial Q-Table:\n", agent.q_table)

for episode in range(10):
    current_state = sample_sleep_stage
    action = agent.get_action(current_state)
    action_name = agent.actions[action]
    
    # Simulate feedback (reward)
    if stage_map[current_state] == 'Light' and action_name == 'Play':
        reward = 1  # good timing
    else:
        reward = -1  # bad timing
    
    agent.update(current_state, action, reward)
    print(f"Episode {episode+1}: Stage={stage_map[current_state]}, Action={action_name}, Reward={reward}")

print("Final Q-Table:\n", agent.q_table)
from tts.speaker import text_to_speech

# Example learning content
learning_text = "The capital of France is Paris. The chemical formula of water is H2O."

# Generate TTS audio
text_to_speech(learning_text, filename='data/learning_content.wav')
