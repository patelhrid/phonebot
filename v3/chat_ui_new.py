import pandas as pd
import joblib
import streamlit as st
from openai import OpenAI

from main import clean_text, tokenize_and_remove_stopwords
from main_knn import df

# Load model and vectorizer (SBERT-based KNN model)
knn = joblib.load('knn_sbert_model.pkl')
sbert_model = joblib.load('sbert_model.pkl')


# Initialize LM Studio client
client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")

# Load dataset for solutions
df = pd.read_csv('../tickets_dataset.csv', encoding='latin1')  # Adjust file path

# Define confidence threshold
DISTANCE_THRESHOLD = 0.7


# Function to predict and contextualize solution for a given problem
def handle_problem(problem_description):
    # Preprocess the input description
    preprocessed_description = clean_text(problem_description)
    preprocessed_description = tokenize_and_remove_stopwords(preprocessed_description)

    # Generate SBERT embedding for the problem description
    X_new_embedding = sbert_model.encode([preprocessed_description])

    # Predict the 3 closest solutions using KNN
    distances, indices = knn.kneighbors(X_new_embedding, n_neighbors=3)

    # Collect the 3 closest solutions with their distances
    closest_solutions_with_confidences = []
    for i in range(3):
        solution = df['Solution'].iloc[indices[0][i]]
        confidence = distances[0][i]
        # Only include solutions within the threshold
        if confidence <= DISTANCE_THRESHOLD:
            closest_solutions_with_confidences.append((solution, confidence))

    # Check if all solutions exceed the threshold
    if len(closest_solutions_with_confidences) == 0:
        return "No reliable solution found. Seek help from other sources.", 1.0, None

    # Calculate the average distance of the solutions within the threshold
    average_distance = sum([conf for _, conf in closest_solutions_with_confidences]) / len(closest_solutions_with_confidences)

    # Aggregate the 3 closest solutions into a single response
    contextualized_response = contextualize_response(problem_description, closest_solutions_with_confidences)
    return contextualized_response, average_distance, closest_solutions_with_confidences


# Define a function to contextualize the output using LM Studio
def contextualize_response(problem, solutions_with_confidences):
    labeled_solutions = "\n".join([f"Solution {i+1} (Confidence: {(1 - conf) * 100:.1f}%): {sol}"
                                   for i, (sol, conf) in enumerate(solutions_with_confidences)])
    history = [
        {"role": "system",
         "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct "
                    "and helpful."},
        {"role": "user",
         "content": f"The problem is: '{problem}'. The predicted solutions are: {labeled_solutions}. Your task is to "
                    f"combine, aggregate, and average the predicted solutions, and present them in a clear, "
                    f"structured sentence. When explaining, address the user as an IT Support agent and provide "
                    f"actionable instructions without using phrases like \"the solution\" since the user is "
                    f"unfamiliar with that term. Assume full ownership of the knowledge you are providing, "
                    f"instructing the user as if you are the sole expert. Avoid starting with phrases like \"Sure, "
                    f"here's the solution:\". Do not mention that you are an AI chatbot or reference the fact that "
                    f"solutions were predicted. If the provided solution is detailed, you may use it verbatim instead "
                    f"of summarizing it. Ensure your response is strictly relevant to the predicted solutions without "
                    f"adding any external information. Do not create or fabricate information. Only use the provided "
                    f"solutions as your source. Your response should be authoritative and direct, tailored to help an "
                    f"IT Support agent effectively resolve the issue. Do not use Markdown in your response."
                    f"Ignore the confidence levels in the predicted solutions entirely, they are NOT for your use."}
    ]

    completion = client.chat.completions.create(
        model="model-identifier",
        messages=history,
        temperature=0.7,
        stream=True,
    )

    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content

    return response


# Streamlit UI (same as before)
st.title("ChatGPIT")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'show_solution' not in st.session_state:
    st.session_state['show_solution'] = False


# Function to add a message to the chat history
def add_message(sender, message):
    st.session_state['messages'].append({"sender": sender, "message": message})


# Function to handle sending a message
def send_message():
    problem_description = st.session_state.input_text
    if problem_description:
        add_message("User", problem_description)
        st.session_state.input_text = ""

        # Display "Generating response..." between user input and assistant response
        add_message("Assistant", "Generating response...")
        with st.spinner("Generating response..."):
            response, average_distance, predicted_solutions_with_confidences = handle_problem(problem_description)
            # Remove the "Generating response..." placeholder
            st.session_state['messages'].pop()
            add_message("Assistant", response)
            st.session_state['distance'] = average_distance
            st.session_state['predicted_solutions_with_confidences'] = predicted_solutions_with_confidences


# CSS to style the chat messages and input box
st.markdown(
    """
    <style>
    .user-message {
        background-color: #7E7F83;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .assistant-message {
        background-color: #202030;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .chat-input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        display: flex;
        align-items: center;
        background-color: white;
        padding: 10px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        z-index: 999;
    }
    .chat-input {
        flex-grow: 1;
        margin-right: 10px;
    }
    .chat-send-button {
        flex-shrink: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the chat history
for message in st.session_state['messages']:
    if message['sender'] == "User":
        st.markdown(f"<div class='user-message'>{message['message']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-message'>{message['message']}</div>", unsafe_allow_html=True)

# Display additional information if available
if 'distance' in st.session_state:
    st.write(f"Average Confidence: {(1 - st.session_state['distance']) * 100:.1f}%")

# Toggle button to show/hide predicted solutions
if 'predicted_solutions_with_confidences' in st.session_state:
    st.session_state['show_solution'] = st.checkbox("Show Predicted Solutions", st.session_state['show_solution'])
    if st.session_state['show_solution']:
        solutions_text = "\n\n".join(
            [f"Solution {i+1} (Confidence: {(1 - conf) * 100:.1f}%): {sol}"
             for i, (sol, conf) in enumerate(st.session_state['predicted_solutions_with_confidences'])]
        )
        st.text_area("Predicted Solutions:", solutions_text, height=150, disabled=True)

# Input text box and send button at the bottom
with st.container():
    st.text_input(
        "Please describe your IT problem:",
        key="input_text",
        placeholder="Type your message here...",
        label_visibility="collapsed",
        on_change=send_message,
        args=(),
    )
    st.button("Send", on_click=send_message, key="send_button")


#  Local URL: http://localhost:8501
#  Network URL: http://172.29.8.39:8501
