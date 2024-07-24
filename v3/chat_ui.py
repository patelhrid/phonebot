import joblib
import streamlit as st
from openai import OpenAI

from main import clean_text, tokenize_and_remove_stopwords
from main_knn import df

# Load model and vectorizer
knn = joblib.load('knn_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize LM Studio client
client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")

# Define confidence threshold
DISTANCE_THRESHOLD = 0.7

# Define a function to contextualize the output using LM Studio
def contextualize_response(problem, solutions):
    labeled_solutions = "\n".join([f"Solution {i+1}: {sol}" for i, sol in enumerate(solutions)])
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
                    f"IT Support agent effectively resolve the issue. Do not use Markdown in your response."}
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


# Function to predict and contextualize solution for a given problem
def handle_problem(problem_description):
    preprocessed_description = clean_text(problem_description)
    preprocessed_description = tokenize_and_remove_stopwords(preprocessed_description)

    # Vectorize using TF-IDF vectorizer
    X_new_tfidf = tfidf_vectorizer.transform([preprocessed_description])

    # Predict the 3 closest solutions using KNN
    distances, indices = knn.kneighbors(X_new_tfidf, n_neighbors=3)

    # Check if the closest solution exceeds the distance threshold
    # if distances[0][0] > DISTANCE_THRESHOLD:
    #     return "Seek help from other sources.", distances[0][0], None

    # Collect the 3 closest solutions
    closest_solutions = []
    for i in range(3):
        solution = df['Solution'].iloc[indices[0][i]]
        closest_solutions.append(solution)

    # Aggregate the 3 closest solutions into a single response
    contextualized_response = contextualize_response(problem_description, closest_solutions)
    return contextualized_response, distances[0][0], closest_solutions


# Function to add a message to the chat history
def add_message(sender, message):
    st.session_state['messages'].append({"sender": sender, "message": message})


# Streamlit app layout
st.title("ChatGPIT")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'show_solution' not in st.session_state:
    st.session_state['show_solution'] = False


# Function to handle sending a message
def send_message():
    problem_description = st.session_state.input_text
    if problem_description:
        add_message("User", problem_description)
        st.session_state.input_text = ""

        # Display "Generating response..." between user input and assistant response
        add_message("Assistant", "Generating response...")
        with st.spinner("Generating response..."):
            response, distance, predicted_solutions = handle_problem(problem_description)
            # Remove the "Generating response..." placeholder
            st.session_state['messages'].pop()
            add_message("Assistant", response)
            st.session_state['distance'] = distance
            st.session_state['predicted_solutions'] = predicted_solutions


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
    st.write(f"Confidence: {st.session_state['distance'] * -100 + 100:.1f}%")

# Toggle button to show/hide predicted solution
if 'predicted_solutions' in st.session_state:
    st.session_state['show_solution'] = st.checkbox("Show Predicted Solutions", st.session_state['show_solution'])
    if st.session_state['show_solution']:
        solutions_text = "\n\n".join([f"Solution {i+1}: {sol}" for i, sol in enumerate(st.session_state['predicted_solutions'])])
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
