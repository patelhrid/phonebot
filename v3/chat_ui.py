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
def contextualize_response(problem, solution):
    history = [
        {"role": "system",
         "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct "
                    "and helpful."},
        {"role": "user",
         "content": f"The problem is: '{problem}' The predicted solution is: '{solution}'. Please explain the "
                    f"solution to the user in a well-structured sentence. If it isn't a relevant solution at all, "
                    f"say \"Seek help elsewhere\" and nothing else, just that.\". Don't mention "
                    f"'the solution', because I do not know what that even is, and instead talk about it like it's "
                    f"your own. Your response must be instructional to myself, an IT Support agent, and you should be "
                    f"telling me what to do. The solution belongs to you, so instruct me on the solution like I've "
                    f"never heard about it before. You are to respond as if you are the only one providing knowledge, "
                    f"because you are the front facing part of the flow. So don't say \"Sure here's the solution\" or "
                    f"something like that. You ARE an AI chatbot trained to help IT technicians, so you can be honest "
                    f"if you dont know how to answer something or if the predicted solution is not relevant and why. "
                    f"If the predicted solution is already very detailed, you dont need to summarize it, and instead "
                    f"you can regurgitate the detailed solution. Your answer must be relevant to the predicted "
                    f"solution, and you must NOT say anything that makes it seem as though you are an AI chatbot and "
                    f"not actually an IT technician. Never ever say \"Sure, here's the solution:\" or anything along "
                    f"those lines. Do NOT pretend to know something and make up the instructions, this is a serious"
                    f"corporate setting."}
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

    # Predict the closest solution using KNN
    distances, indices = knn.kneighbors(X_new_tfidf, n_neighbors=1)
    distance = distances[0][0]
    predicted_solution = df['Solution'].iloc[indices[0][0]]

    # Contextualize the response
    if distance > DISTANCE_THRESHOLD:
        return "Seek help from other sources.", distance

    contextualized_response = contextualize_response(problem_description, predicted_solution)
    return contextualized_response, distance, predicted_solution


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
            response, distance, predicted_solution = handle_problem(problem_description)
            # Remove the "Generating response..." placeholder
            st.session_state['messages'].pop()
            add_message("Assistant", response)
            st.session_state['distance'] = distance
            st.session_state['predicted_solution'] = predicted_solution


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
if 'predicted_solution' in st.session_state:
    st.session_state['show_solution'] = st.checkbox("Show Predicted Solution", st.session_state['show_solution'])
    if st.session_state['show_solution']:
        st.text_area("Predicted Solution:", st.session_state['predicted_solution'], height=150, disabled=True)

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
