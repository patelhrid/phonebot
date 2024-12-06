import subprocess
import logging
import re
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import joblib
import streamlit as st
from openai import OpenAI
import sys
from transformers import AutoTokenizer, AutoModel
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable wide mode
st.set_page_config(page_title="ChatGPIT", layout="wide")

def tokenize_and_remove_stopwords(text):
    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)


# Clean text: lowercase, remove extra spaces and punctuation
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def resource_path(relative_path):
    """ Get absolute path to resource, works for both dev and PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores files there
        base_path = sys._MEIPASS
    except AttributeError:
        # If not bundled, use the current directory
        base_path = os.path.abspath(".")

    # Ensure relative_path is a string
    if not isinstance(relative_path, str):
        raise TypeError(f"Expected a string for relative_path, got {type(relative_path)}: {relative_path}")

    # Join the base path and relative path
    absolute_path = os.path.join(base_path, relative_path)

    # Debugging: Print paths for troubleshooting
    # print(f"Base path: {base_path}")
    # print(f"Relative path: {relative_path}")
    # print(f"Absolute path: {absolute_path}")

    return absolute_path


@st.cache_resource
def setup_once():
    if hasattr(sys, '_MEIPASS'):
        # Running in the PyInstaller bundled environment
        base_path = sys._MEIPASS
    else:
        # Running as a normal Python script
        base_path = os.path.dirname(os.path.abspath(__file__))

    cache_dir = os.path.join(base_path, "cache_dir")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2", cache_dir=cache_dir)
    model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2", cache_dir=cache_dir)

    try:
        # Run 'main_knn_new_copy.py' using the same interpreter that runs this script
        logger.info("Running 'main_knn_new_copy.py'...")
        try:
            df = pd.read_csv(resource_path('tickets_dataset_NEW.csv'), encoding='latin1')  # Adjust file path as necessary
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            exit(1)

        df['Problem_cleaned'] = df['Problem'].apply(clean_text)
        df['Problem_cleaned'] = df['Problem_cleaned'].apply(tokenize_and_remove_stopwords)

        # Load Sentence-BERT model for embeddings
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Generate SBERT embeddings for all problems
        print("Generating SBERT embeddings...")
        embeddings = sbert_model.encode(df['Problem_cleaned'].tolist(), convert_to_tensor=True)

        # Convert tensor embeddings to a numpy array
        embeddings = embeddings.cpu().numpy()

        # Train KNN model using SBERT embeddings
        knn = NearestNeighbors(n_neighbors=3, metric='cosine')  # Adjust n_neighbors if needed
        knn.fit(embeddings)

        # Save the trained KNN model and SBERT model
        joblib.dump(knn, 'knn_sbert_model.pkl')
        joblib.dump(sbert_model, 'sbert_model.pkl')

        print("SBERT model and KNN model saved successfully.")
        logger.info("'main_knn_new_copy.py' finished successfully.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running 'main_knn_new_copy.py': {e}")
        exit(-1)

# Call the cached setup function
setup_once()

def setup_streamlit():
    try:
        # Load model and vectorizer (SBERT-based KNN model)
        knn = joblib.load('knn_sbert_model.pkl')
        sbert_model = joblib.load('sbert_model.pkl')

        # Initialize LM Studio client http://172.29.15.223:8000/
        # client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio") # LOCAL LM Studio
        client = OpenAI(base_url="http://172.29.15.223:8000/v1", api_key="lm-studio")  # P15 LM Studio

        # Load dataset for solutions
        df = pd.read_csv(resource_path('tickets_dataset_NEW.csv'), encoding='latin1')  # Adjust file path

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
            average_distance = sum([conf for _, conf in closest_solutions_with_confidences]) / len(
                closest_solutions_with_confidences)

            # Aggregate the 3 closest solutions into a single response
            contextualized_response = contextualize_response(problem_description, closest_solutions_with_confidences)
            return contextualized_response, average_distance, closest_solutions_with_confidences


        # Define a function to contextualize the output using LM Studio
        def contextualize_response(problem, solutions_with_confidences):
            labeled_solutions = "\n".join([f"Solution {i + 1} (Confidence: {(1 - conf) * 100:.1f}%): {sol}"
                                           for i, (sol, conf) in enumerate(solutions_with_confidences)])
            history = [
                {"role": "system",
                 "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct "
                            "and helpful."},
                {"role": "user",
                 "content": f"The problem is: '{problem}'. The predicted solutions are: {labeled_solutions}. Your task is to "
                            f"combine, aggregate, and average the predicted solutions, and present them in a clear, "
                            f"structured sentence. When explaining, address the user as if you are an IT Support agent and "
                            f"provide actionable instructions without using phrases like \"the solution\" since the user is "
                            f"unfamiliar with that term. Assume full ownership of the knowledge you are providing, "
                            f"instructing the user as if you are the sole expert. Avoid starting with phrases like \"Sure, "
                            f"here's the solution:\". Do not mention that you are an AI chatbot or reference the fact that "
                            f"solutions were predicted. If the provided solution is detailed, you may use it verbatim instead "
                            f"of summarizing it. Ensure your response is strictly relevant to the predicted solutions without "
                            f"adding any external information. If a predicted solution is not relevant, you must not use it."
                            f" Do not create or fabricate information. Only use the provided "
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

        # Title of the application
        st.markdown("<h1 style='text-align: center; color: #ffffff; margin-bottom: 20px;'>ChatGPIT</h1>",
                    unsafe_allow_html=True)

        # Initialize session state variables if not already set
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        # Function to add a message to the chat history
        def add_message(sender, message):
            st.session_state['messages'].append({"sender": sender, "message": message})

        # Function to handle sending a message
        def send_message():
            problem_description = st.session_state.input_text
            if problem_description:
                add_message("User", problem_description)
                st.session_state.input_text = ""

                # Placeholder message while generating a response
                add_message("Assistant", "Thinking...")
                with st.spinner("Generating response..."):
                    response, average_distance, predicted_solutions_with_confidences = handle_problem(
                        problem_description)
                    # Replace placeholder with the actual response
                    st.session_state['messages'].pop()
                    add_message("Assistant", response)
                    st.session_state['distance'] = average_distance
                    st.session_state['predicted_solutions_with_confidences'] = predicted_solutions_with_confidences

        # CSS for ChatGPT-like styling
        st.markdown(
            """
            <style>
            body {
                background-color: #202123;
                color: #f1f1f1;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            /* User and assistant messages */
            .message {
                max-width: 50%; /* Reduced max width for better alignment */
                min-width: 100px;
                padding: 12px;
                margin: 8px 0;
                border-radius: 12px;
                word-wrap: break-word;
                font-size: 14px;
            }
            .user-message {
                background-color: #0078FF;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            .assistant-message {
                background-color: #F1F1F1;
                color: #333;
                margin-right: auto;
                text-align: left;
            }
            .chat-input {
                flex-grow: 1;
                padding: 10px;
                border: none;
                font-size: 14px;
                border-radius: 5px;
                outline: none;
                margin-right: 10px;
                background-color: #40414f;
                color: white;
            }
            .chat-send-button {
                background-color: #0078FF;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 14px;
                cursor: pointer;
            }
            .chat-send-button:hover {
                background-color: #005BBB;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Chat messages display
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        if st.session_state['messages']:
            for message in st.session_state['messages']:
                if message['sender'] == "User":
                    st.markdown(
                        f"<div class='message user-message'>{message['message']}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='message assistant-message'>{message['message']}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.markdown("<div class='empty-placeholder'>Start the conversation by typing below.</div>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Input box and send button
        st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)
        st.text_input(
            "Type your message here...",
            key="input_text",
            placeholder="Send a message...",
            label_visibility="collapsed"
        )
        st.button("Send", on_click=send_message, key="send_button")
        st.markdown("</div>", unsafe_allow_html=True)



    except subprocess.CalledProcessError as e:
        logger.error(f"Error running 'chat_ui_new_copy.py' with Streamlit: {e}")

setup_streamlit()

logger.info("ATTEMPTING TO RUN STREAMLIT COMMAND")

try:
    # setup_once()
    # Check if Streamlit is already running
    if os.getenv("STREAMLIT_RUNNING") != "true":
        # Ensure correct path to 'site_launch_v2.py' within the bundled environment
        script = resource_path('site_launch_v2.py')  # Use the appropriate file path
        logger.info(script)

        # Set an environment variable to prevent recursive invocation
        os.environ["STREAMLIT_RUNNING"] = "true"

        logger.info("Running Streamlit app...")

        # Launch the Streamlit app using subprocess
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', script, '--server.enableXsrfProtection=false'],
                       check=True)
        os.system(f"streamlit run {script} --server.enableXsrfProtection=false")
    else:
        logger.info("Streamlit app is already running.")

except subprocess.CalledProcessError as e:
    logger.error(f"Error running Streamlit: {e}")
