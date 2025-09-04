import datetime
import random
import subprocess
import logging
import re
import time

import requests
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
from streamlit_extras.concurrency_limiter import concurrency_limiter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable wide mode
st.set_page_config(page_title="BramBot", layout="wide")

@concurrency_limiter(max_concurrency=1)
def dataset_setup(input_file="latest incident export.xlsx",
                  knowledge_articles_file="knowledge_articles_export.xlsx",
                  output_file="tickets_dataset_NEW.csv"):
    try:
        # Load the main dataset
        df = pd.read_excel(input_file)

        # Keep only the necessary columns
        df = df[['Incident ID', 'Summary', 'Resolution', 'Status']]

        # Filter rows with Closed or Resolved status
        df = df[df['Status'].isin(['Closed', 'Resolved'])]

        # Normalize the Resolution column
        df['Resolution'] = df['Resolution'].astype(str).str.replace(
            " (Automatically Closed)", "", regex=False
        ).str.strip()

        # Remove unwanted solutions
        unwanted_solutions = ['.', '...', 'fixed', 'resolved', 'test',
                              'duplicate', 'other','done', 'completed.']
        df = df[~df['Resolution'].str.strip().str.lower().isin(unwanted_solutions)]

        # Remove rows with empty resolution
        df = df[df['Resolution'].str.strip() != '']

        # Drop the 'Status' column
        df = df.drop(columns=['Status'])

        # Rename columns
        df = df.rename(columns={
            'Incident ID': 'Ticket #',
            'Summary': 'Problem',
            'Resolution': 'Solution'
        })

        # Load knowledge articles
        if knowledge_articles_file.endswith('.xlsx'):
            knowledge_articles = pd.read_excel(knowledge_articles_file)
        else:
            knowledge_articles = pd.read_csv(
                knowledge_articles_file,
                encoding="utf-8",
                on_bad_lines='skip'
            )

        # Filter for Published
        knowledge_articles = knowledge_articles[knowledge_articles['Status'] == "Published"]

        # Create new rows
        knowledge_articles = knowledge_articles[['Title']].copy()
        knowledge_articles['Problem'] = knowledge_articles['Title']
        knowledge_articles['Solution'] = knowledge_articles['Title'].apply(
            lambda title: f"Refer to the '{title}' knowledge article"
        )
        knowledge_articles['Ticket #'] = ""

        knowledge_articles = knowledge_articles[['Ticket #', 'Problem', 'Solution']]

        # Combine tickets + knowledge articles
        updated_tickets_df = pd.concat([df, knowledge_articles], ignore_index=True)

        # Save final dataset
        updated_tickets_df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"✅ Updated tickets dataset saved to {output_file}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")



# Function to save chat history as a log file
@concurrency_limiter(max_concurrency=1)
def save_chat_history():
    # Create Logs folder if it doesn't exist
    logs_folder = "Logs"
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    # Format the chat history
    chat_history = st.session_state.get('messages', [])
    log_content = "\n".join(
        f"{entry['sender']}: {entry['message']}" for entry in chat_history
    )

    # Generate a unique filename based on the timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = os.path.join(logs_folder, f"LOG_{timestamp}.txt")

    # Write the log content to the file
    with open(filename, "w") as log_file:
        log_file.write(log_content)

    st.success(f"Chat history saved to {filename}!")

@concurrency_limiter(max_concurrency=1)
def tokenize_and_remove_stopwords(text):
    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)


@concurrency_limiter(max_concurrency=1)
# Clean text: lowercase, remove extra spaces and punctuation
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

@concurrency_limiter(max_concurrency=1)
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


@concurrency_limiter(max_concurrency=1)
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
        try:
            # Run 'main_knn_new_copy.py' using the same interpreter that runs this script
            logger.info("Running 'setup'...")

            dataset_path = resource_path('tickets_dataset_NEW.csv')

            # Check if the dataset exists
            if not os.path.exists(dataset_path):
                logger.info("Dataset not found. Running dataset_setup()...")
                dataset_setup()
            else:
                # Check last modified time
                last_modified = os.path.getmtime(dataset_path)
                age_hours = (time.time() - last_modified) / 3600

                if age_hours > 24:
                    logger.info(f"Dataset is older than 24 hours ({age_hours:.1f}h). Rebuilding...")
                    dataset_setup()
                else:
                    logger.info(f"Dataset is fresh ({age_hours:.1f}h old). Using existing file.")

            # Now load the dataset
            df = pd.read_csv(dataset_path, encoding='latin1')

        except FileNotFoundError as e:
            logger.error(f"Dataset file missing even after setup: {e}")
            dataset_setup()

        # Now load the dataset
        df = pd.read_csv(dataset_path, encoding='latin1')

        # Preprocess problems
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
        logger.error(f"Error running setup: {e}")
        exit(-1)


# Call the cached setup function
setup_once()


@concurrency_limiter(max_concurrency=1)
def setup_streamlit():
    try:
        # Load model and vectorizer (SBERT-based KNN model)
        knn = joblib.load('knn_sbert_model.pkl')
        sbert_model = joblib.load('sbert_model.pkl')

        # Initialize LM Studio client
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio") # LOCAL LM Studio
        # client = OpenAI(base_url="http://172.30.89.86:8000/v1", api_key="lm-studio")  # P15 LM Studio

        # Load dataset for solutions
        df = pd.read_csv(resource_path('tickets_dataset_NEW.csv'), encoding='latin1')  # Adjust file path

        # Define confidence threshold
        DISTANCE_THRESHOLD = 0.75

        # Function to predict and contextualize solution for a given problem
        @concurrency_limiter(max_concurrency=1)
        def handle_problem(problem_description):
            preprocessed_description = clean_text(problem_description)
            preprocessed_description = tokenize_and_remove_stopwords(preprocessed_description)

            X_new_embedding = sbert_model.encode([preprocessed_description])
            distances, indices = knn.kneighbors(X_new_embedding, n_neighbors=3)

            st.session_state['solution_indices'] = indices[0]

            closest_solutions_with_confidences = [
                (df['Solution'].iloc[indices[0][i]], distances[0][i])
                for i in range(len(indices[0]))
                if distances[0][i] <= DISTANCE_THRESHOLD
            ]

            if len(closest_solutions_with_confidences) == 0:
                logger.info("No reliable solutions found after requery.")
                return (
                    f"Based on the provided details, no immediate solutions could be identified. "
                    f"Consider revisiting the initial context or seeking alternative expertise.",
                    1.0,
                    None,
                )

            average_distance = sum(conf for _, conf in closest_solutions_with_confidences) / len(
                closest_solutions_with_confidences)
            contextualized_response = contextualize_response(problem_description, closest_solutions_with_confidences)

            return contextualized_response, average_distance, closest_solutions_with_confidences, indices

        # Define a function to contextualize the output using LM Studio
        @concurrency_limiter(max_concurrency=1)
        def contextualize_response(problem, solutions_with_confidences):
            conversation_history = [
                {"role": "system",
                 "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."}
            ]

            # Add user and assistant messages to the history
            for message in st.session_state['messages']:
                conversation_history.append({
                    "role": "user" if message['sender'] == "User" else "assistant",
                    "content": message['message']
                })

            labeled_solutions = "\n".join([f"Solution {i + 1} (Confidence: {(1 - conf) * 100:.1f}%): {sol}"
                                           for i, (sol, conf) in enumerate(solutions_with_confidences)])

            conversation_history.append(
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
                            f"Ignore the confidence levels in the predicted solutions entirely, they are NOT for your use."
                            f"Avoid any specific dates, names, serial numbers, or any specific information in general."
                            f"Be incredibly detailed, verbose, thorough and descriptive in your troubleshooting advice."
                            f"Always remember to be detailed in your responses, without worrying about output length."}
            )

            # Get list of loaded models
            response = requests.get("http://localhost:8000/v1/models")
            models_data = response.json()

            # Extract the list from the 'data' key
            model_list = models_data.get("data", [])

            # Randomly select a model
            selected_model = ""
            while selected_model == "text-embedding-nomic-embed-text-v1.5" or selected_model == "":
                selected_model = random.choice(model_list)["id"]

            completion = client.chat.completions.create(
                model=selected_model,
                messages=conversation_history,
                temperature=0.1,
                stream=True,
            )

            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content

            response += "\n"
            response += "\n(Generated by " + selected_model + ")"

            return response

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        if 'all_follow_up' not in st.session_state:
            st.session_state['all_follow_up'] = []

        if 'show_solution' not in st.session_state:
            st.session_state['show_solution'] = False

        # Function to add a message to the chat history
        @concurrency_limiter(max_concurrency=1)
        def add_message(sender, message):
            st.session_state['messages'].append({"sender": sender, "message": message})
            # st.session_state['all_follow_up'].append(message)

        # Place this helper once at the top of your file
        def show_thinking_bubble(container, text="Thinking"):
            html = f"""
            <style>
            .assistant-thinking {{
                background-color: #004b9a;
                color: white;
                border-radius: 15px 15px 15px 0;
                padding: 12px 16px;
                margin: 10px 0;
                width: fit-content;
                max-width: 70%;
                text-align: left;
                margin-right: auto;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                font-size: 14px;
                line-height: 1.5;
                font-family: 'Parkinsans', sans-serif;
                display: inline-flex;
                align-items: center;
                gap: 10px;
            }}
            .think-dots {{
                display: inline-flex;
                gap: 6px;
                align-items: center;
            }}
            .think-dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: white;
                opacity: .5;
                transform: translateY(0);
                animation: think-bounce 0.9s infinite ease-in-out;
            }}
            .think-dot:nth-child(2) {{ animation-delay: 0.12s; }}
            .think-dot:nth-child(3) {{ animation-delay: 0.24s; }}

            @keyframes think-bounce {{
                0%   {{ transform: translateY(0); opacity:.5; }}
                50%  {{ transform: translateY(-6px); opacity:1; }}
                100% {{ transform: translateY(0); opacity:.5; }}
            }}
            </style>

            <div class="assistant-thinking" role="status" aria-live="polite">
              <div style="white-space:nowrap;">{text}</div>
              <div class="think-dots" aria-hidden="true">
                <div class="think-dot"></div>
                <div class="think-dot"></div>
                <div class="think-dot"></div>
              </div>
            </div>
            """
            container.markdown(html, unsafe_allow_html=True)

        # Function to handle sending a message
        @concurrency_limiter(max_concurrency=1)
        def send_message():
            problem_description = st.session_state.input_text
            if problem_description:
                if 'initial_problem' not in st.session_state:
                    st.session_state['initial_problem'] = problem_description
                    st.session_state["all_follow_up"] = []
                    add_message("User", problem_description)  # ✅ User message stays
                    st.session_state.input_text = ""

                    # 🔹 Show thinking bubble (not part of messages)
                    bubble_placeholder = st.empty()
                    show_thinking_bubble(bubble_placeholder)

                    start_time = time.time()
                    response, average_distance, predicted_solutions_with_confidences, _ = handle_problem(
                        problem_description)
                    end_time = time.time()
                    elapsed = end_time - start_time

                    # 🔹 Remove bubble once done
                    bubble_placeholder.empty()

                    # ❌ Do NOT pop user messages anymore
                    add_message("Assistant", f"{response}\n\n(Response time: {elapsed:.2f} seconds)")

                    st.session_state['distance'] = average_distance
                    st.session_state['predicted_solutions_with_confidences'] = predicted_solutions_with_confidences
                else:
                    st.session_state["all_follow_up"].append(problem_description)

                    full_context = f"{st.session_state['initial_problem']} "
                    for fo in st.session_state["all_follow_up"]:
                        full_context += f"Follow-up: {fo}"

                    if should_requery(full_context):
                        st.session_state['initial_problem'] = full_context
                        add_message("User", problem_description)
                        st.session_state.input_text = ""

                        bubble_placeholder = st.empty()
                        show_thinking_bubble(bubble_placeholder)

                        start_time = time.time()
                        response, average_distance, predicted_solutions_with_confidences, _ = handle_problem(
                            full_context)
                        end_time = time.time()
                        elapsed = end_time - start_time

                        bubble_placeholder.empty()

                        add_message("Assistant", f"{response}\n\n(Response time: {elapsed:.2f} seconds)")
                        st.session_state['distance'] = average_distance
                        st.session_state['predicted_solutions_with_confidences'] = predicted_solutions_with_confidences
                    else:
                        add_message("User", problem_description)
                        st.session_state.input_text = ""

                        bubble_placeholder = st.empty()
                        show_thinking_bubble(bubble_placeholder)

                        start_time = time.time()
                        response = handle_follow_up(full_context)
                        end_time = time.time()

                        bubble_placeholder.empty()

                        elapsed = end_time - start_time
                        add_message("Assistant", f"{response}\n\n(Response time: {elapsed:.2f} seconds)")

        # Function to handle follow-up messages and contextualize the response
        @concurrency_limiter(max_concurrency=1)
        def handle_follow_up(full_context):
            # Now full_context includes the initial problem and all follow-up messages
            response = contextualize_response(full_context,
                                              st.session_state['predicted_solutions_with_confidences'])
            return response

        @concurrency_limiter(max_concurrency=1)
        def should_requery(problem_description):
            requery_keywords = ['didn’t work', 'not solved', 'try again', 'recheck', 'failed', 'problem persists']
            retry_indicators = ['retry', 'recheck', 'check again', 'try again']

            if any(keyword in problem_description.lower() for keyword in requery_keywords):
                logger.info("REQUERY triggered by requery keywords!")
                return True

            if any(keyword in problem_description.lower() for keyword in retry_indicators):
                logger.info("REQUERY triggered by retry indicators!")
                return True

            if 'new issue' in problem_description.lower():
                logger.info("REQUERY triggered by new issue!")
                return True

            # Optionally: Detect significant changes in input length or topic.
            # if len(problem_description) > 1.5 * len(st.session_state['initial_problem']):
            #     logger.info("REQUERY triggered by dramatic description change!")
            #     return True

            return False

        # CSS to style the chat messages and input box
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Parkinsans:wght@300..800&display=swap');

            h1 {
                font-family: 'Parkinsans', sans-serif;
                font-size: 2.5rem; /* Adjust size as needed */
                font-weight: 600;  /* Semi-bold for emphasis */
                text-align: center; /* Optional: center-align title */
                margin-bottom: 20px; /* Add spacing below title */
            }

            /* Apply the font to all elements */
            html, body, [class*="css"] {
                font-family: 'Parkinsans', sans-serif;
            }

            /* Specific styles for Streamlit components */
            .stButton>button,
            .stTextInput input,
            .stTextArea textarea,
            .stMarkdown,
            .stSlider,
            .stRadio,
            .stCheckbox,
            .stcheckbox
            .stNumberInput,
            .stFileUploader,
            .stDateInput,
            .stColorPicker,
            .stMetric {
                font-family: 'Parkinsans', sans-serif;
            }

            /* Chat Container */
            .chat-container {
                display: flex;
                flex-direction: column;
                padding: 20px;
                margin: 0 auto;
                margin-bottom: 70px;  /* Ensure there's space for the fixed input area */
                max-width: 800px;
                min-height: 80vh;
                font-family: 'Parkinsans', sans-serif;
            }

            /* User Messages */
            .user-message {
                background-color: #00c1d8;
                color: white;
                border-radius: 15px 15px 0 15px;
                padding: 12px 16px;
                margin: 10px 0;
                width: fit-content;
                max-width: 70%;
                text-align: left;
                margin-left: auto;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                font-size: 14px;
                line-height: 1.5;
                font-family: 'Parkinsans', sans-serif;
            }

            /* Assistant Messages */
            .assistant-message {
                background-color: #004b9a;
                color: white;
                border-radius: 15px 15px 15px 0;
                padding: 12px 16px;
                margin: 10px 0;
                width: fit-content;
                max-width: 70%;
                text-align: left;
                margin-right: auto;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                font-size: 14px;
                line-height: 1.5;
                font-family: 'Parkinsans', sans-serif;
            }

            /* Optional: Modify other areas like the chat input */
            .chat-input-container, .chat-input, .chat-send-button {
                font-family: 'Parkinsans', sans-serif;
            }

            /* Make sure the content doesn't overlap the fixed input box at the bottom */
            .main-container {
                padding-bottom: 80px; /* Space for the fixed input box */
                height: 100%;
                overflow-y: auto; /* Allow scroll for content above the input box */
            }

            /* Style for the chat input container fixed at the bottom */
            .chat-input-container {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                display: flex;
                align-items: center;
                background-color: #ffffff;
                padding: 12px 20px;
                border-top: 1px solid #e0e0e0;
                box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
                z-index: 9999;
            }

            /* Style for the chat input field */
            .chat-input {
                flex-grow: 1;
                border: 1px solid #dcdcdc;
                border-radius: 20px;
                padding: 10px 15px;
                font-size: 14px;
                outline: none;
                box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.05);
                transition: border-color 0.2s;
            }

            .chat-input:focus {
                border-color: #00c1d8;
            }

            /* Style for the send button */
            .chat-send-button {
                background-color: #00c1d8;
                color: white;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-left: 10px;
                cursor: pointer;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                transition: background-color 0.2s, transform 0.2s;
            }

            .chat-send-button:hover {
                background-color: #009fb2;
                transform: scale(1.1);
            }
            """,
            unsafe_allow_html=True
        )

        st.title("Welcome to BramBot")
        st.caption("You are accountable for everything that you say to the client. Use responses from this site to your discretion.")

        # Display the chat history
        for message in st.session_state['messages']:
            if message['sender'] == "User":
                st.markdown(f"<div class='user-message'>{message['message']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'>{message['message']}</div>", unsafe_allow_html=True)

        # Display additional information if available
        if 'distance' in st.session_state:
            st.write(f"Average Confidence: {(1 - st.session_state['distance']) * 100:.0f}%")

        # Toggle button to show/hide predicted solutions
        if 'predicted_solutions_with_confidences' in st.session_state:
            st.session_state['show_solution'] = st.checkbox("Show Predicted Solutions",
                                                            st.session_state['show_solution'])

            if st.session_state['show_solution'] and st.session_state[
                'predicted_solutions_with_confidences'] is not None:
                solutions_text = "\n\n".join(
                    [
                        (
                            f"Solution {i + 1} "
                            f"(Confidence: {(1 - conf) * 100:.1f}% | "
                            f"Ticket #: {(
                                int(df['Ticket #'].iloc[st.session_state['solution_indices'][i]])
                                if pd.notna(df['Ticket #'].iloc[st.session_state['solution_indices'][i]]) and str(df['Ticket #'].iloc[st.session_state['solution_indices'][i]]).strip() != ''
                                else 'N/A'
                            )}): {sol}"
                        )
                        for i, (sol, conf) in enumerate(st.session_state['predicted_solutions_with_confidences'])
                    ]
                )

                st.text_area("Predicted Solutions:", solutions_text, height=150, disabled=True)

        # Input text box and send button at the bottom
        with st.container():
            cols = st.columns([4, 1])  # Adjust the column width ratio
            with cols[0]:
                st.text_input(
                    "Please describe your IT problem:",
                    key="input_text",
                    placeholder="Type your message here...",
                    label_visibility="collapsed",
                )
            with cols[1]:
                st.button("Send", on_click=send_message, key="send_button", use_container_width=True)

        with st.sidebar:
            st.markdown("### Options")
            if st.button("Reset Conversation"):
                st.session_state['messages'] = []
                st.session_state.pop('initial_problem', None)  # Clear the initial problem
                st.session_state.pop('distance', None)
                st.session_state.pop('predicted_solutions_with_confidences', None)
                st.session_state['show_solution'] = False
                st.session_state['all_follow_up'] = []

            if st.button("Report Conversation"):
                # Display a message when the button is clicked
                st.markdown("""
                    <div style='text-align: center;'>
                        <strong style="color: red;">Please submit feedback related to the reported conversation below.</strong>
                    </div>
                    """, unsafe_allow_html=True)

                # Generate log file and store the chat history
                save_chat_history()

            st.markdown("""
            <div style='text-align: center; margin-top: 20px;'>
                <a href="https://forms.gle/RYGRK5c7jhybijGz8" target="_blank">
                    <button style="background-color: #00c1d8; color: white; border: none; padding: 10px 20px; 
                    border-radius: 5px; cursor: pointer;">Feedback</button>
                </a>
            </div>
            """, unsafe_allow_html=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running 'chat_ui_new_copy.py' with Streamlit: {e}")


setup_streamlit()


# try:
#     # Check if Streamlit is already running
#     if os.getenv("STREAMLIT_RUNNING") != "true":
#         script = resource_path('site_launch_conv.py')  # Use the appropriate file path
#         logger.info(script)
#
#         # Set an environment variable to prevent recursive invocation
#         os.environ["STREAMLIT_RUNNING"] = "true"
#
#         logger.info("Running Streamlit app...")
#
#         # Launch the Streamlit app using subprocess
#         subprocess.run([sys.executable, '-m', 'streamlit', 'run', script, '--server.enableXsrfProtection=false', '--server.port', '8501'],
#                        check=True)
#         os.system(f"streamlit run {script} --server.enableXsrfProtection=false")
#     else:
#         logger.info("Streamlit app is already running.")
#
# except subprocess.CalledProcessError as e:
#     logger.error(f"Error running Streamlit: {e}")
