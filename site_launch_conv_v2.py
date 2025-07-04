import subprocess
import logging
import re
from pydoc import classname

import numpy as np
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
import faiss


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable wide mode
st.set_page_config(page_title="BramBot", layout="wide")


def dataset_setup(input_file="MIR Exports2025_04_7_15_09_53.xlsx", knowledge_articles_file="knowledge_articles_export.xlsx", output_file="tickets_dataset_NEW.csv"):
    try:
        # Load the main dataset
        df = pd.read_excel(input_file)

        # Keep only the necessary columns
        columns_to_keep = ['Incident ID', 'Summary', 'Resolution', 'Status']
        df = df[columns_to_keep]

        # Filter rows to keep only the ones with 'Resolved' status
        df = df[df['Status'] == 'Resolved']

        # Remove rows where 'Resolution' has unwanted values (case-insensitive)
        unwanted_solutions = ['.', '...', 'fixed', 'resolved', 'test', 'duplicate', 'other']
        df = df[~df['Resolution'].str.strip().str.lower().isin(unwanted_solutions)]

        # Remove rows where 'Resolution' is empty
        df = df[df['Resolution'].str.strip() != '']

        # Drop the 'Status' column since it's no longer needed
        df = df.drop(columns=['Status'])

        # Rename the columns
        df = df.rename(columns={
            'Incident ID': 'Ticket #',
            'Summary': 'Problem',
            'Resolution': 'Solution'
        })

        # Export the result to a CSV file
        df.to_csv("filtered_tickets_dataset.csv", index=False, encoding="utf-8")

        # Paths to the files
        tickets_file = "filtered_tickets_dataset.csv"  # Update with your file path

        # Load the knowledge articles
        if knowledge_articles_file.endswith('.xlsx'):
            knowledge_articles = pd.read_excel(knowledge_articles_file)
        else:
            knowledge_articles = pd.read_csv(
                knowledge_articles_file,
                encoding="utf-8",  # Use UTF-8 for better character support
                on_bad_lines='skip'  # Skip malformed lines
            )

        # Filter for 'Published' articles
        knowledge_articles = knowledge_articles[knowledge_articles['Status'] == "Published"]

        # Select the required columns
        knowledge_articles = knowledge_articles[['Title']]

        # Create new rows for the tickets dataset
        knowledge_articles['Problem'] = knowledge_articles['Title']
        knowledge_articles['Solution'] = knowledge_articles['Title'].apply(
            lambda title: f"Refer to the '{title}' knowledge article"
        )
        knowledge_articles['Ticket #'] = ""  # Leave Ticket # blank

        # Keep only the required columns
        knowledge_articles = knowledge_articles[['Ticket #', 'Problem', 'Solution']]

        # Load the existing tickets dataset
        tickets_df = pd.read_csv(tickets_file, encoding="utf-8")

        # Append the knowledge articles to the tickets dataset
        updated_tickets_df = pd.concat([tickets_df, knowledge_articles], ignore_index=True)

        # Save the updated dataset
        updated_tickets_df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"Updated tickets dataset saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


dataset_setup()


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
            df = pd.read_csv(resource_path('tickets_dataset_NEW.csv'),
                             encoding='latin1')  # Adjust file path as necessary
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            exit(1)

        df['Problem_cleaned'] = df['Problem'].apply(clean_text)
        df['Problem_cleaned'] = df['Problem_cleaned'].apply(tokenize_and_remove_stopwords)

        # Load SBERT model
        sbert_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", cache_folder=cache_dir)

        # Load dataset
        df = pd.read_csv("tickets_dataset_NEW.csv", encoding="utf-8")
        df['Problem_cleaned'] = df['Problem'].apply(clean_text)

        # Generate embeddings
        print("Generating SBERT embeddings...")
        embeddings = sbert_model.encode(df['Problem_cleaned'].tolist(), convert_to_tensor=False)
        embeddings = np.array(embeddings, dtype=np.float32)  # Convert to NumPy

        # Initialize FAISS index
        dimension = embeddings.shape[1]  # Vector size
        index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        index.add(embeddings)  # Add vectors to FAISS index

        # Save FAISS index and dataset
        faiss.write_index(index, "faiss_index.bin")
        joblib.dump(df, "tickets_dataset.pkl")

        print("FAISS index and dataset saved successfully.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running 'main_knn_new_copy.py': {e}")
        exit(-1)


# Call the cached setup function
setup_once()


def setup_streamlit():
    try:
        # Load model and vectorizer (SBERT-based KNN model)
        index = faiss.read_index("faiss_index.bin")
        df = joblib.load("tickets_dataset.pkl")
        sbert_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

        # Initialize LM Studio client http://172.29.15.223:8000/
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio") # LOCAL LM Studio
        # client = OpenAI(base_url="http://172.29.15.223:8000/v1", api_key="lm-studio")  # P15 LM Studio

        # Define confidence threshold
        DISTANCE_THRESHOLD = 0.5

        # Function to predict and contextualize solution for a given problem
        def handle_problem(problem_description):
            """Finds the 3 most relevant solutions using FAISS vector search."""
            preprocessed_description = clean_text(problem_description)
            query_embedding = sbert_model.encode([preprocessed_description], convert_to_tensor=False)

            # FAISS search for top 3 results
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), 3)

            # Extract solutions with confidence scores
            closest_solutions_with_confidences = [
                (df.iloc[indices[0][i]]['Solution'], distances[0][i])
                for i in range(len(indices[0]))
            ]

            # Filter based on confidence threshold
            filtered_solutions = [
                (solution, confidence)
                for solution, confidence in closest_solutions_with_confidences
                if confidence <= DISTANCE_THRESHOLD
            ]

            if not filtered_solutions:
                logger.info("No reliable solutions found.")
                return (
                    "No immediate solutions found. Consider refining your query or consulting an expert.",
                    1.0,
                    closest_solutions_with_confidences,  # Return closest even if over threshold
                )

            # Compute average confidence
            average_distance = sum(conf for _, conf in filtered_solutions) / len(filtered_solutions)

            # Contextualize response (optional, if you have a function for it)
            contextualized_response = contextualize_response(problem_description, filtered_solutions)

            return contextualized_response, average_distance, filtered_solutions

        # Define a function to contextualize the output using LM Studio
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

            completion = client.chat.completions.create(
                model="model-identifier",
                messages=conversation_history,
                temperature=0.7,
                stream=True,
            )

            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content

            return response

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        if 'all_follow_up' not in st.session_state:
            st.session_state['all_follow_up'] = []

        if 'show_solution' not in st.session_state:
            st.session_state['show_solution'] = False

        # Function to add a message to the chat history
        def add_message(sender, message):
            st.session_state['messages'].append({"sender": sender, "message": message})
            # st.session_state['all_follow_up'].append(message)

        # Function to handle sending a message
        def send_message():
            problem_description = st.session_state.input_text
            if problem_description:
                if 'initial_problem' not in st.session_state:
                    st.session_state['initial_problem'] = problem_description
                    st.session_state["all_follow_up"] = []  # Initialize follow-up messages
                    add_message("User", problem_description)
                    st.session_state.input_text = ""

                    add_message("Assistant", "Thinking...")
                    with st.spinner("Generating response..."):
                        response, average_distance, predicted_solutions_with_confidences = handle_problem(
                            problem_description)
                        st.session_state['messages'].pop()
                        add_message("Assistant", response)
                        st.session_state['distance'] = average_distance
                        st.session_state['predicted_solutions_with_confidences'] = predicted_solutions_with_confidences
                else:
                    # Append new message to all_follow_up
                    st.session_state["all_follow_up"].append(problem_description)

                    # Combine the initial problem and all follow-ups for context
                    full_context = f"{st.session_state['initial_problem']} "  # Include initial problem first
                    for fo in st.session_state["all_follow_up"]:  # Include all follow-ups
                        full_context += f"Follow-up: {fo} "

                    if should_requery(full_context):
                        # Re-query with updated context
                        st.session_state['initial_problem'] = full_context
                        add_message("User", problem_description)
                        st.session_state.input_text = ""

                        add_message("Assistant", "Thinking...")
                        with st.spinner("Generating response..."):
                            response, average_distance, predicted_solutions_with_confidences = handle_problem(
                                full_context)
                            st.session_state['messages'].pop()
                            add_message("Assistant", response)
                            st.session_state['distance'] = average_distance
                            st.session_state[
                                'predicted_solutions_with_confidences'] = predicted_solutions_with_confidences
                    else:
                        # If it's a non-query follow-up, use the full context to handle response
                        add_message("User", problem_description)
                        st.session_state.input_text = ""

                        add_message("Assistant", "Thinking...")
                        with st.spinner("Generating response..."):
                            response = handle_follow_up(full_context)  # Pass the full context, including all follow-ups
                            st.session_state['messages'].pop()
                            add_message("Assistant", response)

        # Function to handle follow-up messages and contextualize the response
        def handle_follow_up(full_context):
            # Now full_context includes the initial problem and all follow-up messages
            response = contextualize_response(full_context,
                                              st.session_state['predicted_solutions_with_confidences'])
            return response

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

        st.title("Welcome to ChatGPIT")

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
            if st.session_state['show_solution']:
                solutions_text = "\n\n".join(
                    [f"Solution {i + 1} (Confidence: {(1 - conf) * 100:.1f}%): {sol}"
                     for i, (sol, conf) in enumerate(st.session_state['predicted_solutions_with_confidences'])]
                )
                st.text_area("Predicted Solutions:", solutions_text, height=150, disabled=True)

        # Input text box and send button at the bottom
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

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running 'chat_ui_new_copy.py' with Streamlit: {e}")

setup_streamlit()

# logger.info("ATTEMPTING TO RUN STREAMLIT COMMAND")

try:
    # setup_once()
    # Check if Streamlit is already running
    if os.getenv("STREAMLIT_RUNNING") != "true":
        # Ensure correct path to 'site_launch_v2.py' within the bundled environment
        script = resource_path('site_launch_conv.py')  # Use the appropriate file path
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
