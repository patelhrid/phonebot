# chainlit_app.py
import os
import re
import sys
import time
import datetime
import random
import logging
import joblib
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from openai import OpenAI  # used for your LM Studio local client
import chainlit as cl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Globals to hold models/data (loaded at startup)
knn = None
sbert_model = None
df = None

# Basic in-memory session store (keyed by chainlit session id)
sessions = {}

# Lock to protect expensive ops
startup_lock = asyncio.Lock()
compute_lock = asyncio.Lock()

# Executor to run blocking code
executor = ThreadPoolExecutor(max_workers=2)

# ----------------- Utilities (port from Streamlit) -----------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in stop_words]
    return " ".join(filtered)

def resource_path(relative_path: str) -> str:
    # works in dev and when packaged by PyInstaller
    try:
        base = sys._MEIPASS
    except Exception:
        base = os.path.abspath(".")
    return os.path.join(base, relative_path)

def save_chat_history_to_file(session_id: str) -> str:
    logs_folder = "Logs"
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    messages = sessions.get(session_id, {}).get("messages", [])
    log_content = "\n".join(f"{m['sender']}: {m['message']}" for m in messages)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = os.path.join(logs_folder, f"LOG_{session_id}_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(log_content)
    return filename

# ----------------- Dataset / model setup (synchronous, CPU heavy) -----------------
def dataset_setup(input_file="MIR Exports2025_19_8_16_47_24.xlsx",
                  knowledge_articles_file="knowledge_articles_export.xlsx",
                  output_file="tickets_dataset_NEW.csv"):
    """Port of your Streamlit dataset_setup. Keep the same logic."""
    try:
        df_local = pd.read_excel(input_file)
        columns_to_keep = ['Incident ID', 'Summary', 'Resolution', 'Status']
        df_local = df_local[columns_to_keep]
        df_local = df_local[df_local['Status'] == 'Closed']
        df_local['Resolution'] = df_local['Resolution'].astype(str).str.replace(" (Automatically Closed)", "", regex=False).str.strip()
        unwanted_solutions = ['.', '...', 'fixed', 'resolved', 'test', 'duplicate', 'other','done', "completed."]
        df_local = df_local[~df_local['Resolution'].str.strip().str.lower().isin(unwanted_solutions)]
        df_local = df_local[df_local['Resolution'].str.strip() != '']
        df_local = df_local.drop(columns=['Status'])
        df_local = df_local.rename(columns={'Incident ID': 'Ticket #', 'Summary': 'Problem', 'Resolution': 'Solution'})
        df_local.to_csv("filtered_tickets_dataset.csv", index=False, encoding="utf-8")

        # knowledge articles handling (same as Streamlit)
        if knowledge_articles_file.endswith('.xlsx'):
            knowledge_articles = pd.read_excel(knowledge_articles_file)
        else:
            knowledge_articles = pd.read_csv(knowledge_articles_file, encoding="utf-8", on_bad_lines='skip')

        knowledge_articles = knowledge_articles[knowledge_articles['Status'] == "Published"]
        knowledge_articles = knowledge_articles[['Title']]
        knowledge_articles['Problem'] = knowledge_articles['Title']
        knowledge_articles['Solution'] = knowledge_articles['Title'].apply(lambda title: f"Refer to the '{title}' knowledge article")
        knowledge_articles['Ticket #'] = ""
        knowledge_articles = knowledge_articles[['Ticket #', 'Problem', 'Solution']]

        tickets_df = pd.read_csv("filtered_tickets_dataset.csv", encoding="utf-8")
        updated_tickets_df = pd.concat([tickets_df, knowledge_articles], ignore_index=True)
        updated_tickets_df.to_csv(output_file, index=False, encoding="utf-8")

        logger.info(f"Updated tickets dataset saved to {output_file}")
    except Exception as e:
        logger.exception("dataset_setup failed: %s", e)
        raise

def setup_once_sync():
    """Load dataset, preprocess, build or load SBERT embeddings + KNN."""
    global knn, sbert_model, df
    logger.info("Running setup_once_sync()")
    # If prebuilt files exist use them, otherwise build
    dataset_path = "tickets_dataset_NEW.csv"
    if not os.path.exists(dataset_path):
        logger.info("Dataset not found — running dataset_setup()")
        dataset_setup()

    # Load dataset
    df_local = pd.read_csv(dataset_path, encoding='latin1')
    # Preprocess problems
    df_local['Problem_cleaned'] = df_local['Problem'].astype(str).apply(clean_text)
    # Ensure NLTK data is available (user should run once on environment)
    try:
        df_local['Problem_cleaned'] = df_local['Problem_cleaned'].apply(tokenize_and_remove_stopwords)
    except Exception as e:
        logger.warning("NLTK data may be missing. Please run: nltk.download('punkt'); nltk.download('stopwords')")
        raise

    # Load or compute SBERT and KNN
    if os.path.exists("knn_sbert_model.pkl") and os.path.exists("sbert_model.pkl"):
        logger.info("Loading existing knn and sbert model from disk")
        knn = joblib.load("knn_sbert_model.pkl")
        sbert_model = joblib.load("sbert_model.pkl")
    else:
        logger.info("Training SBERT embeddings and KNN (this may take a while)...")
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = sbert_model.encode(df_local['Problem_cleaned'].tolist(), convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
        knn = NearestNeighbors(n_neighbors=3, metric='cosine')
        knn.fit(embeddings)
        joblib.dump(knn, 'knn_sbert_model.pkl')
        joblib.dump(sbert_model, 'sbert_model.pkl')
        logger.info("Saved knn and sbert models.")

    # set globals
    df = df_local
    logger.info("setup_once_sync finished.")

# ----------------- Retrieval and contextualization (sync functions to be run in executor) -----------------
DISTANCE_THRESHOLD = 0.75
LM_STUDIO_URL="http://localhost:8000"

logger = logging.getLogger(__name__)

def _lm_base_v1():
    """
    Normalizes LM_STUDIO_URL into a base URL that ends with /v1 (no trailing slash beyond that).
    Accepts values like:
      - http://localhost:8000
      - http://localhost:8000/
      - http://localhost:8000/v1
      - http://localhost:8000/v1/
    Returns: e.g. "http://localhost:8000/v1"
    """
    raw = os.getenv(LM_STUDIO_URL, "http://localhost:8000").rstrip('/')
    if raw.endswith('/v1'):
        return raw
    return raw + '/v1'


def contextualize_response_sync(problem, solutions_with_confidences, session_id):
    """
    Direct port of your original contextualize_response into Chainlit.
    Uses LM Studio on localhost:8000, streams completions, and preserves conversation history.
    """
    # Build conversation history
    conversation_history = [
        {"role": "system",
         "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."}
    ]

    # Add past messages from this Chainlit session
    for message in sessions[session_id]["messages"]:
        conversation_history.append({
            "role": "user" if message['sender'] == "User" else "assistant",
            "content": message['message']
        })

    # Label the predicted solutions
    labeled_solutions = "\n".join([
        f"Solution {i+1} (Confidence: {(1-conf)*100:.1f}%): {sol}"
        for i, (sol, conf) in enumerate(solutions_with_confidences)
    ])

    conversation_history.append({
        "role": "user",
        "content": (
            f"The problem is: '{problem}'. The predicted solutions are: {labeled_solutions}. "
            f"Your task is to combine, aggregate, and average the predicted solutions, and present them "
            f"in a clear, structured sentence. When explaining, address the user as if you are an IT Support "
            f"agent and provide actionable instructions without using phrases like \"the solution\". "
            f"Assume full ownership of the knowledge you are providing, instructing the user as if you are the sole expert. "
            f"Avoid starting with phrases like \"Sure, here's the solution:\". Do not mention that you are an AI chatbot "
            f"or reference the fact that solutions were predicted. If the provided solution is detailed, you may use it verbatim "
            f"instead of summarizing it. Ensure your response is strictly relevant to the predicted solutions without adding "
            f"any external information. If a predicted solution is not relevant, you must not use it. "
            f"Do not create or fabricate information. Only use the provided solutions as your source. "
            f"Your response should be authoritative and direct, tailored to help an IT Support agent effectively resolve the issue. "
            f"Do not use Markdown in your response. Ignore the confidence levels in the predicted solutions entirely, "
            f"they are NOT for your use. Avoid any specific dates, names, serial numbers, or any specific information in general. "
            f"Be incredibly detailed, verbose, thorough and descriptive in your troubleshooting advice. "
            f"Always remember to be detailed in your responses, without worrying about output length."
        )
    })

    # --- LLM call to LM Studio (port 8000) ---
    lm_base = "http://localhost:8000/v1"
    response = requests.get(f"{lm_base}/models")
    model_list = response.json().get("data", [])

    # Randomly select a model (avoid embedding-only model)
    selected_model = ""
    while selected_model in ["", "text-embedding-nomic-embed-text-v1.5"]:
        selected_model = random.choice(model_list)["id"]

    client = OpenAI(base_url=lm_base, api_key="lm-studio")

    completion = client.chat.completions.create(
        model=selected_model,
        messages=conversation_history,
        temperature=0.1,
        stream=True,
    )

    # Collect streamed response
    full_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content

    full_response += f"\n\n(Generated by {selected_model})"
    return full_response


def handle_problem_sync(problem_description: str, session_id: str):
    global knn, sbert_model, df
    txt = clean_text(problem_description)
    txt = tokenize_and_remove_stopwords(txt)
    X_new_embedding = sbert_model.encode([txt])
    distances, indices = knn.kneighbors(X_new_embedding, n_neighbors=3)
    st_indices = indices[0]

    solutions_with_confidences = [
        (df['Solution'].iloc[st_indices[i]], distances[0][i])
        for i in range(len(st_indices))
        if distances[0][i] <= DISTANCE_THRESHOLD
    ]

    if not solutions_with_confidences:
        return ("Based on the provided details, no immediate solutions could be identified. "
                "Consider revisiting the initial context or seeking alternative expertise.",
                1.0, None, indices)

    avg_distance = sum(conf for _, conf in solutions_with_confidences) / len(solutions_with_confidences)

    # 🔑 pass session_id here
    context_resp = contextualize_response_sync(problem_description, solutions_with_confidences, session_id)
    return context_resp, avg_distance, solutions_with_confidences, indices


# ----------------- Chainlit handlers -----------------

@cl.on_chat_start
async def on_chat_start():
    """
    Called when a user starts a chat session. We lazy-load the models once.
    """
    session_id = cl.user_session.get("id")
    if session_id is None:
        session_id = str(time.time())
        cl.user_session.set("id", session_id)

    # init per-session structure
    sessions[session_id] = {"messages": [], "initial_problem": None, "all_follow_up": []}

    await cl.Message(content="Welcome to BramBot (Chainlit). Initializing resources...").send()

    async with startup_lock:
        # Run the heavy synchronous setup in a thread so we don't block the event loop
        try:
            await asyncio.to_thread(setup_once_sync)
            await cl.Message(content="Models and data loaded. Ready to accept your IT question.").send()
        except Exception as e:
            await cl.Message(content=f"Initialization failed: {e}").send()
            raise


@cl.on_message
async def on_message(message):
    user_text = message if isinstance(message, str) else message.content
    session_id = cl.user_session.get("id", str(time.time()))

    session = sessions.setdefault(session_id, {"messages": [], "initial_problem": None, "all_follow_up": [],
                                               "solutions_shown": False})

    if user_text.strip().lower() == "/save_log":
        fname = save_chat_history_to_file(session_id)
        await cl.Message(content=f"Saved chat log to: {fname}").send()
        return

    session["messages"].append({"sender": "User", "message": user_text})

    await cl.Message(content="Thinking...").send()

    async with compute_lock:
        try:
            response_text, avg_distance, solutions_with_confidences, indices = await asyncio.to_thread(
                handle_problem_sync, user_text, session_id)
        except Exception as e:
            logger.exception("Error in handle_problem_sync: %s", e)
            await cl.Message(content=f"Oops — failed to process your input: {e}").send()
            return

    session["messages"].append({"sender": "Assistant", "message": response_text})

    reply = response_text
    if isinstance(avg_distance, float):
        reply += f"\n\nAverage Confidence: {(1 - avg_distance) * 100:.1f}%"

    msg = cl.Message(content=reply)

    # Only attach button if solutions exist and haven't been shown yet
    if solutions_with_confidences and not session.get("solutions_shown", False):
        sols_text = "\n".join(
            f"*Solution {i + 1} (Confidence {(1 - conf) * 100:.1f}%):* {sol}"
            for i, (sol, conf) in enumerate(solutions_with_confidences)
        )

        msg.actions = [
            cl.Action(
                name="show_solutions",
                payload={"solutions": sols_text},
                label="Show Predicted Solutions"
            )
        ]

    await msg.send()


@cl.action_callback("show_solutions")
async def on_show_solutions(action):
    session_id = cl.user_session.get("id")
    session = sessions.get(session_id, {})

    # Do nothing if solutions were already shown
    if session.get("solutions_shown", False):
        return

    sols_text = action.payload["solutions"]
    await cl.Message(content=f"**Predicted Solutions:**\n{sols_text}").send()

    # Mark as shown
    session["solutions_shown"] = True

