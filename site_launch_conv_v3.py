import os
import sys
import re
import time
import json
import datetime
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Concurrency & clients
from streamlit_extras.concurrency_limiter import concurrency_limiter
import requests
from openai import OpenAI

# NLP / embeddings / retrieval
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import joblib

# Optional hybrid deps
try:
    import faiss  # pip install faiss-cpu
except Exception:
    faiss = None

try:
    from rank_bm25 import BM25Okapi  # pip install rank_bm25
except Exception:
    BM25Okapi = None

# -----------------------------
# Logging & Page
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# v2 UI: keep the working page config and look & feel
st.set_page_config(page_title="BramBot", layout="wide")  # from v2

# -----------------------------
# NLTK + helpers (v3 core)
# -----------------------------
def ensure_nltk():
    try:
        _ = stopwords.words("english")
    except:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except:
        nltk.download("punkt", quiet=True)

ensure_nltk()
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_no_stop(text: str) -> List[str]:
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOP_WORDS]

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # for PyInstaller bundles
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def l2_normalize(mat: np.ndarray, axis: int = 1) -> np.ndarray:
    norms = np.linalg.norm(mat, ord=2, axis=axis, keepdims=True) + 1e-12
    return mat / norms

# -----------------------------
# Dataset setup (from v3)
# -----------------------------
@concurrency_limiter(max_concurrency=1)
def dataset_setup(
    input_file: str = "MIR Exports2025_19_8_16_47_24.xlsx",
    knowledge_articles_file: str = "knowledge_articles_export.xlsx",
    output_file: str = "tickets_dataset_NEW.csv",
):
    """
    Build/refresh tickets_dataset_NEW.csv by filtering MIR export and
    appending published KA titles as ticket-like rows.
    """
    try:
        df = pd.read_excel(input_file)
        columns_to_keep = ["Incident ID", "Summary", "Resolution", "Status"]
        df = df[columns_to_keep]
        df = df[df["Status"] == "Closed"]
        df["Resolution"] = (
            df["Resolution"].astype(str)
            .str.replace(" (Automatically Closed)", "", regex=False)
            .str.strip()
        )

        unwanted_solutions = [".", "...", "fixed", "resolved", "test", "duplicate", "other", "done", "completed."]
        df = df[~df["Resolution"].str.strip().str.lower().isin(unwanted_solutions)]
        df = df[df["Resolution"].str.strip() != ""]

        df = df.drop(columns=["Status"])
        df = df.rename(columns={"Incident ID": "Ticket #", "Summary": "Problem", "Resolution": "Solution"})
        df.to_csv("filtered_tickets_dataset.csv", index=False, encoding="utf-8")

        # Knowledge articles -> pseudo tickets
        if knowledge_articles_file.endswith(".xlsx"):
            ka = pd.read_excel(knowledge_articles_file)
        else:
            ka = pd.read_csv(knowledge_articles_file, encoding="utf-8", on_bad_lines="skip")
        ka = ka[ka["Status"] == "Published"][["Title"]].copy()
        ka["Problem"] = ka["Title"]
        ka["Solution"] = ka["Title"].apply(lambda t: f"Refer to the '{t}' knowledge article")
        ka["Ticket #"] = ""
        ka = ka[["Ticket #", "Problem", "Solution"]]

        tickets_df = pd.read_csv("filtered_tickets_dataset.csv", encoding="utf-8")
        updated = pd.concat([tickets_df, ka], ignore_index=True)
        updated.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Updated tickets dataset saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# -----------------------------
# Config
# -----------------------------
DATASET_CSV = "tickets_dataset_NEW.csv"
EMBED_BASE = "sentence-transformers/paraphrase-MiniLM-L6-v2"
FINETUNED_DIR = "models/domain-sbert"
FAISS_INDEX_PATH = "faiss.index"
EMB_CACHE_PATH = "embeddings.npy"
BM25_STORE = "bm25.pkl"
META_PATH = "meta.json"

TOP_K_DENSE = 12
TOP_K_SPARSE = 12
TOP_K_FINAL = 6
DENSE_WEIGHT = 0.6  # alpha for dense vs sparse

# -----------------------------
# Fine-tuning (light, v3)
# -----------------------------
def train_or_load_embedder(df: pd.DataFrame) -> SentenceTransformer:
    os.makedirs(os.path.dirname(FINETUNED_DIR), exist_ok=True)
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(FINETUNED_DIR) and any(os.scandir(FINETUNED_DIR)):
        logger.info("Loading fine-tuned embedder from %s", FINETUNED_DIR)
        return SentenceTransformer(FINETUNED_DIR).to(torch_device)

    base = SentenceTransformer(EMBED_BASE).to(torch_device)
    logger.info("No fine-tuned model found. Starting a light domain fine-tune...")

    examples = []
    for _, row in df.iterrows():
        p = str(row.get("Problem", "")); s = str(row.get("Solution", ""))
        p_clean = clean_text(p)
        if p_clean and s:
            examples.append(InputExample(texts=[p_clean, s]))

    if len(examples) < 16:
        logger.warning("Insufficient examples (%d). Using base embedder.", len(examples))
        return base

    train_loader = DataLoader(examples, shuffle=True, batch_size=32, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(base)
    base.fit(train_objectives=[(train_loader, train_loss)], epochs=1, warmup_steps=0, show_progress_bar=True)
    base.save(FINETUNED_DIR)
    logger.info("Saved fine-tuned embedder to %s", FINETUNED_DIR)
    return base

# -----------------------------
# Index build/load (v3)
# -----------------------------
@dataclass
class HybridArtifacts:
    df: pd.DataFrame
    embedder: SentenceTransformer
    faiss_index: Optional["faiss.Index"]
    corpus_embeddings: np.ndarray
    bm25: Optional[BM25Okapi]
    tokenized_corpus: List[List[str]]

def build_or_load_indexes() -> HybridArtifacts:
    dataset_path = resource_path(DATASET_CSV)
    if not os.path.exists(dataset_path):
        logger.info("Dataset not found -> dataset_setup()")
        dataset_setup()
    else:
        age_hours = (time.time() - os.path.getmtime(dataset_path)) / 3600
        if age_hours > 24:
            logger.info("Dataset older than 24h (%.1fh). Rebuilding...", age_hours)
            dataset_setup()

    df = pd.read_csv(dataset_path, encoding="latin1").dropna(subset=["Problem", "Solution"]).reset_index(drop=True)
    df["Problem_cleaned"] = df["Problem"].map(clean_text)
    df["Problem_tokens"] = df["Problem_cleaned"].map(tokenize_no_stop)

    embedder = train_or_load_embedder(df)

    # Embedding cache
    if os.path.exists(EMB_CACHE_PATH) and os.path.exists(META_PATH):
        try:
            meta = json.load(open(META_PATH, "r"))
        except Exception:
            meta = {}
        model_id = str(embedder)
        if meta.get("model_id") == model_id and meta.get("dataset_rows") == len(df):
            corpus_embeddings = np.load(EMB_CACHE_PATH)
        else:
            corpus_embeddings = embedder.encode(df["Problem_cleaned"].tolist(), batch_size=64,
                                               convert_to_numpy=True, show_progress_bar=True)
            np.save(EMB_CACHE_PATH, corpus_embeddings)
            json.dump({"model_id": model_id, "dataset_rows": len(df)}, open(META_PATH, "w"))
    else:
        corpus_embeddings = embedder.encode(df["Problem_cleaned"].tolist(), batch_size=64,
                                            convert_to_numpy=True, show_progress_bar=True)
        np.save(EMB_CACHE_PATH, corpus_embeddings)
        json.dump({"model_id": str(embedder), "dataset_rows": len(df)}, open(META_PATH, "w"))

    # FAISS (cosine via inner product on L2-normalized vectors)
    norm_emb = l2_normalize(corpus_embeddings)
    faiss_index = None
    if faiss is None:
        logger.warning("faiss not installed. Dense retrieval disabled.")
    else:
        dim = norm_emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(norm_emb.astype(np.float32))
        faiss.write_index(index, FAISS_INDEX_PATH)
        faiss_index = index

    # BM25
    bm25 = None
    if BM25Okapi is None:
        logger.warning("rank_bm25 not installed. Sparse retrieval disabled.")
    else:
        tokenized = df["Problem_tokens"].tolist()
        bm25 = BM25Okapi(tokenized)
        joblib.dump({"bm25": bm25}, BM25_STORE)

    return HybridArtifacts(
        df=df,
        embedder=embedder,
        faiss_index=faiss_index,
        corpus_embeddings=norm_emb,
        bm25=bm25,
        tokenized_corpus=df["Problem_tokens"].tolist(),
    )

@st.cache_resource
def setup_once_cached() -> HybridArtifacts:
    return build_or_load_indexes()

# -----------------------------
# Hybrid search (v3)
# -----------------------------
def hybrid_search(
    artifacts: HybridArtifacts,
    query: str,
    top_k_dense: int = TOP_K_DENSE,
    top_k_sparse: int = TOP_K_SPARSE,
    top_k_final: int = TOP_K_FINAL,
    dense_weight: float = DENSE_WEIGHT,
) -> List[Tuple[int, float, float, float]]:
    q_clean = clean_text(query)
    q_tokens = tokenize_no_stop(q_clean)

    # Dense via FAISS
    dense_scores: Dict[int, float] = {}
    if artifacts.faiss_index is not None:
        q_emb = artifacts.embedder.encode([q_clean], convert_to_numpy=True)
        q_emb = l2_normalize(q_emb)
        D, I = artifacts.faiss_index.search(q_emb.astype(np.float32), top_k_dense)
        for score, idx in zip(D[0], I[0]):
            dense_scores[int(idx)] = float(score)

    # Sparse via BM25
    sparse_scores: Dict[int, float] = {}
    if artifacts.bm25 is not None and len(q_tokens) > 0:
        s = artifacts.bm25.get_scores(q_tokens)
        s = np.array(s) if isinstance(s, list) else s
        top_idx = np.argpartition(-s, min(top_k_sparse, len(s)-1))[:top_k_sparse]
        for idx in top_idx:
            sparse_scores[int(idx)] = float(s[idx])

    # Union & normalize each modality
    candidate_ids = set(dense_scores) | set(sparse_scores)
    if not candidate_ids:
        return []

    def normalize_map(m: Dict[int, float]) -> Dict[int, float]:
        if not m:
            return {}
        vals = np.array(list(m.values()), dtype=float)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-12:
            return {k: 0.0 for k in m}
        return {k: (v - vmin) / (vmax - vmin) for k, v in m.items()}

    dense_n = normalize_map(dense_scores)
    sparse_n = normalize_map(sparse_scores)

    combined: List[Tuple[int, float, float, float]] = []
    for idx in candidate_ids:
        ds = dense_n.get(idx, 0.0)
        ss = sparse_n.get(idx, 0.0)
        final = dense_weight * ds + (1.0 - dense_weight) * ss
        combined.append((idx, final, dense_scores.get(idx, 0.0), sparse_scores.get(idx, 0.0)))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k_final]

# -----------------------------
# LLM orchestration (v3)
# -----------------------------
def pick_llm_model(client: OpenAI) -> str:
    try:
        resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        models_data = resp.json()

        # Extract the list from the 'data' key
        model_list = models_data.get("data", [])

        # Randomly select a model
        selected_model = ""
        while selected_model == "text-embedding-nomic-embed-text-v1.5" or selected_model == "" or selected_model == "liquid/lfm2-1.2b":
            selected_model = random.choice(model_list)["id"]
        return selected_model
    except Exception:
        return "error"


def build_sources_prompt(df: pd.DataFrame, hits: List[Tuple[int, float, float, float]]) -> str:
    lines = []
    for rank, (idx, comb, ds, ss) in enumerate(hits, 1):
        row = df.iloc[idx]
        ticket_no = row.get("Ticket #", "N/A")
        problem = str(row.get("Problem", ""))[:500]
        solution = str(row.get("Solution", ""))[:1500]
        lines.append(
            f"[Candidate {rank}] Ticket: {ticket_no} | CombinedScore: {comb:.3f} | Dense: {ds:.3f} | Lexical: {ss:.3f}\n"
            f"Problem: {problem}\nResolution: {solution}\n"
        )
    return "\n".join(lines)

def contextualize_with_llm(query: str, sources: str, client: OpenAI, model_id: str) -> str:
    conversation = [
        {
            "role": "system",
            "content": (
                "You are an expert IT support assistant. Synthesize and explain how to fix the user's issue using ONLY the provided candidate resolutions. "
                "Combine overlap, remove contradictions, and produce clear, step-by-step instructions. "
                "Do not invent facts. If candidates are irrelevant, say so briefly and ask for the missing signal. "
                "Address the user directly and avoid meta-talk. No Markdown in your response."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User issue:\n{query}\n\n"
                f"Candidate resolutions & context:\n{sources}\n\n"
                "Write a single cohesive set of instructions tailored to the user's issue, drawing strictly from the candidates above. "
                "Avoid specific dates, names, serial numbers, or confidential details. Ignore irrelevant candidates."
            ),
        },
    ]

    model_id = pick_llm_model(client)

    completion = client.chat.completions.create(
        model=model_id, messages=conversation, temperature=0.1, stream=True
    )
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
    response += "\n\n(Generated by " + model_id + ")"
    return response

# -----------------------------
# Init clients + artifacts (v3)
# -----------------------------
@st.cache_resource
def init_clients_and_artifacts():
    artifacts = setup_once_cached()
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")
    model_id = pick_llm_model(client)
    return artifacts, client, model_id

artifacts, client, default_llm_model = init_clients_and_artifacts()

# -----------------------------
# v2-style app state & helpers
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "all_follow_up" not in st.session_state:
    st.session_state["all_follow_up"] = []
if "show_solution" not in st.session_state:
    st.session_state["show_solution"] = False

@concurrency_limiter(max_concurrency=1)
def save_chat_history():
    logs_folder = "Logs"
    os.makedirs(logs_folder, exist_ok=True)
    chat_history = st.session_state.get("messages", [])
    log_content = "\n".join(f"{m['sender']}: {m['message']}" for m in chat_history)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = os.path.join(logs_folder, f"LOG_{timestamp}.txt")
    with open(filename, "w") as f:
        f.write(log_content)
    st.success(f"Chat history saved to {filename}!")

@concurrency_limiter(max_concurrency=1)
def add_message(sender: str, message: str):
    st.session_state["messages"].append({"sender": sender, "message": message})

# Requery logic = v2 triggers + v3 low-confidence gate
@concurrency_limiter(max_concurrency=1)
def should_requery(problem_description: str) -> bool:
    requery_keywords = ['didn’t work', 'not solved', 'try again', 'recheck', 'failed', 'problem persists']
    retry_indicators = ['retry', 'recheck', 'check again', 'try again']
    low_confidence = st.session_state.get("avg_confidence", 1.0) < 0.35

    text = problem_description.lower()
    if any(k in text for k in requery_keywords): return True
    if any(k in text for k in retry_indicators): return True
    if "new issue" in text: return True
    if low_confidence: return True
    return False

# Core handler: v3 retrieval + v2 return shape
@concurrency_limiter(max_concurrency=1)
def handle_problem(problem_description: str):
    hits = hybrid_search(artifacts, problem_description)
    if not hits:
        return (
            "Based on the provided details, no immediate solutions could be identified. "
            "Consider revisiting the initial context or seeking alternative expertise.",
            0.0, None, None
        )

    st.session_state["solution_indices"] = [h[0] for h in hits]

    predicted_solutions = []
    for idx, comb, ds, ss in hits:
        if comb >= 0.3:  # keep only 30%+
            predicted_solutions.append((artifacts.df["Solution"].iloc[idx], float(comb)))

    avg_conf = float(np.mean([h[1] for h in hits]))
    sources = build_sources_prompt(artifacts.df, hits)
    response = contextualize_with_llm(problem_description, sources, client, default_llm_model)
    return response, avg_conf, predicted_solutions, hits

@concurrency_limiter(max_concurrency=1)
def handle_follow_up(full_context: str):
    last_hits = st.session_state.get("last_hits", [])
    if not last_hits:
        return handle_problem(full_context)[0]
    sources = build_sources_prompt(artifacts.df, last_hits)
    return contextualize_with_llm(full_context, sources, client, default_llm_model)

# -----------------------------
# v2 UI styling
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Parkinsans:wght@300..800&display=swap');
    h1 { font-family: 'Parkinsans', sans-serif; font-size: 2.5rem; font-weight: 600; text-align: center; margin-bottom: 20px; }
    html, body, [class*="css"] { font-family: 'Parkinsans', sans-serif; }
    .chat-container { display: flex; flex-direction: column; padding: 20px; margin: 0 auto; margin-bottom: 70px; max-width: 800px; min-height: 80vh; }
    .user-message { background-color: #00c1d8; color: white; border-radius: 15px 15px 0 15px; padding: 12px 16px; margin: 10px 0; width: fit-content; max-width: 70%; text-align: left; margin-left: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); font-size: 14px; line-height: 1.5; }
    .assistant-message { background-color: #004b9a; color: white; border-radius: 15px 15px 15px 0; padding: 12px 16px; margin: 10px 0; width: fit-content; max-width: 70%; text-align: left; margin-right: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); font-size: 14px; line-height: 1.5; }
    .main-container { padding-bottom: 80px; height: 100%; overflow-y: auto; }
    .chat-input-container { position: fixed; bottom: 0; left: 0; width: 100%; display: flex; align-items: center; background-color: #ffffff; padding: 12px 20px; border-top: 1px solid #e0e0e0; box-shadow: 0 -2px 5px rgba(0,0,0,0.1); z-index: 9999; }
    .chat-input { flex-grow: 1; border: 1px solid #dcdcdc; border-radius: 20px; padding: 10px 15px; font-size: 14px; outline: none; box-shadow: inset 0 2px 5px rgba(0,0,0,0.05); transition: border-color 0.2s; }
    .chat-input:focus { border-color: #00c1d8; }
    .chat-send-button { background-color: #00c1d8; color: white; border: none; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-left: 10px; cursor: pointer; box-shadow: 0 2px 5px rgba(0,0,0,0.1); transition: background-color 0.2s, transform 0.2s; }
    .chat-send-button:hover { background-color: #009fb2; transform: scale(1.1); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Welcome to BramBot")
st.caption("You are accountable for everything that you say to the client. Use responses from this site to your discretion.")

# Render chat history (v2 style)
for message in st.session_state["messages"]:
    if message["sender"] == "User":
        st.markdown(f"<div class='user-message'>{message['message']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-message'>{message['message']}</div>", unsafe_allow_html=True)

# Diagnostics: confidence
if "avg_confidence" in st.session_state:
    st.write(f"Average Confidence: {st.session_state['avg_confidence']*100:.0f}%")

# Toggle predicted solutions (show combined score as “confidence”)
if "predicted_solutions_with_confidences" in st.session_state:
    st.session_state["show_solution"] = st.checkbox("Show Predicted Solutions", st.session_state["show_solution"])
    if st.session_state["show_solution"] and st.session_state["predicted_solutions_with_confidences"] is not None:
        solutions_text = "\n\n".join(
            [
                (
                    f"Solution {i+1} "
                    f"(Confidence: {conf*100:.1f}% | "
                    f"Ticket #: { (int(artifacts.df['Ticket #'].iloc[st.session_state['solution_indices'][i]])\
                        if pd.notna(artifacts.df['Ticket #'].iloc[st.session_state['solution_indices'][i]]) and str(artifacts.df['Ticket #'].iloc[st.session_state['solution_indices'][i]]).strip() != '' else 'N/A' ) }): {sol}"
                )
                for i, (sol, conf) in enumerate(st.session_state["predicted_solutions_with_confidences"])
            ]
        )
        st.text_area("Predicted Solutions:", solutions_text, height=150, disabled=True)

# -----------------------------
# v2 Input + Send button flow
# -----------------------------
@concurrency_limiter(max_concurrency=1)
def send_message():
    problem_description = st.session_state.input_text
    if not problem_description:
        return

    if "initial_problem" not in st.session_state:
        st.session_state["initial_problem"] = problem_description
        st.session_state["all_follow_up"] = []
        add_message("User", problem_description)
        st.session_state.input_text = ""

        add_message("Assistant", "Thinking...")
        with st.spinner("Generating response..."):
            t0 = time.time()
            response, avg_conf, predicted, hits = handle_problem(problem_description)
            elapsed = time.time() - t0

        st.session_state["messages"].pop()
        add_message("Assistant", f"{response}\n\n(Response time: {elapsed:.2f} seconds)")
        st.session_state["avg_confidence"] = avg_conf
        st.session_state["predicted_solutions_with_confidences"] = predicted
        st.session_state["last_hits"] = hits
    else:
        # Follow-ups keep v2 behavior with v3 requery logic
        st.session_state["all_follow_up"].append(problem_description)
        full_context = f"{st.session_state['initial_problem']} " + " ".join(
            [f"Follow-up: {fo}" for fo in st.session_state["all_follow_up"]]
        )

        add_message("User", problem_description)
        st.session_state.input_text = ""

        if should_requery(full_context):
            st.session_state["initial_problem"] = full_context
            add_message("Assistant", "Thinking...")
            with st.spinner("Generating response..."):
                response, avg_conf, predicted, hits = handle_problem(full_context)
            st.session_state["messages"].pop()
            add_message("Assistant", response)
            st.session_state["avg_confidence"] = avg_conf
            st.session_state["predicted_solutions_with_confidences"] = predicted
            st.session_state["last_hits"] = hits
        else:
            add_message("Assistant", "Thinking...")
            with st.spinner("Generating response..."):
                response = handle_follow_up(full_context)
            st.session_state["messages"].pop()
            add_message("Assistant", response)

# Input row (v2 style)
with st.container():
    cols = st.columns([4, 1])
    with cols[0]:
        st.text_input(
            "Please describe your IT problem:",
            key="input_text",
            placeholder="Type your message here...",
            label_visibility="collapsed",
        )
    with cols[1]:
        st.button("Send", on_click=send_message, key="send_button", use_container_width=True)

# Sidebar (v2)
with st.sidebar:
    st.markdown("### Options")
    if st.button("Reset Conversation"):
        st.session_state["messages"] = []
        st.session_state.pop("initial_problem", None)
        st.session_state.pop("avg_confidence", None)
        st.session_state.pop("predicted_solutions_with_confidences", None)
        st.session_state.pop("last_hits", None)
        st.session_state["show_solution"] = False
        st.session_state["all_follow_up"] = []

    if st.button("Report Conversation"):
        st.markdown(
            "<div style='text-align: center;'><strong style='color: red;'>Please submit feedback related to the reported conversation below.</strong></div>",
            unsafe_allow_html=True,
        )
        save_chat_history()

    st.markdown(
        """
        <div style='text-align: center; margin-top: 20px;'>
            <a href="https://forms.gle/RYGRK5c7jhybijGz8" target="_blank">
                <button style="background-color: #00c1d8; color: white; border: none; padding: 10px 20px; 
                border-radius: 5px; cursor: pointer;">Feedback</button>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
