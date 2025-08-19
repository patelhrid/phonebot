import textwrap

import joblib
from openai import OpenAI

from v3.main import clean_text, tokenize_and_remove_stopwords

# Load model and vectorizer
clf = joblib.load('knn_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize LM Studio client
client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")

# Define confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Define a function to contextualize the output using LM Studio
def contextualize_response(problem, solution, confidence):
    history = [
        {"role": "system",
         "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
        {"role": "user",
         "content": f"The problem is: '{problem}' The predicted solution is: '{solution}'. Please explain the "
                    f"solution to the user in a well-structured sentence. If it isn't a good solution,"
                    f"say \"Seek help elsewhere.\". Get straight to the point. Don't mention "
                    f"'the solution', because I do not know what that even is, and instead talk about it like it's "
                    f"your own. Your response must be instructional to myself, an IT Support agent, and you should be "
                    f"telling me what to do. The solution belongs to you, so instruct me on the solution like I've"
                    f"never heard about it before."}
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

    # Predict solution and get confidence score
    predicted_solution = clf.predict(X_new_tfidf)[0]
    confidence_score = clf.predict_proba(X_new_tfidf).max()

    # Contextualize the response
    if confidence_score < CONFIDENCE_THRESHOLD:
        return "Seek help from other sources.", confidence_score

    contextualized_response = contextualize_response(problem_description, predicted_solution, confidence_score)
    return contextualized_response, confidence_score


# Main loop to interact with the user
while True:
    problem_description = input("Please describe your IT problem: ")
    if problem_description.lower() in ["exit", "quit"]:
        break

    print("Generating response...")
    response, confidence = handle_problem(problem_description)

    if confidence != 1.00:
        print(f"\nConfidence Score: {confidence:.2f}")

    wrapped_response = textwrap.fill(response, width=80, initial_indent='    ', subsequent_indent='    ')
    print(f"\nGenerated Response:\n{wrapped_response}\n")
    print("-" * 75 + "\n")
