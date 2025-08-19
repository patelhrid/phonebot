import textwrap
import joblib
from openai import OpenAI
from v3.main import clean_text, tokenize_and_remove_stopwords
from v3.main_knn import df

# Load model and vectorizer
knn = joblib.load('knn_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize LM Studio client
client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")

# Define confidence threshold
DISTANCE_THRESHOLD = 0.8

# Define a function to contextualize the output using LM Studio
def contextualize_response(problem, solution, distance):
    history = [
        {"role": "system",
         "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct "
                    "and helpful."},
        {"role": "user",
         "content": f"The problem is: '{problem}' The predicted solution is: '{solution}'. Please explain the "
                    f"solution to the user in a well-structured sentence. If it isn't a relevant solution at all, "
                    f"say \"Seek help elsewhere.\". Don't mention "
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
                    f"those lines."}
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

    contextualized_response = contextualize_response(problem_description, predicted_solution, distance)
    return contextualized_response, distance, predicted_solution

# Main loop to interact with the user
while True:
    problem_description = input("Please describe your IT problem: ")
    if problem_description.lower() in ["exit", "quit"]:
        break

    print("Generating response...")
    response, distance, predicted_solution = handle_problem(problem_description)

    print(f"\nDistance: {distance:.2f}")

    wrapped_response = textwrap.fill(response, width=80, initial_indent='    ', subsequent_indent='    ')
    print(f"\nGenerated Response:\n{wrapped_response}\n")

    detailed_prediction = input("Would you like to see the actual predicted solution? (y/n): ")
    if detailed_prediction.lower() == "y":
        print(f"\nPredicted Solution:\n{predicted_solution}\n")
    print("-" * 75 + "\n")
