import asyncio
import sys
import textwrap

import joblib
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QMessageBox
from openai import OpenAI
from qasync import QEventLoop, asyncSlot
from qt_material import apply_stylesheet

from v3.main import clean_text, tokenize_and_remove_stopwords
from v3.main_knn import df

# Load model and vectorizer
knn = joblib.load('knn_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize LM Studio client
client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")

# Define confidence threshold
DISTANCE_THRESHOLD = 0.8

class ITSupportApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('IT Support Assistant')

        # Layouts
        main_layout = QVBoxLayout()
        form_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # Problem description input
        self.problem_label = QLabel('Please describe your IT problem:')
        self.problem_input = QTextEdit()
        form_layout.addWidget(self.problem_label)
        form_layout.addWidget(self.problem_input)

        # Generate response button
        self.generate_button = QPushButton('Generate Response')
        self.generate_button.clicked.connect(self.on_generate_response)
        button_layout.addWidget(self.generate_button)

        # Display area for the response
        self.response_label = QLabel('Generated Response:')
        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True)
        form_layout.addWidget(self.response_label)
        form_layout.addWidget(self.response_output)

        # Detailed prediction button
        self.detailed_button = QPushButton('Show Predicted Solution')
        self.detailed_button.clicked.connect(self.show_predicted_solution)
        button_layout.addWidget(self.detailed_button)

        # Add layouts to the main layout
        main_layout.addLayout(form_layout)
        main_layout.addLayout(button_layout)

        # Spinner for loading
        self.spinner = QLabel(self)
        self.spinner.setMovie(QMovie('spinner.gif'))
        self.spinner.hide()
        form_layout.addWidget(self.spinner)

        self.setLayout(main_layout)

    @asyncSlot()
    async def on_generate_response(self):
        problem_description = self.problem_input.toPlainText()
        if not problem_description.strip():
            QMessageBox.warning(self, 'Input Error', 'Please enter a problem description.')
            return

        self.response_output.setText("")
        self.spinner.show()
        self.spinner.movie().start()
        await asyncio.sleep(0)  # Allow the event loop to process the spinner

        response, distance, predicted_solution = await asyncio.to_thread(handle_problem, problem_description)

        self.spinner.movie().stop()
        self.spinner.hide()
        wrapped_response = textwrap.fill(response, width=120, initial_indent='    ', subsequent_indent='    ')
        self.response_output.setText(wrapped_response)
        self.predicted_solution = predicted_solution

        if distance > DISTANCE_THRESHOLD:
            self.response_output.append("\nNote: The distance to the nearest neighbor is high, consider seeking additional help.")
        self.response_output.append(f"\nDistance: {distance:.2f}")

    def show_predicted_solution(self):
        if hasattr(self, 'predicted_solution'):
            QMessageBox.information(self, 'Predicted Solution', self.predicted_solution)
        else:
            QMessageBox.warning(self, 'No Solution', 'Please generate a response first.')

# Define the function to handle the problem
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
        return "Seek help from other sources.", distance, predicted_solution

    contextualized_response = contextualize_response(problem_description, predicted_solution, distance)
    return contextualized_response, distance, predicted_solution

# Define the function to contextualize the output using LM Studio
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

# Main function to run the application
def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    apply_stylesheet(app, theme='dark_teal.xml')
    ex = ITSupportApp()
    ex.show()
    with loop:
        loop.run_forever()

if __name__ == '__main__':
    main()
