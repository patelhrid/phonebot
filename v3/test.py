import joblib

from v3.main import clean_text, tokenize_and_remove_stopwords

# Load model and vectorizer
clf = joblib.load('decision_tree_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

new_problem_descriptions = [
    "My laptop screen is flickering and showing strange colors",
    "Cannot access network drives from my office computer",
    "Emails are not syncing on my mobile device",
    "Printer is not printing documents properly",
    "Software crashes when I open it on my work computer",
    "Wi-Fi connection drops frequently on my laptop",
    "My monitor is displaying 'no signal' despite being connected",
    "Mouse cursor is frozen and unresponsive on my desktop",
    "Webcam is not recognized by any application on my laptop",
    "Keyboard keys are sticking and not registering properly",
    "Sound is not coming through headphones connected to my computer",
    "Battery on my laptop drains very quickly even when fully charged",
    "External hard drive is not showing up on my computer",
    "My computer is running very slow and applications are lagging",
    "Cannot login to MacBook - Password not accepted",
    "Laptop just has spinning circle not able to login; requesting for someone to go down to have a look",
    "When I try to make a call it says 'your home cellular company has not authorized us to provide service.'",
    "Please release any emails from support@cateater.com",
    "Need the license key ASAP for the recently purchased software Stop Motion Pro",
    "Android phone not updating",
    "Issue with transit self-service, unable to log in",
    "Client was informed that PIR would be loaded automatically for Camper website, but it has not appeared",
    "22H2 update keeps failing",
    "Issues with authenticator app number for Austin",
    "Unable to log into city branch email for completing trainings",
    "TLMS is not working for new employee Athavi Jeyakumar",
    "Client's USB-C charger has stopped working, needs replacement",
    "Outlook not loading",
    "Laptop will not turn on, stuck on black screen with blue and gray menu options",
    "PM Scan not allowing user to login, needs password reset",
    "New phone switch causing issues with MF and logins to Teams and Outlook",
    "Unable to access MeOnline after changing password",
    "Outlook on phone not syncing all meetings",
    "Not able to log in to MeOnline after password reset",
    "Issue with 22H2 update not appearing in Software Center",
    "Unable to update or cancel own meeting in webmail",
    "Unable to log into TLMS, needs assistance",
    "Sunsecutor app not sending the code, unable to log in",
    "Cannot log into account, keeps saying password is incorrect despite multiple changes",
    "Edge keeps crashing, IP address error",
    "Software Center failing to load applications",
    "Unable to connect to VPN on City Laptop",
    "MFA issue on personal device, closed all applications to resolve",
    "PC password change not reflecting on Macbook",
    "Error message when accessing secondary mailbox, cannot display folder",
    "Waiting to load profile message on laptop startup",
    "Charging cable for docking station not working",
    "Unable to use Scan to Send function on Canon printer",
    "All WiFi and cellular data turned off on Panasonic Toughbook",
    "Printing foggy grey line on pages from Ricoh MP printer",
    "Issues connecting to hotspot on Toughbook",
    "Second printer in print room has line through middle of page",
    "Via One X agent not logging in, login error",
    "Trouble logging into laptop on domain",
    "Teams missing on shared computer after Office 365 update",
    "ICON remote not working",
    "POS system not powering on at location, need replacement unit",
    "Teams keeps kicking out of meetings, updated app version",
    "Invalid username or password error when trying to log into desktop PC",
    "Bitlocker popup preventing login",
    "Unable to reset password for account access at Sports Center",
    "Teams and Outlook not working on personal devices, functioning at facility",
    "Removed from exception list but still unable to set up MFA",
    "Invalid credentials error when trying to access MeOnline",
    "Outlook app not opening after adding general email account",
    "Extremely slow performance from Amanda's computer",
    "Calls not recording for CSA at Service Brampton",
    "Moneris terminals not initializing, unable to process payments",
    "Cannot login to ICON TATGB501, currently in use",
    "Printer has over 300 jobs in queue, not printing anything"
]


# Preprocess new problem descriptions
preprocessed_new_descriptions = [clean_text(text) for text in new_problem_descriptions]
preprocessed_new_descriptions = [tokenize_and_remove_stopwords(text) for text in preprocessed_new_descriptions]

# Vectorize using TF-IDF vectorizer
X_new_tfidf = tfidf_vectorizer.transform(preprocessed_new_descriptions)

# Predict solutions
predicted_solutions = clf.predict(X_new_tfidf)

# Print predictions
print("Predictions:")
for description, solution in zip(new_problem_descriptions, predicted_solutions):
    print(f"Problem: {description}")
    print(f"Predicted Solution: {solution}")
    print("-" * 75)

# # Optionally, evaluate the classifier on the test set
# y_pred = clf.predict(X_test_tfidf)
# print("\nClassification Report on Test Set:")
# print(classification_report(y_test, y_pred, zero_division=0))
