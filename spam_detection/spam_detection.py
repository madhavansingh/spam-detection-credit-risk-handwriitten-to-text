import pandas as pd
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ------------------ LOAD DATA ------------------
df = pd.read_csv("spam.csv")

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

# ------------------ VECTORIZATION ------------------
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# ------------------ TRAIN MODEL ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# ------------------ GUI FUNCTION ------------------
def predict_spam():
    message = text_entry.get("1.0", tk.END).strip()

    if message == "":
        messagebox.showwarning("Input Error", "Please enter a message!")
        return

    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        messagebox.showinfo("Result", "🚨 This message is SPAM")
    else:
        messagebox.showinfo("Result", "✅ This message is NOT SPAM")

# ------------------ TKINTER UI ------------------
root = tk.Tk()
root.title("Spam Detection System")
root.geometry("400x300")

tk.Label(
    root,
    text="Enter message to check spam:",
    font=("Arial", 12)
).pack(pady=10)

text_entry = tk.Text(root, height=6, width=40)
text_entry.pack(pady=5)

tk.Button(
    root,
    text="Check Spam",
    command=predict_spam,
    font=("Arial", 11),
    bg="#4CAF50",
    fg="white"
).pack(pady=20)

root.mainloop()
