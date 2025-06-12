import tkinter as tk
from tkinter import messagebox
import joblib

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Predict function
def predict():
    msg = entry.get("1.0", tk.END).strip()
    if not msg:
        messagebox.showwarning("Input Error", "Please enter a message.")
        return
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)
    result = "Spam ❌" if prediction[0] == 1 else "Not Spam ✅"
    result_label.config(text=f"{result}", fg='#ff4d4d' if prediction[0] == 1 else '#28a745')

# Exit fullscreen on Escape
def exit_fullscreen(event):
    root.attributes('-fullscreen', False)

# Setup main window
root = tk.Tk()
root.title("Modern Spam Detector")
root.configure(bg="#f0f2f5")
root.attributes('-fullscreen', True)
root.bind("<Escape>", exit_fullscreen)

# Padded container (just padding without rounded corners)
container = tk.Frame(root, bg="white", padx=40, pady=30)
container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Title
title = tk.Label(container, text="🔍 Spam Detector", font=("Segoe UI", 26, "bold"),
                 bg="white", fg="#1f2937")
title.pack(pady=(0, 30))

# Text Entry
entry = tk.Text(container, height=8, width=70, font=("Segoe UI", 14),
                bg="#f9fafb", fg="#111827", bd=0, highlightthickness=2,
                highlightbackground="#cbd5e1", highlightcolor="#6366f1")
entry.pack(pady=10)

# Predict Button
predict_btn = tk.Button(container, text="Detect Message", command=predict,
                        font=("Segoe UI", 14, "bold"), bg="#6366f1", fg="white",
                        activebackground="#4f46e5", activeforeground="white",
                        padx=20, pady=10, bd=0, relief="flat", cursor="hand2")
predict_btn.pack(pady=20)

# Result Label
result_label = tk.Label(container, text="", font=("Segoe UI", 18, "bold"),
                        bg="white", fg="#1f2937")
result_label.pack(pady=(10, 0))

# Run the app
root.mainloop()
