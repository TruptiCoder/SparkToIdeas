import tkinter as tk
from tkinter import font as tkFont
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Predict function
def predict_diabetes():
    try:
        input_data = [float(entry.get()) for entry in entries]
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        result = "Diabetic ❌" if prediction[0] == 1 else "Non-Diabetic ✅"
        result_label.config(text=f"➤ Result: {result}", fg="#dc2626" if prediction[0] == 1 else "#16a34a")
    except ValueError:
        result_label.config(text="➤ Please enter valid numeric values!", fg="#dc2626")

# GUI window
root = tk.Tk()
root.title("Diabetes Prediction App")
root.configure(bg="#f0f2f5")
root.state("zoomed")

# Main container frame
container = tk.Frame(root, bg="white", padx=40, pady=40)
container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Heading
heading = tk.Label(container, text="🔬 Diabetes Prediction App", 
                   font=("Segoe UI", 24, "bold"), bg="white", fg="#1f2937")
heading.pack(pady=(0, 30))

# Input fields
labels = [
    "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
    "Insulin", "BMI", "Diabetes Pedigree Function", "Age"
]

entries = []

# Grid-style layout: 2 per row
form_frame = tk.Frame(container, bg="white")
form_frame.pack()

for i in range(0, len(labels), 2):
    row_frame = tk.Frame(form_frame, bg="white")
    row_frame.pack(pady=8, fill="x")

    for j in range(2):
        if i + j < len(labels):
            field_frame = tk.Frame(row_frame, bg="white")
            field_frame.pack(side="left", padx=20)

            label = tk.Label(field_frame, text=labels[i + j], font=("Segoe UI", 12), 
                             bg="white", fg="#374151", anchor="w")
            label.pack(anchor="w")

            entry = tk.Entry(field_frame, font=("Segoe UI", 12), width=25,
                             bg="#f9fafb", fg="#111827", bd=1, relief="solid")
            entry.pack()
            entries.append(entry)

# Predict Button
predict_btn = tk.Button(container, text="Predict Diabetes", command=predict_diabetes,
                        font=("Segoe UI", 14, "bold"), bg="#6366f1", fg="white",
                        activebackground="#4f46e5", activeforeground="white",
                        padx=20, pady=10, bd=0, relief="flat", cursor="hand2")
predict_btn.pack(pady=30)

# Result display
result_label = tk.Label(container, text="", font=("Segoe UI", 16, "bold"),
                        bg="white", fg="#1f2937")
result_label.pack(pady=10)

# Run the GUI loop
root.mainloop()
