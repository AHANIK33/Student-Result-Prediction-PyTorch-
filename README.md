# Student Result Prediction (PyTorch)

A feedforward neural network built in PyTorch to predict whether a student passes or fails based on study habits, attendance, and past performance.

## Overview

This project trains a small neural network on a compact student dataset to classify `result` (Pass/Fail) from four numeric features: study hours, attendance rate, previous marks, and assignment score. It also demonstrates saving/loading the trained model and scaler for inference on new students.

## Dataset

- **File:** `student.csv`
- **Rows:** 300 students, 5 columns, no missing values
- **Target:** `result` (binary: `1` = Pass, `0` = Fail) — fairly balanced (162 Pass / 138 Fail)

| Column | Description |
|---|---|
| `study_hours` | Hours studied (1–10) |
| `attendance` | Attendance rate (45–100) |
| `previous_marks` | Prior academic marks (35–90) |
| `assignment_score` | Assignment score (35–100) |
| `result` | Target: Pass (1) / Fail (0) |

> Note: the dataset is loaded from `/content/student.csv` (a Google Colab path). Update this path if running locally.

## Approach

1. **Preprocessing:**
   - Split: 80% train, 10% validation, 10% test (`random_state=42`)
   - Feature scaling with `StandardScaler`
   - Wrapped as PyTorch `TensorDataset`/`DataLoader` (`batch_size=32`)
2. **Model architecture** (`MyNN`, fully connected):
   - Input (4) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(2)
3. **Training:**
   - Loss: `CrossEntropyLoss`
   - Optimizer: Adam, `learning_rate=0.001`
   - Epochs: 100 (initial run), with an additional 20-epoch loop tracking train/val loss and accuracy
4. **Evaluation:** Accuracy, Precision, Recall, F1-score, and full classification report on the test set
5. **Persistence:** Model weights saved to `student_model.pth` (`torch.save`), scaler saved to `student_scaler.pkl` (`joblib`), both reloaded to run inference on a new student

## Results (Test Set)

| Metric | Score |
|---|---|
| Accuracy | 83.33% |
| Precision | 92.31% |
| Recall | 75.00% |
| F1 Score | 82.76% |

Validation accuracy during training reached 100%, while test accuracy settled at 83.3% — a sign of some overfitting, which isn't surprising given the dataset only has 300 rows. The model favors precision over recall: it's fairly conservative about predicting "Pass," missing some actual passes (lower recall) but rarely mislabeling a fail as a pass.

## Requirements

```
torch
pandas
numpy
scikit-learn
joblib
```

Install with:
```bash
pip install torch pandas numpy scikit-learn joblib
```

## Usage

1. Place `student.csv` in your working directory (update the path in the notebook if not using Colab).
2. Run `Student_result.ipynb` top to bottom. This trains the model and saves `student_model.pth` and `student_scaler.pkl`.
3. To predict a new student's result using the saved model:

```python
import torch
import joblib
import numpy as np

scaler = joblib.load("student_scaler.pkl")
model = MyNN(num_features=4)
model.load_state_dict(torch.load("student_model.pth", weights_only=True))
model.eval()

new_student = np.array([[6, 85, 70, 75]])  # study_hours, attendance, previous_marks, assignment_score
new_student_scaled = scaler.transform(new_student)
new_student_tensor = torch.tensor(new_student_scaled, dtype=torch.float32)

with torch.no_grad():
    output = model(new_student_tensor)
    _, prediction = torch.max(output, 1)

print("Pass" if prediction.item() == 1 else "Fail")
```

## Project Structure

```
.
├── Student_result.ipynb     # Data loading, model, training, evaluation, inference
├── student.csv                # Dataset (add your own)
├── student_model.pth          # Saved model weights (generated after running)
├── student_scaler.pkl         # Saved StandardScaler (generated after running)
└── README.md
```

## Future Improvements

- Use k-fold cross-validation given the small dataset size (300 rows) to get a more reliable performance estimate
- Add dropout or L2 regularization to reduce the train/test gap and overfitting
- Try simpler baselines (Logistic Regression, Random Forest) for comparison — a 4-feature dataset this small may not need a neural network
- Plot training/validation loss and accuracy curves over epochs
