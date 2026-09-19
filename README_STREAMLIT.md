# Streamlit Student Support Predictor

## Files

Keep these files together:

- `app.py`
- `student-mat.csv`
- `requirements.txt`

## Run

Open the VS Code terminal in this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interface design

The Streamlit interface intentionally does NOT ask the user to enter all
dataset features. It asks a small number of understandable questions:

- Age
- Weekly study time
- Previous class failures
- Absences
- Intention to continue to higher education
- School academic support
- Family academic support
- Internet access at home

The trained Random Forest still uses the full set of non-leakage predictors.
Features not requested in the interface are filled with typical reference
values from the dataset. This keeps the demonstration simple for a non-ML
user while preserving the capstone model.

G1 and G2 are excluded from the main model because they are previous grades
and would make the prediction less useful as an early-support tool.

## Presentation demonstration

1. Open the app.
2. Explain that a normal teacher does not need to understand machine learning.
3. Enter a hypothetical student's information.
4. Click "Estimate Final Grade".
5. Explain that the result is a support signal, not a final judgement.
