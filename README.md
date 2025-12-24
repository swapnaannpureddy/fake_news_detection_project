# Attention-Driven Deep Neural Networks for Context-Aware Fake News Detection

## Features
- Clean, modular codebase
- BERT+Attention and BiLSTM+Attention options (choose at runtime)
- Data cleaning, contextual features, explainability via attention, LIME, SHAP
- Robust training with validation, checkpointing, and evaluation
- Streamlit front-end for demo
- Easily extensible for ensembles

## Instructions

1. Place your datasets (ISOT, FakeNewsNet, LIAR) in `/project/data`
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Preprocess data and train a model:
   ```
   python main.py
   ```
4. Run the web UI:
   ```
   streamlit run app/streamlit_app.py
   ```

## Collaboration
- Assign modules (data, model, train, frontend, explain) to different group members
- Use Git for version control
- Comment code and add docstrings for clarity

## Advanced
- Use LIME/SHAP explanations in app
- Try the ensemble mode for better performance