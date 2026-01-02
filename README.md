# Cancer Treatment Response Prediction

This project uses a machine learning model to explore whether differences in clinical and biological features are associated with treatment response in breast cancer patients.

The model is trained on real-world clinical data and deployed as an interactive Streamlit application for exploratory and educational purposes.

## Motivation

Treatment response in breast cancer varies widely between patients, and it can be challenging to predict outcomes using clinical information alone. In this project, I explore whether commonly available clinicopathologic features—such as tumor cellularity, lymph node involvement, histological subtype, menopausal status, and age at diagnosis—contain useful signal for estimating treatment response

## Dataset

- **Source:** METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)
- **Data used:** Clinical and pathological patient features
- **Outcome:** Recurrence-free survival status, used to derive a binary treatment response label

> Note: Gene expression data was not used in this version of the model.

## Features Used

- Age at diagnosis  
- Number of positive lymph nodes  
- Tumor cellularity  
- Histological subtype  
- Menopausal status  
- Integrative cluster assignment  

## Model

- Random Forest Classifier
- Binary classification (likely vs less likely to respond)
- **Evaluation:** Accuracy, precision, recall, F1-score
- **Performance:** ~61% accuracy using clinical features only
