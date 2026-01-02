# Cancer Treatment Response Prediction

This project uses a machine learning model to explore whether differences in clinical and biological features are associated with treatment response in breast cancer patients.

The model is trained on real-world clinical data and deployed as an interactive Streamlit application for educational purposes.

Test the deployed app [here](https://sarumu2421-cancer-treatment-prediction-streamlit-app-y3jbm1.streamlit.app/)!

**Motivation:**
Treatment response in breast cancer varies widely between patients, and it can be challenging to predict outcomes using clinical information alone. In this project, I explore whether commonly available clinicopathologic features, like tumor cellularity, lymph node involvement, histological subtype, menopausal status, and age at diagnosis, contain useful signal for estimating treatment response. A tool like this could hat help doctor better understand patient risk profiles, compare treatment options, and identify patients who may benefit from certain therapies.

**Dataset:** METABRIC (Molecular Taxonomy of Breast Cancer International Consortium), 2,509 patients


**Features Used:**
- Age at diagnosis  
- Number of positive lymph nodes  
- Tumor cellularity  
- Histological subtype  
- Menopausal status  
- Integrative cluster assignment  

**Model:**
- Random Forest Classifier
- Binary classification (likely vs less likely to respond)
- Evaluation: Accuracy, precision, recall, F1-score
- Performance: ~62% accuracy using clinical features only

**Future:**
This project only uses clinical features, but treatment response in breast cancer is also influenced by genetic factors. In the future, this model could be extended by including gene expression data from the METABRIC dataset to see if it improves prediction performance.
