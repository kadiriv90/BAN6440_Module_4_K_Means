# K-Means Clustering for IT Audit Monitoring & Fraud Detection

## Project Overview  
This project implements a **K-Means clustering application** to analyze network traffic data for IT auditing and fraud detection. The main goal is to group network activity into clusters, helping auditors detect unusual behavior or potential fraudulent transactions by super users.  

The application uses a subset of the **CSE-CIC-IDS2018 dataset** (Processed Traffic Data) and demonstrates how unsupervised machine learning can highlight anomalies without prior labeling.

---

## Key Features  
- Loads and combines multiple CSV files from a specified dataset folder.  
- Preprocesses data: converts to numeric, handles missing or infinite values, and clips extreme values.  
- Standardizes features using **feature scaling** to ensure all variables contribute equally.  
- Performs **K-Means clustering** on scaled data to identify groups of similar network behaviors.  
- Uses **Principal Component Analysis (PCA)** to reduce dimensions for visualization.  
- Provides a visual representation of clusters to easily spot anomalies.  
- Unit tests ensure reliability and robustness of the application.

---

## Project Structure  
```
KMeans_IT_Audit/
│
├─ data/
│   └─ processed_traffic/         # Folder containing CSV files from CSE-CIC-IDS2018
│
├─ src/
│   └─ kmeans_clustering.py       # Main Python application
│
├─ tests/
│   └─ kmeans_unit_testing.py  # Unit tests for the application
│
├─ README.md                      # Project documentation
└─ .gitignore                     # Optional: ignore files/folders from Git
```

---

## Dataset  
- **Name:** CSE-CIC-IDS2018 (Processed Traffic Data)  
- **Source:** [AWS Open Data Registry](https://registry.opendata.aws/cse-cic-ids2018/) 
- **Subset Used:** A representative subset (max 2 CSV files) for computational efficiency while preserving behavioral diversity.  
- **Selected csv files:** 1.Friday-02-03-2018_TrafficForML_CICFlowMeter.csv and 2. Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
- **Features:** Numeric features including packet statistics, flow information, timestamps, and protocol data.  
- **Important Note:** The data files sourced from AWS Open Data Registry (CSE-CIC-IDS2018) were not pushed to Github due to size constraints
---

## Installation & Setup  

1. **Clone the repository:**
```bash
git clone https://github.com/YourUsername/KMeans_IT_Audit.git
cd KMeans_IT_Audit
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate       # Windows
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```
*(Example packages include `pandas`, `numpy`, `scikit-learn`, `matplotlib`.)*

4. **Add dataset:** Place the CSV files inside `data/processed_traffic/`.

---

## Usage  

Run the main application:
```bash
python src/kmeans_clustering.py
```
Workflow:  
1. Loads a subset of CSV files from `data/processed_traffic`.  
2. Preprocesses numeric features and handles missing/extreme values.  
3. Scales features using `StandardScaler`.  
4. Applies K-Means clustering to group network traffic into 4 clusters.  
5. Reduces dimensions using PCA and visualizes clusters in a 2D plot.  

---

## Unit Testing  

The project includes unit tests to ensure robustness:
```bash
python -m unittest discover -s tests
```
Tests verify:  
- Correct loading of data  
- Proper preprocessing (handling NaN, inf, and extreme values)  
- Correct feature scaling  
- Valid clustering output  
- Successful visualization without errors  

Unit testing ensures that the application behaves reliably for IT auditing purposes.

---

## Results & Insights  
- K-Means clustering successfully divided network traffic into four clusters.  
- PCA visualization showed normal traffic grouped together, while clusters with unusual or extreme behavior highlighted potential security threats.  
- The approach helps auditors prioritize investigation and detect potential fraud or misuse efficiently.

---

## Challenges and Solutions  
- **Extreme values & missing data:** Handled during preprocessing using clipping and NaN replacement.  
- **High-dimensional data:** PCA reduced dimensions to 2 for visualization.  
- **Bias from outliers:** Preprocessing and scaling ensured fair contribution of all features.  
- **Reliability:** Unit testing validated each step of the workflow.

---

## References  
Aggarwal, C. C. (2015). *Data mining: The textbook*. Springer. https://doi.org/10.1007/978-3-319-14142-8  

Han, J., Kamber, M., & Pei, J. (2012). *Data mining: Concepts and techniques* (3rd ed.). Morgan Kaufmann. https://www.sciencedirect.com/book/9780123814791/data-mining  

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 108–116. https://doi.org/10.5220/0006633801080116  

---

## License  
This project is for academic purposes. You may use it for learning or research but please cite the references above when reusing content.

