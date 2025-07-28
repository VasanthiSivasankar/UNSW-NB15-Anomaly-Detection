# Anomaly Detection on UNSW-NB15 Dataset using IsolationForest

This project demonstrates how to detect anomalies in the UNSW-NB15 network intrusion dataset using the IsolationForest algorithm. The workflow includes data preprocessing, model training, result analysis, and visualization.

## Dataset
- **File:** `UNSW-NB15_4.csv`
- **Description:** A subset of the UNSW-NB15 dataset, commonly used for network intrusion detection research.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
Run the anomaly detection script:
```bash
python anomaly_det.py
```

## Workflow Overview
1. **Data Loading & Inspection**
   - Loads the CSV file and prints dataset shape, columns, and sample rows.
   - Checks for missing values and data types.

2. **Feature Preparation**
   - Identifies numerical and categorical columns.
   - Encodes categorical features using `LabelEncoder`.
   - Excludes target columns (like `attack_cat`, `label`, `id`) from features.

3. **Model Training**
   - Trains an `IsolationForest` model with a contamination rate of 10% (i.e., expects 10% anomalies).
   - Predicts anomaly labels and computes anomaly scores for each sample.

4. **Result Analysis**
   - Adds anomaly labels and scores to the original dataframe.
   - Prints statistics: total samples, number of anomalies, anomaly rate.
   - Saves results to `anomaly_detection_results.csv`.

5. **Feature Importance**
   - Calculates the absolute correlation between each numerical feature and the anomaly score.
   - Prints and plots the top 10 most important features for anomaly detection.
   - Saves the plot as `feature_importance.png`.

6. **Visualization**
   - Generates and saves plots:
     - Distribution of anomaly scores
     - Pie chart of anomaly vs. normal samples
     - Boxplot of anomaly scores by class
     - Cumulative distribution of anomaly scores
   - All plots are saved as `anomaly_detection_results.png`.
   - Plots are shown at the end of the script (after all results are printed and saved).

## Output Files
- `anomaly_detection_results.csv`: Data with anomaly labels and scores
- `anomaly_detection_results.png`: Visualization of results
- `feature_importance.png`: Feature importance plot

## Notes
- The script is designed to print all results and save files before showing any plot windows, so you can review the output in the terminal first.
- The contamination rate in IsolationForest can be adjusted in the script if you expect a different proportion of anomalies.

## Customization
- To change the contamination rate, modify the `contamination` parameter in the `train_isolation_forest` function call in `main()`.
- To use a different dataset, update the `file_path` variable in `main()`.

## References
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [IsolationForest Documentation (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) 