import pandas as pd

if __name__ == "__main__":
    file_name = "classification_results/labeled_classifications_v4_improved.csv"
    df = pd.read_csv(file_name)
    print(df.columns)
    
    n_na = len(df[pd.isna(df["status"])])
    n_correct = len(df[df["status"] == 'c'])
    n_incorrect = len(df[df["status"] == 'i'])
    
    n_total = len(df)
    
    p_correct = (n_correct / n_total) * 100
    p_incorrect = (n_incorrect / n_total) * 100
    p_na = (n_na / n_total) * 100
    
    print(f"""
          Results for {file_name}
          Percent Correct: {p_correct:.2f}%
          Percent Incorrect: {p_incorrect:.2f}%
          Percent Unanswered: {p_na:.2f}%
          """)


# Results for labeled_classifications_improved_tree.csv
# Percent Correct: 49.00%
# Percent Incorrect: 18.00%
# Percent Unanswered: 33.00%


# Results for labeled_classifications_cleaned_tree.csv
# Percent Correct: 56.00%
# Percent Incorrect: 15.00%
# Percent Unanswered: 29.00%


# Results for labeled_classifications_v4.csv
# Percent Correct: 44.00%
# Percent Incorrect: 46.00%
# Percent Unanswered: 10.00%


# Results for classification_results/labeled_classifications_v4_improved.csv
# Percent Correct: 54.00%
# Percent Incorrect: 14.00%
# Percent Unanswered: 32.00%
