import pandas as pd

if __name__ == "__main__":
    file_name = "classifications_improved_tree.csv"
    df = pd.read_csv(file_name)
    print(df.columns)
    statuses = []
    flags = []
    
    for i,row in df.iterrows():
        if pd.isna(row["classification"]):
            statuses.append(None)
            continue
        print("======================")
        print(f"Original item name: {row["item"]}")
        print(f"Imputed item name: {row["imputed_desc"]}")
        print(f"Classification: {row["classification"]}")
        user_status = None
        while user_status is None:
            temp_status = input("Enter i/c for incorrect/correct: ").strip().lower()
            if temp_status in ["i", "c"]:
                user_status = temp_status
                
        statuses.append(user_status)
        
        user_flag = None
        while user_flag is None:
            temp_flag = input("Enter y/n to flag or not flag: ").strip().lower()
            if temp_flag in ["y", "n"]:
                user_flag = temp_flag
        
        flags.append(user_flag)
        
    df["status"] = statuses
    df.to_csv(f"labeled_{file_name}")
            
    