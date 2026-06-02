import pandas as pd
import numpy as np
import os
from predict import predict_gameweek

def evaluate_all():
    df_hist = pd.read_csv("data/epl_clean.csv")
    df_hist["Date"] = pd.to_datetime(df_hist["Date"])
    
    # Load old log
    old_log = pd.read_csv("predictions/accuracy_log.csv")
    old_log.set_index("gameweek", inplace=True)
    
    new_results = []
    
    for gw in range(31, 39):
        fixture_path = f"predictions/gw{gw}_fixtures.csv"
        fallback_pred_path = f"predictions/gw{gw}_predictions.csv"
        if os.path.exists(fixture_path):
            fix_df = pd.read_csv(fixture_path)
        elif os.path.exists(fallback_pred_path):
            # GW32 trong project hien co prediction nhung khong co fixture file.
            # Dung home/away/date tu prediction cu de backtest model moi.
            fix_df = pd.read_csv(fallback_pred_path)
        else:
            continue
            
        fixtures = fix_df[["home", "away", "date"]].to_dict(orient="records")
        
        # Predict using new model
        print(f"\n--- Predicting GW {gw} ---")
        pred_df = predict_gameweek(fixtures, gameweek=gw, season="2025/26")
        
        # Now find actual results in df_hist
        correct = 0
        total = 0
        for _, row in pred_df.iterrows():
            home = row["home"]
            away = row["away"]
            # match by home/away and season
            actual = df_hist[(df_hist["HomeTeam"] == home) & (df_hist["AwayTeam"] == away) & (df_hist["Season"] == "2025/26")]
            if len(actual) > 0:
                actual_ftr = actual.iloc[0]["FTR"]
                label_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
                actual_label = label_map[actual_ftr]
                
                if row["prediction"] == actual_label:
                    correct += 1
                total += 1
                
        if total > 0:
            acc = correct / total
            old_acc = old_log.loc[gw, "accuracy"] if gw in old_log.index else np.nan
            new_results.append({
                "gameweek": gw,
                "old_accuracy": old_acc,
                "new_accuracy": acc,
                "new_correct": correct,
                "total": total
            })
            
    res_df = pd.DataFrame(new_results)
    print("\n=================================================================")
    print("SO SANH ACCURACY LOG (Cũ vs Mới)")
    print("=================================================================")
    print(res_df.to_string(index=False))
    
    # Print average
    avg_old = res_df["old_accuracy"].mean()
    avg_new = res_df["new_accuracy"].mean()
    print(f"\nAverage Old Accuracy (GW31-38) : {avg_old:.4f}")
    print(f"Average New Accuracy (GW31-38) : {avg_new:.4f}")
    
    # Save comparison
    res_df.to_csv("predictions/accuracy_comparison.csv", index=False)

if __name__ == "__main__":
    evaluate_all()
