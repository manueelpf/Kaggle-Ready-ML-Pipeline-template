import os
import pandas as pd
import matplotlib.pyplot as plt


def run_eda(train_df: pd.DataFrame):
    os.makedirs("outputs", exist_ok=True)

    # Missing values overview
    missing = train_df.isna().mean().sort_values(ascending=False)
    missing.to_csv("outputs/missing_values.csv")

    # Survival rate by Sex
    surv_by_sex = train_df.groupby("Sex")["Survived"].mean()
    plt.figure()
    surv_by_sex.plot(kind="bar")
    plt.title("Survival Rate by Sex")
    plt.ylabel("Survival rate")
    plt.tight_layout()
    plt.savefig("outputs/survival_by_sex.png")
    plt.close()

    # Survival rate by Pclass
    surv_by_class = train_df.groupby("Pclass")["Survived"].mean()
    plt.figure()
    surv_by_class.plot(kind="bar")
    plt.title("Survival Rate by Passenger Class")
    plt.ylabel("Survival rate")
    plt.tight_layout()
    plt.savefig("outputs/survival_by_pclass.png")
    plt.close()