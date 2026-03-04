import re
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Family features
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Title from Name
    # Example: "Braund, Mr. Owen Harris" -> "Mr"
    def extract_title(name: str) -> str:
        match = re.search(r",\s*([A-Za-z]+)\.", str(name))
        return match.group(1) if match else "Unknown"

    df["Title"] = df["Name"].apply(extract_title)

    # Group rare titles
    common_titles = {"Mr", "Mrs", "Miss", "Master"}
    df["Title"] = df["Title"].apply(lambda t: t if t in common_titles else "Other")

    return df