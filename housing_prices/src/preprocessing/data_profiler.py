from ydata_profiling import ProfileReport
import pandas as pd
from pathlib import Path

def df_profiler(df: pd.DataFrame, out: Path ):
    print(f"generating profile report... ")
    profile = ProfileReport(df,title = "Housing EDA", explorative=True)
    profile.to_file(out)
    print(f"Profile saved to {out}")