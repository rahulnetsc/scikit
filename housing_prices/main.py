# main.py

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from src.data.load_data import load_data, load_configs
from src.utils.logger import get_logger
from src.utils.plot_data import plot_hist, plot_correlation, plot_geospatial,plot_geospatial_with_map
from src.preprocessing import split_data, fit_imputer,fit_encoder,transform_data, df_profiler, build_preprocessor
import os
logger = get_logger("main")

def main():
    parser = argparse.ArgumentParser(description="Load housing data based on config environment")
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"],
                        help="Configuration environment to use")
    parser.add_argument("--model", type= str, choices= {'linear', 'rfr', 'gbr', 'hgb', 'svr'}, default= 'linear',
                        help= 'Model type, choices= {linear, rfg, gbr, hgb, svr}')
    parser.add_argument("--cv", type=int, default=None, help="Number of K-Fold splits (e.g. 5 for 5-Fold CV)")

    args = parser.parse_args()

    config_path = Path("configs/config.yaml")
    config = load_configs(config_path)

    env_config = config.get(args.env)
    if env_config is None:
        raise RuntimeError(f"Environment '{args.env}' not found in config.")

    url = env_config["data"]["url"]
    csv_path = Path(env_config["data"]["raw_csv_path"])
    profile_path: Path = Path(env_config["data"]["profiler_path"])

    logger.info(f"Loading housing dataset for environment: {args.env}")
    housing: pd.DataFrame = load_data(url=url, csv_path=csv_path)
    
    if not profile_path.is_file():
        df_profiler(housing,profile_path)

    housing['income_cat'] = pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6.0, np.inf],labels=[1,2,3,4,5])

    train_data, test_data, _ = split_data(housing,strat_col=housing["income_cat"]) 

    train_labels = train_data['median_house_value'].copy()
    train_data = train_data.drop(columns =['median_house_value'] )
    test_labels = test_data['median_house_value'].copy()
    test_data = test_data.drop(columns= ['median_house_value'])
    
    pipeline = build_preprocessor(train_data)
    pipeline.fit(train_data)
    train_data = pipeline.transform(train_data)
    test_data = pipeline.transform(test_data)

    from src.models import benchmark

    results = benchmark(train_data, train_labels, test_data, test_labels)

    # Only keep valid entries (type: list or tuple)
    cleaned = {k: v for k, v in results.items() if isinstance(v, (list, tuple))}

    df = pd.DataFrame.from_dict(cleaned, orient='index', columns=['RMSE', 'Time (s)'])
    df = df.sort_values("RMSE")  # or sort by time

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot RMSE
    ax1.set_xlabel("Model")
    ax1.set_ylabel("RMSE", color="tab:blue")
    ax1.bar(df.index, df["RMSE"], color="tab:blue", alpha=0.6)
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Create second axis for Time
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time (s)", color="tab:red")
    ax2.plot(df.index, df["Time (s)"], color="tab:red", marker="o", linewidth=2)
    ax2.tick_params(axis='y', labelcolor="tab:red")

    plt.title("Model Comparison: RMSE and Time")
    # plt.xticks(rotation=45)
    plt.xticks(rotation=30, ha="right")  # 30° tilt, right-aligned

    plt.tight_layout()
    plt.show()
    
    df.to_csv(Path("data/processed/benchmark_results.csv"), index_label="Model")



if __name__ == "__main__":
    main()


