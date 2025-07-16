# main.py

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from src.data.load_data import load_data, load_configs
from src.utils.logger import get_logger
from src.utils.plot_data import plot_hist, plot_correlation, plot_geospatial,plot_geospatial_with_map
from src.preprocessing import split_data, fit_imputer, transform_data

logger = get_logger("main")

def main():
    parser = argparse.ArgumentParser(description="Load housing data based on config environment")
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"],
                        help="Configuration environment to use")
    args = parser.parse_args()

    config_path = Path("configs/config.yaml")
    config = load_configs(config_path)

    env_config = config.get(args.env)
    if env_config is None:
        raise RuntimeError(f"Environment '{args.env}' not found in config.")

    url = env_config["data"]["url"]
    csv_path = Path(env_config["data"]["raw_csv_path"])

    logger.info(f"Loading housing dataset for environment: {args.env}")
    housing: pd.DataFrame = load_data(url=url, csv_path=csv_path)

    housing['income_cat'] = pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6.0, np.inf],labels=[1,2,3,4,5])

    logger.info("Top 5 rows:\n%s", housing.head().to_string())
    logger.info("DataFrame Info:")
    housing.info(buf=None)  # This prints directly; we'll log a summary below

    logger.info("Column names: %s", list(housing.columns))
    logger.info("Ocean proximity value counts:\n%s", housing["ocean_proximity"].value_counts().to_string())
    logger.info("Describe summary:\n%s", housing.describe().to_string())
  
    train_data, test_data, _ = split_data(housing,strat_col=housing["income_cat"]) 
    imputer = fit_imputer(train_data)
    train_data = transform_data(train_data, imputer=imputer)
    test_data = transform_data(test_data,imputer=imputer)

    
    # plot_hist(housing)
    # plot_correlation(housing)
    # # plot_geospatial(housing)
    # plot_geospatial_with_map(housing)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.countplot(x = 'income_cat', data=housing)
    # plt.xlabel("Category")
    # plt.ylabel("Number of districts")
    # plt.show()

    
    # plot_hist(data_train)
    # plot_correlation(data_train)
    # plot_geospatial(housing)
    # plot_geospatial_with_map(data_train)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.countplot(x = 'income_cat', data=data_test,)
    # plt.grid(True)
    # plt.xlabel("Category")
    # plt.ylabel("Number of districts")
    # plt.show()
    # print(data_test["income_cat"].value_counts() / len(data_test))
    # housing = housing.drop(columns=["income_cat"])
    
    # housing["rooms_per_house"] = housing["total_rooms"]/housing["households"] 
    # housing["bedrooms_ratio"] = housing["total_bedrooms"]/housing["total_rooms"]
    # housing['people_per_house'] = housing['population']/housing['households']
    # corr_matrix = housing.corr(numeric_only=True)
    # print(sorted(corr_matrix['median_house_value'], reverse=True))
    # housing_labels = housing['median_house_value']
    # housing = housing.drop(columns=['median_house_value'])
    # print(housing.keys())


if __name__ == "__main__":
    main()
