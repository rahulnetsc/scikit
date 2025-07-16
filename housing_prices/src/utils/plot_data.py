import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import contextily as ctx
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_hist(df: pd.DataFrame, columns: list[str]|None = None, bins= 50):

    if not columns:
        columns = df.select_dtypes(include=["int","float"]).columns
    df[columns].hist(bins=bins,)
    plt.tight_layout()
    plt.show() 

def plot_correlation(df: pd.DataFrame):
    
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_geospatial(df: pd.DataFrame):

    plt.scatter(df["longitude"], df["latitude"],alpha=0.4,s= df["population"]/100,
                c=df["median_house_value"], cmap="jet",)
    plt.colorbar(label="Median House Value")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geospatial Distribution of House Prices")
    plt.show()

def plot_geospatial_with_map(df: pd.DataFrame, map_style="light"):

    gdf = gpd.GeoDataFrame(df,
                           geometry= gpd.points_from_xy(df["longitude"],df['latitude']),    
                           crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    cmap = cm.get_cmap("plasma")
    norm = colors.Normalize(vmin=df["median_house_value"].min(),
                            vmax=df["median_house_value"].max())


    # ax = gdf.plot(
    #     alpha= 0.4,
    #     c=df["median_house_value"],
    #     cmap="plasma",
    #     markersize= 5,
    #     legend=True
    # )

    # ctx.add_basemap(ax,source= ctx.providers["CartoDB"]["Positron"])
    # ax.set_axis_off()
    # plt.title("House Prices with Map Background", fontsize=14)
    # plt.tight_layout()
    # plt.show()
    ax = gdf.plot(markersize=df["population"]/100,
        figsize=(12, 10),
        alpha=0.6,
        # markersize=5,
        color=[cmap(norm(val)) for val in df["median_house_value"]]
    )

    # Step 4: Add colorbar manually
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # dummy array for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Median House Value")

    # Step 5: Add basemap
    provider = {
        "light": ctx.providers["CartoDB"]["Positron"],
        "dark": ctx.providers["CartoDB"]["DarkMatter"],
        "satellite": ctx.providers["Esri"]["WorldImagery"],
        "terrain": ctx.providers["Esri"]["WorldTopoMap"]
    }.get(map_style, ctx.providers["CartoDB"]["Positron"])

    ctx.add_basemap(ax, source=provider)

    ax.set_axis_off()
    plt.title("House Prices with Map Background", fontsize=14)
    plt.tight_layout()
    plt.show()
