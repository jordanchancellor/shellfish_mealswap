# %%
# Set up environment
import numpy as np 
import pandas as pd
import requests, zipfile, io
from bs4 import BeautifulSoup
import re
from io import StringIO
from functools import reduce
import streamlit as st
import plotly.graph_objects as go
# %%
# --- 1. Pull data & transform ---
# nutrient data: https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_[date].zip
DOWNLOAD_PAGE = "https://fdc.nal.usda.gov/download-datasets.html"
BASE_URL = "https://fdc.nal.usda.gov"

def get_latest_usda_url():
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/139.0.0.0 Safari/537.36"
    }

    r = requests.get("https://fdc.nal.usda.gov/download-datasets.html", headers=headers)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "foundation_food_csv" in href.lower() and href.lower().endswith(".zip"):
            # Make absolute URL if needed
            full_url = href if href.startswith("http") else "https://fdc.nal.usda.gov" + href
            links.append(full_url)

    if not links:
        raise RuntimeError("No Foundation Foods CSV ZIP links found on USDA page.")

    # Sort by date in filename: ..._YYYY-MM-DD.zip
    def extract_date_from_filename(url):
        m = re.search(r"_(\d{4}-\d{2}-\d{2})\.zip", url)
        return m.group(1) if m else "1900-01-01"

    latest = sorted(links, key=lambda x: extract_date_from_filename(x), reverse=True)[0]
    return latest
# %%
@st.cache_data
def load_usda_nutrient_from_zip():
    url = get_latest_usda_url()
    print(f"Downloading: {url}")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Pull relevent data files
    food_categories = pd.read_csv(z.open("FoodData_Central_foundation_food_csv_2025-04-24/food_category.csv"))
    nutrient_ids = pd.read_csv(z.open("FoodData_Central_foundation_food_csv_2025-04-24/nutrient.csv"))
    food_ids = pd.read_csv(z.open("FoodData_Central_foundation_food_csv_2025-04-24/food.csv"))
    nutrient_values = pd.read_csv(z.open("FoodData_Central_foundation_food_csv_2025-04-24/food_nutrient.csv")) # amount it per 100g of food

    # Clean and rename 
    food_categories = food_categories.rename(columns={"id": "food_category_id"})
    food_categories = food_categories[["food_category_id", "description"]]
    food_categories = food_categories.rename(columns={"description": "food_category"})

    nutrient_ids = nutrient_ids.rename(columns={"id": "nutrient_id"})
    nutrient_ids = nutrient_ids.drop(columns=["nutrient_nbr", "rank"])

    food_ids = food_ids.drop(columns=["data_type", "publication_date"])
    nutrient_values = nutrient_values[["fdc_id", "nutrient_id", "amount"]]

    # Merge into master file
    food_tmp = pd.merge(food_categories, food_ids, on="food_category_id")
    nutrient_tmp = pd.merge(nutrient_ids, nutrient_values, on="nutrient_id")
    master_data = pd.merge(food_tmp, nutrient_tmp, on="fdc_id")

    # Subset to protein sources
    protein_subset = master_data[
        master_data["food_category"].isin([
            "Finfish and Shellfish Products",
            "Poultry Products",
            "Beef Products"
        ])
    ]

    # Summarize nutrients within food categories
    nutrient_df = (
        protein_subset
        .drop(columns=["description", "unit_name", "food_category_id", "fdc_id", "nutrient_id"])
        .assign(amount=lambda d: pd.to_numeric(d["amount"], errors="coerce"))
        .groupby(["food_category", "name"])["amount"]
        .mean()
        .reset_index()
        .pivot(index="food_category", columns="name", values="amount")
        .reset_index()
    )

    # Compress fatty acids
    nutrient_df["Unsaturated Fatty Acids"] = nutrient_df[
        ["Fatty acids, total monounsaturated", "Fatty acids, total polyunsaturated"]
    ].mean(axis=1)

    nutrient_df["Saturated Fatty Acids"] = nutrient_df[
        [
            "Fatty acids, total saturated",
            "Fatty acids, total trans",
            "Fatty acids, total trans-dienoic",
            "Fatty acids, total trans-monoenoic",
        ]
    ].mean(axis=1)

    # Final data + ordering
    custom_order = ["Beef Products", "Poultry Products", "Finfish and Shellfish Products"]
    nutrient_df = nutrient_df.set_index("food_category").loc[custom_order].reset_index()

    nutrient_df = nutrient_df[
        [
            "food_category", "Protein", "Cholesterol", "Calcium, Ca", "Iron, Fe",
            "Manganese, Mn", "Magnesium, Mg", "Phosphorus, P", "Zinc, Zn", "Sodium, Na",
            "Copper, Cu", "Vitamin B-12", "Vitamin D (D2 + D3)", "Vitamin E (alpha-tocopherol)",
            "Vitamin B-6", "Vitamin A, RAE", "Vitamin C, total ascorbic acid",
            "Unsaturated Fatty Acids", "Saturated Fatty Acids"
        ]
    ]

    # rename columns to include correct units 
    nutrient_df = nutrient_df.rename(columns={"Protein": "Protein (g)"})
    nutrient_df = nutrient_df.rename(columns={"Cholesterol": "Cholesterol (g)"})
    nutrient_df = nutrient_df.rename(columns={"Calcium, Ca": "Calcium, Ca (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Iron, Fe": "Iron, Fe (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Manganese, Mn": "Manganese, Mn (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Magnesium, Mg": "Magnesium, Mg (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Phosphorus, P": "Phosphorus, P, Mg (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Zinc, Zn": "Zinc, Zn (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Sodium, Na": "Sodium, Na (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Copper, Cu": "Copper, Cu (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Vitamin B-12": "Vitamin B-12 (ug)"})
    nutrient_df = nutrient_df.rename(columns={"Vitamin D (D2 + D3)": "Vitamin D (D2 + D3) (ug)"})
    nutrient_df = nutrient_df.rename(columns={"Vitamin E (alpha-tocopherol)": "Vitamin E (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Vitamin B-6": "Vitamin B-6 (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Vitamin A, RAE": "Vitamin A (ug)"})
    nutrient_df = nutrient_df.rename(columns={"Vitamin C, total ascorbic acid": "Vitamin C (mg)"})
    nutrient_df = nutrient_df.rename(columns={"Unsaturated Fatty Acids": "Unsaturated Fatty Acids (g)"})
    nutrient_df = nutrient_df.rename(columns={"Saturated Fatty Acids": "Saturated Fatty Acids (g)"})

    # calculate nutrients per 1g of food (data is in 100g of food)
    exclude_cols = ["food_category"]
    nutrient_cols = [c for c in nutrient_df.columns if c not in exclude_cols]
    nutrient_df[nutrient_cols] = nutrient_df[nutrient_cols] / 100

    return nutrient_df
# %%
# price data: https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/109093/FMAP.zip?v=12423
@st.cache_data
def load_usda_prices_from_zip(url):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Pull data file
    food_prices = pd.read_csv(z.open("FMAP-Data.csv"))
#    print(food_prices)

    # Clean-up data
    food_prices = food_prices[["EFPG_code", "Year", "Attribute", "Value"]]
    food_prices["EFPG_code"] = food_prices["EFPG_code"].astype(str)

    # Define & select attributes
    attr_condition = food_prices["Attribute"].isin(["Unit_value_mean_wtd"])
    code_condition = food_prices["EFPG_code"].isin([
        "53000", "53050", "53075",
        "50000", "50050", "50075",
        "51500", "51550", "51575"
    ])

    # Get most recent price data
    max_year = food_prices[attr_condition & code_condition]["Year"].max()

    # Subset data
    prices_subset = food_prices[
    attr_condition &
    code_condition &
    (food_prices["Year"] == max_year)
    ]
    
    # Re-name columns & codes with food categories
    code_replacements = {
    "53000": "Beef Products",
    "50000": "Beef Products",
    "53075": "Beef Products",
    "51500": "Poultry Products",
    "51550": "Poultry Products",
    "50050": "Poultry Products",
    "50075": "Finfish and Shellfish Products",
    "53050": "Finfish and Shellfish Products",
    "51575": "Finfish and Shellfish Products",
    }

    prices_subset["EFPG_code"] = prices_subset["EFPG_code"].replace(code_replacements)
    prices_subset = prices_subset.rename(columns={"EFPG_code": "food_category"})

    # Calculate price-per-gram (mean weight in data is price/100g)
    if not np.issubdtype(food_prices["Value"].dtype, np.number):
        food_prices["Value"] = pd.to_numeric(food_prices["Value"], errors='coerce')
    prices_subset["unit_price_per_gram"] = prices_subset["Value"] / 100

    # Calculate average price per gram for each protein category
    price_df = (
        prices_subset
        .drop(columns=["Year", "Attribute", "Value"])
        .groupby(["food_category"])["unit_price_per_gram"]
        .mean()
        .reset_index()
    )

    # Return final dataframe
    return(price_df)
# %%
# emissions data: https://ourworldindata.org/grapher/ghg-per-protein-poore
@st.cache_data
def load_emissions_from_web(url):
    r = requests.get(url)
    r.raise_for_status()

    # Pull data 
    emissions = pd.read_csv(StringIO(r.text))

    # Select attributes
    category_condition = emissions["Entity"].isin([
        "Beef (beef herd)",
        "Pig Meat",
        "Lamb & Mutton",
        "Poultry Meat",
        "Fish (farmed)",
        "Prawns (farmed)",
    ])

    co2_subset = emissions[category_condition]

    # Re-name columns & codes with food categories
    replacements = {
    "Beef (beef herd)": "Beef Products",
    "Pig Meat": "Beef Products",
    "Lamb & Mutton": "Beef Products",
    "Poultry Meat": "Poultry Products",
    "Fish (farmed)": "Finfish and Shellfish Products",
    "Prawns (farmed)": "Finfish and Shellfish Products"
    }

    # Subset dataframe and rename columns
    co2_subset["Entity"] = co2_subset["Entity"].replace(replacements)
    co2_subset = co2_subset.rename(columns={"Entity": "food_category"})

    # Calculate GHG emissions per gram protien (data is per 100g)
    if not np.issubdtype(co2_subset["GHG emissions per 100g protein (Poore & Nemecek, 2018)"].dtype, np.number):
        co2_subset["GHG emissions per 100g protein (Poore & Nemecek, 2018)"] = pd.to_numeric(co2_subset["GHG emissions per 100g protein (Poore & Nemecek, 2018)"], errors='coerce')
    co2_subset["GHG_emission_per_gram"] = co2_subset["GHG emissions per 100g protein (Poore & Nemecek, 2018)"] / 100

    # Return final dataframe
    emissions_df = (
        co2_subset
        .drop(columns=["Year", "Code", "GHG emissions per 100g protein (Poore & Nemecek, 2018)"])
        .groupby(["food_category"])["GHG_emission_per_gram"]
        .mean()
        .reset_index()
    )

    return emissions_df
# Merge all data frames into master
# %%
@st.cache_data
def build_master_df():
    # nutrient
    nutrient_df = load_usda_nutrient_from_zip()

    # price (FMAP static URL you used)
    PRICE_PAGE = "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/109093/FMAP.zip?v=12423"
    price_df = load_usda_prices_from_zip(PRICE_PAGE)

    # emissions
    EMISSIONS_PAGE = "https://ourworldindata.org/grapher/ghg-per-protein-poore.csv?v=1&csvType=full&useColumnShortNames=true"
    emissions_df = load_emissions_from_web(EMISSIONS_PAGE)

    # merge (inner join on categories; if a category is missing from any source it will be dropped)
    dataframes = [price_df, emissions_df, nutrient_df]
    master = reduce(lambda left, right: pd.merge(left, right, on='food_category'), dataframes)
    master.set_index("food_category", inplace=True) # set index for value retrieval
    return master

master_df = build_master_df()
# %%
# Health benefit factors; many different studies with conflicting results --> estimates here are theoreticl & conservative 
# THIS IS NOT MEDICAL ADVICE, THIS IS SIMPLY AN ATTEMPT AT QUANTIFYING HEALTH EFFECTS OF MEAL-SWAPPING
health_benefit_factor = {
    "heart_disease_risk": 0.05,  # Estimated heart disease risk reduction percentage
    "mortality": 0.08, # Estimated all-purpose mortality risk reduction percentage (doi: 10.1017/S0007114523002040)
    "brain function" : 0.05 # Estimated decrease in risk of diminished brain health
}

# --- 2. Impact Model --- 
def calculate_impacts(master_df, n_swaps, timeframe="Week", weight_type="Current", swap_from="Beef Products"):
    # Meal weights
    if weight_type == "Current":
        meal_weights = {"Beef Products": 41, "Poultry Products": 44, "Finfish and Shellfish Products": 16}
    else:
        meal_weights = {"Beef Products": 52.5, "Poultry Products": 52.5, "Finfish and Shellfish Products": 32}

    scale = 52 if timeframe == "Year" else 1
    n = n_swaps * scale

    co2_saved = (master_df.at[swap_from, "GHG_emission_per_gram"] * meal_weights[swap_from]
                 - master_df.at["Finfish and Shellfish Products", "GHG_emission_per_gram"] * meal_weights["Finfish and Shellfish Products"]) * n

    cost_diff = (master_df.at["Finfish and Shellfish Products", "unit_price_per_gram"] * meal_weights["Finfish and Shellfish Products"]
                 - master_df.at[swap_from, "unit_price_per_gram"] * meal_weights[swap_from]) * n

    health = {k: v * n for k, v in health_benefit_factor.items()}

    return co2_saved, cost_diff, health

# nutrient comparison
def nutrient_comparison(master_df, n_swaps, timeframe="Week", weight_type="current", swap_from="Beef Products"):
    # Meal weights
    if weight_type == "Current":
        meal_weights = {"Beef Products": 41, "Poultry Products": 44, "Finfish and Shellfish Products": 16}
    else:
        meal_weights = {"Beef Products": 52.5, "Poultry Products": 52.5, "Finfish and Shellfish Products": 32}

    factor = n_swaps if timeframe=="Week" else n_swaps*52

    # Select nutrients to display
    nutrients = ["Protein (g)", "Cholesterol (g)", "Iron, Fe (mg)", "Vitamin B-12 (ug)", "Sodium, Na (mg)",
                 "Vitamin B-6 (mg)", "Zinc, Zn (mg)", "Copper, Cu (mg)"
                 ]

    # compute per-meal values for swaps
    swap_vals = master_df.loc[swap_from, nutrients] * meal_weights[swap_from]
    shell_vals = master_df.loc["Finfish and Shellfish Products", nutrients] * meal_weights["Finfish and Shellfish Products"]
    diff = (shell_vals - swap_vals) * factor

    # compute Unsat/Sat ratio
    swap_ratio = master_df.at[swap_from, "Unsaturated Fatty Acids (g)"] / master_df.at[swap_from, "Saturated Fatty Acids (g)"]
    shell_ratio = master_df.at["Finfish and Shellfish Products", "Unsaturated Fatty Acids (g)"] / master_df.at["Finfish and Shellfish Products", "Saturated Fatty Acids (g)"]
    diff_ratio = shell_ratio - swap_ratio

    swap_series = pd.concat([swap_vals, pd.Series({"Unsat/Sat ratio": swap_ratio})])
    shell_series = pd.concat([shell_vals, pd.Series({"Unsat/Sat ratio": shell_ratio})])
    diff_series = pd.concat([diff, pd.Series({"Unsat/Sat ratio": diff_ratio})])

    df = pd.DataFrame({
        f"{swap_from} meal": swap_series,
        "Shellfish meal": shell_series,
        f"Change per {timeframe}": diff_series
    })

    return df

# --- 3. Streamlit Application --- 
st.title("Seafood Meal Swap Impact Calculator")

# User inputs
swap_from = st.selectbox("Swap from:", ["Beef Products", "Poultry Products"])
timeframe = st.radio("Timeframe:", ["Week", "Year"], horizontal=True)
weight_type = st.radio("Meal weights:", ["Current", "Recommended"], horizontal=True)
health_metric = st.selectbox("Health metric:", list(health_benefit_factor.keys()))
n_swaps = st.slider("Meals swapped per week:", 0, 21, 2)

# Load cached master_df
master_df = build_master_df()

# Calculate impacts
co2, cost, health = calculate_impacts(master_df, n_swaps, timeframe, weight_type, swap_from)

col_metrics, col_plot = st.columns([1, 2])  # adjust widths as you like
with col_metrics: # print metrics
    st.subheader(f"Swapping {swap_from} → Shellfish")
    st.metric("CO₂ saved (kg)", round(co2, 2))
    st.metric("Cost difference ($)", round(cost, 2))
    st.metric(f"Health improvement ({health_metric})", round(health[health_metric], 3))

with col_plot: # interactive figure
    x = list(range(0, 22))
    co2_vals, cost_vals, health_vals = [], [], []
    for n in x:
        c, cost_val, h = calculate_impacts(master_df, n, timeframe, weight_type, swap_from)
        co2_vals.append(c)
        cost_vals.append(cost_val)
        health_vals.append(h[health_metric])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=co2_vals, mode='lines+markers', name='CO₂ saved (kg)'))
    fig.add_trace(go.Scatter(x=x, y=cost_vals, mode='lines+markers', name='Cost difference ($)'))
    fig.add_trace(go.Scatter(x=x, y=health_vals, mode='lines+markers', name=f'Health improvement ({health_metric})'))

    fig.update_layout(
        title=f"Impact of swapping {swap_from} → Shellfish",
        xaxis_title="Meals swapped per week",
        yaxis_title="Impact",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Nutrient comparison table
st.subheader(f"Nutrient Comparison: {swap_from} → Shellfish")
nutrient_df = nutrient_comparison(master_df, n_swaps, timeframe, weight_type, swap_from)
st.dataframe(nutrient_df.style.format("{:.2f}"))

