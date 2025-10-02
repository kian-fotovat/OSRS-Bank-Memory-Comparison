import io
import json
import os
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# --- Cache Configuration ---
PRICE_CACHE_FILE = "price_cache.json"
TRADEABLE_CACHE_FILE = "tradeable_ids_cache.json"
HEADERS = {"User-Agent": "OSRS Bank Analyzer Script/2.1-fix"}

# Set the page layout to wide
st.set_page_config(layout="wide")
st.title("Old School RuneScape Bank Value Analyzer")


# --- Helper for Dismissible Alerts ---
def show_alert(message, level="info", duration="long"):
    """Displays a dismissible and auto-dismissing alert using st.toast."""
    icon_map = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    st.toast(message, icon=icon_map.get(level, "ℹ️"), duration=duration)


# --- Caching Functions (Prices) ---
def load_price_cache():
    """Loads the price cache from a JSON file if it exists."""
    if os.path.exists(PRICE_CACHE_FILE):
        with open(PRICE_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_price_cache(data):
    """Saves the given data to the price cache JSON file."""
    with open(PRICE_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=4)


# --- Caching Functions (Tradeable IDs) ---
def load_tradeable_ids_cache():
    """Loads the tradeable IDs cache from a JSON file."""
    if os.path.exists(TRADEABLE_CACHE_FILE):
        with open(TRADEABLE_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_tradeable_ids_cache(data):
    """Saves the tradeable IDs list to its cache file."""
    with open(TRADEABLE_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=4)


# --- API and Data Processing Functions ---
@st.cache_data(ttl=3600)  # Cache API calls for an hour to avoid re-fetching on minor changes
def execute_bucket_query(lua_query):
    """Executes a given Lua query against the OSRS Wiki's Bucket API."""
    api_url = "https://oldschool.runescape.wiki/api.php"
    params = {"action": "bucket", "format": "json", "formatversion": 2, "query": lua_query}
    try:
        response = requests.get(api_url, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            show_alert(f"Wiki API Error: {data['error']}", "error")
            return None
        return [int(item["id"]) for item in data.get("bucket", [])]
    except requests.exceptions.RequestException as e:
        show_alert(f"Network error fetching tradeable items: {e}", "error")
        return None
    except (KeyError, ValueError):
        show_alert("Failed to parse response from Wiki API for tradeable items.", "error")
        return None


def _fetch_tradeable_item_ids_from_api():
    """Helper function to query the API for all tradeable item IDs."""
    return execute_bucket_query("bucket('exchange').select('id').limit(10000).run()")


def get_tradeable_item_ids():
    """Gets the list of tradeable item IDs, using a 30-day cache."""
    cache = load_tradeable_ids_cache()
    today = date.today()
    if last_fetched_str := cache.get("last_fetched"):
        if (today - date.fromisoformat(last_fetched_str)).days < 30:
            show_alert("Loaded tradeable item list from cache.", "info")
            return cache.get("ids", [])

    show_alert("Fetching fresh list of tradeable items from OSRS Wiki API...", "info")
    if ids := _fetch_tradeable_item_ids_from_api():
        show_alert(f"Successfully fetched {len(ids)} tradeable item IDs.", "success")
        save_tradeable_ids_cache({"last_fetched": today.isoformat(), "ids": ids})
        return ids
    else:
        show_alert("Could not retrieve tradeable items. Using stale cache if available.", "error")
        return cache.get("ids", [])


def get_item_history(item_id):
    """Fetches the last 365 days of price data for a single item ID."""
    url = f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=24h&id={item_id}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json().get("data", [])
        price_map = {}
        for record in data:
            record_date = datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d")
            avg_price = (record.get("avgHighPrice", 0) + record.get("avgLowPrice", 0)) / 2 if record.get("avgHighPrice") and record.get("avgLowPrice") else 0
            price_map[record_date] = avg_price
        return price_map
    except requests.exceptions.RequestException as e:
        show_alert(f"API call failed for item {item_id}: {e}", "error")
        return None


def fetch_and_cache_item_histories(item_ids, all_selected_dates):
    """Manages fetching and caching price data."""
    if not item_ids:
        show_alert("No tradeable items found to fetch price data for.", "warning")
        return {}

    cache = load_price_cache()
    items_to_fetch = []
    latest_file_date = max(all_selected_dates)
    latest_file_date_str = latest_file_date.strftime("%Y-%m-%d")

    for item_id in item_ids:
        # Do not attempt to fetch price history for coins
        if item_id == 995:
            continue
        item_id_str = str(item_id)
        if item_id_str not in cache or cache[item_id_str].get("latest_file_date", "1970-01-01") < latest_file_date_str:
            items_to_fetch.append(item_id)

    if items_to_fetch:
        show_alert(f"Updating price cache for {len(items_to_fetch)} item(s).", "info")
        progress_bar = st.progress(0, text="Fetching price data...")
        for i, item_id in enumerate(items_to_fetch):
            history = get_item_history(item_id)
            if history is None:
                show_alert(f"Failed to fetch data for item ID {item_id}. Halting execution.", "error")
                st.stop()
            cache[str(item_id)] = {"latest_file_date": latest_file_date_str, "price_map": history}
            progress_bar.progress((i + 1) / len(items_to_fetch), text=f"Updating item {i + 1}/{len(items_to_fetch)}...")
        progress_bar.empty()
        save_price_cache(cache)
        show_alert("Price cache updated.", "success")

    return {item_id: cache[str(item_id)]["price_map"] for item_id in item_ids if str(item_id) in cache and item_id != 995}


def process_bank_file(df, selected_date, all_price_histories):
    """Processes a bank dataframe using pre-fetched price histories."""
    date_str = selected_date.strftime("%Y-%m-%d")
    df["date"] = date_str

    def get_price_for_item(item_id):
        # Handle coins as a special case with a fixed value of 1
        if item_id == 995:
            return 1
        return all_price_histories.get(item_id, {}).get(date_str, 0)

    df["price"] = df["Item id"].apply(get_price_for_item)
    df["stack_value"] = df["Item quantity"] * df["price"]
    return df


# --- START OF UI LOGIC ---

# --- Sidebar UI ---
st.sidebar.header("Upload Bank Files")
uploaded_files = st.sidebar.file_uploader("Upload tab-separated bank files", accept_multiple_files=True, type=["txt"])

selected_dates = []
if uploaded_files:
    st.sidebar.subheader("Assign a Date to Each File")
    yesterday = date.today() - timedelta(days=1)
    for i, file in enumerate(uploaded_files):
        selected_date = st.sidebar.date_input(f"Date for '{file.name}'", key=f"date_{i}", value=yesterday, max_value=yesterday)
        selected_dates.append(selected_date)

if st.sidebar.button("Compare Banks"):
    if len(uploaded_files) < 2:
        show_alert("Please upload at least two bank files to compare.", "warning")
    else:
        with st.spinner("Processing banks... This may take a moment."):
            tradeable_ids_set = set(get_tradeable_item_ids())
            if not tradeable_ids_set:
                show_alert("Cannot proceed without the list of tradeable items.", "error")
                st.stop()

            all_item_ids, all_items_map, bank_dfs = set(), {}, {}
            for file in uploaded_files:
                try:
                    df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")), sep="\t")
                    if not all(col in df.columns for col in ["Item id", "Item name", "Item quantity"]):
                        show_alert(f"File '{file.name}' is missing required columns. Skipping.", "error")
                        continue

                    original_count = len(df)
                    # Keep items that are either in the tradeable set OR are coins (ID 995)
                    df = df[df["Item id"].isin(tradeable_ids_set) | (df["Item id"] == 995)]
                    if len(df) < original_count:
                        show_alert(f"Ignoring {original_count - len(df)} untradeable item(s) from '{file.name}'.", "info")

                    bank_dfs[file.name] = df
                    all_item_ids.update(df["Item id"])
                    all_items_map.update(pd.Series(df["Item name"].values, index=df["Item id"]).to_dict())
                except Exception as e:
                    show_alert(f"Error reading file '{file.name}': {e}", "error")
                    continue

            all_price_histories = fetch_and_cache_item_histories(list(all_item_ids), selected_dates)
            processed_dfs = [process_bank_file(bank_dfs[file.name].copy(), s_date, all_price_histories) for file, s_date in sorted(zip(uploaded_files, selected_dates), key=lambda p: p[1]) if file.name in bank_dfs]

            if len(processed_dfs) < 2:
                show_alert("Fewer than two valid bank files could be processed.", "error")
            else:
                all_banks_df = pd.concat(processed_dfs)
                value_pivot = all_banks_df.pivot_table(index=["Item id", "Item name"], columns="date", values="stack_value").fillna(0)
                quantity_pivot = all_banks_df.pivot_table(index=["Item id", "Item name"], columns="date", values="Item quantity").fillna(0)
                oldest_date, latest_date = value_pivot.columns[0], value_pivot.columns[-1]
                value_pivot["value_change"] = value_pivot[latest_date] - value_pivot[oldest_date]

                # --- MARKET MOVER LOGIC WITH DEBUGGING ---
                shared_items_mask = (quantity_pivot[oldest_date] > 0) & (quantity_pivot[latest_date] > 0)
                market_df = pd.DataFrame()

                if shared_items_mask.any():
                    q1 = quantity_pivot.loc[shared_items_mask, oldest_date]
                    q2 = quantity_pivot.loc[shared_items_mask, latest_date]
                    v1 = value_pivot.loc[shared_items_mask, oldest_date]
                    v2 = value_pivot.loc[shared_items_mask, latest_date]

                    p1 = v1.divide(q1).fillna(0)
                    p2 = v2.divide(q2).fillna(0)

                    q_bar = (q1 + q2) / 2
                    delta_p = p2 - p1

                    price_effect = delta_p * q_bar
                    total_v_change = v2 - v1
                    qty_effect = total_v_change - price_effect
                    is_mover = (price_effect.abs() >= qty_effect.abs()) & (total_v_change.abs() > 0)

                    mover_indices = p1[is_mover].index
                    if not mover_indices.empty:
                        market_df = pd.DataFrame(index=mover_indices)
                        market_df[f"Qty {oldest_date}"] = q1[is_mover]
                        market_df[f"Qty {latest_date}"] = q2[is_mover]
                        market_df[f"Price {oldest_date}"] = p1[is_mover]
                        market_df[f"Price {latest_date}"] = p2[is_mover]
                        market_df["Total Value Change"] = total_v_change[is_mover]
                        market_df["Value Change from Market"] = price_effect[is_mover]
                        market_df["Value Change from Quantity"] = qty_effect[is_mover]

                st.session_state.market_movers_df = market_df

                # --- Original Comparison Logic ---
                combined_df = pd.DataFrame(index=value_pivot.index)
                for col_date in value_pivot.columns.drop("value_change"):
                    combined_df[col_date] = value_pivot[col_date]
                    combined_df[f"Qty {col_date}"] = quantity_pivot[col_date]
                combined_df["value_change"] = value_pivot["value_change"]

                new_items_mask = (value_pivot[oldest_date] == 0) & (value_pivot[latest_date] > 0)

                st.session_state.new_items_df = combined_df[new_items_mask].copy().drop(columns=["value_change"])
                st.session_state.comparison_df = combined_df[~new_items_mask].copy()
                st.session_state.oldest_date = oldest_date
                st.session_state.latest_date = latest_date
                st.session_state.all_dates = value_pivot.columns.drop("value_change").tolist()
                st.session_state.comparison_complete = True

if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.rerun()

if st.session_state.get("comparison_complete", False):
    comparison_df = st.session_state.comparison_df
    new_items_df = st.session_state.new_items_df
    market_movers_df = st.session_state.get("market_movers_df", pd.DataFrame())
    oldest_date = st.session_state.oldest_date
    latest_date = st.session_state.latest_date
    all_dates = st.session_state.all_dates

    show_alert("Bank comparison complete!", "success")

    # --- MAIN UI ---
    st.header("Filters")
    with st.expander("Filter Comparison & New Items Tables", expanded=True):
        st.subheader("Comparison Table Filters")
        f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
        loss_threshold = f_col1.number_input("Show items with value change less than", value=0.0, key="comp_loss_thresh", format="%.0f")
        gain_threshold = f_col2.number_input("Show items with value change greater than", value=0.0, key="comp_gain_thresh", format="%.0f")
        filter_mode_comp = f_col3.radio("Filter Mode", ["Exclusive", "Inclusive"], key="comp_filter_mode", horizontal=True)
        selected_items_comp = st.multiselect("Filter by Item Name", options=comparison_df.index.get_level_values("Item name").unique(), key="comp_item_select")

        st.subheader("New Items Table Filters")
        if not new_items_df.empty:
            fn_col1, fn_col2 = st.columns([2, 1])
            min_value_new = fn_col1.number_input("Show items with a value greater than", value=0.0, min_value=0.0, key="new_item_min_val", format="%.0f")
            filter_mode_new = fn_col2.radio("Filter Mode", ["Exclusive", "Inclusive"], key="new_item_filter_mode", horizontal=True)
            selected_items_new = st.multiselect("Filter by Item Name", options=new_items_df.index.get_level_values("Item name").unique(), key="new_item_select")

    # --- Apply Filters ---
    filtered_comp_df = comparison_df.copy()
    if selected_items_comp:
        is_in_selection = filtered_comp_df.index.get_level_values("Item name").isin(selected_items_comp)
        filtered_comp_df = filtered_comp_df[is_in_selection if filter_mode_comp == "Inclusive" else ~is_in_selection]

    if loss_threshold != 0 or gain_threshold != 0:
        filtered_comp_df = filtered_comp_df[(filtered_comp_df["value_change"] < loss_threshold) | (filtered_comp_df["value_change"] > gain_threshold)]

    filtered_new_items_df = new_items_df.copy()
    if not new_items_df.empty:
        if selected_items_new:
            is_in_selection = filtered_new_items_df.index.get_level_values("Item name").isin(selected_items_new)
            filtered_new_items_df = filtered_new_items_df[is_in_selection if filter_mode_new == "Inclusive" else ~is_in_selection]

        if min_value_new > 0:
            filtered_new_items_df = filtered_new_items_df[filtered_new_items_df[latest_date] > min_value_new]

    # --- Dynamic Summary ---
    st.header("Filtered Summary")
    filtered_total_values = {}
    for date_col in all_dates:
        comp_sum = filtered_comp_df[date_col].sum()
        new_sum = filtered_new_items_df[date_col].sum() if not filtered_new_items_df.empty else 0
        filtered_total_values[date_col] = comp_sum + new_sum
    filtered_total_difference = filtered_total_values[latest_date] - filtered_total_values[oldest_date]
    cols = st.columns(len(filtered_total_values))
    for i, (date_col, value) in enumerate(filtered_total_values.items()):
        cols[i].metric(label=f"Filtered Value on {date_col}", value=f"{value:,.0f} GP")
    st.metric(label=f"Filtered Value Change ({oldest_date} to {latest_date})", value=f"{filtered_total_difference:,.0f} GP", delta=f"{filtered_total_difference:,.0f} GP")

    # --- Data Tables ---
    st.header(f"Per-Item Value Changes ({oldest_date} vs {latest_date})")
    comp_formatter = {col: "{:,.0f} GP" for col in [d for d in all_dates] + ["value_change"]}
    for col in comparison_df.columns:
        if "Qty" in str(col):
            comp_formatter[col] = "{:,.0f}"
    st.dataframe(filtered_comp_df.style.format(comp_formatter), use_container_width=True)

    if not new_items_df.empty:
        st.header(f"New Items Added Between {oldest_date} and {latest_date}")
        st.dataframe(filtered_new_items_df.style.format(comp_formatter), use_container_width=True)

    st.header("Market-Driven Value Changes")
    if not market_movers_df.empty:
        with st.expander("Filter Market Movers Table", expanded=True):
            m_f_col1, m_f_col2, m_f_col3 = st.columns([2, 2, 1])
            m_loss_thresh = m_f_col1.number_input("Show items with value change less than", value=0.0, key="market_loss_thresh", format="%.0f")
            m_gain_thresh = m_f_col2.number_input("Show items with value change greater than", value=0.0, key="market_gain_thresh", format="%.0f")
            m_filter_mode = m_f_col3.radio("Filter Mode", ["Exclusive", "Inclusive"], key="market_filter_mode", horizontal=True)
            m_selected_items = st.multiselect("Filter by Item Name", options=market_movers_df.index.get_level_values("Item name").unique(), key="market_item_select")

        filtered_market_df = market_movers_df.copy()
        if m_selected_items:
            is_in_selection = filtered_market_df.index.get_level_values("Item name").isin(m_selected_items)
            filtered_market_df = filtered_market_df[is_in_selection if m_filter_mode == "Inclusive" else ~is_in_selection]

        if m_loss_thresh != 0 or m_gain_thresh != 0:
            filtered_market_df = filtered_market_df[(filtered_market_df["Total Value Change"] < m_loss_thresh) | (filtered_market_df["Total Value Change"] > m_gain_thresh)]

        market_formatter = {
            f"Price {oldest_date}": "{:,.2f} GP",
            f"Price {latest_date}": "{:,.2f} GP",
            "Total Value Change": "{:,.0f} GP",
            "Value Change from Market": "{:,.0f} GP",
            "Value Change from Quantity": "{:,.0f} GP",
            f"Qty {oldest_date}": "{:,.0f}",
            f"Qty {latest_date}": "{:,.0f}",
        }
        col_order = [f"Qty {oldest_date}", f"Qty {latest_date}", f"Price {oldest_date}", f"Price {latest_date}", "Value Change from Market", "Value Change from Quantity", "Total Value Change"]
        st.dataframe(filtered_market_df[col_order].style.format(market_formatter), use_container_width=True)
    else:
        st.info(f"No items were found where price change impact was greater than quantity change impact between {oldest_date} and {latest_date}.")

    # --- Visualizations ---
    st.header("Visualizations")
    gainers_df = comparison_df[comparison_df["value_change"] > 0].sort_values(by="value_change", ascending=False)
    losers_df = comparison_df[comparison_df["value_change"] < 0].sort_values(by="value_change", ascending=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Item Value Gainers")
        with st.expander("Filter Gainers Chart"):
            num_gainers = st.slider("Number of items to show", 1, 50, 10, key="num_gainers")
            filter_mode_gainer = st.radio("Filter Mode", ["Exclusive", "Inclusive"], key="gainer_filter_mode", horizontal=True)
            gainer_items = st.multiselect("Filter by Item Name", options=gainers_df.index.get_level_values("Item name").unique(), key="gainer_item_select")

        display_gainers = gainers_df
        if gainer_items:
            is_in_selection = display_gainers.index.get_level_values("Item name").isin(gainer_items)
            display_gainers = display_gainers[is_in_selection if filter_mode_gainer == "Inclusive" else ~is_in_selection]

        if not display_gainers.empty:
            fig = px.bar(
                display_gainers.head(num_gainers),
                x=display_gainers.head(num_gainers).index.get_level_values("Item name"),
                y="value_change",
                labels={"x": "Item Name", "value_change": "Value Change (GP)"},
                color_discrete_sequence=["green"],
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Item Value Losers")
        with st.expander("Filter Losers Chart"):
            num_losers = st.slider("Number of items to show", 1, 50, 10, key="num_losers")
            filter_mode_loser = st.radio("Filter Mode", ["Exclusive", "Inclusive"], key="loser_filter_mode", horizontal=True)
            loser_items = st.multiselect("Filter by Item Name", options=losers_df.index.get_level_values("Item name").unique(), key="loser_item_select")

        display_losers = losers_df
        if loser_items:
            is_in_selection = display_losers.index.get_level_values("Item name").isin(loser_items)
            display_losers = display_losers[is_in_selection if filter_mode_loser == "Inclusive" else ~is_in_selection]

        if not display_losers.empty:
            fig = px.bar(
                display_losers.head(num_losers),
                x=display_losers.head(num_losers).index.get_level_values("Item name"),
                y="value_change",
                labels={"x": "Item Name", "value_change": "Value Change (GP)"},
                color_discrete_sequence=["red"],
            )
            st.plotly_chart(fig, use_container_width=True)
