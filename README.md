# Old School RuneScape Bank Value Analyzer

This script is a [Streamlit](https://streamlit.io/) web application designed to analyze and compare the value of an Old School RuneScape (OSRS) bank over time. It is built to work with data exported from the [Bank Memory](https://runelite.net/plugin-hub/show/bank-memory) plugin for [RuneLite](https://runelite.net/). By uploading multiple bank data files, users can visualize changes in item quantities and values, identify market-driven price fluctuations, and track the introduction of new items to their bank.

## How to Use

1.  **Export Your Bank Data**: Use the **Bank Memory** plugin within RuneLite to export your OSRS bank contents. A brief overview of the process:
    *   Right-click on a saved bank in RuneLite.
    *   Select "Copy bank data" from the context menu.
    *   Paste the copied data into a text editor.
    *   Save the file with a `.txt` extension.

    The file will contain the following columns:
    *   `Item id`: The unique identifier for each item in the bank.
    *   `Item name`: The human-readable name of the item.
    *   `Item quantity`: The number of items of this type in the bank.

2.  **Upload Files**: In the web application's sidebar, click the "Upload tab-separated bank files" button and select two or more of your exported bank files.

3.  **Assign Dates**: For each uploaded file, assign the date on which the bank snapshot was taken. This is crucial for accurate historical price lookups.

4.  **Compare Banks**: Click the "Compare Banks" button to initiate the analysis. The script will fetch historical price data from the OSRS Wiki API, process your bank files, and display a detailed comparison.

5.  **Analyze the Results**: The application will present several tables and charts:
    *   **Filtered Summary**: An overview of your bank's total value on the selected dates and the overall change.
    *   **Per-Item Value Changes**: A detailed table showing the change in value for each item present in both the oldest and newest snapshots.
    *   **New Items Added**: A list of items that were not present in the oldest snapshot but appear in the newest one.
    *   **Market-Driven Value Changes**: A table that isolates value changes caused by market price fluctuations versus changes in item quantity.
    *   **Visualizations**: Bar charts illustrating the top item value gainers and losers.

## Features

*   **Bank Value Comparison**: Compares the total value of your bank between different points in time.
*   **Detailed Item Analysis**: Provides a per-item breakdown of value and quantity changes.
*   **New Item Tracking**: Identifies items that have been added to your bank between snapshots.
*   **Market Mover Identification**: Differentiates between value changes due to price fluctuations (market effect) and changes due to buying/selling items (quantity effect).
*   **Interactive Filtering**: Allows for dynamic filtering of the results tables based on value change thresholds and item names.
*   **Data Visualization**: Includes charts to easily spot the most significant gainers and losers in your bank's value.
*   **Caching**: Caches price data and the list of tradeable item IDs to speed up subsequent analyses and reduce the load on the OSRS Wiki API.

## Known Issues and Limitations

*   **Ornamented Items**: The script currently does not handle ornamented tradeable items correctly. These items are treated as untradeable and are therefore excluded from the value analysis. Additionally, the script does not "track" the ornamentation status of an item. For example, if an item transitions from unornamented to ornamented between snapshots, the script will interpret this as losing the unornamented item and gaining a new, untradeable (and thus valueless) ornamented item. This can result in an incorrect representation of value changes.

## Credits

*   **Bank Data Export**: Bank data is exported using the [Bank Memory](https://runelite.net/plugin-hub/show/bank-memory) plugin for [RuneLite](https://runelite.net/).
*   **Price Data Source**: Historical item price data is sourced from the [OSRS Wiki API](https://prices.runescape.wiki/api).
*   **Application Framework**: This application is built using the [Streamlit](https://streamlit.io/) framework.
*   **Core Libraries**: Python libraries used include [Pandas](https://pandas.pydata.org/), [Plotly Express](https://plotly.com/python/plotly-express/), [requests](https://github.com/psf/requests/), and all their dependencies.

## Disclaimer
*   This application was quickly put together using [Google AI Studio](https://ai.google.dev/). Expect to find bugs, bad code, or other issues. Feel free to open a pull request with any improvements.