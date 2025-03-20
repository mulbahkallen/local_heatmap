import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import os

# ------------------------------------------------------------------
# 1) Helper Functions (Grid, Basic SERP Scraper, etc.)
# ------------------------------------------------------------------

def generate_square_grid(center_lat: float, center_lon: float, radius_miles: float, grid_size: int = 5):
    """
    Generate a grid_size x grid_size grid of points around (center_lat, center_lon)
    within +/- radius_miles. Returns [(lat, lon), (lat, lon), ...].
    """
    if grid_size < 1:
        return []
    # Approx. 69 miles per degree lat
    lat_extent = radius_miles / 69.0
    # For longitude, factor in cos(latitude)
    lon_extent = radius_miles / (69.0 * np.cos(np.radians(center_lat)))

    lat_values = np.linspace(center_lat - lat_extent, center_lat + lat_extent, grid_size)
    lon_values = np.linspace(center_lon - lon_extent, center_lon + lon_extent, grid_size)

    points = []
    for lat in lat_values:
        for lon in lon_values:
            points.append((lat, lon))

    return points

def generate_circular_grid(center_lat: float, center_lon: float, radius_miles: float, num_points: int = 25):
    """
    Generate a set of points in a circular pattern around (center_lat, center_lon).
    """
    if num_points < 1:
        return []

    lat_degs = radius_miles / 69.0
    lon_degs = radius_miles / (69.0 * np.cos(np.radians(center_lat)))

    points = []
    for i in range(num_points):
        angle = 2.0 * np.pi * (i / num_points)
        lat_offset = lat_degs * np.sin(angle)
        lon_offset = lon_degs * np.cos(angle)
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        points.append((lat, lon))

    return points

def scrape_google_serp(query: str, lat: float, lon: float):
    """
    Attempt to perform a "local" Google search with simple parameters.
    WARNING: Not an official or reliable way to get local results.
    Return a list of dicts with { 'title', 'link' } for top organic results.
    
    We try 'near' in the query or some param like &gl=us, but this 
    won't truly localize to lat/lon unless you incorporate advanced 
    'UULE' parameters or a paid SERP API. This may also trigger CAPTCHAs.
    """
    # WARNING: This might get blocked or show a captcha
    # A naive approach: e.g. "coffee shop near 34.0522, -118.2437"
    # gl=us might help some localization for US searches.
    # We're also turning off personalization with pws=0 if possible.

    # Construct a naive search URL. You might try user-agents or proxies if blocked.
    search_query = f"{query} near {lat:.4f},{lon:.4f}"
    url = (
        "https://www.google.com/search"
        f"?q={search_query}"
        "&gl=us"  # Attempt to set geolocation = US
        "&pws=0"  # Turn off personalized results
        "&hl=en"  # Set language to English
    )

    # Optional: Provide a custom User-Agent to mimic a real browser
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=5)
        # If we get a captcha, the HTML might contain "unusual traffic" text
        html = resp.text
    except Exception as e:
        return []  # Return empty on error

    soup = BeautifulSoup(html, "html.parser")

    # This is extremely fragile—Google changes HTML often.
    # Let's attempt to parse the organic results in <div class="tF2Cxc"> or similar containers:
    # Titles often in <h3> with class "LC20lb MBeuO DKV0Md" or similar.
    # This can break at any time. It's just an example.

    results = []
    for g in soup.find_all("div", class_="tF2Cxc"):
        # Title
        h3 = g.find("h3")
        if not h3:
            continue
        title = h3.get_text(strip=True)

        # Link in <a> within that container
        link_tag = g.find("a")
        link = link_tag["href"] if link_tag else ""

        results.append({"title": title, "link": link})

    return results

def local_serp_rank_for_business(lat: float, lon: float, keyword: str, target_business: str):
    """
    Perform a SERP scrape for the given keyword near lat/lon, 
    then see if target_business name is in the top 10 or so. 
    Return (rank, topN).
    - rank: integer rank if found (1-based), else None
    - topN: the list of top results with 'title'/'link'
    """
    top_results = scrape_google_serp(keyword, lat, lon)
    if not top_results:
        return None, top_results

    # We simply look for the target_business substring in the 'title' (case-insensitive).
    rank = None
    for idx, item in enumerate(top_results):
        if target_business.lower() in item['title'].lower():
            rank = idx + 1
            break

    # If you want to limit to e.g. top 20 only, we do that:
    topN = top_results[:20]
    return rank, topN


def create_heatmap(df: pd.DataFrame, center_lat: float, center_lon: float):
    import plotly.graph_objects as go

    fig = go.Figure()

    for _, row in df.iterrows():
        rank_val = row['rank']
        if rank_val is None:
            marker_color = 'red'
            text_label = "X"
        elif rank_val <= 3:
            marker_color = 'green'
            text_label = str(rank_val)
        elif rank_val <= 10:
            marker_color = 'orange'
            text_label = str(rank_val)
        else:
            marker_color = 'red'
            text_label = str(rank_val)

        hover_text = ""
        topN = row['top_serp']
        if topN:
            # Create a short hover text from topN
            snippet = []
            for i, r in enumerate(topN[:5]):  # show first 5 in hover
                snippet.append(f"{i+1}. {r['title']}")
            hover_text = "\n".join(snippet)
        else:
            hover_text = "No SERP data or blocked."

        fig.add_trace(
            go.Scattermapbox(
                lat=[row['latitude']],
                lon=[row['longitude']],
                mode='markers+text',
                marker=dict(size=20, color=marker_color),
                text=[text_label],
                textposition="middle center",
                textfont=dict(size=14, color="black", family="Arial Black"),
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False
            )
        )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="SERP Rank Heatmap (Scraped)"
    )
    return fig

def generate_rank_report(df: pd.DataFrame, client_gbp: str):
    """
    Summarize coverage: top 3, top 10, or not found.
    """
    total_points = len(df)
    df_found = df.dropna(subset=['rank'])

    top_3_points = df_found[df_found['rank'] <= 3].shape[0]
    top_10_points = df_found[df_found['rank'] <= 10].shape[0]

    pct_top_3 = 0
    pct_top_10 = 0
    if total_points > 0:
        pct_top_3 = 100.0 * top_3_points / total_points
        pct_top_10 = 100.0 * top_10_points / total_points

    avg_rank = df_found['rank'].mean() if not df_found.empty else None

    lines = [
        f"**{client_gbp} Coverage (SERP-Scraped)**",
        f"- **Total Grid Points:** {total_points}",
        f"- **Business Found at:** {len(df_found)} points",
        f"- **In Top 3:** {top_3_points} points ({pct_top_3:.1f}% of total)",
        f"- **In Top 10:** {top_10_points} points ({pct_top_10:.1f}% of total)",
        (f"- **Average Rank (where found):** {avg_rank:.2f}"
         if avg_rank is not None else "- Average Rank: N/A"),
    ]
    return "\n".join(lines)

# ------------------------------------------------------------------
# 2) STREAMLIT APP
# ------------------------------------------------------------------

def main():
    st.title("Local SERP Scraper Heatmap (Proof-of-Concept)")
    st.write("""
    **Disclaimer**: This example tries to scrape Google's search results for each point in a geo-grid.
    This approach:
    - Violates Google's Terms of Service.
    - Can fail easily (CAPTCHAs, IP blocks).
    - Is not recommended or stable for production.
    
    **If you need reliable local SERP data**, use a paid API like SerpApi, DataForSEO, etc.
    """)

    # Google Maps key is still used for geocoding the center address
    if "places_api_key" not in st.session_state:
        st.subheader("Enter Google Maps API Key (for Geocoding Center Address)")
        google_key_input = st.text_input("Google Maps/Places API Key", type="password")
        if st.button("Save API Key"):
            if google_key_input:
                st.session_state["places_api_key"] = google_key_input
                # Instead of st.experimental_rerun(), we do this:
                raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
            else:
                st.warning("Please provide a valid API key to proceed.")
        st.stop()

    places_api_key = st.session_state["places_api_key"]
    gmaps = googlemaps.Client(key=places_api_key)

    # Snapshot label
    snapshot_name = st.text_input("Snapshot Name", value=f"serp_snapshot_{int(time.time())}")
    client_gbp = st.text_input("Your Business Name", "Starbucks")
    keyword = st.text_input("Keyword to Search", "Coffee Shop")
    business_address = st.text_input("Center Address or City", "Los Angeles, CA")

    grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
    radius = st.slider("Radius (miles)", 1, 20, 5)
    if grid_shape == "Square":
        grid_size = st.slider("Grid Size", 3, 11, 5)
    else:
        grid_size = st.slider("Circle Points", 8, 60, 25)

    if st.button("Run SERP Crawl"):
        # 1) Geocode center
        try:
            geocode_result = gmaps.geocode(business_address)
            if geocode_result:
                center_lat = geocode_result[0]['geometry']['location']['lat']
                center_lon = geocode_result[0]['geometry']['location']['lng']
            else:
                center_lat, center_lon = None, None
        except Exception as e:
            st.error(f"Geocoding error: {e}")
            center_lat, center_lon = None, None

        if not center_lat or not center_lon:
            st.error("Could not geocode address. Try again.")
            return

        st.write(f"**Center**: {business_address} → (Lat: {center_lat:.4f}, Lon: {center_lon:.4f})")

        # 2) Generate grid
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Generated {len(grid_points)} points in a {grid_size}x{grid_size} square grid.")
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Generated {len(grid_points)} points in a circle pattern.")

        serp_data = []
        progress_bar = st.progress(0)
        total_pts = len(grid_points)

        for i, (lat, lon) in enumerate(grid_points, start=1):
            rank, top_serp = local_serp_rank_for_business(lat, lon, keyword, client_gbp)
            serp_data.append({
                "latitude": lat,
                "longitude": lon,
                "rank": rank,
                "top_serp": top_serp
            })
            progress_bar.progress(int((i / total_pts)*100))

        progress_bar.empty()

        df = pd.DataFrame(serp_data)
        df["snapshot_name"] = snapshot_name
        df["timestamp"] = pd.Timestamp.now()

        # 3) Show Heatmap
        st.plotly_chart(create_heatmap(df, center_lat, center_lon), use_container_width=True)

        # 4) Rank Report
        st.write("### Coverage Report")
        st.markdown(generate_rank_report(df, client_gbp))

        # 5) Save snapshot
        history_file = "serp_scrape_history.csv"
        if not os.path.isfile(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)
        st.success(f"Saved snapshot '{snapshot_name}' in '{history_file}'.")

        # 6) Download
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download This Snapshot (CSV)",
            data=csv_data,
            file_name=f"{snapshot_name}_serp_data.csv",
            mime="text/csv"
        )

        # 7) Show Detailed Results
        st.write("### Detailed Results per Grid Point")
        for idx, row in df.iterrows():
            rank_str = row["rank"]
            if rank_str is None:
                rank_str = "X"  # Not found
            details = row["top_serp"]
            snippet = ""
            if details:
                for i, item in enumerate(details[:5], start=1):
                    snippet += f"\n  {i}. {item['title']}"
            else:
                snippet = "No results or blocked."

            st.markdown(f"""
**Point {idx+1}**  
- **Coords**: ({row['latitude']:.4f}, {row['longitude']:.4f})  
- **Rank**: {rank_str}  
- **Top SERP** (first 5 results): {snippet}
""")

    # Compare old snapshots
    st.write("---")
    st.subheader("Compare Old Snapshots")
    uploaded_file = st.file_uploader("Upload a prior serp_scrape_history.csv", type=["csv"])
    if uploaded_file:
        old_df = pd.read_csv(uploaded_file)
        st.write("Snapshots Found:", old_df['snapshot_name'].unique())
        st.markdown("You can implement your own diff or side-by-side comparison here.")

if __name__ == "__main__":
    main()
