import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import requests
import time
import os
from bs4 import BeautifulSoup

# -------------------------------------------------------------------
# 1) HELPER FUNCTIONS
# -------------------------------------------------------------------

def generate_square_grid(center_lat: float, center_lon: float, radius_miles: float, grid_size: int = 5):
    """
    Build a square grid of lat/lon around (center_lat, center_lon).
    """
    if grid_size < 1:
        return []

    lat_extent = radius_miles / 69.0
    lon_extent = radius_miles / (69.0 * np.cos(np.radians(center_lat)))

    lat_vals = np.linspace(center_lat - lat_extent, center_lat + lat_extent, grid_size)
    lon_vals = np.linspace(center_lon - lon_extent, center_lon + lon_extent, grid_size)

    points = []
    for lat in lat_vals:
        for lon in lon_vals:
            points.append((lat, lon))
    return points

def generate_circular_grid(center_lat: float, center_lon: float, radius_miles: float, num_points: int = 25):
    """
    Build a circular pattern of lat/lon around (center_lat, center_lon).
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

def reverse_geocode_city(lat: float, lon: float, gmaps_client) -> str:
    """
    Reverse geocode (lat, lon) to approximate city or region name using Google Maps.
    Return a string like "Los Angeles, California" or an empty string if not found.
    """
    try:
        result = gmaps_client.reverse_geocode((lat, lon))
        if not result:
            return ""
        # parse out something like 'locality' or 'administrative_area_level_1'
        # (the exact structure can vary, so we do a quick approach):
        locality = ""
        admin_area = ""

        for component in result[0]["address_components"]:
            if "locality" in component["types"]:
                locality = component["long_name"]
            if "administrative_area_level_1" in component["types"]:
                admin_area = component["long_name"]

        # fallback: if no 'locality', maybe use 'sublocality' or 'postal_town', etc.
        if not locality:
            for component in result[0]["address_components"]:
                if "postal_town" in component["types"] or "sublocality" in component["types"]:
                    locality = component["long_name"]
                    break

        # build something like "Locality, AdminArea"
        if locality and admin_area:
            return f"{locality},{admin_area}"
        elif locality:
            return locality
        elif admin_area:
            return admin_area
        else:
            # fallback to formatted_address truncated?
            return result[0].get("formatted_address", "")
    except:
        return ""

def serpstack_search(api_key: str, query: str, location: str, num: int = 10):
    """
    Call serpstack's Google Search API with given query and location.
    Return the parsed JSON or None if error.
    Example: 
      GET https://api.serpstack.com/search
          ? access_key=YOUR_KEY
          & query=coffee shop
          & location=Los Angeles
          & output=json
    We'll limit to num=10 results per page for simplicity.
    """
    base_url = "https://api.serpstack.com/search"
    params = {
        "access_key": api_key,
        "query": query,
        "location": location,
        "output": "json",
        "type": "web",
        "num": num,        # how many results per page
        "page": 1,         # just first page in this example
        "auto_location": 0 # turn off auto-loc so we rely on our location param
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # serpstack returns { "request": {...}, "search_parameters": {...}, "search_information": {...}, "organic_results": [...], etc. }
        if data.get("success", True) is False:
            # means there's an error
            st.error(f"Serpstack error: {data.get('error', {}).get('info', 'Unknown')}")
            return None
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Request to serpstack failed: {e}")
        return None

def find_business_rank(api_json: dict, target_business: str) -> (int, list):
    """
    From the serpstack search JSON, find the rank of target_business in 'organic_results'.
    Return (rank, top_results), where rank is 1-based index or None if not found,
    and top_results is the array of organic_results themselves.
    """
    if not api_json or "organic_results" not in api_json:
        return None, []

    organic = api_json["organic_results"]
    rank = None

    # we look for the business by substring match in title
    for i, item in enumerate(organic, start=1):
        title = item.get("title", "")
        if target_business.lower() in title.lower():
            rank = i
            break

    return rank, organic

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

        # Build a short hover text
        serp_info = row['organic_results']
        hover_text = ""
        if serp_info:
            # let's just show the top 3 titles
            snippet_lines = []
            for idx, r in enumerate(serp_info[:3], start=1):
                snippet_lines.append(f"{idx}. {r.get('title', '')}")
            hover_text = "\n".join(snippet_lines)
        else:
            hover_text = "No results or API error."

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
            zoom=9
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="SERPstack Local Approx Heatmap"
    )
    return fig

def generate_report(df: pd.DataFrame, client_name: str):
    total_pts = len(df)
    found_df = df.dropna(subset=['rank'])

    top3 = found_df[found_df['rank'] <= 3].shape[0]
    top10 = found_df[found_df['rank'] <= 10].shape[0]

    pct_top3 = (100 * top3 / total_pts) if total_pts else 0
    pct_top10 = (100 * top10 / total_pts) if total_pts else 0

    avg_rank = found_df['rank'].mean() if not found_df.empty else None

    lines = [
        f"**{client_name} Coverage (SERPstack)**",
        f"- **Total Grid Points:** {total_pts}",
        f"- **Business Found:** {len(found_df)} points",
        f"- **In Top 3:** {top3} points ({pct_top3:.1f}%)",
        f"- **In Top 10:** {top10} points ({pct_top10:.1f}%)",
    ]
    if avg_rank is not None:
        lines.append(f"- **Average Rank (where found):** {avg_rank:.2f}")
    else:
        lines.append("- **Average Rank:** N/A (Not found at any grid point)")

    return "\n".join(lines)

# -------------------------------------------------------------------
# 2) STREAMLIT APP
# -------------------------------------------------------------------

def main():
    st.title("Local SERPstack Geo-Grid Demo")

    st.write("""
    This tool uses the **serpstack** API (a paid/freemium service) to approximate local SERP ranking
    based on city-level location parameters. We generate a grid of coordinates around a center address,
    reverse-geocode each point to a nearby city, then query serpstack for `query + location`. 
    Finally, we check the position of your business name in the top organic results.
    
    **Disclaimer**: 
    - This is an approximation. serpstack location queries do not accept raw lat/lon. 
    - We rely on city-level strings from reverse-geocoding each grid point, which may not perfectly replicate real lat-lon local SERPs.
    - If you need truly accurate local rank at lat-lon resolution, consider serpstack’s advanced plan or other local SERP APIs.
    """)

    # 1) serpstack API key
    serpstack_key = st.text_input("Serpstack API Key", type="password", help="Get this from serpstack.com dashboard.")
    if not serpstack_key:
        st.info("Enter a serpstack Access Key to proceed.")
        st.stop()

    # 2) Google Maps key for geocoding
    if "places_api_key" not in st.session_state:
        st.subheader("Enter Google Maps API Key (for geocoding)")
        google_key_input = st.text_input("Google Maps/Places API Key", type="password")
        if st.button("Save Google API Key"):
            if google_key_input:
                st.session_state["places_api_key"] = google_key_input
                # Force the script to re-run with updated session_state
                raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
            else:
                st.warning("Please provide a valid Google API key.")
        st.stop()

    gmaps_client = googlemaps.Client(key=st.session_state["places_api_key"])

    # 3) Basic inputs
    snapshot_name = st.text_input("Snapshot Name", value=f"serpstack_snapshot_{int(time.time())}")
    target_business = st.text_input("Target Business Name", "Starbucks")
    keyword = st.text_input("Keyword to Search", "Coffee Shop")
    center_address = st.text_input("Center Address/City", "Los Angeles, CA")

    grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
    radius = st.slider("Radius (Miles)", 1, 20, 5)
    if grid_shape == "Square":
        grid_size = st.slider("Square Grid Size", 3, 11, 5)
    else:
        grid_size = st.slider("Number of Circle Points", 8, 60, 25)

    if st.button("Run SERPstack Analysis"):
        # A) Forward geocode center
        try:
            geocode_result = gmaps_client.geocode(center_address)
            if geocode_result:
                center_lat = geocode_result[0]["geometry"]["location"]["lat"]
                center_lon = geocode_result[0]["geometry"]["location"]["lng"]
            else:
                center_lat, center_lon = None, None
        except Exception as e:
            st.error(f"Error geocoding center address: {e}")
            return

        if not center_lat or not center_lon:
            st.error("Could not geocode center address. Please check spelling.")
            return

        st.write(f"**Center**: {center_address} → (Lat: {center_lat:.5f}, Lon: {center_lon:.5f})")

        # B) Generate grid
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Generated {len(grid_points)} points in a {grid_size}x{grid_size} square grid.")
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Generated {len(grid_points)} points in a circular pattern ({len(grid_points)}).")

        # C) For each point, reverse-geocode -> city -> serpstack -> find rank
        df_rows = []
        progress_bar = st.progress(0)
        total_pts = len(grid_points)

        for i, (lat, lon) in enumerate(grid_points, start=1):
            # Reverse geocode to approximate city
            city_str = reverse_geocode_city(lat, lon, gmaps_client)
            # If no city found, skip
            if not city_str:
                rank = None
                top_res = []
            else:
                # call serpstack
                data = serpstack_search(serpstack_key, keyword, city_str)
                if data is None:
                    rank = None
                    top_res = []
                else:
                    rank, organic = find_business_rank(data, target_business)
                    top_res = organic

            df_rows.append({
                "latitude": lat,
                "longitude": lon,
                "city_used": city_str,
                "rank": rank,
                "organic_results": top_res
            })

            progress_bar.progress(int((i / total_pts) * 100))
            time.sleep(0.2)  # slight pause to avoid spamming serpstack too quickly

        progress_bar.empty()

        df = pd.DataFrame(df_rows)
        df["snapshot_name"] = snapshot_name
        df["timestamp"] = pd.Timestamp.now()

        # D) Heatmap
        st.plotly_chart(create_heatmap(df, center_lat, center_lon), use_container_width=True)

        # E) Report
        st.write("### Rank Coverage Report")
        st.markdown(generate_report(df, target_business))

        # F) Save CSV
        history_file = "serpstack_history.csv"
        if not os.path.isfile(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)
        st.success(f"Saved snapshot '{snapshot_name}' in {history_file}.")

        # G) Download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download This Snapshot as CSV",
            data=csv_data,
            file_name=f"{snapshot_name}_serpstack.csv",
            mime="text/csv"
        )

        # H) Detailed View
        st.write("### Detailed Results")
        for idx, row in df.iterrows():
            rank_str = row["rank"] if row["rank"] is not None else "X"
            st.markdown(f"""
**Grid Point {idx+1}**  
- Location: (Lat: {row['latitude']:.5f}, Lon: {row['longitude']:.5f})  
- Reverse-Geocoded City: {row['city_used']}  
- **Rank**: {rank_str}
""")

    # Let user upload old CSV for comparison
    st.write("---")
    st.subheader("Compare Old Snapshots")
    uploaded = st.file_uploader("Upload a serpstack_history CSV", type=["csv"])
    if uploaded:
        old_df = pd.read_csv(uploaded)
        st.write("Found snapshots:", old_df["snapshot_name"].unique())
        st.markdown("""
        You could implement side-by-side or difference mapping here if desired.
        """)

if __name__ == "__main__":
    main()
