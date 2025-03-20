import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import requests
import time
import os

import plotly.graph_objects as go

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
    Return something like "Los Angeles, California" or empty string if not found.
    """
    try:
        result = gmaps_client.reverse_geocode((lat, lon))
        if not result:
            return ""
        locality = ""
        admin_area = ""

        for component in result[0]["address_components"]:
            if "locality" in component["types"]:
                locality = component["long_name"]
            if "administrative_area_level_1" in component["types"]:
                admin_area = component["long_name"]

        # fallback to sublocalities or postal_town if needed
        if not locality:
            for component in result[0]["address_components"]:
                if "postal_town" in component["types"] or "sublocality" in component["types"]:
                    locality = component["long_name"]
                    break

        # build "Locality, AdminArea"
        if locality and admin_area:
            return f"{locality},{admin_area}"
        elif locality:
            return locality
        elif admin_area:
            return admin_area
        else:
            return result[0].get("formatted_address", "")
    except:
        return ""

def serpstack_search(api_key: str, query: str, location: str, num: int = 10):
    """
    Call serpstack's Google Search API with the given query + location.
    Returns the JSON or None if error.
    We only request the first 'num' results from page=1 for simplicity.
    """
    base_url = "https://api.serpstack.com/search"
    params = {
        "access_key": api_key,
        "query": query,
        "location": location,
        "output": "json",
        "type": "web",
        "num": num,        
        "page": 1,         
        "auto_location": 0 
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success", True) is False:
            # means there's an error per serpstack's doc
            st.error(f"Serpstack error: {data.get('error', {}).get('info', 'Unknown')}")
            return None
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Request to serpstack failed: {e}")
        return None

def find_business_rank(api_json: dict, target_business: str) -> (int, list):
    """
    From the serpstack search JSON, find the rank of target_business in 'organic_results'.
    Return (rank, topN), where rank is 1-based or None if not found,
    and topN is the list of organic results themselves.
    """
    if not api_json or "organic_results" not in api_json:
        return None, []

    organic = api_json["organic_results"]
    rank = None

    for i, item in enumerate(organic, start=1):
        title = item.get("title", "")
        # naive substring match
        if target_business.lower() in title.lower():
            rank = i
            break

    return rank, organic

def create_scattermap(df: pd.DataFrame, center_lat: float, center_lon: float):
    """
    Build a Plotly Scattermap for the SERP rank data (green/orange/red).
    """
    fig = go.Figure()

    for _, row in df.iterrows():
        rank_val = row["rank"]
        if rank_val is None:
            marker_color = "red"
            text_label = "X"
        elif rank_val <= 3:
            marker_color = "green"
            text_label = str(rank_val)
        elif rank_val <= 10:
            marker_color = "orange"
            text_label = str(rank_val)
        else:
            marker_color = "red"
            text_label = str(rank_val)

        # build hover text from topN
        topN = row["organic_results"] or []
        if topN:
            snippet_lines = []
            for i, item in enumerate(topN[:3], start=1):  # show top 3 in hover
                snippet_lines.append(f"{i}. {item.get('title', '')}")
            hover_text = "\n".join(snippet_lines)
        else:
            hover_text = "No results or API error."

        fig.add_trace(
            go.Scattermap(
                lat=[row["latitude"]],
                lon=[row["longitude"]],
                mode="markers+text",
                marker=dict(size=20, color=marker_color),
                text=[text_label],
                textposition="middle center",
                textfont=dict(size=14, color="black"),
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False
            )
        )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=9,
        margin=dict(r=0, t=0, l=0, b=0),
        title="SERPstack Local Approx Heatmap"
    )
    return fig

def generate_report(df: pd.DataFrame, target_business: str):
    """
    Summarize coverage in top3, top10, etc.
    """
    total_pts = len(df)
    df_found = df.dropna(subset=["rank"])

    top3 = df_found[df_found["rank"] <= 3].shape[0]
    top10 = df_found[df_found["rank"] <= 10].shape[0]
    pct_top3 = (100*top3/total_pts) if total_pts else 0
    pct_top10 = (100*top10/total_pts) if total_pts else 0

    avg_rank = df_found["rank"].mean() if not df_found.empty else None

    lines = [
        f"**{target_business} SERP Coverage**",
        f"- **Total Grid Points:** {total_pts}",
        f"- **Found at:** {len(df_found)} points",
        f"- **In Top 3:** {top3} ({pct_top3:.1f}%)",
        f"- **In Top 10:** {top10} ({pct_top10:.1f}%)"
    ]
    if avg_rank is not None:
        lines.append(f"- **Average Rank (where found):** {avg_rank:.2f}")
    else:
        lines.append("- **Average Rank:** N/A (Not found anywhere)")

    return "\n".join(lines)

# -------------------------------------------------------------------
# 2) STREAMLIT APP
# -------------------------------------------------------------------

def main():
    st.title("Local SERP Heatmap via Serpstack (Approx)")

    st.write("""
    This tool uses your **Serpstack** API (paid/freemium) + Google Geocoding:
    1. We geocode a center address to get (lat, lon).
    2. Build a geo-grid around that center.
    3. Reverse-geocode each point to an approximate city name (Serpstack does not accept raw lat/lon).
    4. Call Serpstack's "Google Search" endpoint with `query` and `location=CityName`.
    5. Check where your business name appears in the organic results (if at all).
    
    **Disclaimer**:
    - Because Serpstack does not accept raw lat/lon, each point's "location" is approximate (city-level).
    - For truly lat/lon-precise local SERP data, you'd need advanced or alternative solutions.
    """)

    # 1) serpstack key
    st.subheader("Enter Serpstack API Key")
    if "serpstack_key" not in st.session_state:
        serp_key_input = st.text_input("Serpstack Access Key", type="password",
                                       help="Sign up at serpstack.com to get an API key.")
        if st.button("Save Serpstack Key"):
            if serp_key_input:
                st.session_state["serpstack_key"] = serp_key_input
                st.warning("Serpstack key saved! Please reload or rerun to proceed.")
            else:
                st.warning("Please provide a valid serpstack key.")
        st.stop()

    serpstack_key = st.session_state["serpstack_key"]

    # 2) Google Maps key for geocoding
    st.subheader("Enter Google Maps API Key (for geocoding)")
    if "google_maps_key" not in st.session_state:
        google_key_input = st.text_input("Google Maps/Places API Key", type="password")
        if st.button("Save Google Maps Key"):
            if google_key_input:
                st.session_state["google_maps_key"] = google_key_input
                st.warning("Google Maps key saved! Please reload or rerun to proceed.")
            else:
                st.warning("Please provide a valid Google Maps key.")
        st.stop()

    gmaps_client = googlemaps.Client(key=st.session_state["google_maps_key"])

    # 3) Basic inputs
    snapshot_name = st.text_input("Snapshot Name", value=f"serpstack_snapshot_{int(time.time())}",
                                  help="Used to label your scan in the CSV.")
    target_business = st.text_input("Target Business Name", value="Starbucks",
                                    help="Substring match in the SERP result's title.")
    keyword = st.text_input("Keyword to Search", value="Coffee Shop",
                            help="We'll pass this to serpstack as 'query'.")
    center_address = st.text_input("Center Address/City", value="Los Angeles, CA",
                                   help="We geocode this to get the center lat/lon.")
    grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
    radius = st.slider("Radius (Miles)", 1, 20, 5)
    if grid_shape == "Square":
        grid_size = st.slider("Square Grid Size", 3, 11, 5, help="5 => 25 total points.")
    else:
        grid_size = st.slider("Circle Points", 8, 60, 25)

    if st.button("Run Serpstack Analysis"):
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
            center_lat, center_lon = None, None

        if not center_lat or not center_lon:
            st.error("Could not geocode center address. Try again.")
            return

        st.write(f"**Center**: {center_address} → (Lat: {center_lat:.5f}, Lon: {center_lon:.5f})")

        # B) Generate grid
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"**Generated** {len(grid_points)} points in a {grid_size}×{grid_size} square.")
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"**Generated** {len(grid_points)} points in a circle pattern.")

        # C) For each point -> reverse geocode -> serpstack -> find rank
        df_rows = []
        progress_bar = st.progress(0)
        total_pts = len(grid_points)

        for i, (lat, lon) in enumerate(grid_points, start=1):
            # Reverse geocode to city
            city_str = reverse_geocode_city(lat, lon, gmaps_client)
            if not city_str:
                rank = None
                organic = []
            else:
                # call serpstack
                data = serpstack_search(serpstack_key, keyword, city_str)
                if not data:
                    rank = None
                    organic = []
                else:
                    rank, organic = find_business_rank(data, target_business)

            df_rows.append({
                "latitude": lat,
                "longitude": lon,
                "city_used": city_str,
                "rank": rank,
                "organic_results": organic
            })

            progress_bar.progress(int((i / total_pts) * 100))
            time.sleep(0.2)  # slight delay to avoid spamming

        progress_bar.empty()

        df = pd.DataFrame(df_rows)
        df["snapshot_name"] = snapshot_name
        df["timestamp"] = pd.Timestamp.now()

        # D) Plot
        st.plotly_chart(create_scattermap(df, center_lat, center_lon), use_container_width=True)

        # E) Coverage Report
        st.write("### SERP Coverage Report")
        st.markdown(generate_report(df, target_business))

        # F) Save CSV
        history_file = "serpstack_history.csv"
        if not os.path.isfile(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode="a", header=False, index=False)
        st.success(f"Saved snapshot '{snapshot_name}' → {history_file}.")

        # G) Download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Snapshot as CSV",
            data=csv_data,
            file_name=f"{snapshot_name}_serpstack.csv",
            mime="text/csv"
        )

        # H) Detailed Results
        st.write("### Detailed Per-Point Data")
        for idx, row in df.iterrows():
            rank_str = row["rank"] if row["rank"] is not None else "X"
            st.markdown(f"""
**Grid Point {idx+1}**  
- **Coords**: (Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f})  
- **City**: {row['city_used']}  
- **Rank**: {rank_str}
""")

    # Past Snapshots
    st.write("---")
    st.subheader("Compare Old Snapshots")
    uploaded = st.file_uploader("Upload serpstack_history CSV", type=["csv"])
    if uploaded:
        old_df = pd.read_csv(uploaded)
        st.write("Snapshots found:", old_df["snapshot_name"].unique())
        st.markdown("Implement side-by-side or difference logic here if desired.")

if __name__ == "__main__":
    main()
