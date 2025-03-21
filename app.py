import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import requests
import time
import os
import plotly.graph_objects as go

# ----------------------------------------------
# 1) HELPER FUNCTIONS
# ----------------------------------------------

def generate_square_grid(center_lat: float, center_lon: float, radius_miles: float, grid_size: int = 5):
    """
    Generate a square grid of lat/lon around (center_lat, center_lon).
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
    Generate a circular pattern of lat/lon around (center_lat, center_lon).
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
    Reverse geocode lat/lon to approximate city or region name (for serpstack location).
    """
    try:
        results = gmaps_client.reverse_geocode((lat, lon))
        if not results:
            return ""
        # Attempt to extract "locality" and "administrative_area_level_1"
        locality = ""
        admin_area = ""
        for comp in results[0].get("address_components", []):
            if "locality" in comp["types"]:
                locality = comp["long_name"]
            if "administrative_area_level_1" in comp["types"]:
                admin_area = comp["long_name"]

        # fallback if needed
        if not locality:
            for comp in results[0].get("address_components", []):
                if "postal_town" in comp["types"] or "sublocality" in comp["types"]:
                    locality = comp["long_name"]
                    break

        if locality and admin_area:
            return f"{locality},{admin_area}"
        elif locality:
            return locality
        elif admin_area:
            return admin_area
        else:
            return results[0].get("formatted_address", "")
    except:
        return ""

def serpstack_search(api_key: str, query: str, location: str, num: int = 10):
    """
    Query serpstack with the given search query + city location.
    Returns JSON or None on error.
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
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success", True) is False:
            st.error(f"Serpstack error: {data.get('error', {}).get('info', 'Unknown')}")
            return None
        return data
    except requests.RequestException as e:
        st.error(f"Error calling serpstack: {e}")
        return None

def find_map_pack_rank(data: dict, gbp_name: str):
    """
    Parse serpstack's local_results to find the GBP name in the 'title'.
    Return 1-based rank or None if not found.
    """
    if "local_results" not in data:
        return None
    locs = data["local_results"]
    for i, loc in enumerate(locs, start=1):
        title = loc.get("title", "")
        if gbp_name.lower() in title.lower():
            return i
    return None

def find_organic_rank_by_url(data: dict, domain_name: str):
    """
    Parse serpstack's organic_results, matching 'domain_name' substring in each result's 'url'.
    Return 1-based rank or None if not found.
    """
    if "organic_results" not in data:
        return None
    orgs = data["organic_results"]
    for i, item in enumerate(orgs, start=1):
        url = item.get("url", "")
        if domain_name.lower() in url.lower():
            return i
    return None

def create_scattermap(df: pd.DataFrame, center_lat: float, center_lon: float, rank_type: str):
    """
    Build a plotly Scattermap. 
    rank_type can be 'map_pack_rank' or 'organic_rank'.
    """
    fig = go.Figure()

    for _, row in df.iterrows():
        rank_val = row[rank_type]
        if pd.isna(rank_val):
            marker_color = "red"
            text_label = "X"
        else:
            rank_val = int(rank_val)
            if rank_val <= 3:
                marker_color = "green"
            elif rank_val <= 10:
                marker_color = "orange"
            else:
                marker_color = "red"
            text_label = str(rank_val)

        hover_text = f"City: {row['city_used']}\n{rank_type}: {text_label}"

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
        title=f"Serpstack Heatmap - {rank_type}"
    )
    return fig

def generate_coverage_report(df: pd.DataFrame, column: str, label: str):
    """
    Summarize coverage for the specified rank column (map_pack_rank or organic_rank).
    """
    total_pts = len(df)
    df_found = df.dropna(subset=[column])

    top3 = df_found[df_found[column] <= 3].shape[0]
    top10 = df_found[df_found[column] <= 10].shape[0]

    pct_top3 = (100 * top3 / total_pts) if total_pts else 0
    pct_top10 = (100 * top10 / total_pts) if total_pts else 0

    avg_rank = df_found[column].mean() if not df_found.empty else None

    lines = [
        f"**Coverage Report: {label}**",
        f"- **Total Grid Points:** {total_pts}",
        f"- **Found at:** {len(df_found)} points",
        f"- **In Top 3:** {top3} ({pct_top3:.1f}%)",
        f"- **In Top 10:** {top10} ({pct_top10:.1f}%)"
    ]
    if avg_rank is not None:
        lines.append(f"- **Average Rank (where found):** {avg_rank:.2f}")
    else:
        lines.append("- **Average Rank:** N/A")

    return "\n".join(lines)

# ----------------------------------------------
# 2) STREAMLIT APP
# ----------------------------------------------

def main():
    st.title("SERPstack Geo-Grid: Map Pack + Organic URL Matching")

    st.write("""
    This app uses **Serpstack** for SERP data and **Google Geocoding** for location data. 
    - We check **Map Pack** (local_results) by matching your **Google Business Profile** name. 
    - We check **Organic** results by matching your **website domain** in the 'url' field.
    """)

    # 1) serpstack key
    if "serpstack_key" not in st.session_state:
        st.subheader("Enter Serpstack API Key")
        key_input = st.text_input("Serpstack Access Key", type="password")
        if st.button("Save Serpstack Key"):
            if key_input:
                st.session_state["serpstack_key"] = key_input
                st.warning("Key saved! Please refresh or rerun the app to continue.")
            else:
                st.warning("Please provide a valid Serpstack key.")
        st.stop()

    serpstack_key = st.session_state["serpstack_key"]

    # 2) Google Maps key
    if "google_maps_key" not in st.session_state:
        st.subheader("Enter Google Maps API Key")
        gkey_input = st.text_input("Google Maps/Places API Key", type="password")
        if st.button("Save Google Maps Key"):
            if gkey_input:
                st.session_state["google_maps_key"] = gkey_input
                st.warning("Maps key saved! Please refresh or rerun the app to continue.")
            else:
                st.warning("Please provide a valid Google Maps key.")
        st.stop()

    gmaps = googlemaps.Client(key=st.session_state["google_maps_key"])

    # 3) Inputs for business
    snapshot_name = st.text_input("Snapshot Name", value=f"serpstack_snapshot_{int(time.time())}")
    gbp_name = st.text_input("Google Business Profile Name", value="Starbucks",
                             help="We'll look for this name in local_results 'title'.")
    domain_name = st.text_input("Website Domain for Organic", value="starbucks.com",
                                help="We'll match this substring in the organic 'url' field.")
    search_keyword = st.text_input("Keyword to Search", value="coffee shop")
    center_addr = st.text_input("Center Address/City", value="Los Angeles, CA")

    grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
    radius_mi = st.slider("Radius (miles)", 1, 20, 5)
    if grid_shape == "Square":
        grid_sz = st.slider("Square Grid Size", 3, 11, 5)
    else:
        grid_sz = st.slider("Circle Points", 8, 60, 25)

    if st.button("Run Serpstack Analysis"):
        # A) Geocode center
        try:
            geo_res = gmaps.geocode(center_addr)
            if geo_res:
                center_lat = geo_res[0]["geometry"]["location"]["lat"]
                center_lon = geo_res[0]["geometry"]["location"]["lng"]
            else:
                center_lat, center_lon = None, None
        except Exception as e:
            st.error(f"Error geocoding center address: {e}")
            center_lat, center_lon = None, None

        if not center_lat or not center_lon:
            st.error("Could not geocode center address. Try again.")
            return

        st.write(f"**Center**: {center_addr} → (Lat: {center_lat:.4f}, Lon: {center_lon:.4f})")

        # B) Generate grid
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius_mi, grid_sz)
            st.write(f"Generated {len(grid_points)} points in a {grid_sz}x{grid_sz} square.")
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius_mi, grid_sz)
            st.write(f"Generated {len(grid_points)} points in a circle pattern ({len(grid_points)}).")

        # C) For each point -> city -> serpstack -> parse map pack & organic
        rows = []
        bar = st.progress(0)
        total = len(grid_points)

        for i, (lat, lon) in enumerate(grid_points, start=1):
            city_str = reverse_geocode_city(lat, lon, gmaps)
            if not city_str:
                # no city found
                map_rank = None
                org_rank = None
            else:
                data = serpstack_search(serpstack_key, search_keyword, city_str)
                if data is None:
                    map_rank = None
                    org_rank = None
                else:
                    map_rank = find_map_pack_rank(data, gbp_name)
                    org_rank = find_organic_rank_by_url(data, domain_name)

            rows.append({
                "latitude": lat,
                "longitude": lon,
                "city_used": city_str,
                "map_pack_rank": map_rank,
                "organic_rank": org_rank
            })
            bar.progress(int((i / total) * 100))
            time.sleep(0.2)

        bar.empty()

        df = pd.DataFrame(rows)
        df["snapshot_name"] = snapshot_name
        df["timestamp"] = pd.Timestamp.now()

        # D) Let user pick which rank to visualize
        rank_choice = st.selectbox("Which rank do you want to plot?", ["map_pack_rank", "organic_rank"])
        fig = create_scattermap(df, center_lat, center_lon, rank_choice)
        st.plotly_chart(fig, use_container_width=True)

        # Show coverage for both
        st.write("### Map Pack Coverage")
        st.markdown(generate_coverage_report(df, "map_pack_rank", label="Map Pack Rank"))

        st.write("### Organic Coverage")
        st.markdown(generate_coverage_report(df, "organic_rank", label="Organic URL Match"))

        # E) Save CSV
        hist_file = "serpstack_map_organic.csv"
        if not os.path.isfile(hist_file):
            df.to_csv(hist_file, index=False)
        else:
            df.to_csv(hist_file, mode='a', header=False, index=False)
        st.success(f"Snapshot '{snapshot_name}' saved to '{hist_file}'.")

        # F) Download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Snapshot as CSV",
            data=csv_data,
            file_name=f"{snapshot_name}_serpstack_map_organic.csv",
            mime="text/csv"
        )

        # G) Detailed Rows
        st.write("### Detailed Results")
        for idx, row in df.iterrows():
            mp = row["map_pack_rank"]
            mp_str = str(int(mp)) if pd.notna(mp) else "X"
            org = row["organic_rank"]
            org_str = str(int(org)) if pd.notna(org) else "X"

            st.markdown(f"""
**Grid Point {idx+1}**  
- Coordinates: (Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f})  
- City Used: {row['city_used']}  
- Map Pack Rank: {mp_str}  
- Organic Rank: {org_str}  
""")

    # Compare old snapshots
    st.write("---")
    st.subheader("Compare Old Snapshots")
    uploaded_file = st.file_uploader("Upload a previously saved CSV (serpstack_map_organic.csv)", type=["csv"])
    if uploaded_file:
        old_df = pd.read_csv(uploaded_file)
        st.write("Found Snapshots:", old_df["snapshot_name"].unique())
        st.markdown("You could implement side-by-side or difference mapping here.")

if __name__ == "__main__":
    main()
