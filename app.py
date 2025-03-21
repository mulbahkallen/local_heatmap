import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import requests
import time
import os
import plotly.graph_objects as go

from requests.adapters import HTTPAdapter, Retry

# -----------------------------------------------------------
# 1) GLOBAL RETRY SESSION
# -----------------------------------------------------------
session = requests.Session()
retries = Retry(
    total=3,             
    backoff_factor=1,    
    status_forcelist=[429, 500, 502, 503, 504]
)
session.mount('https://', HTTPAdapter(max_retries=retries))


# -----------------------------------------------------------
# 2) HELPER FUNCTIONS
# -----------------------------------------------------------

def serpstack_location_api(api_key: str, location_query: str, limit=1):
    """
    Query serpstack's Locations API to get a canonical_name for a city/state/country.
    E.g. "New York" -> "New York,NY,United States"
    """
    base_url = "https://api.serpstack.com/locations"
    params = {
        "access_key": api_key,
        "query": location_query,
        "limit": limit
    }
    try:
        resp = session.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            # means there's an error
            st.error(f"Serpstack location error: {data['error'].get('info', 'Unknown')}")
            return None
        elif isinstance(data, list) and len(data) > 0:
            # pick the first location
            return data[0].get("canonical_name", "")
        else:
            return None
    except requests.RequestException as e:
        st.error(f"Error calling serpstack Locations API: {e}")
        return None


def generate_square_grid(center_lat: float, center_lon: float, radius_miles: float, grid_size: int = 5):
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
    Attempt to get a 'City, State' or 'Locality, AdminArea'.
    We'll pass this to serpstack's *Locations API* next to get a canonical name.
    """
    try:
        results = gmaps_client.reverse_geocode((lat, lon))
        if not results:
            return ""
        locality = ""
        admin_area = ""
        for comp in results[0].get("address_components", []):
            if "locality" in comp["types"]:
                locality = comp["long_name"]
            if "administrative_area_level_1" in comp["types"]:
                admin_area = comp["long_name"]

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

def serpstack_search(api_key: str, query: str, canonical_loc: str, num: int = 10):
    """
    Query serpstack's 'search' endpoint with a *canonical location* from the Locations API.
    """
    base_url = "https://api.serpstack.com/search"
    params = {
        "access_key": api_key,
        "query": query,
        "location": canonical_loc,
        "output": "json",
        "type": "web",
        "num": num,
        "page": 1,
        "auto_location": 0
    }
    try:
        resp = session.get(base_url, params=params, timeout=30)
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
    if "local_results" not in data:
        return None
    for i, loc in enumerate(data["local_results"], start=1):
        title = loc.get("title", "")
        if gbp_name.lower() in title.lower():
            return i
    return None

def find_organic_rank_by_url(data: dict, domain_name: str):
    if "organic_results" not in data:
        return None
    for i, item in enumerate(data["organic_results"], start=1):
        url = item.get("url", "")
        if domain_name.lower() in url.lower():
            return i
    return None

def create_scattermap(df: pd.DataFrame, center_lat: float, center_lon: float, rank_type: str):
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

        hover_text = f"Used City: {row['approx_city']}\n{rank_type}: {text_label}"

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
        mapbox_zoom=10,
        margin=dict(r=0, t=0, l=0, b=0),
        title=f"SERPstack Heatmap – {rank_type}"
    )
    return fig

def generate_coverage_report(df: pd.DataFrame, column: str, label: str):
    total_pts = len(df)
    df_found = df.dropna(subset=[column])

    top3 = df_found[df_found[column] <= 3].shape[0]
    top10 = df_found[df_found[column] <= 10].shape[0]

    pct_top3 = (100.0 * top3 / total_pts) if total_pts else 0
    pct_top10 = (100.0 * top10 / total_pts) if total_pts else 0

    avg_rank = df_found[column].mean() if not df_found.empty else None

    lines = [
        f"**Coverage Report: {label}**",
        f"- **Total Grid Points:** {total_pts}",
        f"- **Found at:** {len(df_found)}",
        f"- **In Top 3:** {top3} ({pct_top3:.1f}%)",
        f"- **In Top 10:** {top10} ({pct_top10:.1f}%)"
    ]
    if avg_rank is not None:
        lines.append(f"- **Average Rank:** {avg_rank:.2f}")
    else:
        lines.append("- **Average Rank:** N/A")
    return "\n".join(lines)


# -----------------------------------------------------------
# 3) STREAMLIT APP
# -----------------------------------------------------------

def main():
    st.title("Local SERP Analysis with Serpstack (Using the Locations API)")

    st.write("""
    This version uses Serpstack's **Locations API** to get a recognized canonical location string for each grid point,
    which often helps retrieve map pack (local_results) data if Google shows a local pack for that query.

    **Steps**:
    1. Provide Serpstack key + Google Maps key.
    2. Enter your business address (we center the grid on it).
    3. We'll build a grid, reverse-geocode each point to e.g. "City,State".
    4. We'll call Serpstack's Locations API to get a canonical_name for that city.
    5. We'll query the SERP with that canonical_name.
    6. We rank your business by **GBP name** in map pack and by **domain** in organic.

    If you're still seeing "X" everywhere:
    - Possibly the query does not trigger a local pack,
    - The domain or GBP name doesn't match how it appears in results,
    - Or your plan doesn't return that data.
    """)

    # 1) Serpstack key
    if "serpstack_key" not in st.session_state:
        st.subheader("Enter Serpstack API Key")
        serp_input = st.text_input("Serpstack Access Key", type="password")
        if st.button("Save Serpstack Key"):
            if serp_input:
                st.session_state["serpstack_key"] = serp_input
                st.warning("Key saved. Please rerun or refresh.")
            else:
                st.warning("Please provide a valid key.")
        st.stop()

    serpstack_key = st.session_state["serpstack_key"]

    # 2) Google Maps key
    if "google_maps_key" not in st.session_state:
        st.subheader("Enter Google Maps API Key")
        gkey_input = st.text_input("Google Maps/Places API Key", type="password")
        if st.button("Save Google Maps Key"):
            if gkey_input:
                st.session_state["google_maps_key"] = gkey_input
                st.warning("Key saved. Please rerun or refresh.")
            else:
                st.warning("Please provide a valid key.")
        st.stop()

    gmaps_client = googlemaps.Client(key=st.session_state["google_maps_key"])

    # 3) Basic user inputs
    snapshot_name = st.text_input("Snapshot Name", value=f"serpstack_snapshot_{int(time.time())}")
    gbp_name = st.text_input("Google Business Profile Name", "Starbucks",
                             help="We'll look for this name in local_results 'title'.")
    domain_name = st.text_input("Website Domain for Organic", "starbucks.com",
                                help="We'll match this substring in the 'url' field of organic results.")
    keyword = st.text_input("Keyword to Search", "coffee shop")
    business_address = st.text_input("Business Address", "Los Angeles, CA",
                                     help="We geocode this for the center of the grid.")

    grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
    radius_miles = st.slider("Radius (Miles)", 1, 20, 5)
    if grid_shape == "Square":
        grid_size = st.slider("Square Grid Size", 3, 11, 5)
    else:
        grid_size = st.slider("Circle Points", 8, 60, 25)

    if st.button("Run Serpstack Analysis"):
        # A) Geocode center
        try:
            geo_result = gmaps_client.geocode(business_address)
            if geo_result:
                center_lat = geo_result[0]["geometry"]["location"]["lat"]
                center_lon = geo_result[0]["geometry"]["location"]["lng"]
            else:
                center_lat, center_lon = None, None
        except Exception as e:
            st.error(f"Error geocoding address: {e}")
            center_lat, center_lon = None, None

        if not center_lat or not center_lon:
            st.error("Could not geocode that address. Try again.")
            return

        st.write(f"**Center**: {business_address} → (Lat: {center_lat:.4f}, Lon: {center_lon:.4f})")

        # B) Generate grid
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius_miles, grid_size)
            st.write(f"**Generated** {len(grid_points)} points in a {grid_size}x{grid_size} square.")
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius_miles, grid_size)
            st.write(f"**Generated** {len(grid_points)} points in a circle pattern.")

        # C) For each point -> 1) reverse geocode to city -> 2) call serpstack Locations API
        #    -> 3) do serpstack search -> parse map + organic rank
        results_data = []
        progress_bar = st.progress(0)
        total_pts = len(grid_points)

        for i, (lat, lon) in enumerate(grid_points, start=1):
            # 1) Reverse geocode to "City,State"
            approx_city = reverse_geocode_city(lat, lon, gmaps_client)

            if not approx_city:
                # no city found
                map_rank = None
                org_rank = None
                canonical_loc = ""
            else:
                # 2) call serpstack location API to get canonical name
                canonical_loc = serpstack_location_api(serpstack_key, approx_city, limit=1)
                if not canonical_loc:
                    # fallback
                    map_rank = None
                    org_rank = None
                else:
                    # 3) call serpstack search
                    data = serpstack_search(serpstack_key, keyword, canonical_loc)
                    if not data:
                        map_rank = None
                        org_rank = None
                    else:
                        map_rank = find_map_pack_rank(data, gbp_name)
                        org_rank = find_organic_rank_by_url(data, domain_name)

            results_data.append({
                "latitude": lat,
                "longitude": lon,
                "approx_city": approx_city,
                "canonical_loc": canonical_loc if canonical_loc else "",
                "map_pack_rank": map_rank,
                "organic_rank": org_rank
            })

            progress_bar.progress(int((i / total_pts)*100))
            time.sleep(1)  # 1s delay to reduce timeouts

        progress_bar.empty()

        # Turn into DataFrame
        df = pd.DataFrame(results_data)
        df["snapshot_name"] = snapshot_name
        df["timestamp"] = pd.Timestamp.now()

        # D) Let user pick which rank to visualize
        rank_choice = st.selectbox("Which rank to plot on the heatmap?",
                                   ["map_pack_rank", "organic_rank"])
        fig = create_scattermap(df, center_lat, center_lon, rank_choice)
        st.plotly_chart(fig, use_container_width=True)

        # E) Coverage Reports
        st.write("### Map Pack Coverage")
        st.markdown(generate_coverage_report(df, "map_pack_rank", label="Map Pack Rank"))

        st.write("### Organic Coverage")
        st.markdown(generate_coverage_report(df, "organic_rank", label="Organic URL Match"))

        # F) Save to local CSV
        history_file = "serpstack_canonical_history.csv"
        if not os.path.isfile(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode="a", header=False, index=False)

        st.success(f"Snapshot '{snapshot_name}' saved to '{history_file}'.")

        # G) Download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Snapshot CSV",
            data=csv_data,
            file_name=f"{snapshot_name}_serpstack_canonical.csv",
            mime="text/csv"
        )

        # H) Detailed Rows
        st.write("### Detailed Grid Results")
        for idx, row in df.iterrows():
            mp = row["map_pack_rank"]
            org = row["organic_rank"]
            mp_str = str(int(mp)) if pd.notna(mp) else "X"
            org_str = str(int(org)) if pd.notna(org) else "X"

            st.markdown(f"""
**Grid Point {idx+1}**  
- Coordinates: ({row['latitude']:.4f}, {row['longitude']:.4f})  
- Reverse-Geocoded City: {row['approx_city']}  
- Canonical Location: {row['canonical_loc']}  
- **Map Pack Rank**: {mp_str}  
- **Organic Rank**: {org_str}
""")

    # Compare old snapshots
    st.write("---")
    st.subheader("Compare Old Snapshots")
    upfile = st.file_uploader("Upload a previously saved CSV", type=["csv"])
    if upfile:
        old_df = pd.read_csv(upfile)
        st.write("Found Snapshots:", old_df["snapshot_name"].unique())
        st.markdown("Implement side-by-side or difference logic here if desired.")


if __name__ == "__main__":
    main()
