import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import requests
import time
import os
import plotly.graph_objects as go

from requests.adapters import HTTPAdapter, Retry

# ----------------------------------------------
# 1) GLOBAL RETRY/SESSION
# ----------------------------------------------
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
session.mount('https://', HTTPAdapter(max_retries=retries))


# ----------------------------------------------
# 2) HELPER FUNCTIONS
# ----------------------------------------------

def serpstack_location_api(api_key: str, location_query: str, limit=1):
    """Use Serpstack's Locations API to fetch a canonical location string."""
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
        # If data is a list with at least 1 item, return the 'canonical_name'
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("canonical_name", "")
        return None
    except requests.RequestException:
        return None

def generate_square_grid(center_lat: float, center_lon: float, radius_miles: float, grid_size: int):
    """Build a grid_size x grid_size square of lat/lon points."""
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

def generate_circular_grid(center_lat: float, center_lon: float, radius_miles: float, num_points: int):
    """Build a circular pattern of lat/lon points."""
    if num_points < 1:
        return []
    lat_degs = radius_miles / 69.0
    lon_degs = radius_miles / (69.0 * np.cos(np.radians(center_lat)))
    points = []
    for i in range(num_points):
        angle = 2.0 * np.pi * (i / num_points)
        lat_offset = lat_degs * np.sin(angle)
        lon_offset = lon_degs * np.cos(angle)
        points.append((center_lat + lat_offset, center_lon + lon_offset))
    return points

def reverse_geocode_city(lat: float, lon: float, gmaps_client) -> str:
    """Approximate city/state from lat/lon using Google Maps."""
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

        # fallback to other location types
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

def serpstack_search(api_key: str, query: str, canonical_loc: str, num=10):
    """Query serpstack's main 'search' endpoint with a canonical location."""
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
            return None
        return data
    except requests.RequestException:
        return None

def find_map_pack_rank(serp_data: dict, gbp_name: str):
    """Partial match the GBP name in local_results 'title'."""
    if "local_results" not in serp_data:
        return None
    target_sub = gbp_name.lower()
    for i, item in enumerate(serp_data["local_results"], start=1):
        title = item.get("title", "").lower()
        if target_sub in title:
            return i
    return None

def find_organic_rank(serp_data: dict, domain_name: str, fallback_biz_name: str):
    """
    Identify best matching link in 'organic_results'.
    We do partial match for domain_name in 'domain', 'url', 'displayed_url'
    OR fallback_biz_name in 'title'.
    """
    if "organic_results" not in serp_data:
        return None

    dsub = domain_name.lower()
    bsub = fallback_biz_name.lower()

    for i, item in enumerate(serp_data["organic_results"], start=1):
        dom = item.get("domain", "").lower()
        url = item.get("url", "").lower()
        disp = item.get("displayed_url", "").lower()
        title = item.get("title", "").lower()

        # If domain substring is in domain/url/displayed_url, or fallback name in title
        if dsub in dom or dsub in url or dsub in disp or bsub in title:
            return i
    return None

def create_scattermap(df: pd.DataFrame, center_lat: float, center_lon: float, rank_type: str, map_title: str):
    """Build a scattermap with a more pleasing style (carto-positron) and moderate zoom."""
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

        hover_text = (
            f"City: {row['approx_city']}\n"
            f"Rank: {text_label}"
        )

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
        mapbox_style="carto-positron",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=12,
        margin=dict(r=0, t=40, l=0, b=0),
        title=map_title
    )
    return fig

def coverage_report(df: pd.DataFrame, column: str, label: str):
    """Summarize coverage of a particular rank column (map_pack_rank or organic_rank)."""
    total_pts = len(df)
    df_found = df.dropna(subset=[column])

    top3 = df_found[df_found[column] <= 3].shape[0]
    top10 = df_found[df_found[column] <= 10].shape[0]

    pct_top3 = 100 * top3 / total_pts if total_pts else 0
    pct_top10 = 100 * top10 / total_pts if total_pts else 0

    avg_rank = df_found[column].mean() if not df_found.empty else None

    lines = [
        f"**{label}**",
        f"- Total Grid Points: {total_pts}",
        f"- Found at: {len(df_found)}",
        f"- In Top 3: {top3}  ({pct_top3:.1f}%)",
        f"- In Top 10: {top10} ({pct_top10:.1f}%)"
    ]
    if avg_rank is not None:
        lines.append(f"- Average Rank: {avg_rank:.2f}")
    else:
        lines.append("- Average Rank: N/A")

    return "\n".join(lines)


# ----------------------------------------------
# 3) STREAMLIT APP (Client-Facing)
# ----------------------------------------------
def main():
    st.title("Local SEO Rank Checker")

    # Minimal instructions, client-facing
    st.write("""
    Enter the required details below to generate two side-by-side heatmaps:
    1. **Map Pack Rank** based on your Business Name.
    2. **Organic Rank** based on your Domain and Fallback Name.
    """)

    # 1) Retrieve or prompt for API keys
    if "serpstack_key" not in st.session_state:
        st.subheader("Enter Serpstack API Key")
        serp_input = st.text_input("Serpstack Access Key (e.g. 32f0...)", type="password")
        if st.button("Save Serpstack Key"):
            if serp_input:
                st.session_state["serpstack_key"] = serp_input
                st.info("Key saved. Please rerun the tool.")
            else:
                st.warning("Please enter a valid Serpstack key.")
        st.stop()

    if "google_maps_key" not in st.session_state:
        st.subheader("Enter Google Maps API Key")
        gmaps_input = st.text_input("Google Maps/Places API Key (e.g. AIza...)", type="password")
        if st.button("Save Google Maps Key"):
            if gmaps_input:
                st.session_state["google_maps_key"] = gmaps_input
                st.info("Key saved. Please rerun the tool.")
            else:
                st.warning("Please enter a valid Google Maps key.")
        st.stop()

    serpstack_key = st.session_state["serpstack_key"]
    google_maps_key = st.session_state["google_maps_key"]
    gmaps_client = googlemaps.Client(key=google_maps_key)

    # 2) User inputs for business & grid
    with st.expander("Project / Business Details"):
        snapshot_name = st.text_input("Project/Snapshot Name", value=f"local_seo_snapshot_{int(time.time())}")
        domain_name = st.text_input("Short Domain (for organic)", help="e.g., 'mydomain.com' (no http/https).")
        fallback_biz_name = st.text_input("Fallback Name (for organic)", help="Used if domain not found in URL.")
        gbp_name = st.text_input("GBP Name (Map Pack)", help="Business name as it appears in local pack (partial match).")
        keyword = st.text_input("Keyword to Search", value="coffee shop")
        business_address = st.text_input("Business Address", help="We'll center the grid on this address.")

    with st.expander("Grid Settings"):
        grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
        radius_miles = st.slider("Radius (Miles)", 1, 20, 5)
        if grid_shape == "Square":
            grid_size = st.slider("Square Grid Size", 3, 11, 5, help="5 => 25 points, etc.")
        else:
            grid_size = st.slider("Circle Points", 8, 60, 25)

    # 3) RUN
    if st.button("Run Analysis"):
        # A) Geocode the center
        try:
            geo_result = gmaps_client.geocode(business_address)
            if geo_result:
                center_lat = geo_result[0]["geometry"]["location"]["lat"]
                center_lon = geo_result[0]["geometry"]["location"]["lng"]
            else:
                center_lat, center_lon = None, None
        except Exception as e:
            st.error(f"Error with Geocoding: {e}")
            return

        if not center_lat or not center_lon:
            st.error("Unable to geocode that address. Please try again.")
            return

        # B) Generate grid
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius_miles, grid_size)
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius_miles, grid_size)

        # C) For each grid point -> reverse geocode -> location API -> serpstack search -> parse results
        data_rows = []
        progress = st.progress(0)
        total_points = len(grid_points)

        for i, (lat, lon) in enumerate(grid_points, start=1):
            approx_city = reverse_geocode_city(lat, lon, gmaps_client)
            if approx_city:
                # get canonical name
                canonical_loc = serpstack_location_api(serpstack_key, approx_city)
                if canonical_loc:
                    serp_data = serpstack_search(serpstack_key, keyword, canonical_loc)
                    if serp_data:
                        map_rank = find_map_pack_rank(serp_data, gbp_name)
                        org_rank = find_organic_rank(serp_data, domain_name, fallback_biz_name)
                    else:
                        map_rank = None
                        org_rank = None
                else:
                    map_rank = None
                    org_rank = None
            else:
                map_rank = None
                org_rank = None
                canonical_loc = ""

            data_rows.append({
                "latitude": lat,
                "longitude": lon,
                "approx_city": approx_city,
                "canonical_loc": canonical_loc if canonical_loc else "",
                "map_pack_rank": map_rank,
                "organic_rank": org_rank
            })

            progress.progress(int((i / total_points) * 100))
            time.sleep(1)  # 1s delay to avoid rate/timeouts

        progress.empty()

        df = pd.DataFrame(data_rows)
        df["snapshot_name"] = snapshot_name
        df["timestamp"] = pd.Timestamp.now()

        # D) Display 2 side-by-side maps: one for Map Pack, one for Organic
        col1, col2 = st.columns(2)
        with col1:
            fig_map_pack = create_scattermap(df, center_lat, center_lon, "map_pack_rank", "Map Pack Coverage")
            st.plotly_chart(fig_map_pack, use_container_width=True)
        with col2:
            fig_org = create_scattermap(df, center_lat, center_lon, "organic_rank", "Organic Coverage")
            st.plotly_chart(fig_org, use_container_width=True)

        # E) Coverage reports
        st.subheader("Coverage Summaries")
        st.markdown(coverage_report(df, "map_pack_rank", "Map Pack Rank"))
        st.markdown(coverage_report(df, "organic_rank", "Organic Rank"))

        # F) Save CSV
        history_file = "local_seo_history.csv"
        if not os.path.isfile(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode="a", header=False, index=False)
        st.success(f"Snapshot '{snapshot_name}' saved to '{history_file}'.")

        # G) Download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download This Snapshot (CSV)",
            data=csv_data,
            file_name=f"{snapshot_name}_local_seo.csv",
            mime="text/csv"
        )

        # H) Detailed results
        st.subheader("Detailed Results")
        for idx, row in df.iterrows():
            mp_rank = row["map_pack_rank"]
            org_rank = row["organic_rank"]
            mp_str = str(int(mp_rank)) if pd.notna(mp_rank) else "X"
            org_str = str(int(org_rank)) if pd.notna(org_rank) else "X"

            st.markdown(f"""
**Grid Point {idx+1}**  
- Coordinates: ({row['latitude']:.4f}, {row['longitude']:.4f})  
- Reverse-Geocoded City: {row['approx_city']}  
- Map Pack Rank: {mp_str}  
- Organic Rank: {org_str}
""")

    # Compare old snapshots
    st.write("---")
    st.subheader("Compare Old Snapshots")
    file_upload = st.file_uploader("Upload a previously saved CSV", type=["csv"])
    if file_upload:
        old_data = pd.read_csv(file_upload)
        st.write("Found Snapshots:", old_data["snapshot_name"].unique())
        st.write("Implement additional comparison logic here if desired.")


if __name__ == "__main__":
    main()
