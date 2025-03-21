import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import requests
import time
import os
import plotly.graph_objects as go

from requests.adapters import HTTPAdapter, Retry

# ------------------------------------------------
# 1) Global Retry Session
# ------------------------------------------------
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
session.mount('https://', HTTPAdapter(max_retries=retries))


# ------------------------------------------------
# 2) Helper Functions: Common
# ------------------------------------------------

def generate_square_grid(center_lat, center_lon, radius_miles, grid_size):
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

def generate_circular_grid(center_lat, center_lon, radius_miles, num_points):
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

def create_scattermap(df, center_lat, center_lon, rank_col, map_title):
    fig = go.Figure()
    for _, row in df.iterrows():
        rank_val = row[rank_col]
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

        hover_text = f"Rank: {text_label}"

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

def coverage_report(df, rank_col, label):
    total = len(df)
    df_found = df.dropna(subset=[rank_col])

    top3 = df_found[df_found[rank_col] <= 3].shape[0]
    top10 = df_found[df_found[rank_col] <= 10].shape[0]
    pct3 = (100 * top3 / total) if total else 0
    pct10 = (100 * top10 / total) if total else 0

    avg_rank = df_found[rank_col].mean() if not df_found.empty else None
    lines = [
        f"**{label}**",
        f"- Total Points: {total}",
        f"- Found at: {len(df_found)}",
        f"- In Top 3: {top3} ({pct3:.1f}%)",
        f"- In Top 10: {top10} ({pct10:.1f}%)"
    ]
    if avg_rank is not None:
        lines.append(f"- Average Rank: {avg_rank:.2f}")
    else:
        lines.append("- Average Rank: N/A")
    return "\n".join(lines)


# ------------------------------------------------
# 3) Serpstack Approach
# ------------------------------------------------

def serpstack_location_api(api_key, location_query):
    base_url = "https://api.serpstack.com/locations"
    params = {
        "access_key": api_key,
        "query": location_query,
        "limit": 1
    }
    try:
        r = session.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("canonical_name", "")
        return None
    except:
        return None

def serpstack_search(api_key, query, canonical_loc):
    base_url = "https://api.serpstack.com/search"
    params = {
        "access_key": api_key,
        "query": query,
        "location": canonical_loc,
        "output": "json",
        "type": "web",
        "num": 10,
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
    except:
        return None

def serpstack_map_pack_rank(serp_data, gbp_name):
    if "local_results" not in serp_data:
        return None
    target = gbp_name.lower()
    for i, item in enumerate(serp_data["local_results"], start=1):
        title = item.get("title", "").lower()
        if target in title:
            return i
    return None

def serpstack_organic_rank(serp_data, domain, fallback_name):
    if "organic_results" not in serp_data:
        return None

    dsub = domain.lower()
    fsub = fallback_name.lower()
    for i, item in enumerate(serp_data["organic_results"], start=1):
        dom = item.get("domain", "").lower()
        url = item.get("url", "").lower()
        disp = item.get("displayed_url", "").lower()
        title = item.get("title", "").lower()

        if (dsub in dom) or (dsub in url) or (dsub in disp) or (fsub in title):
            return i
    return None


# ------------------------------------------------
# 4) Google Places Approach
# ------------------------------------------------

def google_places_fetch(lat, lon, keyword, api_key):
    """
    Use the Places NearbySearch with rankby=distance (or radius?), 
    then re-sort results by rating desc. 
    This does NOT reflect actual search rank in Google Maps.
    """
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    loc_str = f"{lat},{lon}"
    # We use rankby=distance to get places. Then we'll do a custom re-sort by rating desc.
    params = {
        "location": loc_str,
        "keyword": keyword,
        "rankby": "distance",
        "key": api_key
    }
    all_results = []
    page_token = None

    for _ in range(3):
        try:
            r = session.get(base_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            all_results.extend(results)

            if "next_page_token" in data:
                page_token = data["next_page_token"]
                time.sleep(2)
                params["pagetoken"] = page_token
            else:
                break
        except:
            break

    # Sort by rating desc, then reviews desc
    structured = []
    for place in all_results:
        name = place.get("name", "").lower()
        place_id = place.get("place_id", "")
        rating = place.get("rating", 0) or 0
        reviews = place.get("user_ratings_total", 0) or 0
        structured.append({
            "place_id": place_id,
            "name": name,
            "rating": float(rating),
            "reviews": int(reviews)
        })

    structured.sort(key=lambda x: (-x["rating"], -x["reviews"], x["name"]))
    return structured

def google_places_rank(lat, lon, keyword, gbp_name, domain_sub, fallback_name, api_key):
    """
    1) Fetch up to 60 places near (lat, lon).
    2) Re-sort them by rating desc.
    3) Find "map_pack_rank" by partial matching gbp_name in place name.
    4) Find "organic_rank" by partial matching domain_sub or fallback_name in place name 
       (very approximate, not truly 'organic'). 
       Or we can treat it as a second "ranking" for demonstration.
    """
    structured = google_places_fetch(lat, lon, keyword, api_key)
    if not structured:
        return (None, None)

    map_rank = None
    org_rank = None
    target_gbp = gbp_name.lower()
    target_dom = domain_sub.lower()
    fallback_biz = fallback_name.lower()

    for idx, item in enumerate(structured, start=1):
        place_name = item["name"]
        # Map rank if gbp_name is in place name
        if (map_rank is None) and (target_gbp in place_name):
            map_rank = idx
        # For "organic rank," we fake a partial match if domain or fallback is in place name
        # (this is obviously not real "organic" but just for demonstration)
        if (org_rank is None) and ((target_dom in place_name) or (fallback_biz in place_name)):
            org_rank = idx

        if map_rank and org_rank:
            break

    return (map_rank, org_rank)


# ------------------------------------------------
# 5) Streamlit App (Client-Facing)
# ------------------------------------------------
def main():
    st.title("Local SEO Rank Checker")
    st.write("Select the approach you want to use: **Serpstack** or **Google Places**.")

    # 1) Approach
    approach = st.radio("Choose Method", ["Serpstack", "Google Places"])

    # 2) API Keys
    if approach == "Serpstack":
        # we need serpstack & google maps
        if "serpstack_key" not in st.session_state:
            st.subheader("Enter Serpstack API Key")
            serp_input = st.text_input("Serpstack Access Key", type="password")
            if st.button("Save Serpstack Key"):
                if serp_input:
                    st.session_state["serpstack_key"] = serp_input
                    st.info("Key saved. Please rerun.")
                else:
                    st.warning("Please enter a valid Serpstack key.")
            st.stop()

        if "google_maps_key" not in st.session_state:
            st.subheader("Enter Google Maps API Key")
            gmaps_input = st.text_input("Google Maps/Places API Key", type="password")
            if st.button("Save Google Maps Key"):
                if gmaps_input:
                    st.session_state["google_maps_key"] = gmaps_input
                    st.info("Key saved. Please rerun.")
                else:
                    st.warning("Please enter a valid Google Maps key.")
            st.stop()

        serpstack_key = st.session_state["serpstack_key"]
        google_maps_key = st.session_state["google_maps_key"]
        gmaps_client = googlemaps.Client(key=google_maps_key)

    else:
        # "Google Places" approach only needs google maps key
        if "google_maps_key" not in st.session_state:
            st.subheader("Enter Google Maps API Key")
            gmaps_input = st.text_input("Google Maps/Places API Key", type="password")
            if st.button("Save Google Maps Key"):
                if gmaps_input:
                    st.session_state["google_maps_key"] = gmaps_input
                    st.info("Key saved. Please rerun.")
                else:
                    st.warning("Please enter a valid Google Maps key.")
            st.stop()

        google_maps_key = st.session_state["google_maps_key"]
        gmaps_client = googlemaps.Client(key=google_maps_key)

    # 3) Basic user input
    snapshot_name = st.text_input("Snapshot Name", value=f"local_seo_snapshot_{int(time.time())}")
    keyword = st.text_input("Keyword", "coffee shop")
    gbp_name = st.text_input("GBP/Business Name (Map Pack partial match)", "My Business")
    domain_name = st.text_input("Short Domain (e.g. 'mydomain.com')", "mydomain.com")
    fallback_name = st.text_input("Fallback Name (for organic)", "My Brand")
    business_address = st.text_input("Business Address", "Los Angeles, CA")

    # 4) Grid Settings
    grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
    radius_miles = st.slider("Radius (Miles)", 1, 20, 5)
    if grid_shape == "Square":
        grid_size = st.slider("Square Grid Size", 3, 11, 5)
    else:
        grid_size = st.slider("Circle Points", 8, 60, 25)

    if st.button("Run Analysis"):
        # Geocode center
        try:
            geo_res = gmaps_client.geocode(business_address)
            if geo_res:
                center_lat = geo_res[0]["geometry"]["location"]["lat"]
                center_lon = geo_res[0]["geometry"]["location"]["lng"]
            else:
                center_lat, center_lon = None, None
        except Exception as e:
            st.error(f"Error geocoding: {e}")
            return

        if not center_lat or not center_lon:
            st.error("Could not geocode address.")
            return

        # Generate points
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius_miles, grid_size)
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius_miles, grid_size)

        results = []
        progress_bar = st.progress(0)
        total_pts = len(grid_points)

        for i, (lat, lon) in enumerate(grid_points, start=1):
            if approach == "Serpstack":
                # 1) reverse geocode
                approx_city = reverse_geocode_city(lat, lon, gmaps_client)
                if approx_city:
                    # 2) location API
                    canonical_loc = serpstack_location_api(serpstack_key, approx_city)
                    if canonical_loc:
                        serp_data = serpstack_search(serpstack_key, keyword, canonical_loc)
                        if serp_data:
                            map_rank = serpstack_map_pack_rank(serp_data, gbp_name)
                            org_rank = serpstack_organic_rank(serp_data, domain_name, fallback_name)
                        else:
                            map_rank = None
                            org_rank = None
                    else:
                        map_rank = None
                        org_rank = None
                else:
                    map_rank = None
                    org_rank = None
            else:
                # "Google Places" approach
                map_rank, org_rank = google_places_rank(
                    lat, lon, keyword,
                    gbp_name, domain_name, fallback_name,
                    google_maps_key
                )

            results.append({
                "latitude": lat,
                "longitude": lon,
                "map_pack_rank": map_rank,
                "organic_rank": org_rank
            })

            progress_bar.progress(int((i / total_pts)*100))
            time.sleep(1)

        progress_bar.empty()

        df = pd.DataFrame(results)
        df["snapshot_name"] = snapshot_name
        df["timestamp"] = pd.Timestamp.now()

        # Show side-by-side maps
        col1, col2 = st.columns(2)
        with col1:
            fig_map = create_scattermap(df, center_lat, center_lon, "map_pack_rank", "Map Pack Coverage")
            st.plotly_chart(fig_map, use_container_width=True)
        with col2:
            fig_org = create_scattermap(df, center_lat, center_lon, "organic_rank", "Organic Coverage")
            st.plotly_chart(fig_org, use_container_width=True)

        # Coverage Summaries
        st.subheader("Coverage Reports")
        st.markdown(coverage_report(df, "map_pack_rank", "Map Pack Rank"))
        st.markdown(coverage_report(df, "organic_rank", "Organic Rank"))

        # Save CSV
        history_file = f"{approach.lower()}_local_seo_history.csv"
        if not os.path.isfile(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)
        st.success(f"Snapshot '{snapshot_name}' saved to '{history_file}'.")

        # Download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download This Snapshot (CSV)",
            data=csv_data,
            file_name=f"{snapshot_name}_{approach.lower()}_local_seo.csv",
            mime="text/csv"
        )

        # Detailed Output
        st.subheader("Detailed Points")
        for idx, row in df.iterrows():
            mp = row["map_pack_rank"]
            org = row["organic_rank"]
            mp_str = str(int(mp)) if pd.notna(mp) else "X"
            org_str = str(int(org)) if pd.notna(org) else "X"

            st.markdown(f"""
**Point {idx+1}**  
- Lat/Lon: ({row['latitude']:.4f}, {row['longitude']:.4f})  
- Map Rank: {mp_str}  
- Organic Rank: {org_str}
""")

    # Compare old snapshots
    st.write("---")
    st.subheader("Compare Old Snapshots")
    upfile = st.file_uploader("Upload CSV", type=["csv"])
    if upfile:
        old_data = pd.read_csv(upfile)
        st.write("Snapshots Found:", old_data["snapshot_name"].unique())
        st.write("Implement further comparison or difference logic as needed.")


if __name__ == "__main__":
    main()
