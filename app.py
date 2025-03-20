import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import time
import os

# ------------------------------------------------------------
# 1. HELPER FUNCTIONS
# ------------------------------------------------------------

def generate_square_grid(center_lat: float, center_lon: float, radius_miles: float, grid_size: int = 5):
    """
    Generate a grid_size x grid_size grid of points within a square
    bounding box of +/- radius_miles around (center_lat, center_lon).
    """
    if grid_size < 1:
        return []

    lat_extent = radius_miles / 69.0
    lon_extent = radius_miles / (69.0 * np.cos(np.radians(center_lat)))

    lat_values = np.linspace(center_lat - lat_extent, center_lat + lat_extent, grid_size)
    lon_values = np.linspace(center_lon - lon_extent, center_lon + lon_extent, grid_size)

    grid_points = []
    for lat in lat_values:
        for lon in lon_values:
            grid_points.append((lat, lon))

    return grid_points

def generate_circular_grid(center_lat: float, center_lon: float, radius_miles: float, num_points: int = 25):
    """
    Generate lat/lon coordinates in a circular pattern around (center_lat, center_lon).
    """
    if num_points < 1:
        return []

    lat_degs = radius_miles / 69.0
    lon_degs = radius_miles / (69.0 * np.cos(np.radians(center_lat)))

    grid_points = []
    for i in range(num_points):
        angle = 2.0 * np.pi * (i / num_points)
        lat_offset = lat_degs * np.sin(angle)
        lon_offset = lon_degs * np.cos(angle)
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        grid_points.append((lat, lon))

    return grid_points

def fetch_nearby_places(lat: float, lon: float, keyword: str, api_key: str):
    """
    Collect up to 60 results from the Google Places Nearby Search,
    using rankby=distance (closest first).
    """
    location = f"{lat},{lon}"
    base_url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={location}&keyword={keyword}"
        f"&rankby=distance&key={api_key}"
    )

    all_results = []
    page_url = base_url
    for _ in range(3):  # up to 3 pages
        try:
            resp = requests.get(page_url)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Request error while fetching from Google Places API: {e}")
            break

        results = data.get("results", [])
        all_results.extend(results)

        if "next_page_token" in data:
            next_token = data["next_page_token"]
            # short delay to let the token activate
            time.sleep(2)
            page_url = base_url + f"&pagetoken={next_token}"
        else:
            break

    return all_results

def search_places_top3_by_rating(lat: float, lon: float, keyword: str, target_business: str, api_key: str):
    """
    1) Fetch places around (lat, lon).
    2) Sort them by rating (desc), then by reviews (desc), then name (asc).
    3) Return the top 3 places, plus the rank of the target business if found.
    """
    all_results = fetch_nearby_places(lat, lon, keyword, api_key)
    structured = []

    for place in all_results:
        name = place.get("name", "Unknown")
        place_id = place.get("place_id", "")
        rating = place.get("rating", 0)
        reviews = place.get("user_ratings_total", 0)
        types_ = place.get("types", [])
        open_now = place.get("opening_hours", {}).get("open_now", None)
        business_status = place.get("business_status", None)

        if rating is None:
            rating = 0
        if reviews is None:
            reviews = 0

        structured.append({
            "place_id": place_id,
            "name": name,
            "rating": float(rating),
            "reviews": int(reviews),
            "types": types_,
            "open_now": open_now,
            "business_status": business_status
        })

    # Sort by rating desc, then reviews desc, then name asc
    structured.sort(key=lambda x: (-x["rating"], -x["reviews"], x["name"]))

    top_3 = structured[:3]
    rank = None
    client_details = None
    for idx, biz in enumerate(structured):
        # match business by name substring
        if target_business.lower() in biz["name"].lower():
            rank = idx + 1
            client_details = biz
            break

    return rank, top_3, client_details

def create_heatmap(df: pd.DataFrame, center_lat: float, center_lon: float):
    """
    Creates a Scattermapbox-based heatmap of the "rating-based rank" results.
    """
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

        hover_text = "No competitor data."
        if row['top_3']:
            hover_items = []
            for i, biz in enumerate(row['top_3']):
                cats = ", ".join(biz.get("types", []))
                status = biz.get("business_status", "Unknown")
                open_status = biz.get("open_now", None)
                open_str = ("Open now" if open_status is True
                            else "Closed now" if open_status is False
                            else "Unknown")
                hover_items.append(
                    f"{i+1}. {biz['name']} "
                    f"({biz['rating']}⭐, {biz['reviews']} reviews)<br>"
                    f"Types: {cats}<br>"
                    f"Status: {status} - {open_str}"
                )
            hover_text = "<br><br>".join(hover_items)

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
            zoom=12
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Rating-Based Coverage Heatmap"
    )
    return fig

def generate_growth_report(df: pd.DataFrame, client_gbp: str):
    """
    Summarize coverage in top 3, top 10, or not found (X).
    """
    total_points = len(df)
    df_found = df.dropna(subset=['rank'])  # points where business is found

    top_3_points = df_found[df_found['rank'] <= 3].shape[0]
    top_10_points = df_found[df_found['rank'] <= 10].shape[0]

    pct_top_3 = 0
    pct_top_10 = 0
    if total_points > 0:
        pct_top_3 = 100.0 * top_3_points / total_points
        pct_top_10 = 100.0 * top_10_points / total_points

    average_rank = df_found['rank'].mean() if not df_found.empty else None

    lines = [
        f"**{client_gbp} Coverage Report**",
        f"- **Total Grid Points:** {total_points}",
        f"- **Business Found at:** {len(df_found)} points",
        f"- **In Top 3:** {top_3_points} points ({pct_top_3:.1f}% of total)",
        f"- **In Top 10:** {top_10_points} points ({pct_top_10:.1f}% of total)",
        f"- **Average Rank (where found):** {average_rank:.2f}" if average_rank else "- Average Rank: N/A",
    ]
    return "\n".join(lines)

# ------------------------------------------------------------
# 2. STREAMLIT APP
# ------------------------------------------------------------

def main():
    st.title("Local SEO: Rating-Based Geo-Grid Tool")
    st.write("""
    This tool uses **Google Places Nearby Search** to collect local businesses near each grid point, 
    sorts them by **star rating**, and shows where your business appears in that *rating-based* list.
    
    **Disclaimer**: This does **not** reflect the actual order of Google's map pack or local 3-Pack. 
    It's simply a quick, free way to see how your business's star rating compares regionally.
    """)

    # ---- 1) Ask for Google Maps/Places API key ----
    if "places_api_key" not in st.session_state:
        st.subheader("Enter Your Google Maps Places API Key")
        google_key_input = st.text_input("Google Maps/Places API Key", type="password")
        if st.button("Save API Key"):
            if google_key_input:
                st.session_state["places_api_key"] = google_key_input
                st.experimental_rerun()
            else:
                st.warning("Please provide a valid API key to proceed.")
        st.stop()

    places_api_key = st.session_state["places_api_key"]
    gmaps = googlemaps.Client(key=places_api_key)

    # ---- 2) Basic Input Fields ----
    snapshot_name = st.text_input("Snapshot Name", value=f"Snapshot_{int(time.time())}")
    client_gbp = st.text_input("Your Business Name (as on Google)", "Starbucks")
    keyword = st.text_input("Keyword to Explore (e.g., 'Coffee Shop')", "Coffee Shop")
    business_address = st.text_input("Business Address", "Los Angeles, CA")

    grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
    radius = st.slider("Radius (miles)", 1, 20, 5, help="Approx. distance from center to boundary of the grid.")
    if grid_shape == "Square":
        grid_size = st.slider("Square Grid Size", 3, 11, 5, help="Grid points = size * size.")
    else:
        grid_size = st.slider("Number of Circle Points", 8, 60, 25)

    # ---- 3) Generate Heatmap Action ----
    if st.button("🔍 Generate Rating-Based Heatmap"):
        # Geocode the address
        try:
            geocode_result = gmaps.geocode(business_address)
            if geocode_result:
                center_lat = geocode_result[0]['geometry']['location']['lat']
                center_lon = geocode_result[0]['geometry']['location']['lng']
            else:
                center_lat, center_lon = None, None
        except Exception as e:
            st.error(f"Error with Geocoding API: {e}")
            center_lat, center_lon = None, None

        if not center_lat or not center_lon:
            st.error("❌ Could not find the address. Check spelling or try a more specific address.")
            return

        st.success(f"Address Found: {business_address} (Lat: {center_lat:.5f}, Lon: {center_lon:.5f})")

        # Generate the grid
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Using {len(grid_points)} grid points (Square: {grid_size} x {grid_size}).")
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Using {len(grid_points)} grid points (Circle).")

        # Gather data
        grid_data = []
        competitor_place_ids = set()  # we won't do anything with them now that GPT analysis is removed
        client_info_global = {}
        
        progress_bar = st.progress(0)
        total_points = len(grid_points)

        for i, (lat, lon) in enumerate(grid_points):
            rank, top_3, client_details = search_places_top3_by_rating(
                lat, lon, keyword, client_gbp, places_api_key
            )
            if client_details is not None:
                client_info_global = client_details

            for c in top_3:
                competitor_place_ids.add(c["place_id"])

            grid_data.append({
                'latitude': lat,
                'longitude': lon,
                'rank': rank,
                'top_3': top_3,
            })

            # Update progress bar
            progress_bar.progress(int(((i+1)/total_points)*100))

        progress_bar.empty()

        df = pd.DataFrame(grid_data)
        df['snapshot_name'] = snapshot_name
        df['timestamp'] = pd.Timestamp.now()

        # Plot the heatmap
        st.plotly_chart(create_heatmap(df, center_lat, center_lon), use_container_width=True)

        # Display a growth report
        st.write("### Coverage Report (Rating-Based)")
        st.markdown(generate_growth_report(df, client_gbp))

        # Save to local CSV for future comparison
        history_file = "rating_based_history.csv"
        if not os.path.isfile(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)
        st.success(f"Snapshot '{snapshot_name}' saved in '{history_file}'.")

        # Download button for this data
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Current Snapshot",
            data=csv_data,
            file_name=f"{snapshot_name}_rating_data.csv",
            mime="text/csv"
        )

        # Detailed listing
        st.write("### Detailed Grid Point Results")
        for i, row in df.iterrows():
            rank_str = str(row['rank']) if row['rank'] else "X"
            top3_text = ""
            for comp_idx, comp in enumerate(row['top_3'], start=1):
                top3_text += (
                    f"\n   {comp_idx}. {comp['name']} "
                    f"(Rating: {comp['rating']}, {comp['reviews']} reviews) "
                    f"Types: {', '.join(comp.get('types', []))}, "
                    f"Status: {comp.get('business_status', 'Unknown')}, "
                    f"Open Now: {comp.get('open_now', 'Unknown')}"
                )

            st.markdown(f"""
**Grid Point {i+1}**  
- Coordinates: ({row['latitude']:.5f}, {row['longitude']:.5f})  
- **{client_gbp}** Rating-Based Rank: {rank_str if rank_str else "X"}  
- Top 3 by Rating: {top3_text if top3_text else "N/A"}
""")

    # ---- 4) Let user upload old data for comparisons ----
    st.write("---")
    st.subheader("Compare Past Snapshots")
    uploaded_file = st.file_uploader("Upload a previously saved CSV (e.g., rating_based_history.csv)", type=["csv"])
    if uploaded_file:
        old_data = pd.read_csv(uploaded_file)
        st.write("**Found Snapshots**:", old_data['snapshot_name'].unique())
        st.markdown("""
        You can implement advanced comparison logic here if you want to show
        how your rating-based rank coverage has changed over time.
        """)

if __name__ == "__main__":
    main()
