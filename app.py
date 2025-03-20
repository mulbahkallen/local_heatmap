import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import os

# ------------------------------------------------------------
# 1) HELPER FUNCTIONS
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
    sorted by distance (rankby=distance).
    """
    location = f"{lat},{lon}"
    base_url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={location}&keyword={keyword}"
        f"&rankby=distance&key={api_key}"
    )

    all_results = []
    page_url = base_url
    for _ in range(3):
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
            time.sleep(2)  # short delay for token activation
            page_url = base_url + f"&pagetoken={next_token}"
        else:
            break

    return all_results

def search_places_top3_by_rating(lat: float, lon: float, keyword: str, target_business: str, api_key: str):
    """
    1) Fetch places around (lat, lon).
    2) Sort them by rating (desc), then reviews (desc), then name (asc).
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

    # find target business rank if name is found
    for idx, biz in enumerate(structured):
        if target_business.lower() in biz["name"].lower():
            rank = idx + 1
            break

    return rank, top_3

def create_heatmap(df: pd.DataFrame, center_lat: float, center_lon: float):
    """
    Creates a Scattermap-based heatmap of the rating-based rank results.

    Uses go.Scattermap (MapLibre-based), which replaces the older go.Scattermapbox.
    """
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

        # Build hover text from top_3
        top_3 = row.get("top_3", [])
        hover_text = "No competitor data."
        if top_3:
            lines = []
            for i, biz in enumerate(top_3, start=1):
                lines.append(f"{i}. {biz['name']} ({biz['rating']}⭐, {biz['reviews']} reviews)")
            hover_text = "\n".join(lines)

        # Instead of Scattermapbox, use Scattermap
        fig.add_trace(
            go.Scattermap(
                lat=[row['latitude']],
                lon=[row['longitude']],
                mode='markers+text',
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
        mapbox_zoom=12,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Rating-Based Coverage Heatmap"
    )
    return fig

def generate_growth_report(df: pd.DataFrame, business_name: str):
    """
    Summarize how many grid points rank top 3, top 10, or not found.
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

    average_rank = df_found['rank'].mean() if not df_found.empty else None

    lines = [
        f"**{business_name} Coverage Report**",
        f"- **Total Grid Points:** {total_points}",
        f"- **Business Found:** {len(df_found)} points",
        f"- **In Top 3:** {top_3_points} ({pct_top_3:.1f}% of total)",
        f"- **In Top 10:** {top_10_points} ({pct_top_10:.1f}% of total)",
    ]
    if average_rank is not None:
        lines.append(f"- **Average Rank (where found):** {average_rank:.2f}")
    else:
        lines.append("- **Average Rank:** N/A (Business not found in any grid point)")

    return "\n".join(lines)

# ------------------------------------------------------------
# 2) STREAMLIT APP
# ------------------------------------------------------------

def main():
    st.title("Local SEO: Rating-Based Heatmap (No ChatGPT, Updated)")

    st.write("""
    This app uses the **Google Places Nearby Search API** (free within certain limits) to see how your
    business compares by **star rating** around a specific address.  
    - **Disclaimer**: This does **not** reflect actual Local 3-Pack ranking.  
    - Instead, it shows how your star rating stacks up against nearby places, 
      re-sorted by rating, for each grid point.
    """)

    # 1. Ask for Google Maps/Places API key if not in session
    if "places_api_key" not in st.session_state:
        st.subheader("Enter Your Google Maps Places API Key")
        st.write("Create or retrieve an API key from https://console.cloud.google.com/apis/credentials")
        google_key_input = st.text_input("Google Maps/Places API Key", type="password")
        if st.button("Save API Key"):
            if google_key_input:
                # store the key in session_state
                st.session_state["places_api_key"] = google_key_input
                st.warning("Your API Key was saved. Please refresh or rerun the app to continue.")
            else:
                st.warning("Please provide a valid API key.")
        st.stop()

    # 2. Basic user inputs
    snapshot_name = st.text_input("Snapshot Name/Label", value=f"rating_snapshot_{int(time.time())}",
                                  help="Used to label this run in the CSV history.")
    business_name = st.text_input("Your Business Name (as it appears in Google)", help="Ex: Starbucks, ACME Plumbing, etc.")
    keyword = st.text_input("Keyword to Search For", value="Coffee Shop",
                            help="E.g. 'Coffee Shop', 'Plumber', 'Pizza near me'")

    business_address = st.text_input("Enter a Full Address or City for Center", value="Los Angeles, CA",
                                     help="We will geocode this to find the center lat/lon of your grid.")

    grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
    radius = st.slider("Radius (Miles)", 1, 20, 5, help="Approx distance from center address to edge of your grid.")
    if grid_shape == "Square":
        grid_size = st.slider("Square Grid Size", 3, 11, 5, help="Number of points along each side (Grid=5x5=25).")
    else:
        grid_size = st.slider("Number of Circle Points", 8, 60, 25, help="Points on the perimeter of the circle.")

    # 3. Generate Heatmap button
    if st.button("Generate Rating Heatmap"):
        try:
            # Geocode center
            gmaps_client = googlemaps.Client(key=st.session_state["places_api_key"])
            geocode_result = gmaps_client.geocode(business_address)
            if geocode_result:
                center_lat = geocode_result[0]['geometry']['location']['lat']
                center_lon = geocode_result[0]['geometry']['location']['lng']
            else:
                center_lat, center_lon = None, None
        except Exception as e:
            st.error(f"Geocoding error: {e}")
            center_lat, center_lon = None, None

        if not center_lat or not center_lon:
            st.error("Failed to geocode the address. Please try a more specific address.")
            return

        st.success(f"Address Found: {business_address} → (Lat: {center_lat:.5f}, Lon: {center_lon:.5f})")

        # Build the grid
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Using a {grid_size}x{grid_size} square = {len(grid_points)} total points.")
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Using a circular pattern with {len(grid_points)} total points.")

        # Gather data
        df_data = []
        progress = st.progress(0)

        for i, (lat, lon) in enumerate(grid_points, start=1):
            rank, top_3 = search_places_top3_by_rating(
                lat, lon, keyword, business_name, st.session_state["places_api_key"]
            )
            df_data.append({
                "latitude": lat,
                "longitude": lon,
                "rank": rank,
                "top_3": top_3
            })
            progress.progress(int((i / len(grid_points)) * 100))

        progress.empty()

        # Create DataFrame
        df = pd.DataFrame(df_data)
        df["snapshot_name"] = snapshot_name
        df["timestamp"] = pd.Timestamp.now()

        # Plot the heatmap
        fig = create_heatmap(df, center_lat, center_lon)
        st.plotly_chart(fig, use_container_width=True)

        # Show summary stats
        st.write("### Coverage Report")
        st.markdown(generate_growth_report(df, business_name))

        # Save to local CSV
        history_file = "rating_based_history.csv"
        if not os.path.isfile(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)
        st.success(f"Snapshot '{snapshot_name}' appended to '{history_file}'.")

        # Download current snapshot
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download This Snapshot (CSV)",
            data=csv_data,
            file_name=f"{snapshot_name}_rating_data.csv",
            mime="text/csv"
        )

        # Detailed listing
        st.write("### Detailed Results Per Grid Point")
        for idx, row in df.iterrows():
            rank_str = row["rank"] if row["rank"] is not None else "X"
            top3_details = row["top_3"]
            comp_list = ""
            for c_idx, comp in enumerate(top3_details, start=1):
                comp_list += (f"\n   {c_idx}. {comp['name']} "
                              f"(Rating: {comp['rating']}, {comp['reviews']} reviews)")

            st.markdown(f"""
**Grid Point {idx+1}**  
- Coordinates: (Lat: {row['latitude']:.5f}, Lon: {row['longitude']:.5f})  
- **{business_name}** Rank: {rank_str}  
- Top 3 by Rating: {comp_list if comp_list else "None"}
""")

    # Optionally, let user upload old CSV for comparisons
    st.write("---")
    st.subheader("Compare Past Snapshots")
    uploaded_file = st.file_uploader("Upload a previously saved CSV (e.g. rating_based_history.csv)", type=["csv"])
    if uploaded_file:
        old_data = pd.read_csv(uploaded_file)
        st.write("Found Snapshots:", old_data['snapshot_name'].unique())
        st.markdown("""
        Implement custom difference or side-by-side comparisons here if desired.
        """)


if __name__ == "__main__":
    main()
