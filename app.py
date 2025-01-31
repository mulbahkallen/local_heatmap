import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# 1. Load Google Maps API Key securely
places_api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
gmaps = googlemaps.Client(key=places_api_key)

# 2. Latitude/Longitude Retrieval (cached)
@st.cache_data(show_spinner=False)
def get_lat_long_google(location_name):
    """
    Get the latitude and longitude for a given address/string location 
    using the Google Maps Geocoding API. 
    """
    try:
        geocode_result = gmaps.geocode(location_name)
        if geocode_result:
            lat = geocode_result[0]['geometry']['location']['lat']
            lon = geocode_result[0]['geometry']['location']['lng']
            return lat, lon
        else:
            return None, None
    except Exception as e:
        st.error(f"Error with Geocoding API: {e}")
        return None, None

# 3. Generate a Grid of Points (square) Around a Location
def generate_square_grid(center_lat, center_lon, radius_miles, grid_size=5):
    """
    Generates a list of (latitude, longitude) pairs in a square grid 
    around the center (center_lat, center_lon).
    radius_miles: the distance in miles from the center to extend.
    grid_size: how many points on one side of the grid (odd number recommended).
    """
    half_grid = grid_size // 2
    # Approx: 1 degree lat ~ 69 miles
    lat_step = radius_miles / 69.0 / half_grid  
    # Approx: 1 degree lon ~ 69 * cos(latitude) miles
    lon_step = radius_miles / (69.0 * np.cos(np.radians(center_lat))) / half_grid

    grid_points = []
    for i in range(-half_grid, half_grid + 1):
        for j in range(-half_grid, half_grid + 1):
            grid_points.append((center_lat + i * lat_step, center_lon + j * lon_step))
    return grid_points

# 4. Places API Search (cached)
@st.cache_data(show_spinner=False)
def search_places_api(lat, lon, keyword, client_gbp, places_api_key):
    """
    Uses Google Places Nearby Search to find the rank of a specific 
    business (client_gbp) among up to 100 results sorted by distance.
    Returns:
        rank (int or None): The position of the client's business in the results.
        top_3 (list of dict): The top 3 businesses with name, rating, and review count.
        client_details (dict or None): Additional info about the client if found (rating, reviews).
    """
    location = f"{lat},{lon}"
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={location}&keyword={keyword}"
        f"&rankby=distance&key={places_api_key}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json().get('results', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Request error while searching Places API: {e}")
        return None, [], None

    rank = None
    top_3 = []
    client_details = None

    for idx, result in enumerate(results[:100]):  # Limit to top 100 for performance
        business_name = result.get('name', 'Unknown').lower()
        rating = result.get('rating', 'N/A')
        reviews = result.get('user_ratings_total', 'N/A')
        
        # Collect top 3 for competitor insight
        if idx < 3:
            top_3.append({
                'name': result.get('name', 'Unknown'),
                'rating': rating,
                'reviews': reviews
            })
        
        # Check if this is the target business
        if client_gbp.lower() in business_name:
            rank = idx + 1
            # Keep track of extra client info if needed
            client_details = {
                'name': result.get('name', ''),
                'rating': rating,
                'reviews': reviews
            }

    return rank, top_3, client_details

# 5. Create a Heatmap (Plotly)
def create_heatmap(df, center_lat, center_lon):
    """
    Creates a Scattermapbox-based heatmap with color-coding:
        - green if rank <= 3
        - orange if 4 <= rank <= 10
        - red if rank is None or rank > 10
    Hover tooltip shows the top 3 competitor info at each point.
    """
    fig = go.Figure()

    for _, row in df.iterrows():
        rank_val = row['rank']
        if rank_val is None:
            color = 'red'
            text_label = "X"
        elif rank_val <= 3:
            color = 'green'
            text_label = str(rank_val)
        elif rank_val <= 10:
            color = 'orange'
            text_label = str(rank_val)
        else:
            color = 'red'
            text_label = str(rank_val)

        hover_text = "No businesses found."
        if row['top_3']:
            hover_items = []
            for i, biz in enumerate(row['top_3']):
                hover_items.append(
                    f"{i+1}. {biz['name']} "
                    f"({biz['rating']}⭐, {biz['reviews']} reviews)"
                )
            hover_text = "<br>".join(hover_items)

        fig.add_trace(
            go.Scattermapbox(
                lat=[row['latitude']],
                lon=[row['longitude']],
                mode='markers+text',
                marker=dict(size=20, color=color),
                text=[text_label],
                textposition="middle center",
                textfont=dict(size=12, color="black", family="Arial Black"),
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
        title="Target Business Ranking Heatmap"
    )
    return fig

# 6. Growth / SEO Report
def generate_growth_report(df, client_gbp):
    """
    Generate a textual report highlighting how many grid points 
    have the client in top 3, 4-10, and not found or >10. 
    Includes an average rank calculation for the points where 
    the business was actually found.
    """
    green_count = df[df['rank'].notna() & (df['rank'] <= 3)].shape[0]
    orange_count = df[df['rank'].notna() & (df['rank'] > 3) & (df['rank'] <= 10)].shape[0]
    red_count = df[df['rank'].isna() | (df['rank'] > 10)].shape[0]

    # For average rank, only consider points where rank is not None
    found_df = df[df['rank'].notna()]
    average_rank = found_df['rank'].mean() if not found_df.empty else None

    total_points = len(df)
    coverage_percentage = (len(found_df) / total_points) * 100 if total_points > 0 else 0

    report_lines = [
        f"**{client_gbp} Coverage Report**",
        f"- Total Grid Points: **{total_points}**",
        f"- Found in: **{len(found_df)}** points ({coverage_percentage:.2f}% coverage)",
        f"- Average Rank (where found): **{average_rank:.2f}**" if average_rank else "- Average Rank: N/A",
        "",
        f"✅ **{green_count} areas** (rank 1–3)",
        f"🟠 **{orange_count} areas** (rank 4–10)",
        f"🔴 **{red_count} areas** (rank > 10 or not found)",
        "",
        "### Recommendations:",
        "- **Improve Red/Orange Zones**: Focus on on-page optimizations & local signals in these regions.",
        "- **Optimize GBP Listing**: Ensure categories, photos, descriptions, and Q&A are updated regularly.",
        "- **Review Generation**: Encourage new reviews to stay competitive and increase click-through rates.",
        "- **Citations & Local Links**: Build consistent NAP (Name/Address/Phone) listings on local directories.",
        "- **Geo-Targeted Content**: Publish localized posts or pages to boost presence in target areas."
    ]

    return "\n".join(report_lines)

# 7. Streamlit App
def main():
    st.title("📍 Google Business Profile Ranking Heatmap")
    st.write("Analyze how your business ranks across different grid points using Google Places API.")

    # --- User Inputs ---
    client_gbp = st.text_input("Enter Your Business Name (Google Business Profile)", "Starbucks")
    keyword = st.text_input("Enter Target Keyword (e.g., 'Coffee Shop')", "Coffee Shop")
    business_address = st.text_input("Enter Your Business Address (Full Address)", "Los Angeles, CA")
    radius = st.slider("Select Search Radius (miles)", 1, 10, 5)
    grid_size = st.slider("Select Grid Size", 3, 11, 5)

    # --- Generate Heatmap when button clicked ---
    if st.button("🔍 Generate Heatmap"):
        center_lat, center_lon = get_lat_long_google(business_address)

        if not center_lat or not center_lon:
            st.error("❌ Could not find the address. Please try again.")
            return

        st.success(f"📍 Address Found: {business_address} ({center_lat}, {center_lon})")
        
        # Generate the grid
        grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size=grid_size)

        # Collect ranking data
        grid_data = []
        for lat, lon in grid_points:
            rank, top_3, client_info = search_places_api(lat, lon, keyword, client_gbp, places_api_key)
            grid_data.append({
                'latitude': lat,
                'longitude': lon,
                'rank': rank,
                'top_3': top_3,
                'client_name': client_info['name'] if client_info else None,
                'client_rating': client_info['rating'] if client_info else None,
                'client_reviews': client_info['reviews'] if client_info else None
            })

        # Create DataFrame
        df = pd.DataFrame(grid_data)
        
        # Display Heatmap
        st.plotly_chart(create_heatmap(df, center_lat, center_lon), use_container_width=True)

        # Show Table of Results
        st.write("### 📊 Ranking Data")
        # Convert top_3 from list of dicts to string
        df['top_3_summary'] = df['top_3'].apply(
            lambda x: ", ".join([biz['name'] for biz in x]) if x else "No data"
        )

        # Display the DataFrame in Streamlit
        st.dataframe(df[[
            'latitude', 'longitude', 'rank', 'client_rating', 'client_reviews', 'top_3_summary'
        ]])

        # Growth / SEO Report
        st.write("### 📈 Growth Report")
        st.markdown(generate_growth_report(df, client_gbp))

        # Option to Download CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Ranking Data",
            data=csv_data,
            file_name="ranking_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
