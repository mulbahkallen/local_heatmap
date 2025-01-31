import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# Load Google Maps API Key securely from Streamlit secrets
places_api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
gmaps = googlemaps.Client(key=places_api_key)

# Function to Get Latitude and Longitude using Google Maps API
def get_lat_long_google(location_name):
    geocode_result = gmaps.geocode(location_name)
    if geocode_result:
        lat = geocode_result[0]['geometry']['location']['lat']
        lon = geocode_result[0]['geometry']['location']['lng']
        return lat, lon
    else:
        return None, None

# Function to Generate a Grid of Points Around a Location
def generate_square_grid(center_lat, center_lon, radius, grid_size=5):
    half_grid = grid_size // 2
    lat_step = radius / 69.0 / grid_size
    lon_step = radius / (69.0 * np.cos(np.radians(center_lat))) / grid_size

    grid_points = []
    for i in range(-half_grid, half_grid + 1):
        for j in range(-half_grid, half_grid + 1):
            grid_points.append((center_lat + i * lat_step, center_lon + j * lon_step))

    return grid_points

# Function to Search Google Places API for a Specific Business
def search_places_api(lat, lon, keyword, client_gbp):
    location = f"{lat},{lon}"
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&keyword={keyword}&rankby=distance&key={places_api_key}"

    response = requests.get(url)
    if response.status_code != 200:
        return None, []

    results = response.json().get('results', [])
    rank = None
    top_3 = []

    for idx, result in enumerate(results[:100]):  # Check up to top 100 results
        # Check for the target business
        if client_gbp.lower() in result.get('name', '').lower():
            rank = idx + 1

        # Collect top 3 businesses
        if idx < 3:
            top_3.append({
                'name': result.get('name', 'Unknown'),
                'rating': result.get('rating', 'N/A'),
                'reviews': result.get('user_ratings_total', 'N/A')
            })

    return rank, top_3

# Function to Create the Heatmap
def create_heatmap(df, center_lat, center_lon):
    fig = go.Figure()

    for _, row in df.iterrows():
        # Assign color based on rank
        if row['rank'] is None:
            color = 'red'  # No ranking
            text = "X"  # Display "X" for no rank
        elif row['rank'] <= 3:
            color = 'green'  # High visibility
            text = f"{row['rank']}"
        elif row['rank'] <= 10:
            color = 'orange'  # Moderate visibility
            text = f"{row['rank']}"
        else:
            color = 'red'  # Low visibility
            text = f"{row['rank']}"

        # Hover text with top 3 businesses
        hover_text = "<br>".join(
            [f"{idx + 1}. {biz['name']} ({biz['rating']}⭐, {biz['reviews']} reviews)"
             for idx, biz in enumerate(row['top_3'])]
        ) if row['top_3'] else "No businesses found."

        # Add point to map
        fig.add_trace(go.Scattermapbox(
            lat=[row['latitude']],
            lon=[row['longitude']],
            mode='markers+text',
            marker=dict(size=20, color=color),
            text=[text],  # Display rank or "X"
            textfont=dict(size=12, color="black", family="Arial Black"),
            textposition="middle center",
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False
        ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=12),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Target Business Ranking Heatmap"
    )

    return fig

# Function to Generate a Growth Report
def generate_growth_report(df, client_gbp):
    """Generate a report highlighting ranking insights and recommendations."""
    green_count = df[df['rank'] <= 3].shape[0]
    orange_count = df[(df['rank'] > 3) & (df['rank'] <= 10)].shape[0]
    red_count = df[df['rank'].isna() | (df['rank'] > 10)].shape[0]

    report = []
    report.append(f"✅ **{green_count} areas where {client_gbp} ranks highly (1–3)**.")
    report.append(f"🟠 **{orange_count} areas where {client_gbp} has moderate visibility (4–10)**.")
    report.append(f"🔴 **{red_count} areas where {client_gbp} ranks poorly or is not found.**")
    report.append("### Recommendations:")
    report.append("- Focus on improving visibility in red and orange zones.")
    report.append("- Encourage customers to leave reviews to boost credibility.")
    report.append("- Optimize GBP details (e.g., categories, photos, keywords).")
    report.append("- Engage with users by responding to reviews and updating content.")

    return "\n".join(report)

# Streamlit UI
st.title("📍 Google Business Profile Ranking Heatmap")
st.write("Analyze how your business ranks across different grid points using Google Places API.")

# User Inputs
client_gbp = st.text_input("Enter Your Business Name (Google Business Profile)", "Starbucks")
keyword = st.text_input("Enter Target Keyword (e.g., 'Coffee Shop')", "Coffee Shop")
business_address = st.text_input("Enter Your Business Address (Full Address)", "Los Angeles, CA")
radius = st.slider("Select Search Radius (miles)", 1, 10, 5)
grid_size = st.slider("Select Grid Size", 3, 11, 5)

if st.button("🔍 Generate Heatmap"):
    center_lat, center_lon = get_lat_long_google(business_address)

    if not center_lat or not center_lon:
        st.error("❌ Could not find the address. Please try again.")
    else:
        st.success(f"📍 Address Found: {business_address} ({center_lat}, {center_lon})")

        # Generate grid points
        grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
        grid_data = []

        # Search rankings for each grid point
        for lat, lon in grid_points:
            rank, top_3 = search_places_api(lat, lon, keyword, client_gbp)
            grid_data.append({'latitude': lat, 'longitude': lon, 'rank': rank, 'top_3': top_3})

        # Create DataFrame for results
        df = pd.DataFrame(grid_data)

        # Display heatmap
        st.plotly_chart(create_heatmap(df, center_lat, center_lon))

        # Display ranking data table
        st.write("### 📊 Ranking Data")
        detailed_df = df.copy()
        detailed_df['top_3'] = detailed_df['top_3'].apply(lambda x: ", ".join([biz['name'] for biz in x]) if x else "No data")
        st.dataframe(detailed_df[['latitude', 'longitude', 'rank', 'top_3']])

        # Generate growth report
        st.write("### 📈 Growth Report")
        st.markdown(generate_growth_report(df, client_gbp))

        # Option to download ranking data as CSV
        st.download_button(
            label="📥 Download Ranking Data",
            data=detailed_df.to_csv(index=False).encode('utf-8'),
            file_name="ranking_data.csv",
            mime="text/csv"
        )
