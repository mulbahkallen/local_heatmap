import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
import zipfile

# Load Google Maps API Key (Securely from Streamlit Secrets)
places_api_key = st.secrets.get("GOOGLE_MAPS_API_KEY")
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
def generate_square_grid(center_lat, center_lon, radius, grid_size=3):
    half_grid = grid_size // 2
    lat_step = radius / 69.0 / grid_size
    grid_points = []

    for i in range(-half_grid, half_grid + 1):
        for j in range(-half_grid, half_grid + 1):
            lon_step = radius / (69.0 * np.cos(np.radians(center_lat))) / grid_size
            grid_points.append((center_lat + i * lat_step, center_lon + j * lon_step))

    return grid_points

# Function to Search Google Places API
def search_places_api(lat, lon, keyword):
    location = f"{lat},{lon}"
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&keyword={keyword}&rankby=distance&key={places_api_key}"

    response = requests.get(url)
    if response.status_code != 200:
        return [], []

    results = response.json().get('results', [])

    businesses = []
    for idx, result in enumerate(results[:10]):  # Limit to top 10 results
        businesses.append({
            'rank': idx + 1,
            'name': result.get('name', 'Unknown'),
            'rating': result.get('rating', 'N/A'),
            'user_ratings_total': result.get('user_ratings_total', 'N/A'),
            'vicinity': result.get('vicinity', 'N/A'),
            'categories': result.get('types', [])
        })

    return businesses

# Function to Create Plotly Map
def create_map(df, center_lat, center_lon):
    fig = go.Figure()

    for index, row in df.iterrows():
        marker_color = 'green' if row['rank'] <= 3 else 'orange' if row['rank'] <= 6 else 'red'
        hover_text = f"{row['rank']}: {row['name']} ({row['rating']}⭐ {row['user_ratings_total']} reviews)"

        fig.add_trace(go.Scattermapbox(
            lat=[row['latitude']],
            lon=[row['longitude']],
            mode='markers+text',
            marker=dict(size=15, color=marker_color),
            text=[str(row['rank'])],
            hovertext=hover_text,
            hoverinfo='text'
        ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=12),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    return fig

# Streamlit UI
st.title("📍 Local Business Audit & Competitor Analysis")
st.write("Analyze your business ranking and find competitor insights using Google Places API.")

# User Inputs
keyword = st.text_input("Enter Business Keyword (e.g., 'Coffee Shop')", "Coffee Shop")
central_location = st.text_input("Enter Central Location (e.g., 'Los Angeles, CA')", "Los Angeles, CA")
radius = st.slider("Select Search Radius (miles)", 1, 10, 5)
grid_size = st.slider("Select Grid Size", 3, 7, 5)

if st.button("🔍 Run Analysis"):
    center_lat, center_lon = get_lat_long_google(central_location)

    if not center_lat or not center_lon:
        st.error("❌ Could not find the location. Please try again.")
    else:
        st.success(f"📍 Center Location Found: {central_location} ({center_lat}, {center_lon})")

        # Generate grid points
        grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
        business_data = []

        for lat, lon in grid_points:
            businesses = search_places_api(lat, lon, keyword)
            for business in businesses:
                business.update({"latitude": lat, "longitude": lon})
                business_data.append(business)

        if business_data:
            df = pd.DataFrame(business_data)
            st.plotly_chart(create_map(df, center_lat, center_lon))
            
            # Display top businesses
            st.write("### 📊 Top Competitors")
            st.dataframe(df[['rank', 'name', 'rating', 'user_ratings_total', 'vicinity']])

            # Create a ZIP file for download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                zipf.writestr("competitor_analysis.csv", df.to_csv(index=False))

            st.download_button("📥 Download Report", zip_buffer.getvalue(), "business_audit.zip", "application/zip")

