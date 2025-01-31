import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
import zipfile

# Load Google Maps API Key securely from Streamlit Secrets
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

# Function to Search Google Places API for Competitors
def search_places_api(lat, lon, keyword, client_gbp):
    location = f"{lat},{lon}"
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&keyword={keyword}&rankby=distance&key={places_api_key}"

    response = requests.get(url)
    if response.status_code != 200:
        return None, []

    results = response.json().get('results', [])
    
    businesses = []
    client_rank = None

    for idx, result in enumerate(results[:10]):  # Limit to top 10 results
        business_name = result.get('name', 'Unknown')
        business_data = {
            'rank': idx + 1,
            'name': business_name,
            'rating': result.get('rating', 'N/A'),
            'user_ratings_total': result.get('user_ratings_total', 'N/A'),
            'vicinity': result.get('vicinity', 'N/A'),
            'categories': ", ".join(result.get('types', [])),
            'latitude': result.get('geometry', {}).get('location', {}).get('lat', None),
            'longitude': result.get('geometry', {}).get('location', {}).get('lng', None)
        }
        businesses.append(business_data)

        # Check if the client's business is found in the results
        if client_gbp.lower() in business_name.lower():
            client_rank = idx + 1  # Rank starts from 1

    return client_rank, businesses

# Function to Create a Plotly Map
def create_map(df, center_lat, center_lon, client_gbp, client_rank):
    fig = go.Figure()

    for _, row in df.iterrows():
        marker_color = "blue" if row["name"].lower() == client_gbp.lower() else "red"
        hover_text = f"{row['rank']}: {row['name']} ({row['rating']}⭐ {row['user_ratings_total']} reviews)"

        fig.add_trace(go.Scattermapbox(
            lat=[row["latitude"]],
            lon=[row["longitude"]],
            mode="markers+text",
            marker=dict(size=15, color=marker_color),
            text=[str(row["rank"])],
            hovertext=hover_text,
            hoverinfo="text"
        ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=12),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title=f"Competitor Rankings for '{client_gbp}' ({'Ranked' if client_rank else 'Not Found'})"
    )
    
    return fig

# Streamlit UI
st.title("📍 Google Business Profile Rank & Competitor Analysis")
st.write("Check your business ranking and analyze competitors using Google Places API.")

# User Inputs
client_gbp = st.text_input("Enter Your Business Name (Google Business Profile)", "Starbucks")
keyword = st.text_input("Enter Target Keyword (e.g., 'Coffee Shop')", "Coffee Shop")
location = st.text_input("Enter Search Location (City or Zip Code)", "Los Angeles, CA")

if st.button("🔍 Analyze Rankings"):
    center_lat, center_lon = get_lat_long_google(location)

    if not center_lat or not center_lon:
        st.error("❌ Could not find the location. Please try again.")
    else:
        st.success(f"📍 Location Found: {location} ({center_lat}, {center_lon})")

        # Search Google Places API for businesses
        client_rank, businesses = search_places_api(center_lat, center_lon, keyword, client_gbp)

        if businesses:
            df = pd.DataFrame(businesses)
            st.plotly_chart(create_map(df, center_lat, center_lon, client_gbp, client_rank))

            # Display results
            st.write("### 📊 Competitor Rankings")
            st.dataframe(df[['rank', 'name', 'rating', 'user_ratings_total', 'vicinity']])

            # Show client ranking
            if client_rank:
                st.success(f"✅ **{client_gbp} is ranked #{client_rank} for '{keyword}' in {location}.**")
            else:
                st.warning(f"⚠️ **{client_gbp} was NOT found in the top 10 results for '{keyword}'.**")

            # Create a ZIP file for download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                zipf.writestr("competitor_analysis.csv", df.to_csv(index=False))

            st.download_button("📥 Download Report", zip_buffer.getvalue(), "business_audit.zip", "application/zip")

