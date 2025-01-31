import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import openai  # for ChatGPT calls

# -------------------------------------------------------------------------
# 1. Load API Keys from Streamlit Secrets
# -------------------------------------------------------------------------
places_api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key

# Initialize Google Maps Client
gmaps = googlemaps.Client(key=places_api_key)


# -------------------------------------------------------------------------
# 2. Define Helper Functions
# -------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_lat_long_google(location_name: str):
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


def generate_square_grid(center_lat: float, center_lon: float, radius_miles: float, grid_size: int = 5):
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
            lat = center_lat + i * lat_step
            lon = center_lon + j * lon_step
            grid_points.append((lat, lon))

    return grid_points


@st.cache_data(show_spinner=False)
def search_places_api(
    lat: float,
    lon: float,
    keyword: str,
    target_business: str,
    api_key: str
):
    """
    Uses Google Places Nearby Search to find the rank of a specific 
    business (target_business) among up to 100 results sorted by distance.
    Returns:
        rank (int or None): The position of the client's business in the results.
        top_3 (list of dict): The top 3 competitor businesses 
                              with {name, rating, reviews}.
        client_details (dict or None): Additional info about the client if found.
    """
    location = f"{lat},{lon}"
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={location}&keyword={keyword}"
        f"&rankby=distance&key={api_key}"
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

    # Check up to 100 results
    for idx, result in enumerate(results[:100]):
        business_name = result.get('name', 'Unknown').lower()
        rating = result.get('rating', 'N/A')
        reviews = result.get('user_ratings_total', 'N/A')
        
        # Collect top 3 for competitor insight (exclude if it's the target business)
        if idx < 3:
            if target_business.lower() not in business_name:
                top_3.append({
                    'name': result.get('name', 'Unknown'),
                    'rating': rating,
                    'reviews': reviews
                })

        # Identify the target business rank
        if target_business.lower() in business_name:
            rank = idx + 1
            client_details = {
                'name': result.get('name', ''),
                'rating': rating,
                'reviews': reviews
            }

    return rank, top_3, client_details


def create_heatmap(df: pd.DataFrame, center_lat: float, center_lon: float):
    """
    Creates a Scattermapbox-based heatmap. The text (rank or X) is shown in black.
    """
    fig = go.Figure()

    for _, row in df.iterrows():
        rank_val = row['rank']
        # The marker color can still represent performance if desired.
        # But the text is explicitly in black per instructions.
        if rank_val is None:
            marker_color = 'red'
            text_label = "X"  # black X if not found
        elif rank_val <= 3:
            marker_color = 'green'
            text_label = str(rank_val)  # black rank number
        elif rank_val <= 10:
            marker_color = 'orange'
            text_label = str(rank_val)
        else:
            marker_color = 'red'
            text_label = str(rank_val)

        # Build hover text from top_3 competitor info
        if row['top_3']:
            hover_items = []
            for i, biz in enumerate(row['top_3']):
                hover_items.append(
                    f"{i+1}. {biz['name']} "
                    f"({biz['rating']}⭐, {biz['reviews']} reviews)"
                )
            hover_text = "<br>".join(hover_items)
        else:
            hover_text = "No competitor data in top 3."

        fig.add_trace(
            go.Scattermapbox(
                lat=[row['latitude']],
                lon=[row['longitude']],
                mode='markers+text',
                marker=dict(size=20, color=marker_color),
                text=[text_label],
                textposition="middle center",
                textfont=dict(size=14, color="black", family="Arial Black"),  # black text
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


def generate_growth_report(df: pd.DataFrame, client_gbp: str):
    """
    Generate a textual report highlighting:
    - Number of grid points in top 3, top 10, or not found.
    - Percent of grid points in top 3 and top 10.
    - Average rank.
    """
    total_points = len(df)

    # Where the business is actually found
    df_found = df.dropna(subset=['rank'])

    # Number of grid points in top 3
    top_3_points = df_found[df_found['rank'] <= 3].shape[0]
    # Number in top 10
    top_10_points = df_found[df_found['rank'] <= 10].shape[0]

    # Percentages
    pct_top_3 = 100.0 * top_3_points / total_points if total_points else 0
    pct_top_10 = 100.0 * top_10_points / total_points if total_points else 0

    # Summaries for green/orange/red
    green_count = top_3_points
    orange_count = df_found[(df_found['rank'] > 3) & (df_found['rank'] <= 10)].shape[0]
    red_count = total_points - (green_count + orange_count)

    average_rank = df_found['rank'].mean() if not df_found.empty else None

    lines = [
        f"**{client_gbp} Coverage Report**",
        f"- **Total Grid Points:** {total_points}",
        f"- **Business Found at:** {len(df_found)} points",
        f"- **In Top 3:** {top_3_points} points ({pct_top_3:.1f}% of total)",
        f"- **In Top 10:** {top_10_points} points ({pct_top_10:.1f}% of total)",
        f"- **Average Rank (where found):** {average_rank:.2f}" if average_rank else "- Average Rank: N/A",
        "",
        f"✅ **{green_count} areas** (rank 1–3).",
        f"🟠 **{orange_count} areas** (rank 4–10).",
        f"🔴 **{red_count} areas** (rank > 10 or not found).",
        "",
        "### Recommendations:",
        "- Improve presence in Red/Orange zones with local SEO tactics (on-page, link building, citations).",
        "- Encourage reviews to boost trust & ranking signals for localized searches.",
        "- Keep GBP (Google Business Profile) updated with new photos, Q&A, and posts."
    ]

    return "\n".join(lines)


def analyze_competitors_with_gpt(client_gbp: str, competitor_df: pd.DataFrame) -> str:
    """
    Sends competitor information to ChatGPT for an SEO comparison
    with the target business (client_gbp).
    competitor_df must have columns: 'name', 'rating', 'reviews', 
    plus any others you'd like to feed in.
    Returns the GPT-generated analysis as a string.
    """
    # Build a summary of competitor details
    competitor_summaries = []
    for _, row in competitor_df.iterrows():
        name = row['name']
        rating = row['rating']
        reviews = row['reviews']
        competitor_summaries.append(f"- {name}: {rating} stars, {reviews} reviews")

    # Turn them into a single prompt block
    competitor_text = "\n".join(competitor_summaries)

    prompt = f"""
You are a local SEO specialist. You have a target business named "{client_gbp}".
Here is a list of local competitor businesses, along with their rating and number of reviews:

{competitor_text}

Based on the above data, how do these competitors compare to "{client_gbp}" in terms of
online presence? Provide specific recommendations for "{client_gbp}" to improve
its local SEO against these competitors.
    """

    try:
        # ChatCompletion with the openai API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful SEO consultant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        gpt_answer = response.choices[0].message.content
        return gpt_answer
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "Could not analyze competitors with ChatGPT."


# -------------------------------------------------------------------------
# 3. Streamlit Main App
# -------------------------------------------------------------------------
def main():
    st.title("📍 Google Business Profile Ranking Heatmap & Analysis")
    st.write("Analyze how your business ranks across different grid points using Google Places API.\n"
             "Then compare competitor profiles using ChatGPT for deeper insights.")

    # --- User Inputs ---
    client_gbp = st.text_input("Enter Your Business Name (Google Business Profile)", "Starbucks")
    keyword = st.text_input("Enter Target Keyword (e.g., 'Coffee Shop')", "Coffee Shop")
    business_address = st.text_input("Enter Your Business Address (Full Address)", "Los Angeles, CA")
    radius = st.slider("Select Search Radius (miles)", 1, 20, 5)
    grid_size = st.slider("Select Grid Size", 3, 11, 5)

    # --- Generate Heatmap when button clicked ---
    if st.button("🔍 Generate Heatmap"):
        center_lat, center_lon = get_lat_long_google(business_address)

        if not center_lat or not center_lon:
            st.error("❌ Could not find the address. Please try again.")
            return

        st.success(f"📍 Address Found: {business_address} (Lat: {center_lat}, Lon: {center_lon})")
        
        # Generate the grid
        grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size=grid_size)

        # Collect ranking data
        grid_data = []
        all_competitors = []  # We will collect competitor data here for ChatGPT analysis

        for lat, lon in grid_points:
            rank, top_3, client_info = search_places_api(lat, lon, keyword, client_gbp, places_api_key)
            # Add competitor info to a global list (if any top_3 is found)
            all_competitors.extend(top_3)

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
        
        # --- Display Heatmap ---
        st.plotly_chart(create_heatmap(df, center_lat, center_lon), use_container_width=True)

        # --- Show Table of Results ---
        st.write("### 📊 Ranking Data")
        df['top_3_summary'] = df['top_3'].apply(
            lambda comps: ", ".join([f"{biz['name']} ({biz['reviews']} reviews)"
                                     for biz in comps]) if comps else "No data"
        )

        st.dataframe(
            df[['latitude', 'longitude', 'rank', 'client_rating', 'client_reviews', 'top_3_summary']]
        )

        # --- Growth / SEO Report ---
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

        # --- ChatGPT Competitor Analysis ---
        # Convert all_competitors into a unique DataFrame 
        # so we don't feed duplicate competitor entries multiple times.
        if all_competitors:
            competitor_df = pd.DataFrame(all_competitors)
            # Some businesses might appear multiple times in different grid points. 
            # We can group by name, taking the highest review count, or average rating, etc.
            # For simplicity, let’s keep the first appearance or deduplicate by name:
            competitor_df.drop_duplicates(subset=["name"], keep="first", inplace=True)

            # Let the user decide if they want to analyze competitor data with ChatGPT
            if st.button("Analyze Competitors with ChatGPT"):
                with st.spinner("Analyzing competitor data..."):
                    gpt_analysis = analyze_competitors_with_gpt(client_gbp, competitor_df)
                st.write("### ChatGPT Competitor Comparison")
                st.write(gpt_analysis)
        else:
            st.info("No competitor data found to analyze with ChatGPT.")


if __name__ == "__main__":
    main()
