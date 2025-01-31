import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import openai
from bs4 import BeautifulSoup

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
    Generate a grid_size x grid_size grid of points within a square
    bounding box of +/- radius_miles around (center_lat, center_lon).

    If grid_size=4, you get 16 total points.
    If grid_size=5, you get 25 total points, etc.
    """
    if grid_size < 1:
        return []

    # Approximate degrees per mile for latitude and longitude
    lat_extent = radius_miles / 69.0
    lon_extent = radius_miles / (69.0 * np.cos(np.radians(center_lat)))

    # Create arrays of size grid_size spaced between [center - extent, center + extent]
    lat_values = np.linspace(center_lat - lat_extent, center_lat + lat_extent, grid_size)
    lon_values = np.linspace(center_lon - lon_extent, center_lon + lon_extent, grid_size)

    grid_points = []
    for lat in lat_values:
        for lon in lon_values:
            grid_points.append((lat, lon))

    return grid_points



def search_places_nearby(lat: float, lon: float, keyword: str, target_business: str, api_key: str):
    """
    Uses Google Places Nearby Search to find the rank of a specific
    business (target_business) among up to 100 results sorted by distance.
    Returns:
        rank (int or None): The position of the client's business in the results.
        top_3 (list of dict): The top 3 competitor businesses
                              with {place_id, name, rating, reviews}.
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
        place_id = result.get('place_id')
        name = result.get('name', 'Unknown')
        lower_name = name.lower()
        rating = result.get('rating', 'N/A')
        reviews = result.get('user_ratings_total', 'N/A')

        # Identify the target business rank
        if target_business.lower() in lower_name:
            rank = idx + 1
            client_details = {
                'place_id': place_id,
                'name': name,
                'rating': rating,
                'reviews': reviews
            }

        # For the top 3 (competitors), skip if it's the target
        if idx < 3 and target_business.lower() not in lower_name:
            top_3.append({
                'place_id': place_id,
                'name': name,
                'rating': rating,
                'reviews': reviews
            })

    return rank, top_3, client_details


def get_place_details(place_id: str, api_key: str):
    """
    Fetch additional details about a place using the Google Places Details API.
    Returns a dict with fields like {address, phone, website, etc.}
    """
    url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}&key={api_key}"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json().get('result', {})
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching place details: {e}")
        return {}

    return {
        'address': result.get('formatted_address', ''),
        'phone': result.get('formatted_phone_number', ''),
        'website': result.get('website', ''),
        'name': result.get('name', ''),
        'rating': result.get('rating', 'N/A'),
        'reviews': result.get('user_ratings_total', 'N/A')
    }


def scrape_website(url: str, max_chars: int = 2000):
    """
    Attempt to scrape a competitor's website to get textual data for AI analysis.
    Return raw text (limited to max_chars).
    Note: This may fail if the site blocks scraping or needs JavaScript.
    """
    if not url:
        return ""

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Grab visible text
        texts = soup.find_all(["p", "h1", "h2", "h3", "h4", "li"], limit=None)
        combined = " ".join(t.get_text(separator=" ", strip=True) for t in texts)
        # Limit the size
        return combined[:max_chars]
    except Exception:
        # Gracefully handle any scraping errors
        return ""


def create_heatmap(df: pd.DataFrame, center_lat: float, center_lon: float):
    """
    Creates a Scattermapbox-based heatmap. The text (rank or X) is shown in black.
    """
    fig = go.Figure()

    for _, row in df.iterrows():
        rank_val = row['rank']
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
    - Percentage of grid points in top 3 and top 10.
    - Average rank.
    """
    total_points = len(df)
    df_found = df.dropna(subset=['rank'])  # Only points where business is found

    top_3_points = df_found[df_found['rank'] <= 3].shape[0]
    top_10_points = df_found[df_found['rank'] <= 10].shape[0]

    pct_top_3 = 100.0 * top_3_points / total_points if total_points else 0
    pct_top_10 = 100.0 * top_10_points / total_points if total_points else 0

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


def analyze_competitors_with_gpt(client_gbp: str, competitor_details: list) -> str:
    """
    Sends competitor information to ChatGPT for an SEO comparison
    with the target business (client_gbp).
    competitor_details: a list of dicts, each having keys like
        { 'name', 'rating', 'reviews', 'address', 'phone', 'website_content', ... }
    Returns the GPT-generated analysis as a string.
    """
    # Build a summary of competitor details
    competitor_summaries = []
    for comp in competitor_details:
        summary_str = (
            f"- Name: {comp.get('name', 'N/A')}\n"
            f"  Address: {comp.get('address', 'N/A')}\n"
            f"  Phone: {comp.get('phone', 'N/A')}\n"
            f"  Rating: {comp.get('rating', 'N/A')} with {comp.get('reviews', '0')} reviews\n"
        )
        if comp.get('website_content'):
            # Summarize the presence of website text (truncated in scrape)
            summary_str += f"  Website Snippet: {comp.get('website_content')[:200]}...\n"
        competitor_summaries.append(summary_str)

    competitor_text = "\n".join(competitor_summaries)

    # Create a system + user prompt for best results
    prompt = f"""
You are a local SEO consultant. The target business is "{client_gbp}".
Below is competitor data (addresses, phone, reviews, ratings, snippet from website):
{competitor_text}
Provide a comparison and specific SEO recommendations for "{client_gbp}" to improve 
its local presence. Consider GMB/GBP optimization, website improvements, 
and local citation strategies.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a highly skilled local SEO consultant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=700
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
    st.write("Analyze how your business ranks in your target area.\n"
             "Then compare competitor profiles using AI for deeper insights.")

    # --- User Inputs ---
    client_gbp = st.text_input("Enter Your Business Name (Google Business Profile)", "Starbucks")
    keyword = st.text_input("Enter Target Keyword (e.g., 'Coffee Shop')", "Coffee Shop")
    business_address = st.text_input("Enter Your Business Address (Full Address)", "Los Angeles, CA")
    radius = st.slider("Select Search Radius (miles)", 1, 20, 5)
    grid_size = st.slider("Select Grid Size (total points = grid_size^2)", 3, 11, 5)

    # --- Generate Heatmap when button clicked ---
    if st.button("🔍 Generate Heatmap"):
        center_lat, center_lon = get_lat_long_google(business_address)

        if not center_lat or not center_lon:
            st.error("❌ Could not find the address. Please try again.")
            return

        st.success(f"📍 Address Found: {business_address} (Lat: {center_lat}, Lon: {center_lon})")

        # Generate the grid
        grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
        st.write(f"Using **{len(grid_points)}** grid points (grid_size={grid_size}×{grid_size}).")

        # Collect ranking data
        grid_data = []
        competitor_place_ids = set()

        for lat, lon in grid_points:
            rank, top_3, client_info = search_places_nearby(
                lat, lon, keyword, client_gbp, places_api_key
            )
            # Gather competitor place_ids for further details
            for c in top_3:
                competitor_place_ids.add(c['place_id'])

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

        # --- Show a Cleaner Table of Results ---
        st.write("### 📊 Ranking Data (Summary)")
        # We'll display it as bullet points for each row
        for i, row in df.iterrows():
            rank_str = str(row['rank']) if row['rank'] else "X"
            top3_text = ""
            for comp_idx, comp in enumerate(row['top_3'], start=1):
                top3_text += f"\n   {comp_idx}. {comp['name']} ({comp['rating']}⭐, {comp['reviews']} reviews)"
            st.markdown(f"""
**Grid Point {i+1}**  
- Location: ({row['latitude']:.5f}, {row['longitude']:.5f})  
- Rank of {client_gbp}: {rank_str}  
- Top 3 Competitors: {top3_text if top3_text else "None"}
            """)

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

        # --- Gather Detailed Competitor Info (Place Details + Scrape) ---
        if competitor_place_ids:
            if st.button("Analyze Competitors with ChatGPT"):
                with st.spinner("Fetching competitor details & scraping websites..."):
                    competitor_details_list = []
                    for pid in competitor_place_ids:
                        details = get_place_details(pid, places_api_key)
                        # Attempt to scrape website for textual info
                        website_content = ""
                        if details.get('website'):
                            website_content = scrape_website(details['website'], max_chars=2000)

                        competitor_details_list.append({
                            'name': details.get('name', ''),
                            'address': details.get('address', ''),
                            'phone': details.get('phone', ''),
                            'rating': details.get('rating', 'N/A'),
                            'reviews': details.get('reviews', '0'),
                            'website': details.get('website', ''),
                            'website_content': website_content
                        })

                    # Now feed these details to GPT
                    gpt_analysis = analyze_competitors_with_gpt(client_gbp, competitor_details_list)
                st.write("### ChatGPT Competitor Comparison")
                st.write(gpt_analysis)
        else:
            st.info("No competitor data found to analyze with ChatGPT.")


if __name__ == "__main__":
    main()
