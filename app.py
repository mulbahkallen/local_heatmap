import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import openai
from bs4 import BeautifulSoup
import time
import os

# -------------------------------------------------------------------------
# 1. Helper Functions (Grid, Places, Analysis)
# -------------------------------------------------------------------------

def generate_square_grid(center_lat: float, center_lon: float, radius_miles: float, grid_size: int = 5):
    """
    Generate a grid_size x grid_size grid of points within a square
    bounding box of +/- radius_miles around (center_lat, center_lon).
    """
    if grid_size < 1:
        return []

    # ~69 miles per degree latitude, vary for longitude
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
    Collect up to 60 results (3 pages) from the Google Places Nearby Search,
    using rankby=distance (i.e. closest first).
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
            # Wait a couple seconds to let the token become active
            time.sleep(2)
            page_url = base_url + f"&pagetoken={next_token}"
        else:
            break

    return all_results

def search_places_top3_by_rating(lat: float, lon: float, keyword: str, target_business: str, api_key: str):
    """
    1) Fetch places around (lat, lon).
    2) Sort them by rating (desc), then by reviews (desc).
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
        if target_business.lower() in biz["name"].lower():
            rank = idx + 1  # 1-based rank
            client_details = biz
            break

    return rank, top_3, client_details

def get_place_details(place_id: str, api_key: str):
    """
    Use Places Details API to get address, phone, website, etc.
    """
    details_url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}&key={api_key}"
    )
    try:
        resp = requests.get(details_url)
        resp.raise_for_status()
        result = resp.json().get("result", {})
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching place details: {e}")
        return {}

    return {
        "address": result.get("formatted_address", ""),
        "phone": result.get("formatted_phone_number", ""),
        "website": result.get("website", ""),
        "name": result.get("name", ""),
        "rating": result.get("rating", "N/A"),
        "reviews": result.get("user_ratings_total", "N/A"),
    }

def scrape_website(url: str, max_chars: int = 2000):
    """
    Attempt to scrape basic textual content from a website for GPT analysis.
    """
    if not url:
        return ""

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        texts = soup.find_all(["p", "h1", "h2", "h3", "h4", "li"], limit=None)
        combined = " ".join(t.get_text(separator=" ", strip=True) for t in texts)
        return combined[:max_chars]
    except Exception:
        return ""

def create_heatmap(df: pd.DataFrame, center_lat: float, center_lon: float):
    """
    Creates a Scattermapbox-based heatmap of the rank results.
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

        # Build hover text
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
        title="Target Business Ranking Heatmap"
    )
    return fig

def generate_growth_report(df: pd.DataFrame, client_gbp: str):
    """
    Summarize coverage: how many points rank top 3, top 10, or not found.
    """
    total_points = len(df)
    df_found = df.dropna(subset=['rank'])

    top_3_points = df_found[df_found['rank'] <= 3].shape[0]
    top_10_points = df_found[df_found['rank'] <= 10].shape[0]

    pct_top_3 = (100.0 * top_3_points / total_points) if total_points else 0
    pct_top_10 = (100.0 * top_10_points / total_points) if total_points else 0

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

def analyze_competitors_with_gpt(client_gbp: str, competitor_details: list, client_info: dict):
    """
    AI-based competitor analysis & recommendations using GPT.
    """
    competitor_summaries = []
    for comp in competitor_details:
        summary_str = (
            f"- Name: {comp.get('name', 'N/A')} "
            f"(Rating: {comp.get('rating', 'N/A')}, {comp.get('reviews', '0')} reviews)\n"
            f"  Address: {comp.get('address', 'N/A')}\n"
            f"  Phone: {comp.get('phone', 'N/A')}\n"
        )
        if comp.get('website_content'):
            snippet = comp.get('website_content')[:200]
            summary_str += f"  Website Snippet: {snippet}...\n"
        competitor_summaries.append(summary_str)

    competitor_text = "\n".join(competitor_summaries)

    target_str = (
        f"Target Business: {client_gbp}\n"
        f"Rating: {client_info.get('rating', 'N/A')} | Reviews: {client_info.get('reviews', '0')}\n"
    )

    prompt = f"""
You are an advanced local SEO consultant. Compare "{client_gbp}" to the competitors below:
Target Business Data:
{target_str}

Competitors Data:
{competitor_text}

Provide a deep, data-driven, actionable analysis:
1. Summarize how the target's rating/review count compares to each competitor.
2. Evaluate each competitor's website snippet (if any) and how the target might improve or differentiate its own content.
3. Provide specific local SEO recommendations (citations, GMB/GBP enhancements, content strategies) 
   with metrics or evidence-based reasoning.
4. Conclude with the top priorities for the target to outrank these competitors.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a highly skilled local SEO consultant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=900
        )
        gpt_answer = response.choices[0].message.content
        return gpt_answer.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "Could not analyze competitors with ChatGPT."

# -------------------------------------------------------------------------
# 2. Streamlit Main App
# -------------------------------------------------------------------------
def main():
    st.title("📍 Google Business Profile Rating-Based Heatmap & Analysis (Free Approach)")

    # 1. Prompt for API keys if not in session
    if "openai_api_key" not in st.session_state or "places_api_key" not in st.session_state:
        st.subheader("Enter Your API Keys (Optional for ChatGPT)")
        openai_key_input = st.text_input("OpenAI API Key (sk-...)", type="password", help="Needed for GPT-based competitor analysis.")
        google_key_input = st.text_input("Google Maps/Places API Key", type="password")
        if st.button("Save API Keys"):
            if google_key_input:
                st.session_state["places_api_key"] = google_key_input
            if openai_key_input:
                st.session_state["openai_api_key"] = openai_key_input
            if not google_key_input:
                st.warning("A Google Maps API key is required to fetch local data.")
        st.stop()

    # 2. Set up keys
    places_api_key = st.session_state["places_api_key"]
    gmaps = googlemaps.Client(key=places_api_key)

    # OpenAI key is optional (only needed for competitor analysis)
    if "openai_api_key" in st.session_state:
        openai.api_key = st.session_state["openai_api_key"]

    # 3. Initialize placeholders in session_state if not exist
    if "competitor_place_ids" not in st.session_state:
        st.session_state["competitor_place_ids"] = set()
    if "client_info" not in st.session_state:
        st.session_state["client_info"] = {}

    # 4. Let user define a "snapshot_name" to track runs over time
    snapshot_name = st.text_input("Snapshot Name", value=f"Snapshot_{int(time.time())}",
                                  help="Label this scan so you can compare it later.")

    # 5. Main user inputs
    client_gbp = st.text_input("Target Business Name", "Starbucks")
    keyword = st.text_input("Target Keyword", "Coffee Shop")
    business_address = st.text_input("Business Address", "Los Angeles, CA")

    grid_shape = st.selectbox("Grid Shape", ["Square", "Circle"])
    radius = st.slider("Radius (miles)", 1, 20, 5)
    if grid_shape == "Square":
        grid_size = st.slider("Square Grid Size", 3, 11, 5)
    else:
        grid_size = st.slider("Number of Circle Points", 8, 60, 25)

    # 6. Button to Generate Heatmap
    if st.button("🔍 Generate Heatmap"):
        # A) Geocode the address
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
            st.error("❌ Could not find the address. Please try again.")
            return
        st.success(f"Found Address: {business_address} (Lat: {center_lat:.5f}, Lon: {center_lon:.5f})")

        # B) Generate Grid Points
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Using {len(grid_points)} grid points (Square: {grid_size}x{grid_size}).")
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Using {len(grid_points)} grid points (Circle).")

        # C) For each grid point, get rank & top3
        grid_data = []
        competitor_place_ids = set()
        client_info_global = {}

        with st.spinner("Collecting data from Google Places API..."):
            for lat, lon in grid_points:
                rank, top_3, client_details = search_places_top3_by_rating(
                    lat, lon, keyword, client_gbp, places_api_key
                )
                # If the target business is found, store details
                if client_details is not None:
                    client_info_global = client_details

                # Gather competitor place_ids from top_3
                for c in top_3:
                    competitor_place_ids.add(c["place_id"])

                grid_data.append({
                    'latitude': lat,
                    'longitude': lon,
                    'rank': rank,
                    'top_3': top_3,
                })

        # D) Update session_state
        st.session_state["competitor_place_ids"] = competitor_place_ids
        st.session_state["client_info"] = client_info_global

        # E) Create DataFrame
        df = pd.DataFrame(grid_data)
        df['snapshot_name'] = snapshot_name
        df['timestamp'] = pd.Timestamp.now()

        # F) Plot Heatmap
        st.plotly_chart(create_heatmap(df, center_lat, center_lon), use_container_width=True)

        # G) Show Growth Report
        st.write("### Growth Report")
        st.markdown(generate_growth_report(df, client_gbp))

        # H) Save to local "ranking_history.csv"
        history_file = "ranking_history.csv"
        if not os.path.isfile(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)
        st.success(f"Snapshot '{snapshot_name}' saved/updated in '{history_file}'.")

        # I) Download button for this snapshot
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download This Snapshot",
            data=csv_data,
            file_name=f"{snapshot_name}_ranking_data.csv",
            mime="text/csv"
        )

        # J) Detailed Grid Summary
        st.write("### Detailed Grid Summary")
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
- **{client_gbp}** Rank By Rating: {rank_str}  
- Top 3 Best-Rated Competitors: {top3_text if top3_text else "None"}
""")

    # 7. Optional: ChatGPT Competitor Analysis
    competitor_place_ids = st.session_state.get("competitor_place_ids", set())
    client_info_global = st.session_state.get("client_info", {})
    if competitor_place_ids:
        if "openai_api_key" in st.session_state and st.session_state["openai_api_key"]:
            if st.button("Analyze Competitors with ChatGPT"):
                with st.spinner("Fetching competitor details & scraping websites..."):
                    competitor_details_list = []
                    for pid in competitor_place_ids:
                        details = get_place_details(pid, places_api_key)
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

                    gpt_analysis = analyze_competitors_with_gpt(client_gbp, competitor_details_list, client_info_global)

                st.write("### Competitor Comparison & Recommendations")
                st.write(gpt_analysis)
        else:
            st.info("Enter an OpenAI API key if you want GPT competitor analysis.")
    else:
        st.info("No competitor data available. Generate Heatmap first.")

    # 8. Compare to Past Snapshots
    st.write("---")
    st.subheader("Compare to Past Snapshots")
    uploaded_file = st.file_uploader("Upload a Past CSV Snapshot (e.g., from ranking_history.csv)", type=["csv"])
    if uploaded_file:
        old_data = pd.read_csv(uploaded_file)
        st.write("**Uploaded Snapshot(s)**:", old_data['snapshot_name'].unique())

        # You could implement merging or difference analysis here:
        # For instance, let the user pick which snapshots to compare side by side.
        # Example:
        # unique_snaps = old_data['snapshot_name'].unique().tolist()
        # pick1 = st.selectbox("Choose Snapshot #1", unique_snaps)
        # pick2 = st.selectbox("Choose Snapshot #2", unique_snaps)
        # Then compare data within those snapshots.
        # This is left as an exercise depending on how you want to visualize differences.

if __name__ == "__main__":
    main()
