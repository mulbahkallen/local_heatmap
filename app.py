import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import openai
from bs4 import BeautifulSoup
import time

# -------------------------------------------------------------------------
# 1. Define Helper Functions (including the NEW circular grid generator)
# -------------------------------------------------------------------------

def generate_square_grid(center_lat: float, center_lon: float, radius_miles: float, grid_size: int = 5):
    """
    Generate a grid_size x grid_size grid of points within a square
    bounding box of +/- radius_miles around (center_lat, center_lon).

    If grid_size=4, you get 16 total points.
    If grid_size=5, you get 25 total points, etc.
    """
    if grid_size < 1:
        return []

    # Approx degrees per mile for latitude and longitude
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
    Generate approximately 'num_points' lat/lon coordinates in a circular
    pattern around (center_lat, center_lon) within 'radius_miles'.

    This is useful for a scenario where the business may be on the edge
    or corner of a city, so we only sample within a circular boundary
    rather than a full square bounding box.
    """
    if num_points < 1:
        return []

    # Convert miles to degrees for lat (roughly ~69 miles per deg)
    # For lon, we factor in cos(latitude).
    lat_degs = radius_miles / 69.0
    lon_degs = radius_miles / (69.0 * np.cos(np.radians(center_lat)))

    # We will spread the points around the circle:
    # We'll place them in equal angular steps from 0..360 deg.
    grid_points = []
    for i in range(num_points):
        angle = 2.0 * np.pi * (i / num_points)
        # random or evenly spaced radius could be used; here we keep everything at the boundary
        # or do a mix of rings. For simplicity, let's just do one ring:
        lat_offset = lat_degs * np.sin(angle)
        lon_offset = lon_degs * np.cos(angle)
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        grid_points.append((lat, lon))

    return grid_points


def fetch_nearby_places(lat: float, lon: float, keyword: str, api_key: str):
    """
    Collect up to 60 results (3 pages) from the Google Places Nearby Search,
    sorted by distance (rankby=distance).
    Return a list of place dicts with rating, reviews, etc.
    """
    location = f"{lat},{lon}"
    base_url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={location}&keyword={keyword}"
        f"&rankby=distance&key={api_key}"
    )

    all_results = []
    page_url = base_url
    for _ in range(3):  # Up to 3 pages
        try:
            resp = requests.get(page_url)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Request error while searching Places API: {e}")
            break

        results = data.get("results", [])
        all_results.extend(results)

        # If there's a next_page_token, we wait 2 seconds and fetch again
        if "next_page_token" in data:
            next_token = data["next_page_token"]
            time.sleep(2)  # recommended short wait
            page_url = base_url + f"&pagetoken={next_token}"
        else:
            break  # No more pages

    return all_results


def search_places_top3_by_rating(lat: float, lon: float, keyword: str, target_business: str, api_key: str):
    """
    1) Fetch up to 60 results around (lat, lon).
    2) Sort them by rating desc, then by reviews desc, ignoring 'None' or missing rating.
    3) Return the top 3 best-rated places as 'top_3'.
    4) Find the rank of the target business within that rating-sorted list (1-based).
    5) Return client_details if found.
    """
    all_results = fetch_nearby_places(lat, lon, keyword, api_key)

    # Convert to a structured list
    structured = []
    for place in all_results:
        name = place.get("name", "Unknown")
        place_id = place.get("place_id", "")
        rating = place.get("rating", 0)
        reviews = place.get("user_ratings_total", 0)
        types_ = place.get("types", [])  # categories
        open_now = place.get("opening_hours", {}).get("open_now", None)
        business_status = place.get("business_status", None)

        # If rating is None, treat as 0
        if rating is None:
            rating = 0
        if reviews is None:
            reviews = 0

        structured.append({
            "place_id": place_id,
            "name": name,
            "rating": float(rating) if isinstance(rating, (int, float)) else 0.0,
            "reviews": int(reviews) if isinstance(reviews, int) else 0,
            "types": types_,
            "open_now": open_now,
            "business_status": business_status
        })

    # Sort by rating desc, then reviews desc, then name asc
    structured.sort(key=lambda x: (-x["rating"], -x["reviews"], x["name"]))

    # Identify top 3 in that sorted list
    top_3 = structured[:3]

    # Find target business rank if it appears in structured
    rank = None
    client_details = None
    for idx, biz in enumerate(structured):
        if target_business.lower() in biz["name"].lower():
            rank = idx + 1  # 1-based
            client_details = biz
            break

    return rank, top_3, client_details


def get_place_details(place_id: str, api_key: str):
    """
    Fetch additional details about a place using the Google Places Details API.
    Returns a dict with fields like {address, phone, website, etc.}
    """
    details_url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}&key={api_key}"
    )
    try:
        response = requests.get(details_url)
        response.raise_for_status()
        result = response.json().get("result", {})
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
    Attempt to scrape a competitor's website to get textual data for AI analysis.
    Return raw text (limited to max_chars).
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
        # Gracefully handle scraping errors
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
        if row['top_3']:
            hover_items = []
            for i, biz in enumerate(row['top_3']):
                # Enrich hover with 'types', 'open_now', 'business_status'
                cats = ", ".join(biz.get("types", []))
                status = biz.get("business_status", "Unknown")
                open_status = biz.get("open_now", None)
                open_str = "Open now" if open_status else ("Closed now" if open_status is False else "Unknown")
                hover_items.append(
                    f"{i+1}. {biz['name']} "
                    f"({biz['rating']}⭐, {biz['reviews']} reviews)<br>"
                    f"Types: {cats}<br>"
                    f"Status: {status} - {open_str}"
                )
            hover_text = "<br><br>".join(hover_items)
        else:
            hover_text = "No competitor data."

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
    Generate a textual report:
    - # of grid points in top 3, top 10, or not found (X).
    - % in top 3, top 10.
    - Average rank (where found).
    """
    total_points = len(df)
    df_found = df.dropna(subset=['rank'])  # points where business is found

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


def analyze_competitors_with_gpt(client_gbp: str, competitor_details: list, client_info: dict):
    """
    Provides deep, data-driven insights about each competitor vs. the target client.
    competitor_details: list of { name, address, phone, rating, reviews, website, website_content }
    client_info: { rating, reviews, place_id, name, etc. } for the target business
    """
    # Build competitor summary lines
    competitor_summaries = []
    for comp in competitor_details:
        summary_str = (
            f"- Name: {comp.get('name', 'N/A')} (Rating: {comp.get('rating', 'N/A')}, "
            f"{comp.get('reviews', '0')} reviews)\n"
            f"  Address: {comp.get('address', 'N/A')}\n"
            f"  Phone: {comp.get('phone', 'N/A')}\n"
        )
        if comp.get('website_content'):
            summary_str += (
                f"  Website Snippet: {comp.get('website_content')[:200]}...\n"
            )
        competitor_summaries.append(summary_str)

    competitor_text = "\n".join(competitor_summaries)

    # Include target business info in the prompt for deeper data-driven analysis
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
   with metrics or evidence-based reasoning (e.g., how many more reviews or the average rating in the top 3).
4. Conclude with the top priorities for the target to outrank these competitors.
    """

    try:
        # Just set openai.api_key once, in main() - but here we rely on it being set
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
    st.title("📍 Google Business Profile Ranking Heatmap & Analysis")

    # First step: Ask for API keys
    # We'll store them in session_state so the user only has to input them once.
    if "openai_api_key" not in st.session_state or "places_api_key" not in st.session_state:
        st.subheader("Enter Your API Keys")
        openai_key_input = st.text_input("OpenAI API Key (sk-...)", type="password")
        google_key_input = st.text_input("Google Maps/Places API Key", type="password")
        if st.button("Save API Keys"):
            if openai_key_input and google_key_input:
                st.session_state["openai_api_key"] = openai_key_input
                st.session_state["places_api_key"] = google_key_input
                st.experimental_rerun()
            else:
                st.warning("Please provide both API keys before proceeding.")
        st.stop()

    # Now we have the keys in session_state, set them for usage
    openai.api_key = st.session_state["openai_api_key"]
    gmaps = googlemaps.Client(key=st.session_state["places_api_key"])

    # Ensure competitor data persists across script runs
    if "competitor_place_ids" not in st.session_state:
        st.session_state["competitor_place_ids"] = set()

    if "client_info" not in st.session_state:
        st.session_state["client_info"] = {}

    st.write("Analyze how your business ranks (by rating) in your target area.\n"
             "Then compare competitor profiles using AI for deeper, data-driven insights.")

    # --- User Inputs ---
    client_gbp = st.text_input("Enter Your Business Name (Google Business Profile)", "Starbucks")
    keyword = st.text_input("Enter Target Keyword (e.g., 'Coffee Shop')", "Coffee Shop")
    business_address = st.text_input("Enter Your Business Address (Full Address)", "Los Angeles, CA")

    # NEW: let the user select grid shape
    grid_shape = st.selectbox("Select Grid Shape", ["Square", "Circle"])
    radius = st.slider("Select Search Radius (miles)", 1, 20, 5)

    if grid_shape == "Square":
        grid_size = st.slider("Select Square Grid Size (grid points = size^2)", 3, 11, 5)
    else:
        # For circle, we just pick how many points on the ring
        grid_size = st.slider("Number of Points in the Circle", 8, 60, 25)

    # --- Generate Heatmap when button clicked ---
    if st.button("🔍 Generate Heatmap"):
        # Geocode to get lat/lon
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

        st.success(f"📍 Address Found: {business_address} (Lat: {center_lat}, Lon: {center_lon})")

        # Generate the grid
        if grid_shape == "Square":
            grid_points = generate_square_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Using **{len(grid_points)}** grid points (square grid: {grid_size}×{grid_size}).")
        else:
            grid_points = generate_circular_grid(center_lat, center_lon, radius, grid_size)
            st.write(f"Using **{len(grid_points)}** grid points in a circular pattern.")

        # Collect ranking data
        grid_data = []
        competitor_place_ids = set()
        client_info_global = {}

        for lat, lon in grid_points:
            rank, top_3, client_details = search_places_top3_by_rating(
                lat, lon, keyword, client_gbp, st.session_state["places_api_key"]
            )

            # If the target business is found, store the place_id, rating, reviews, etc.
            if client_details is not None:
                client_info_global = client_details

            # Collect the top-3 place_ids in a set for later details retrieval
            for c in top_3:
                competitor_place_ids.add(c["place_id"])

            grid_data.append({
                'latitude': lat,
                'longitude': lon,
                'rank': rank,
                'top_3': top_3,
            })

        # Save the competitor data in session_state
        st.session_state["competitor_place_ids"] = competitor_place_ids

        # Also store the last known client info
        st.session_state["client_info"] = client_info_global

        # Create DataFrame
        df = pd.DataFrame(grid_data)

        # --- Display Heatmap ---
        st.plotly_chart(create_heatmap(df, center_lat, center_lon), use_container_width=True)

        # --- Show a Cleaner Table of Results ---
        st.write("### 📊 Rating-Sorted Ranking Data (Summary)")
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
- Location: ({row['latitude']:.5f}, {row['longitude']:.5f})  
- **{client_gbp}** Rank By Rating: {rank_str}  
- Top 3 Best-Rated Competitors: {top3_text if top3_text else "None"}
            """)

        # --- Growth / SEO Report ---
        st.write("### 📈 Growth Report")
        st.markdown(generate_growth_report(df, client_gbp))

        # Option to Download CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Ranking Data",
            data=csv_data,
            file_name="rating_sorted_ranking_data.csv",
            mime="text/csv"
        )

    # Retrieve competitor IDs from session_state for next button
    competitor_place_ids = st.session_state.get("competitor_place_ids", set())
    client_info_global = st.session_state.get("client_info", {})

    # --- Gather Detailed Competitor Info + GPT Analysis ---
    if competitor_place_ids:
        if st.button("Analyze Competitors with ChatGPT"):
            with st.spinner("Fetching competitor details & scraping websites..."):
                competitor_details_list = []
                for pid in competitor_place_ids:
                    details = get_place_details(pid, st.session_state["places_api_key"])
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

                # Now feed these details & the client info to GPT
                gpt_analysis = analyze_competitors_with_gpt(
                    client_gbp, competitor_details_list, client_info_global
                )

            st.write("### 🏆 Competitor Comparison & Recommendations")
            st.write(gpt_analysis)
    else:
        st.info("No competitor data found to analyze with ChatGPT.")


if __name__ == "__main__":
    main()
