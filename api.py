# api.py
# Handles Realtor.com API interactions for searching and selecting properties.
# Uses RapidAPI endpoint for property listings and maps data to analyzer format.

import requests
import logging
from dotenv import load_dotenv
import os

# Configure logging for debugging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
REALTOR_API_KEY = os.getenv("REALTOR_API_KEY")
if not REALTOR_API_KEY:
    raise ValueError("REALTOR_API_KEY environment variable not set")

def search_properties(criteria):
    """
    Search for properties using Realtor.com API via RapidAPI.
    Args:
        criteria (dict): Search filters (e.g., type: "office", location: "Orlando", min_price: 1000000).
    Returns:
        list: List of up to 5 properties with mapped fields for analyzer.
    """
    try:
        # API endpoint and headers
        url = "https://realtor.p.rapidapi.com/properties/v2/list-for-sale"
        headers = {
            "X-RapidAPI-Key": REALTOR_API_KEY,
            "X-RapidAPI-Host": "realtor.p.rapidapi.com"
        }
        # Build query parameters from criteria
        params = {
            "city": criteria.get("location", "Orlando").split(",")[0],
            "property_type": criteria.get("type", "commercial"),
            "limit": 10,
            "offset": 0,
            "sort": "relevance"
        }
        if "min_price" in criteria:
            params["min_price"] = criteria["min_price"]
        if "max_price" in criteria:
            params["max_price"] = criteria["max_price"]
        if "min_sq_ft" in criteria:
            params["min_sq_ft"] = criteria["min_sq_ft"]
        if "max_sq_ft" in criteria:
            params["max_sq_ft"] = criteria["max_sq_ft"]
        
        # Make API request
        logger.debug(f"Sending API request with params: {params}")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Map API response to analyzer-compatible format
        properties = []
        for prop in data.get("data", {}).get("results", [])[:5]:  # Limit to 5
            properties.append({
                "id": prop.get("property_id", "N/A"),
                "type": prop.get("property_type", "Commercial"),
                "location": prop.get("address", {}).get("full", "N/A"),
                "purchase_price": prop.get("list_price", 0),
                "gross_rent": prop.get("list_price", 0) * 0.08,  # Estimate 8% yield
                "vacancy_rate": 0.05,  # Default
                "expenses": prop.get("list_price", 0) * 0.03,  # Estimate 3%
                "expense_growth": 0.02,
                "loan_amount": prop.get("list_price", 0) * 0.8,  # 80% LTV
                "interest_rate": 0.05,
                "loan_term": 25,
                "noi_growth": 0.03,
                "appreciation_rate": 0.03,
                "exit_cap_rate": 0.06,
                "hold_period": 5,
                "sq_ft": prop.get("building_size", {}).get("size", 0)
            })
        logger.debug(f"Found {len(properties)} properties")
        return properties
    except Exception as e:
        logger.error(f"Property search error: {str(e)}")
        return []

def select_properties(selection, search_results):
    """
    Select properties from search results based on user input.
    Args:
        selection (str): Comma-separated indices (e.g., "1,3").
        search_results (list): List of properties from search_properties.
    Returns:
        list: Selected properties.
    """
    try:
        indices = [int(i.strip()) - 1 for i in selection.split(',') if i.strip().isdigit()]
        selected = [search_results[i] for i in indices if 0 <= i < len(search_results)]
        logger.debug(f"Selected {len(selected)} properties: {[p['id'] for p in selected]}")
        return selected
    except Exception as e:
        logger.error(f"Property selection error: {str(e)}")
        return []