# main.py
# Core FastAPI app setup, routing, and state management for RealtyBot.
# Handles user interactions (chat, upload, report) and delegates to other modules.

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import httpx
import os
from dotenv import load_dotenv
import logging
import re
from api import search_properties, select_properties  # Import API functions
from analyzer import calculate_metrics, is_good_deal  # Import analysis functions
from charts import generate_charts  # Import chart functions

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app and mount static/templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load environment variables for APIs
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
REALTOR_API_KEY = os.getenv("REALTOR_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY environment variable not set")
if not REALTOR_API_KEY:
    raise ValueError("REALTOR_API_KEY environment variable not set for property search")

XAI_API_URL = "https://api.x.ai/v1/chat/completions"

# Default assumptions for deal analysis (stored as decimals)
DEFAULT_ASSUMPTIONS = {
    "min_cap_rate": 0.06,
    "max_cap_rate": 0.08,
    "min_cash_on_cash": 0.08,
    "min_dscr": 1.2,
    "min_occupancy": 0.90,
    "min_noi_growth": 0.02,
    "max_noi_growth": 0.04,
    "min_appreciation": 0.02,
    "max_appreciation": 0.05,
    "min_irr": 0.10,
    "default_tax_rate": 0.01,
    "default_depreciation_years": 27.5
}

def reset_state():
    """Reset application state for a new analysis."""
    return {
        "stage": "welcome",  # Tracks user progress: welcome, missing_data, assumptions, search, select, report
        "data": None,  # Stores uploaded CSV or selected property data
        "assumptions": DEFAULT_ASSUMPTIONS.copy(),  # User-defined criteria
        "missing_data": {},  # Missing CSV columns
        "analysis": None,  # Stores analysis results
        "conversation": [  # Chat history
            {"role": "system", "content": (
                "You are RealtyBot, a friendly real estate analyst chatbot. Guide users through analyzing commercial real estate deals in a conversational way. "
                "Use a warm tone, acknowledge changes explicitly (e.g., 'Updated min cap rate to 7%!'), and clarify unclear inputs. "
                "Use phrases like 'Awesome,' 'Got it,' or 'Let’s dive in!' Display metrics as percentages (e.g., 6%). "
                "Include a call to action (e.g., 'What’s next?')."
            )}
        ],
        "search_results": [],  # Stores Realtor API search results
        "selected_properties": []  # Stores user-selected properties
    }

# Initialize state
state = reset_state()

async def call_xai_api(message: str, conversation: list):
    """Call xAI API for conversational responses."""
    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    payload = {
        "model": "grok-3",
        "messages": conversation + [{"role": "user", "content": message}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(XAI_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"xAI API error: {str(e)}")
            return f"Oops, something went wrong with the API: {str(e)}. Let’s try again, shall we?"

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Render the main chat interface."""
    if state["stage"] == "welcome":
        welcome_message = (
            "Welcome to RealtyBot! I’m here to help you analyze commercial real estate deals. Follow these steps:\n"
            "1. Search for properties (e.g., 'search properties office Orlando min price 1M') or upload a CSV (like template.csv).\n"
            "2. Review/adjust assumptions using sliders.\n"
            "3. Chat to refine criteria or select properties (e.g., 'select 1,3').\n"
            "4. Click 'Run Report' to view the analysis.\n"
            "Ready? Try a property search or upload a CSV in Step 1!"
        )
        state["conversation"].append({"role": "assistant", "content": welcome_message})
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation": state["conversation"],
        "message": state["conversation"][-1]["content"],
        "assumptions": state["assumptions"],
        "search_results": state["search_results"]  # Pass search results for display
    })

@app.post("/upload", response_class=HTMLResponse)
async def upload_spreadsheet(request: Request, file: UploadFile = File(...)):
    """Handle CSV upload with robust error handling."""
    try:
        contents = await file.read()
        if len(contents) == 0:
            error_message = "The uploaded file is empty. Please ensure your CSV contains data and try again in Step 1."
            state["conversation"].append({"role": "assistant", "content": error_message})
            logger.error(f"Empty file uploaded: {file.filename}")
            return templates.TemplateResponse("chat.html", {
                "request": request,
                "conversation": state["conversation"],
                "message": error_message,
                "assumptions": state["assumptions"],
                "search_results": state["search_results"]
            })

        # Log file content for debugging
        logger.debug(f"File content (first 200 chars): {contents[:200].decode('utf-8', errors='ignore')}")

        # Try multiple encodings and delimiters
        encodings = ['utf-8', 'utf-8-sig', 'latin1']
        delimiters = [',', '\t', ';']
        df = None
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(BytesIO(contents), encoding=encoding, sep=delimiter)
                    if not df.empty and df.shape[1] > 0:
                        logger.debug(f"Successfully read CSV with encoding={encoding}, delimiter={delimiter}")
                        break
                except Exception as e:
                    logger.debug(f"Failed with encoding={encoding}, delimiter={delimiter}: {str(e)}")
                    continue
            if df is not None and not df.empty and df.shape[1] > 0:
                break

        if df is None or df.empty or df.shape[1] == 0:
            error_message = (
                "The file couldn’t be parsed as a valid CSV. Ensure it has a header row (e.g., 'Property_Type,Location,...') "
                "and data rows, using commas, tabs, or semicolons. Save as UTF-8 CSV and try again in Step 1."
            )
            state["conversation"].append({"role": "assistant", "content": error_message})
            logger.error(f"Invalid CSV: {file.filename}, shape={df.shape if df is not None else 'None'}")
            return templates.TemplateResponse("chat.html", {
                "request": request,
                "conversation": state["conversation"],
                "message": error_message,
                "assumptions": state["assumptions"],
                "search_results": state["search_results"]
            })

        if df.shape[0] == 0:
            error_message = "The CSV has a header but no data rows. Please add at least one row of data and try again in Step 1."
            state["conversation"].append({"role": "assistant", "content": error_message})
            logger.error(f"No data rows in CSV: {file.filename}")
            return templates.TemplateResponse("chat.html", {
                "request": request,
                "conversation": state["conversation"],
                "message": error_message,
                "assumptions": state["assumptions"],
                "search_results": state["search_results"]
            })

        required_columns = ["Property_Type", "Location", "Purchase_Price", "Gross_Rent", "Vacancy_Rate", "Expenses",
                           "Expense_Growth", "Loan_Amount", "Interest_Rate", "Loan_Term", "NOI_Growth",
                           "Appreciation_Rate", "Exit_Cap_Rate", "Hold_Period"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        state["data"] = df.iloc[0].to_dict()
        state["missing_data"] = {col: None for col in missing_columns}
        
        if missing_columns:
            state["stage"] = "missing_data"
            message = (
                f"Great, I’ve got the file! Missing columns: {', '.join(missing_columns)}. "
                "Provide these in Step 3 (e.g., 'tax_rate: 1%') or type 'default'. What’s next?"
            )
        else:
            state["stage"] = "assumptions"
            assumptions_str = "\n".join([
                f"{k.replace('_', ' ')}: {v * 100:.2f}%" if k in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"]
                else f"{k.replace('_', ' ')}: {v}"
                for k, v in state["assumptions"].items()
            ])
            message = (
                f"Awesome, the data looks good! Move to Step 2 to tweak assumptions, or chat in Step 3 (e.g., 'min_cap_rate: 7%'). "
                f"Current assumptions:\n{assumptions_str}\nClick 'Run Report' in Step 4 when ready."
            )
        
        state["conversation"].append({"role": "assistant", "content": message})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": message,
            "assumptions": state["assumptions"],
            "search_results": state["search_results"]
        })
    except Exception as e:
        error_message = f"Oops, upload error: {str(e)}. Check your CSV format and try again in Step 1."
        state["conversation"].append({"role": "assistant", "content": error_message})
        logger.error(f"Upload error: {str(e)} for file: {file.filename}")
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": error_message,
            "assumptions": state["assumptions"],
            "search_results": state["search_results"]
        })

@app.post("/chat", response_class=HTMLResponse)
async def handle_chat(request: Request, message: str = Form(...)):
    """Handle chat input, including property search and selection."""
    state["conversation"].append({"role": "user", "content": message})
    message_lower = message.lower().strip()

    # Parse property search (e.g., "search properties office Orlando min price 1M")
    search_pattern = re.compile(r"search\s+properties\s+([^\d]+)\s*([\w\s,]+)?(?:min\s+price\s+([\d.]+)([mMkK])?)?(?:max\s+price\s+([\d.]+)([mMkK])?)?(?:min\s+sq\s*ft\s+([\d]+))?")
    search_match = search_pattern.search(message_lower)
    if search_match:
        type_ = search_match.group(1).strip()
        location = search_match.group(2).strip() if search_match.group(2) else "Orlando"
        min_price = float(search_match.group(3)) * (1000000 if search_match.group(4) in ['m', 'M'] else 1000) if search_match.group(3) else None
        max_price = float(search_match.group(5)) * (1000000 if search_match.group(6) in ['m', 'M'] else 1000) if search_match.group(5) else None
        min_sq_ft = int(search_match.group(7)) if search_match.group(7) else None
        criteria = {"type": type_, "location": location}
        if min_price:
            criteria["min_price"] = min_price
        if max_price:
            criteria["max_price"] = max_price
        if min_sq_ft:
            criteria["min_sq_ft"] = min_sq_ft
        
        state["stage"] = "search"
        properties = search_properties(criteria)
        if not properties:
            response = "No properties found with those criteria. Try different filters (e.g., 'search properties retail Tampa min price 500K') in Step 3."
        else:
            prop_list = "\n".join([f"{i+1}. {p['type']} at {p['location']}, ${p['purchase_price']:,.0f}, {p['sq_ft']} sq ft" for i, p in enumerate(properties)])
            response = f"Found {len(properties)} properties:\n{prop_list}\nPlease select properties (e.g., 'select 1,3') in Step 3."
        state["conversation"].append({"role": "assistant", "content": response})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": response,
            "assumptions": state["assumptions"],
            "search_results": state["search_results"]
        })

    # Parse property selection (e.g., "select 1,3")
    if message_lower.startswith("select "):
        selection = message_lower.replace("select ", "").strip()
        selected = select_properties(selection)
        if not selected:
            response = "Invalid selection. Please use numbers (e.g., 'select 1,3') based on the property list in Step 3."
        else:
            state["stage"] = "assumptions"
            state["data"] = selected[0] if len(selected) == 1 else None  # Single property for now
            state["selected_properties"] = selected
            prop_list = "\n".join([f"{p['type']} at {p['location']}, ${p['purchase_price']:,.0f}" for p in selected])
            response = (
                f"Selected {len(selected)} properties:\n{prop_list}\n"
                "Move to Step 2 to adjust assumptions, or chat in Step 3 (e.g., 'min_cap_rate: 7%'). Click 'Run Report' in Step 4."
            )
        state["conversation"].append({"role": "assistant", "content": response})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": response,
            "assumptions": state["assumptions"],
            "search_results": state["search_results"]
        })

    # Handle assumption changes (key:value or natural language)
    updated_fields = []
    for line in message.split("\n"):
        line = line.strip()
        if ":" in line:
            try:
                key, value = [part.strip() for part in line.split(":", 1)]
                key_normalized = key.lower().replace(" ", "_")
                if key_normalized in state["assumptions"]:
                    value = float(value[:-1]) / 100 if value.endswith("%") else float(value)
                    state["assumptions"][key_normalized] = value
                    updated_fields.append(f"{key_normalized.replace('_', ' ')} to {value * 100:.2f}%" if key_normalized in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"] else f"{key_normalized.replace('_', ' ')} to {value}")
                    logger.debug(f"Updated {key_normalized} to {value} (key:value)")
            except ValueError as e:
                logger.error(f"Invalid value in key:value format: {line}, error: {str(e)}")
                continue

    natural_pattern = re.compile(r"(set|change|update)\s+(.+?)\s+(to|at)\s+([\d.]+)\s*(%|percent)?", re.IGNORECASE)
    matches = natural_pattern.findall(message)
    for match in matches:
        _, key, _, value, is_percent = match
        key_normalized = key.lower().replace(" ", "_").strip()
        if key_normalized in state["assumptions"]:
            try:
                value = float(value) / 100 if is_percent else float(value)
                state["assumptions"][key_normalized] = value
                updated_fields.append(f"{key_normalized.replace('_', ' ')} to {value * 100:.2f}%" if key_normalized in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"] else f"{key_normalized.replace('_', ' ')} to {value}")
                logger.debug(f"Updated {key_normalized} to {value} (natural language)")
            except ValueError as e:
                logger.error(f"Invalid value in natural language: {match}, error: {str(e)}")
                continue

    assumptions_str = "\n".join([
        f"{k.replace('_', ' ')}: {v * 100:.2f}%" if k in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"]
        else f"{k.replace('_', ' ')}: {v}"
        for k, v in state["assumptions"].items()
    ])

    if updated_fields:
        response = f"Nice, I’ve updated {', '.join(updated_fields)}!\nCurrent assumptions:\n{assumptions_str}\nTweak more in Step 2 or here in Step 3. Ready? Click 'Run Report' in Step 4."
    elif "change" in message_lower or "update" in message_lower or "assumption" in message_lower or ":" in message or natural_pattern.search(message):
        response = (
            f"Got it! I see you’re adjusting assumptions, but I didn’t catch valid changes. Current setup:\n{assumptions_str}\n"
            "Use formats like 'min_cap_rate: 7%' or 'set min cap rate to 7 percent' in Step 3, or adjust sliders in Step 2. What’s next?"
        )
    elif state["stage"] == "welcome":
        response = (
            "Ready to get started? Search for properties (e.g., 'search properties office Orlando min price 1M') or upload a CSV in Step 1. What’s up?"
        )
    elif state["stage"] == "missing_data":
        if message_lower == "default":
            for key in state["missing_data"]:
                state["missing_data"][key] = state["assumptions"].get(f"default_{key}", 0)
            state["stage"] = "assumptions"
            response = (
                f"Got it, using default values for missing data. Head to Step 2 to tweak assumptions, or stay in Step 3 to chat. "
                f"Current assumptions:\n{assumptions_str}\nClick 'Run Report' in Step 4."
            )
        else:
            try:
                updated_fields = []
                for line in message.split("\n"):
                    line = line.strip()
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().replace(" ", "_").lower()
                        if key in state["missing_data"]:
                            value = float(value[:-1]) / 100 if value.endswith("%") else float(value)
                            state["missing_data"][key] = value
                            updated_fields.append(f"{key.replace('_', ' ')} to {value * 100:.2f}%" if key in ["tax_rate"] else f"{key.replace('_', ' ')} to {value}")
                if updated_fields:
                    response = f"Nice, updated {', '.join(updated_fields)}! "
                if all(v is not None for v in state["missing_data"].values()):
                    state["stage"] = "assumptions"
                    response += (
                        f"All missing data filled! Move to Step 2 to adjust assumptions, or stay in Step 3. "
                        f"Current assumptions:\n{assumptions_str}\nClick 'Run Report' in Step 4."
                    )
                else:
                    missing = [k for k, v in state["missing_data"].items() if v is None]
                    response += (
                        f"Still need values for: {', '.join(missing)}. "
                        "Provide them in Step 3 (e.g., 'tax_rate: 1%') or type 'default'. What’s up?"
                    )
            except:
                response = (
                    "Didn’t catch that. Provide missing data in Step 3 (e.g., 'tax_rate: 1%') or type 'default'. Try again?"
                )
    else:
        response = (
            f"Current assumptions:\n{assumptions_str}\n"
            "Tweak them in Step 2 or here in Step 3 (e.g., 'min_cap_rate: 7%'). Click 'Run Report' in Step 4 when ready."
        )
    
    state["conversation"].append({"role": "assistant", "content": response})
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation": state["conversation"],
        "message": response,
        "assumptions": state["assumptions"],
        "search_results": state["search_results"]
    })

@app.post("/update_assumptions", response_class=HTMLResponse)
async def update_assumptions(request: Request):
    """Update assumptions from slider inputs."""
    try:
        form_data = await request.form()
        logger.debug(f"Received form data: {form_data}")
        updated_fields = []
        for key, value in form_data.items():
            if key in state["assumptions"]:
                try:
                    value = float(value) / 100 if key in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"] else float(value)
                    state["assumptions"][key] = value
                    updated_fields.append(f"{key.replace('_', ' ')} to {value * 100:.2f}%" if key in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"] else f"{key.replace('_', ' ')} to {value}")
                except ValueError:
                    logger.error(f"Invalid value for {key}: {value}")
                    continue
        if updated_fields:
            response = (
                f"Nice, updated {', '.join(updated_fields)}! "
                f"Current assumptions:\n" +
                "\n".join([
                    f"{k.replace('_', ' ')}: {v * 100:.2f}%" if k in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"]
                    else f"{k.replace('_', ' ')}: {v}"
                    for k, v in state["assumptions"].items()
                ]) +
                "\nReady? Click 'Run Report' in Step 4 or adjust sliders in Step 2."
            )
        else:
            response = "No valid changes made to assumptions. Adjust sliders in Step 2 and try again."
        state["conversation"].append({"role": "assistant", "content": response})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": response,
            "assumptions": state["assumptions"],
            "search_results": state["search_results"]
        })
    except Exception as e:
        logger.error(f"Error in update_assumptions: {str(e)}")
        response = f"Oops, slider update error: {str(e)}. Try again in Step 2 or use Step 3 to set values."
        state["conversation"].append({"role": "assistant", "content": response})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": response,
            "assumptions": state["assumptions"],
            "search_results": state["search_results"]
        })

@app.post("/run_report", response_class=JSONResponse)
async def run_report(request: Request):
    """Generate reports for selected properties or uploaded CSV."""
    try:
        if not state["data"] and not state["selected_properties"]:
            response = "No deal data or properties selected! Upload a CSV in Step 1 or select properties in Step 3."
            state["conversation"].append({"role": "assistant", "content": response})
            return JSONResponse(content={"success": False, "message": response})

        state["stage"] = "report"
        analyses = []
        
        # Handle multiple selected properties or single CSV
        properties = state["selected_properties"] if state["selected_properties"] else [state["data"]]
        for prop in properties:
            metrics = calculate_metrics(prop, state["assumptions"])
            conclusion, reasons, improvements = is_good_deal(metrics, prop, state["assumptions"])
            charts = generate_charts(metrics, prop)
            analyses.append({
                "Property_Type": prop.get("type", prop.get("Property_Type", "Commercial")),
                "Location": prop.get("location", prop.get("Location", "N/A")),
                "Purchase_Price": prop.get("purchase_price", prop.get("Purchase_Price", 0)),
                "NOI": metrics["NOI"],
                "Cap_Rate": metrics["Cap_Rate"],
                "DSCR": metrics["DSCR"],
                "Cash_on_Cash": metrics["Cash_on_Cash"],
                "IRR": metrics["IRR"],
                "Occupancy_Rate": metrics["Occupancy_Rate"],
                "Future_Value": metrics["Future_Value"],
                "Conclusion": conclusion,
                "Reasons": reasons,
                "Improvements": improvements,
                "Charts": charts
            })
        
        state["analysis"] = analyses
        state["conversation"].append({"role": "assistant", "content": f"Generated {len(analyses)} report(s)! Opening in a new tab..."})
        return JSONResponse(content={"success": True, "message": "Report generated successfully", "url": "/report"})
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        response = f"Oops, report generation error: {str(e)}. Please try again."
        state["conversation"].append({"role": "assistant", "content": response})
        return JSONResponse(content={"success": False, "message": response})

@app.get("/reset", response_class=HTMLResponse)
async def reset(request: Request):
    """Reset the application state."""
    global state
    state = reset_state()
    welcome_message = (
        "Starting fresh! Ready to analyze a new deal? Search for properties (e.g., 'search properties office Orlando min price 1M') or upload a CSV in Step 1."
    )
    state["conversation"].append({"role": "assistant", "content": welcome_message})
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation": state["conversation"],
        "message": welcome_message,
        "assumptions": state["assumptions"],
        "search_results": state["search_results"]
    })

@app.get("/report", response_class=HTMLResponse)
async def view_report(request: Request):
    """Display the analysis report(s)."""
    if not state["analysis"]:
        error_message = (
            "No analysis yet! Search for properties or upload a CSV in Step 1 and complete the steps. Start now?"
        )
        state["conversation"].append({"role": "assistant", "content": error_message})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": error_message,
            "assumptions": state["assumptions"],
            "search_results": state["search_results"]
        })
    return templates.TemplateResponse("report.html", {
        "request": request,
        "reports": state["analysis"]  # Support multiple reports
    })