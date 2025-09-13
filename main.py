from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import httpx
import json
import os
from dotenv import load_dotenv
import logging
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY environment variable not set")
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

# Default assumptions (stored as decimals)
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

# State management
def reset_state():
    """Reset the application state for a new analysis."""
    return {
        "stage": "welcome",
        "data": None,
        "assumptions": DEFAULT_ASSUMPTIONS.copy(),
        "missing_data": {},
        "analysis": None,
        "conversation": [
            {"role": "system", "content": (
                "You are a friendly and professional real estate analyst chatbot named RealtyBot. Your goal is to guide users through analyzing commercial real estate deals in a conversational, engaging way. "
                "Use a warm, approachable tone, like you're chatting with a colleague. Ask clear, concise questions, provide helpful feedback, and adapt to user inputs. "
                "Acknowledge changes to assumptions explicitly (e.g., 'I see you updated min cap rate to 7%!'). If the user is unclear, ask for clarification politely. "
                "Use phrases like 'Awesome,' 'Got it,' or 'Let’s dive in!' to keep the conversation lively. Display metrics like cap rate, cash-on-cash, IRR, and occupancy as percentages (e.g., 6% instead of 0.06). "
                "Always include a call to action (e.g., 'What do you think?' or 'Ready to proceed?')."
            )}
        ]
    }

state = reset_state()

async def call_xai_api(message: str, conversation: list):
    """Call xAI API for conversational response."""
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
            return f"Oops, something went wrong with the API: {str(e)}. Let’s try again, shall we?"

def calculate_metrics(data, assumptions):
    """Calculate key financial metrics for the property."""
    gross_rent = data["Gross_Rent"]
    vacancy_rate = data["Vacancy_Rate"]
    expenses = data["Expenses"]
    purchase_price = data["Purchase_Price"]
    loan_amount = data["Loan_Amount"]
    interest_rate = data["Interest_Rate"]
    loan_term = data["Loan_Term"]
    noi_growth = data["NOI_Growth"]
    appreciation_rate = data["Appreciation_Rate"]
    exit_cap_rate = data["Exit_Cap_Rate"]
    hold_period = data["Hold_Period"]
    tax_rate = state["missing_data"].get("tax_rate", assumptions["default_tax_rate"])

    # Calculate NOI
    effective_gross_income = gross_rent * (1 - vacancy_rate)
    noi = effective_gross_income - expenses

    # Cap Rate
    cap_rate = noi / purchase_price if purchase_price > 0 else 0

    # Debt Service
    monthly_rate = interest_rate / 12
    num_payments = loan_term * 12
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1) if loan_amount > 0 and monthly_rate > 0 else 0
    annual_debt_service = monthly_payment * 12

    # DSCR
    dscr = noi / annual_debt_service if annual_debt_service > 0 else float("inf")

    # Cash-on-Cash Return
    cash_invested = purchase_price - loan_amount
    annual_cash_flow = noi - annual_debt_service
    cash_on_cash = annual_cash_flow / cash_invested if cash_invested > 0 else 0

    # IRR (simplified)
    cash_flows = [-cash_invested]
    current_noi = noi
    for year in range(1, int(hold_period) + 1):
        current_noi = current_noi * (1 + noi_growth)
        cash_flow = current_noi - annual_debt_service
        cash_flows.append(cash_flow)
    exit_value = current_noi / exit_cap_rate if exit_cap_rate > 0 else 0
    cash_flows[-1] += exit_value
    irr = npf.irr(cash_flows) if len(cash_flows) > 1 and all(np.isfinite(cash_flows)) else 0

    # Appreciation
    future_value = purchase_price * (1 + appreciation_rate)**hold_period

    return {
        "NOI": noi,
        "Cap_Rate": cap_rate * 100,
        "DSCR": dscr,
        "Cash_on_Cash": cash_on_cash * 100,
        "IRR": irr * 100,
        "Future_Value": future_value,
        "Annual_Cash_Flow": annual_cash_flow,
        "Occupancy_Rate": (1 - vacancy_rate) * 100
    }

def is_good_deal(metrics, data, assumptions):
    """Determine if the deal is 'good' and provide verbose conclusion with improvement suggestions."""
    reasons = []
    improvements = []
    is_good = True

    # Cap Rate Check
    if metrics["Cap_Rate"] < assumptions["min_cap_rate"] * 100:
        is_good = False
        reasons.append(f"The cap rate of {metrics['Cap_Rate']:.2f}% is below your minimum of {assumptions['min_cap_rate'] * 100:.2f}%.")
        # Suggest improvement: Increase NOI or decrease purchase price
        required_noi = assumptions["min_cap_rate"] * data["Purchase_Price"]
        noi_shortfall = required_noi - metrics["NOI"]
        price_reduction = metrics["NOI"] / assumptions["min_cap_rate"] - data["Purchase_Price"]
        improvements.append(
            f"To meet the minimum cap rate of {assumptions['min_cap_rate'] * 100:.2f}%, increase NOI by ${noi_shortfall:,.2f} (e.g., raise rents or reduce expenses) "
            f"or reduce the purchase price by ${-price_reduction:,.2f} to ${data['Purchase_Price'] + price_reduction:,.2f}."
        )
    elif metrics["Cap_Rate"] > assumptions["max_cap_rate"] * 100:
        is_good = False
        reasons.append(f"The cap rate of {metrics['Cap_Rate']:.2f}% exceeds your maximum of {assumptions['max_cap_rate'] * 100:.2f}%.")
        # Suggest improvement: Decrease NOI or increase purchase price
        required_noi = assumptions["max_cap_rate"] * data["Purchase_Price"]
        noi_excess = metrics["NOI"] - required_noi
        price_increase = metrics["NOI"] / assumptions["max_cap_rate"] - data["Purchase_Price"]
        improvements.append(
            f"To meet the maximum cap rate of {assumptions['max_cap_rate'] * 100:.2f}%, decrease NOI by ${noi_excess:,.2f} (e.g., account for higher expenses) "
            f"or increase the purchase price by ${price_increase:,.2f} to ${data['Purchase_Price'] + price_increase:,.2f}."
        )

    # Cash-on-Cash Return Check
    if metrics["Cash_on_Cash"] < assumptions["min_cash_on_cash"] * 100:
        is_good = False
        reasons.append(f"The cash-on-cash return of {metrics['Cash_on_Cash']:.2f}% is below your target of {assumptions['min_cash_on_cash'] * 100:.2f}%.")
        # Suggest improvement: Increase NOI or reduce loan amount
        cash_invested = data["Purchase_Price"] - data["Loan_Amount"]
        required_cash_flow = assumptions["min_cash_on_cash"] * cash_invested
        cash_flow_shortfall = required_cash_flow - metrics["Annual_Cash_Flow"]
        improvements.append(
            f"To meet the minimum cash-on-cash return of {assumptions['min_cash_on_cash'] * 100:.2f}%, increase annual cash flow by ${cash_flow_shortfall:,.2f} "
            f"(e.g., raise NOI by ${cash_flow_shortfall:,.2f} or reduce annual debt service)."
        )

    # DSCR Check
    if metrics["DSCR"] < assumptions["min_dscr"]:
        is_good = False
        reasons.append(f"The DSCR of {metrics['DSCR']:.2f} is below your target of {assumptions['min_dscr']:.2f}.")
        # Suggest improvement: Increase NOI or reduce debt service
        annual_debt_service = metrics["NOI"] / metrics["DSCR"] if metrics["DSCR"] > 0 else 0
        required_noi = assumptions["min_dscr"] * annual_debt_service
        noi_shortfall = required_noi - metrics["NOI"]
        improvements.append(
            f"To meet the minimum DSCR of {assumptions['min_dscr']:.2f}, increase NOI by ${noi_shortfall:,.2f} "
            f"(e.g., raise rents or reduce expenses) or reduce annual debt service by negotiating a lower loan amount or interest rate."
        )

    # Occupancy Rate Check
    if metrics["Occupancy_Rate"] < assumptions["min_occupancy"] * 100:
        is_good = False
        reasons.append(f"The occupancy rate of {metrics['Occupancy_Rate']:.2f}% is below your target of {assumptions['min_occupancy'] * 100:.2f}%.")
        # Suggest improvement: Increase occupancy
        required_vacancy_rate = 1 - assumptions["min_occupancy"]
        current_vacancy_rate = 1 - metrics["Occupancy_Rate"] / 100
        required_gross_rent = metrics["NOI"] + data["Expenses"] / (1 - required_vacancy_rate)
        rent_increase = required_gross_rent - data["Gross_Rent"]
        improvements.append(
            f"To meet the minimum occupancy rate of {assumptions['min_occupancy'] * 100:.2f}%, increase occupancy by {(current_vacancy_rate - required_vacancy_rate) * 100:.2f}% "
            f"or boost gross rent by ${rent_increase:,.2f} to ${required_gross_rent:,.2f}."
        )

    # IRR Check
    if metrics["IRR"] < assumptions["min_irr"] * 100:
        is_good = False
        reasons.append(f"The IRR of {metrics['IRR']:.2f}% is below your target of {assumptions['min_irr'] * 100:.2f}%.")
        # Suggest improvement: Increase NOI or exit value
        improvements.append(
            f"To meet the minimum IRR of {assumptions['min_irr'] * 100:.2f}%, consider increasing NOI through higher rents or lower expenses, "
            f"or negotiate a lower purchase price to improve future cash flows and exit value."
        )

    # Generate verbose conclusion
    if is_good:
        conclusion = (
            "This is a good deal! It meets all your criteria, with a cap rate of {metrics['Cap_Rate']:.2f}%, "
            "cash-on-cash return of {metrics['Cash_on_Cash']:.2f}%, DSCR of {metrics['DSCR']:.2f}, "
            "occupancy rate of {metrics['Occupancy_Rate']:.2f}%, and IRR of {metrics['IRR']:.2f}%. Great find!"
        ).format(metrics=metrics)
    else:
        conclusion = (
            "This deal doesn’t meet your investment criteria due to the following issues:\n- " +
            "\n- ".join(reasons) +
            "\n\nTo make this deal better, consider these adjustments:\n- " +
            "\n- ".join(improvements) +
            "\n\nYou can tweak these factors and click 'Run Report' again to see if the deal improves!"
        )

    return conclusion, reasons, improvements

def generate_charts(metrics, data):
    """Generate Plotly charts for the report with 90% scaling."""
    charts = []

    # Bar chart: NOI, Cash Flow, Debt Service
    bar_fig = go.Figure(data=[
        go.Bar(name="NOI", x=["Year 1"], y=[metrics["NOI"]]),
        go.Bar(name="Cash Flow", x=["Year 1"], y=[metrics["Annual_Cash_Flow"]]),
        go.Bar(name="Debt Service", x=["Year 1"], y=[metrics["Annual_Cash_Flow"] + metrics["NOI"] - metrics["Annual_Cash_Flow"]])
    ])
    bar_fig.update_layout(title="Year 1 Financials", barmode="group", width=540, height=360)  # 90% of 600x400
    charts.append(bar_fig.to_html(full_html=False))

    # Line chart: Property Value Appreciation
    years = list(range(int(data["Hold_Period"]) + 1))
    values = [data["Purchase_Price"] * (1 + data["Appreciation_Rate"])**t for t in years]
    line_fig = px.line(x=years, y=values, labels={"x": "Year", "y": "Property Value"}, title="Property Value Over Time")
    line_fig.update_layout(width=540, height=360)  # 90% of 600x400
    charts.append(line_fig.to_html(full_html=False))

    # Pie chart: Revenue vs. Expenses
    pie_fig = go.Figure(data=[
        go.Pie(labels=["Gross Rent", "Vacancy Loss", "Expenses"], values=[
            data["Gross_Rent"],
            data["Gross_Rent"] * data["Vacancy_Rate"],
            data["Expenses"]
        ])
    ])
    pie_fig.update_layout(title="Income and Expense Breakdown", width=540, height=360)  # 90% of 600x400
    charts.append(pie_fig.to_html(full_html=False))

    # Gauge chart: Cap Rate
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=metrics["Cap_Rate"],
        title={"text": "Cap Rate (%)"},
        gauge={"axis": {"range": [0, 10]}, "bar": {"color": "darkblue"}, "steps": [
            {"range": [0, state["assumptions"]["min_cap_rate"] * 100], "color": "red"},
            {"range": [state["assumptions"]["min_cap_rate"] * 100, state["assumptions"]["max_cap_rate"] * 100], "color": "green"},
            {"range": [state["assumptions"]["max_cap_rate"] * 100, 10], "color": "red"}
        ]}
    ))
    gauge_fig.update_layout(width=360, height=270)  # 90% of 400x300
    charts.append(gauge_fig.to_html(full_html=False))

    return charts

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Render the chat interface."""
    if state["stage"] == "welcome":
        welcome_message = (
            "Welcome to RealtyBot! I’m here to help you analyze commercial real estate deals like a pro. "
            "Follow these steps:\n"
            "1. Upload your deal data (CSV file, like template.csv).\n"
            "2. Review and adjust assumptions using the sliders.\n"
            "3. Chat with me to ask questions or refine criteria.\n"
            "4. Click 'Run Report' to view the analysis.\n"
            "Ready to start? Upload your CSV in Step 1 above!"
        )
        state["conversation"].append({"role": "assistant", "content": welcome_message})
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation": state["conversation"],
        "message": state["conversation"][-1]["content"],
        "assumptions": state["assumptions"]
    })

@app.post("/upload", response_class=HTMLResponse)
async def upload_spreadsheet(request: Request, file: UploadFile = File(...)):
    """Handle spreadsheet upload."""
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        required_columns = ["Property_Type", "Location", "Purchase_Price", "Gross_Rent", "Vacancy_Rate", "Expenses",
                           "Expense_Growth", "Loan_Amount", "Interest_Rate", "Loan_Term", "NOI_Growth",
                           "Appreciation_Rate", "Exit_Cap_Rate", "Hold_Period"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        state["data"] = df.iloc[0].to_dict()
        state["missing_data"] = {col: None for col in missing_columns}
        
        if missing_columns:
            state["stage"] = "missing_data"
            message = (
                f"Great, I’ve got the file! But I noticed we’re missing a few details: {', '.join(missing_columns)}. "
                "Please provide these values in the chat below (e.g., 'tax_rate: 1%') or type 'default' to use standard values. "
                "Head to Step 3 to let me know what’s next!"
            )
        else:
            state["stage"] = "assumptions"
            assumptions_str = "\n".join([
                f"{k.replace('_', ' ')}: {v * 100:.2f}%" if k in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"]
                else f"{k.replace('_', ' ')}: {v}"
                for k, v in state["assumptions"].items()
            ])
            message = (
                f"Awesome, the data looks good! Move to Step 2 to review and tweak the assumptions using the sliders, or chat with me in Step 3 to adjust them (e.g., 'min_cap_rate: 7%'). "
                f"Current assumptions:\n{assumptions_str}\nWhen you’re ready, click 'Run Report' in Step 4 to view the analysis."
            )
        
        state["conversation"].append({"role": "assistant", "content": message})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": message,
            "assumptions": state["assumptions"]
        })
    except Exception as e:
        error_message = f"Oops, something went wrong with the upload: {str(e)}. Please try uploading the file again in Step 1."
        state["conversation"].append({"role": "assistant", "content": error_message})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": error_message,
            "assumptions": state["assumptions"]
        })

@app.post("/chat", response_class=HTMLResponse)
async def handle_chat(request: Request, message: str = Form(...)):
    """Handle user chat input with enhanced parsing for assumptions."""
    state["conversation"].append({"role": "user", "content": message})
    message_lower = message.lower().strip()
    
    # Handle assumption changes (key:value or natural language)
    updated_fields = []
    # Try key:value format first
    for line in message.split("\n"):
        line = line.strip()
        if ":" in line:
            try:
                key, value = [part.strip() for part in line.split(":", 1)]
                key_normalized = key.lower().replace(" ", "_")
                if key_normalized in state["assumptions"]:
                    value = value.strip()
                    if value.endswith("%"):
                        value = float(value[:-1]) / 100
                    else:
                        value = float(value)
                    state["assumptions"][key_normalized] = value
                    updated_fields.append(f"{key_normalized.replace('_', ' ')} to {value * 100:.2f}%" if key_normalized in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"] else f"{key_normalized.replace('_', ' ')} to {value}")
                    logger.debug(f"Updated {key_normalized} to {value} (key:value)")
            except ValueError as e:
                logger.error(f"Invalid value in key:value format: {line}, error: {str(e)}")
                continue

    # Try natural language (e.g., "set min cap rate to 8 percent")
    natural_pattern = re.compile(r"(set|change|update)\s+(.+?)\s+(to|at)\s+([\d.]+)\s*(%|percent)?", re.IGNORECASE)
    matches = natural_pattern.findall(message)
    for match in matches:
        action, key, _, value, is_percent = match
        key_normalized = key.lower().replace(" ", "_").strip()
        if key_normalized in state["assumptions"]:
            try:
                value = float(value)
                if is_percent:
                    value = value / 100
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
        response = f"Nice, I’ve updated {', '.join(updated_fields)}! Here’s the current setup:\n{assumptions_str}\nYou can tweak more in Step 2 using the sliders or here in Step 3. Ready? Click 'Run Report' in Step 4."
    elif "change" in message_lower or "update" in message_lower or "assumption" in message_lower or ":" in message or natural_pattern.search(message):
        response = (
            f"Got it! I see you’re asking about the assumptions, but I didn’t find valid changes. Current setup:\n{assumptions_str}\n"
            "Please use formats like 'min_cap_rate: 7%' or 'set min cap rate to 7 percent' in Step 3, or adjust sliders in Step 2. What’s next?"
        )
    elif state["stage"] == "welcome":
        response = (
            "Looks like you’re ready to get started! Please upload a CSV file with property data (like template.csv) in Step 1 above. Got one ready?"
        )
    elif state["stage"] == "missing_data":
        if message_lower == "default":
            for key in state["missing_data"]:
                state["missing_data"][key] = state["assumptions"].get(f"default_{key}", 0)
            state["stage"] = "assumptions"
            response = (
                f"Got it, I’ll use default values for the missing data. Head to Step 2 to tweak the assumptions using the sliders, or stay here in Step 3 to adjust them via chat. "
                f"Current assumptions:\n{assumptions_str}\nWhen you’re ready, click 'Run Report' in Step 4."
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
                            value = value.strip()
                            if value.endswith("%"):
                                value = float(value[:-1]) / 100
                            else:
                                value = float(value)
                            state["missing_data"][key] = value
                            updated_fields.append(f"{key.replace('_', ' ')} to {value * 100:.2f}%" if key in ["tax_rate"] else f"{key.replace('_', ' ')} to {value}")
                if updated_fields:
                    response = f"Nice, I’ve updated {', '.join(updated_fields)}! "
                if all(v is not None for v in state["missing_data"].values()):
                    state["stage"] = "assumptions"
                    response += (
                        f"All missing data is filled! Move to Step 2 to adjust assumptions with sliders, or stay here in Step 3 to chat. "
                        f"Current assumptions:\n{assumptions_str}\nClick 'Run Report' in Step 4 to view the analysis."
                    )
                else:
                    missing = [k for k, v in state["missing_data"].items() if v is None]
                    response += (
                        f"Hmm, I still need values for: {', '.join(missing)}. "
                        "Please provide them here in Step 3 (e.g., 'tax_rate: 1%') or type 'default'. What’s up?"
                    )
            except:
                response = (
                    "I didn’t quite catch that. Please provide the missing data in the format 'key: value' (e.g., 'tax_rate: 1%') "
                    "here in Step 3, or type 'default'. Can you try again?"
                )
    else:  # stage == "assumptions"
        response = (
            f"I’m ready to help! Current assumptions:\n{assumptions_str}\n"
            "You can tweak them in Step 2 using the sliders or here in Step 3 (e.g., 'min_cap_rate: 7%' or 'set min cap rate to 7 percent'). "
            "When ready, click 'Run Report' in Step 4 to view the analysis."
        )
    
    state["conversation"].append({"role": "assistant", "content": response})
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation": state["conversation"],
        "message": response,
        "assumptions": state["assumptions"]
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
                    value = float(value)
                    if key in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"]:
                        value = value / 100
                    state["assumptions"][key] = value
                    updated_fields.append(f"{key.replace('_', ' ')} to {value * 100:.2f}%" if key in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"] else f"{key.replace('_', ' ')} to {value}")
                except ValueError:
                    logger.error(f"Invalid value for {key}: {value}")
                    continue
        logger.debug(f"Updated assumptions: {state['assumptions']}")
        if updated_fields:
            response = (
                f"Nice, I’ve updated {', '.join(updated_fields)} based on the sliders! "
                f"Here’s the current setup:\n" +
                "\n".join([
                    f"{k.replace('_', ' ')}: {v * 100:.2f}%" if k in ["min_cap_rate", "max_cap_rate", "min_cash_on_cash", "min_occupancy", "min_noi_growth", "max_noi_growth", "min_appreciation", "max_appreciation", "min_irr", "default_tax_rate"]
                    else f"{k.replace('_', ' ')}: {v}"
                    for k, v in state["assumptions"].items()
                ]) +
                "\nReady to analyze? Click 'Run Report' in Step 4 or adjust the sliders more in Step 2."
            )
        else:
            response = "No valid changes were made to the assumptions. Please adjust the sliders in Step 2 and try again."
        state["conversation"].append({"role": "assistant", "content": response})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": response,
            "assumptions": state["assumptions"]
        })
    except Exception as e:
        logger.error(f"Error in update_assumptions: {str(e)}")
        response = f"Oops, something went wrong with the slider updates: {str(e)}. Please try again in Step 2 or use the chat in Step 3 to set values."
        state["conversation"].append({"role": "assistant", "content": response})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": response,
            "assumptions": state["assumptions"]
        })

@app.post("/run_report", response_class=JSONResponse)
async def run_report(request: Request):
    """Generate the report and signal to open it in a new tab."""
    try:
        if not state["data"]:
            response = "Oops, no deal data uploaded yet! Please upload a CSV in Step 1 to proceed with the analysis."
            state["conversation"].append({"role": "assistant", "content": response})
            return JSONResponse(content={"success": False, "message": response})
        
        state["stage"] = "report"
        metrics = calculate_metrics(state["data"], state["assumptions"])
        conclusion, reasons, improvements = is_good_deal(metrics, state["data"], state["assumptions"])
        charts = generate_charts(metrics, state["data"])
        state["analysis"] = {
            "Property_Type": state["data"]["Property_Type"],
            "Location": state["data"]["Location"],
            "Purchase_Price": state["data"]["Purchase_Price"],
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
        }
        state["conversation"].append({"role": "assistant", "content": "All done! Opening your analysis report in a new tab..."})
        return JSONResponse(content={"success": True, "message": "Report generated successfully", "url": "/report"})
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        response = f"Oops, something went wrong while generating the report: {str(e)}. Please try again."
        state["conversation"].append({"role": "assistant", "content": response})
        return JSONResponse(content={"success": False, "message": response})

@app.get("/reset", response_class=HTMLResponse)
async def reset(request: Request):
    """Reset the application state."""
    global state
    state = reset_state()
    welcome_message = (
        "Alright, we’re starting fresh! Ready to analyze a new property? "
        "Upload a CSV file with property data (like template.csv) in Step 1 above. Let’s do this!"
    )
    state["conversation"].append({"role": "assistant", "content": welcome_message})
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation": state["conversation"],
        "message": welcome_message,
        "assumptions": state["assumptions"]
    })

@app.get("/report", response_class=HTMLResponse)
async def view_report(request: Request):
    """Display the analysis report."""
    if not state["analysis"]:
        error_message = (
            "No analysis yet! Please upload a CSV in Step 1 and complete the steps in the chat. Want to start now? Head back to the main page."
        )
        state["conversation"].append({"role": "assistant", "content": error_message})
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "conversation": state["conversation"],
            "message": error_message,
            "assumptions": state["assumptions"]
        })
    return templates.TemplateResponse("report.html", {"request": request, "report": state["analysis"]})