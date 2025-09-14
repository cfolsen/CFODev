# charts.py
# Generates Plotly charts for RealtyBot reports (e.g., NOI, cash flow, appreciation).
# Uses compact formatting (90% scaling) for consistent UI.

import plotly.express as px
import plotly.graph_objects as go
import logging

# Configure logging for debugging
logger = logging.getLogger(__name__)

def generate_charts(metrics, data, assumptions):
    """
    Generate Plotly charts for the report with 90% scaling.
    Args:
        metrics (dict): Calculated metrics (e.g., NOI, Cap_Rate).
        data (dict): Property data (e.g., purchase_price, gross_rent).
        assumptions (dict): User-defined criteria (e.g., min_cap_rate).
    Returns:
        list: HTML strings of Plotly charts.
    """
    charts = []
    try:
        # Bar chart: NOI, Cash Flow, Debt Service
        bar_fig = go.Figure(data=[
            go.Bar(name="NOI", x=["Year 1"], y=[metrics["NOI"]]),
            go.Bar(name="Cash Flow", x=["Year 1"], y=[metrics["Annual_Cash_Flow"]]),
            go.Bar(name="Debt Service", x=["Year 1"], y=[metrics["Annual_Cash_Flow"] + metrics["NOI"] - metrics["Annual_Cash_Flow"]])
        ])
        bar_fig.update_layout(title="Year 1 Financials", barmode="group", width=540, height=360)
        charts.append(bar_fig.to_html(full_html=False))
        logger.debug("Generated bar chart")

        # Line chart: Property Value Appreciation
        years = list(range(int(data.get("hold_period", 5)) + 1))
        values = [data.get("purchase_price", 0) * (1 + data.get("appreciation_rate", 0.03))**t for t in years]
        line_fig = px.line(x=years, y=values, labels={"x": "Year", "y": "Property Value"}, title="Property Value Over Time")
        line_fig.update_layout(width=540, height=360)
        charts.append(line_fig.to_html(full_html=False))
        logger.debug("Generated line chart")

        # Pie chart: Revenue vs. Expenses
        pie_fig = go.Figure(data=[
            go.Pie(labels=["Gross Rent", "Vacancy Loss", "Expenses"], values=[
                data.get("gross_rent", 0),
                data.get("gross_rent", 0) * data.get("vacancy_rate", 0.05),
                data.get("expenses", 0)
            ])
        ])
        pie_fig.update_layout(title="Income and Expense Breakdown", width=540, height=360)
        charts.append(pie_fig.to_html(full_html=False))
        logger.debug("Generated pie chart")

        # Gauge chart: Cap Rate
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics["Cap_Rate"],
            title={"text": "Cap Rate (%)"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, assumptions["min_cap_rate"] * 100], "color": "red"},
                    {"range": [assumptions["min_cap_rate"] * 100, assumptions["max_cap_rate"] * 100], "color": "green"},
                    {"range": [assumptions["max_cap_rate"] * 100, 10], "color": "red"}
                ]
            }
        ))
        gauge_fig.update_layout(width=360, height=270)
        charts.append(gauge_fig.to_html(full_html=False))
        logger.debug("Generated gauge chart")

        return charts
    except Exception as e:
        logger.error(f"Chart generation error: {str(e)}")
        return []