# analyzer.py
# Handles financial calculations and deal evaluation for RealtyBot.
# Computes metrics (NOI, cap rate, DSCR, etc.) and determines if a deal is good.

import numpy as np
import numpy_financial as npf
import logging

# Configure logging for debugging
logger = logging.getLogger(__name__)

def calculate_metrics(data, assumptions):
    """
    Calculate key financial metrics for a property.
    Args:
        data (dict): Property data (e.g., purchase_price, gross_rent, vacancy_rate).
        assumptions (dict): User-defined criteria (e.g., min_cap_rate, min_dscr).
    Returns:
        dict: Metrics including NOI, Cap_Rate, DSCR, Cash_on_Cash, IRR, etc.
    """
    # Extract data with defaults for missing fields
    gross_rent = data.get("gross_rent", 0)
    vacancy_rate = data.get("vacancy_rate", 0.05)
    expenses = data.get("expenses", 0)
    purchase_price = data.get("purchase_price", 0)
    loan_amount = data.get("loan_amount", 0)
    interest_rate = data.get("interest_rate", 0.05)
    loan_term = data.get("loan_term", 25)
    noi_growth = data.get("noi_growth", 0.03)
    appreciation_rate = data.get("appreciation_rate", 0.03)
    exit_cap_rate = data.get("exit_cap_rate", 0.06)
    hold_period = data.get("hold_period", 5)
    tax_rate = assumptions.get("default_tax_rate", 0.01)

    try:
        # Calculate NOI
        effective_gross_income = gross_rent * (1 - vacancy_rate)
        noi = effective_gross_income - expenses
        logger.debug(f"NOI calculated: ${noi:,.2f}")

        # Cap Rate
        cap_rate = noi / purchase_price if purchase_price > 0 else 0

        # Debt Service
        monthly_rate = interest_rate / 12
        num_payments = loan_term * 12
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1) if loan_amount > 0 and monthly_rate > 0 else 0
        annual_debt_service = monthly_payment * 12
        logger.debug(f"Annual debt service: ${annual_debt_service:,.2f}")

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
        logger.debug(f"IRR calculated: {irr*100:.2f}%")

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
    except Exception as e:
        logger.error(f"Metrics calculation error: {str(e)}")
        return {}

def is_good_deal(metrics, data, assumptions):
    """
    Determine if the deal meets user criteria and suggest improvements.
    Args:
        metrics (dict): Calculated metrics from calculate_metrics.
        data (dict): Property data.
        assumptions (dict): User-defined criteria.
    Returns:
        tuple: (conclusion, reasons, improvements) for the report.
    """
    reasons = []
    improvements = []
    is_good = True

    try:
        # Cap Rate Check
        if metrics["Cap_Rate"] < assumptions["min_cap_rate"] * 100:
            is_good = False
            reasons.append(f"The cap rate of {metrics['Cap_Rate']:.2f}% is below your minimum of {assumptions['min_cap_rate'] * 100:.2f}%.")
            required_noi = assumptions["min_cap_rate"] * data.get("purchase_price", 0)
            noi_shortfall = required_noi - metrics["NOI"]
            price_reduction = metrics["NOI"] / assumptions["min_cap_rate"] - data.get("purchase_price", 0) if assumptions["min_cap_rate"] > 0 else 0
            improvements.append(
                f"To meet the minimum cap rate of {assumptions['min_cap_rate'] * 100:.2f}%, increase NOI by ${noi_shortfall:,.2f} or reduce the purchase price by ${-price_reduction:,.2f}."
            )
        elif metrics["Cap_Rate"] > assumptions["max_cap_rate"] * 100:
            is_good = False
            reasons.append(f"The cap rate of {metrics['Cap_Rate']:.2f}% exceeds your maximum of {assumptions['max_cap_rate'] * 100:.2f}%.")
            required_noi = assumptions["max_cap_rate"] * data.get("purchase_price", 0)
            noi_excess = metrics["NOI"] - required_noi
            price_increase = metrics["NOI"] / assumptions["max_cap_rate"] - data.get("purchase_price", 0) if assumptions["max_cap_rate"] > 0 else 0
            improvements.append(
                f"To meet the maximum cap rate of {assumptions['max_cap_rate'] * 100:.2f}%, decrease NOI by ${noi_excess:,.2f} or increase the purchase price by ${price_increase:,.2f}."
            )

        # Cash-on-Cash Return Check
        if metrics["Cash_on_Cash"] < assumptions["min_cash_on_cash"] * 100:
            is_good = False
            reasons.append(f"The cash-on-cash return of {metrics['Cash_on_Cash']:.2f}% is below your target of {assumptions['min_cash_on_cash'] * 100:.2f}%.")
            cash_invested = data.get("purchase_price", 0) - data.get("loan_amount", 0)
            required_cash_flow = assumptions["min_cash_on_cash"] * cash_invested
            cash_flow_shortfall = required_cash_flow - metrics["Annual_Cash_Flow"]
            improvements.append(
                f"To meet the minimum cash-on-cash return of {assumptions['min_cash_on_cash'] * 100:.2f}%, increase annual cash flow by ${cash_flow_shortfall:,.2f}."
            )

        # DSCR Check
        if metrics["DSCR"] < assumptions["min_dscr"]:
            is_good = False
            reasons.append(f"The DSCR of {metrics['DSCR']:.2f} is below your target of {assumptions['min_dscr']:.2f}.")
            annual_debt_service = metrics["NOI"] / metrics["DSCR"] if metrics["DSCR"] > 0 else 0
            required_noi = assumptions["min_dscr"] * annual_debt_service
            noi_shortfall = required_noi - metrics["NOI"]
            improvements.append(
                f"To meet the minimum DSCR of {assumptions['min_dscr']:.2f}, increase NOI by ${noi_shortfall:,.2f} or reduce debt service."
            )

        # Occupancy Rate Check
        if metrics["Occupancy_Rate"] < assumptions["min_occupancy"] * 100:
            is_good = False
            reasons.append(f"The occupancy rate of {metrics['Occupancy_Rate']:.2f}% is below your target of {assumptions['min_occupancy'] * 100:.2f}%.")
            required_vacancy_rate = 1 - assumptions["min_occupancy"]
            current_vacancy_rate = 1 - metrics["Occupancy_Rate"] / 100
            required_gross_rent = metrics["NOI"] + data.get("expenses", 0) / (1 - required_vacancy_rate)
            rent_increase = required_gross_rent - data.get("gross_rent", 0)
            improvements.append(
                f"To meet the minimum occupancy rate of {assumptions['min_occupancy'] * 100:.2f}%, increase occupancy by {(current_vacancy_rate - required_vacancy_rate) * 100:.2f}% or boost gross rent by ${rent_increase:,.2f}."
            )

        # IRR Check
        if metrics["IRR"] < assumptions["min_irr"] * 100:
            is_good = False
            reasons.append(f"The IRR of {metrics['IRR']:.2f}% is below your target of {assumptions['min_irr'] * 100:.2f}%.")
            improvements.append(
                f"To meet the minimum IRR of {assumptions['min_irr'] * 100:.2f}%, increase NOI or negotiate a lower purchase price."
            )

        # Generate verbose conclusion
        if is_good:
            conclusion = (
                f"This is a good deal! It meets all criteria: cap rate {metrics['Cap_Rate']:.2f}%, "
                f"cash-on-cash {metrics['Cash_on_Cash']:.2f}%, DSCR {metrics['DSCR']:.2f}, "
                f"occupancy {metrics['Occupancy_Rate']:.2f}%, IRR {metrics['IRR']:.2f}%."
            )
        else:
            conclusion = (
                "This deal doesn’t meet your criteria due to:\n- " +
                "\n- ".join(reasons) +
                "\n\nTo improve this deal:\n- " +
                "\n- ".join(improvements) +
                "\n\nTweak these factors and click 'Run Report' again!"
            )

        return conclusion, reasons, improvements
    except Exception as e:
        logger.error(f"Deal evaluation error: {str(e)}")
        return "Error evaluating deal.", [], []