def assess_risk_level(health_index):
    if 80 <= health_index <= 100:
        risk_level = "Green"
        advisory_message = "Normal: Engine operating within healthy parameters. Continue routine monitoring."
    elif 50 <= health_index <= 79:
        risk_level = "Yellow"
        advisory_message = "Degrading: Engine showing signs of degradation. Recommend closer inspection and predictive maintenance planning."
    elif 0 <= health_index <= 49:
        risk_level = "Red"
        advisory_message = "Critical: Engine health is critical. Immediate attention and maintenance required to prevent failure."
    else:
        risk_level = "Unknown"
        advisory_message = "Health index out of range. Cannot assess risk."

    return risk_level, advisory_message

if __name__ == "__main__":
    # Example usage:
    print(assess_risk_level(95))
    print(assess_risk_level(70))
    print(assess_risk_level(25))
    print(assess_risk_level(105)) # Out of range example
    print(assess_risk_level(-10)) # Out of range example
