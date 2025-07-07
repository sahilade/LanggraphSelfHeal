
def GetAPIEndpoint(x: str) -> int:
    """Get API Endpoint based on the job type."""
    if x=="sap":
        return r"endpoint of  SAP is - https://api.sap.com/str"
    elif x=="salesforce":
        return "endpoint of  salesforce is - https://api.salesforce.com/int"
    else:
        return "Other endpoints -https://api.default.com/def"
    

def get_weather(location: str) -> str:
    """Get weather."""
    return f"The weather in {location} is sunny 25°C."

def get_wind(location: str) -> str:
    """get wind speed"""
    return f"The wind speed in {location} is 10 km/h."