
def GetAPIEndpoint(x: str) -> int:
    """Get API Endpoint based on the job type."""
    if x=="sap":
        return "https://api.sap.com/str"
    elif x=="salesforce":
        return "https://api.salesforce.com/int"
    else:
        return "https://api.default.com/def"
    

def get_weather(location: str) -> str:
    """Get weather."""
    return f"The weather in {location} is sunny 25°C."

def get_wind(location: str) -> str:
    """get wind speed"""
    return f"The wind speed in {location} is 10 km/h."