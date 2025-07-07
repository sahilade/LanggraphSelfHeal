
def GetAPIEndpoint(x: str) -> int:
    """Get API Endpoint based on the job type."""
    if x=="sap":
        return "https://api.sap.com/str"
    elif x=="salesforce":
        return "https://api.salesforce.com/int"
    else:
        return "https://api.default.com/def"