import subprocess
import datetime
from utils.Service.ServiceOperation import start_service,stop_service

def get_latest_event_time(source_name):
    # PowerShell command
    ps_command = f"Get-WinEvent -LogName Application | Where-Object {{$_.ProviderName -eq '{source_name}'}} | Select-Object -First 1 -Property TimeCreated"
    result = subprocess.run(["powershell", "-Command", ps_command], capture_output=True, text=True)
    #print(f"RAW PowerShell output:\n{result!r}")
    output = result.stdout.strip()
    #print("output", output)
    if "TimeCreated" not in output:
        return None

    lines = output.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("timecreated"):
            continue
        if all(c == '-' for c in line.replace(' ', '')):
            continue

        print(f"Parsing line: {line!r}")
        try:
            last_time = datetime.datetime.strptime(line, "%d-%m-%Y %H:%M:%S")
            return last_time
        except ValueError as e:
            print(f"Could not parse date: {line}")
            print(f"Error: {e}")
            return None


def is_heartbeat_recent(last_event_time, threshold_minutes=2):
    if last_event_time is None:
        return False
    now = datetime.datetime.now()
    age = now - last_event_time
    return age < datetime.timedelta(minutes=threshold_minutes)

def check_service_status(service_name, threshold_minutes=2):
    last_event_time = get_latest_event_time(service_name)
    if not is_heartbeat_recent(last_event_time, threshold_minutes):
        print(f"Heartbeat missing or too old. Starting service: {service_name}")
        start_service(service_name)
        return {"status": "started"}
    else:
        print(f"Heartbeat is recent. Service is healthy.")
        return {"status": "healthy"}

