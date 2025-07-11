import subprocess

def start_service(service_name):
    subprocess.run(["sc", "start", service_name], check=True)

def stop_service(service_name):
    subprocess.run(["sc", "stop", service_name], check=True)
