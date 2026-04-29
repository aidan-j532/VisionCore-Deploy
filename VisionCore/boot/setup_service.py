import subprocess
import sys
import os
import platform

# Mainly vibe coded but supercool ngl

SERVICE_NAME = "visioncore"

def run(cmd, check=True):
    return subprocess.run(cmd, check=check, text=True, capture_output=True)

def get_platform():
    if platform.system() == "Windows":
        return "windows"
    if platform.system() == "Darwin":
        return "macos"
    # Linux, check if systemd is running
    result = run(["pidof", "systemd"], check=False)
    if result.returncode == 0:
        return "linux_systemd"
    return "linux_other"

def setup_systemd(script_path):
    user = os.environ.get("USER", "pi")
    python = sys.executable
    workdir = os.path.dirname(os.path.abspath(script_path))

    service = f"""[Unit]
Description={SERVICE_NAME}
After=network.target

[Service]
ExecStart={python} {os.path.abspath(script_path)}
Restart=always
RestartSec=5
User={user}
WorkingDirectory={workdir}

[Install]
WantedBy=multi-user.target
"""
    service_file = f"/etc/systemd/system/{SERVICE_NAME}.service"
    
    # Write via tee so we can use sudo
    proc = subprocess.run(
        ["sudo", "tee", service_file],
        input=service,
        text=True,
        capture_output=True
    )
    if proc.returncode != 0:
        print(f"Failed to write service file: {proc.stderr}")
        sys.exit(1)

    run(["sudo", "systemctl", "daemon-reload"])
    run(["sudo", "systemctl", "enable", SERVICE_NAME])
    run(["sudo", "systemctl", "start", SERVICE_NAME])
    print(f"Service '{SERVICE_NAME}' installed and started.")
    print(f"  Logs:    journalctl -u {SERVICE_NAME} -f")
    print(f"  Stop:    sudo systemctl stop {SERVICE_NAME}")
    print(f"  Disable: sudo systemctl disable {SERVICE_NAME}")


def setup_windows(script_path):
    python = sys.executable
    script_path = os.path.abspath(script_path)

    # Register as a scheduled task that runs at startup
    cmd = [
        "schtasks", "/create", "/tn", SERVICE_NAME,
        "/tr", f"{python} {script_path}",
        "/sc", "onstart",
        "/ru", "SYSTEM",
        "/f"  # overwrite if exists
    ]
    result = run(cmd, check=False)
    if result.returncode != 0:
        print(f"Failed to create task: {result.stderr}")
        sys.exit(1)

    print(f"Scheduled task '{SERVICE_NAME}' created.")
    print(f"  Start:  schtasks /run /tn {SERVICE_NAME}")
    print(f"  Stop:   schtasks /end /tn {SERVICE_NAME}")
    print(f"  Remove: schtasks /delete /tn {SERVICE_NAME}")


def setup_macos(script_path):
    python = sys.executable
    script_path = os.path.abspath(script_path)
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/com.{SERVICE_NAME}.plist")

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.{SERVICE_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{script_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
"""
    with open(plist_path, "w") as f:
        f.write(plist)

    run(["launchctl", "load", plist_path])
    print(f"LaunchAgent '{SERVICE_NAME}' installed and started.")
    print(f"  Stop:    launchctl unload {plist_path}")
    print(f"  Remove:  rm {plist_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup_service.py <script.py>")
        sys.exit(1)

    script = sys.argv[1]
    if not os.path.isfile(script):
        print(f"File not found: {script}")
        sys.exit(1)

    detected = get_platform()
    print(f"Detected platform: {detected}")

    if detected == "linux_systemd":
        setup_systemd(script)
    elif detected == "windows":
        setup_windows(script)
    elif detected == "macos":
        setup_macos(script)
    else:
        print("Unsupported platform (no systemd detected). Set up a cron job manually:")
        print(f"  @reboot {sys.executable} {os.path.abspath(script)}")