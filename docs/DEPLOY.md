# Running and deploying

## Modes
| Goal | Command |
|---|---|
| Demo without hardware | `scripts\run_dashboard.bat --sim --scenario gas_leak` |
| Arduino over USB | `scripts\run_dashboard.bat --port COM3` |
| ESP32 over Wi-Fi (HTTP) | `set HAZARD_DEVICE_KEY=<long random secret>` then `scripts\run_dashboard.bat --http-ingest`; flash `arduino/esp32_http/esp32_http.ino` with the same key and your PC's address (untested on a board) |
| + live camera and people | add `--camera 0 --people` (see docs/PERSONNEL.md) |
| + fire CNN | train it first (docs/COLAB.md), then add `--camera 0` |
Console mode (no web page): omit `--dashboard` in `python -m src.main ...`.

## Notifications (Telegram / e-mail / webhook)
Copy `data/notifications.example.json` to `data/notifications.json`, set `"enabled": true` for what you use, and put the
SECRETS in environment variables named in the file (never in the file): `HAZARD_TG_TOKEN`, `HAZARD_SMTP_USER`,
`HAZARD_SMTP_PASS`, `HAZARD_WEBHOOK_SECRET`. Restart, then press **Send test notification** on the Configuration page.
Audiences: warnings go to the *admin* lists; CRITICAL and EMERGENCY go to *admin* and *user* lists. Sending is queued in a
background thread, so a dead network cannot delay the shutdown logic; delivery results appear in the System log.
Telegram: create a bot with @BotFather, send it a message, and read your chat id from the bot's `getUpdates`.

## Behaviour you should know
- The software attempts to shut the relay down if the monitoring loop reaches its consecutive-error limit or the link fails. This is a software command, not proof of physical fail-safe behavior; verify the actual relay polarity, power-loss state, and supervised low-voltage behavior on the target hardware before making a physical fail-safe claim. After an outage press **Re-arm machine** (allowed only when SAFE).
- HTTP ingest: commands such as `RELAY:0` reach the device in the reply to its next reading, so allow up to one poll interval (0.5 s) of delay.
- Dashboard is plain HTTP on this machine only. To expose it to a network use a TLS reverse proxy or a VPN, and keep `--host` at the default otherwise.
- Back up `data/hazard.db` (all settings, users, people templates, logs) and `data/secret_key`. Keep both private.

## Start automatically when Windows starts (Task Scheduler)
```
schtasks /Create /TN "HazardMonitor" /SC ONSTART /RL HIGHEST /TR "\"C:\path\to\hazard_system\scripts\run_dashboard.bat\" --port COM3"
```
Test it by rebooting. Check the System log afterwards.

## Updating
Stop the dashboard, replace the files, run `pip install -r requirements.txt`, `python -m pytest -q`, start again. Your `data/` folder is untouched.
