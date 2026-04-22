## activate venv

```
source .venv/bin/activate
```

## run web server on mac

```
cd uart-http-proxy
npm run dev -- --device /dev/cu.usbmodem101
```

## deploy to coral

```
python3 scripts/flashtool.py -e tracker
```

## UART HTTP camera (host proxy + firmware)

Firmware `uart_camera_http` speaks HTTP/1.1 on the USB serial console so it pairs with `uart-http-proxy` on your Mac. Flash it, then run the Node proxy with a generous idle timeout (JPEGs are large at 115200 baud):

```
bash build.sh
python3 scripts/flashtool.py -e uart_camera_http
cd uart-http-proxy && npm run dev -- --device /dev/cu.YOURPORT --idle-ms 250 --max-ms 60000
```

Open `http://127.0.0.1:8787/` — the Node app serves the UI from `uart-http-proxy/public/`; the page polls `GET /frame`, which the proxy forwards to the device.

## watch uart

```
screen /dev/cu.usbmodem101 115200
```
