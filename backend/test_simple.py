import urllib.request
import os

BASE = "http://localhost:8000"

def run():
    csv_path = "static/simple.csv"
    boundary = '----b'
    with open(csv_path, 'rb') as f:
        file_content = f.read()

    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="simple.csv"\r\n'
        f'Content-Type: text/csv\r\n\r\n'
    ).encode('utf-8') + file_content + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req = urllib.request.Request(f"{BASE}/api/analyze", data=body, method="POST")
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    try:
        r = urllib.request.urlopen(req)
        print(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Error {e.code}: {e.read().decode()}")

if __name__ == "__main__":
    run()
