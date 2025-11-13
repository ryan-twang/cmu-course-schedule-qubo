import requests

def download_pdf(url, out_path):
    r = requests.get(url, stream=True, verify=False)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
