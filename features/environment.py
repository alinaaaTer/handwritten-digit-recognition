import subprocess
import time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"

def before_all(context):
    context.app_proc = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    time.sleep(2)  

def before_scenario(context, scenario):
    context.pw = sync_playwright().start()
    context.browser = context.pw.chromium.launch(headless=True)
    context.page = context.browser.new_page()
    context.page.goto(APP_URL, wait_until="domcontentloaded")

def after_scenario(context, scenario):
    context.page.close()
    context.browser.close()
    context.pw.stop()

def after_all(context):
    context.app_proc.terminate()
