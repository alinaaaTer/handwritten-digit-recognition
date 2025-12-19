from behave import given, when, then
from pathlib import Path
from playwright.sync_api import expect

FEATURES_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FEATURES_DIR.parent

@given("the app is running")
def step_app_running(context):
    pass

@when("I open the app page")
def step_open_page(context):
    context.page.reload()

@when('I upload file "{relative_path}"')
def step_upload_file(context, relative_path):
    file_path = (PROJECT_ROOT / relative_path).resolve()
    assert file_path.exists(), f"File not found: {file_path}"

    context.page.get_by_text("Upload image").wait_for(timeout=15000)
    context.page.locator('input[type="file"]').set_input_files(str(file_path))

@then("I should see the uploaded image preview")
def step_see_preview(context):
    expect(context.page.get_by_text("Preprocessed (28×28)", exact=False)).to_be_visible(timeout=8000)

@then("I should see prediction block")
def step_prediction_block(context):
    expect(context.page.get_by_text("Result", exact=False)).to_be_visible(timeout=8000)
    expect(context.page.get_by_text("Predicted digit", exact=False)).to_be_visible(timeout=8000)

@then("I should see confidence value")
def step_confidence(context):
    expect(context.page.get_by_text("Confidence", exact=False)).to_be_visible(timeout=8000)

@then("I should see top-3 predictions")
def step_top3(context):
    expect(context.page.get_by_text("Top-3", exact=False)).to_be_visible(timeout=8000)

@then("I should see an error message")
def step_error(context):
    expect(context.page.get_by_text("File is too large", exact=False)).to_be_visible(timeout=8000)
