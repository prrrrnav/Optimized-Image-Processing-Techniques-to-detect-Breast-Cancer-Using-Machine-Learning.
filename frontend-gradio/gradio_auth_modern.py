import io, requests, os
import gradio as gr
from PIL import Image

# === CONFIG ===
NODE_BASE = "http://127.0.0.1:3000"

AUTH_SIGNUP = f"{NODE_BASE}/api/auth/signup"
AUTH_LOGIN  = f"{NODE_BASE}/api/auth/login"
PREDICT_URL = f"{NODE_BASE}/predict"
HISTORY_URL = f"{NODE_BASE}/api/history"

# Sample image path
SAMPLE_IMAGE_PATH = r"D:\Code_Playground\Projects\Image-demo\ml-model\dataset\test\benign\8867_idx5_x551_y501_class0.png"

def post_json(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except Exception as e:
        resp = getattr(e, "response", None)
        error_text = resp.text if resp else str(e)
        return {"ok": False, "error": error_text}

def post_file(url, files, headers=None):
    try:
        r = requests.post(url, files=files, headers=headers or {}, timeout=60)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except Exception as e:
        resp = getattr(e, "response", None)
        error_text = resp.text if resp else str(e)
        return {"ok": False, "error": error_text}

def get_json(url, headers=None, params=None):
    try:
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=15)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except Exception as e:
        resp = getattr(e, "response", None)
        error_text = resp.text if resp else str(e)
        return {"ok": False, "error": error_text}

# === Signup ===
def signup(username, email, password):
    if not username or not email or not password:
        return "All fields are required"
    
    res = post_json(AUTH_SIGNUP, {"username": username, "email": email, "password": password})
    if res["ok"]:
        return f"Account created successfully. Please login."
    else:
        return f"Registration failed: {res['error']}"

# === Login ===
def login(email, password, token):
    if not email or not password:
        return "Email and password are required", token
    
    res = post_json(AUTH_LOGIN, {"email": email, "password": password})
    if not res["ok"]:
        return f"Authentication failed: {res['error']}", token
    
    new_token = res["data"]["token"]
    user = res["data"]["user"]
    return f"Welcome back, {user['username']}", new_token

# === Predict ===
def predict(img, token):
    if img is None:
        return "Please upload an image to analyze", None, None, None

    if not token:
        return "Authentication required. Please login first.", None, None, None

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    headers = {"Authorization": f"Bearer {token}"}
    files = {"image": ("image.jpg", buf, "image/jpeg")}

    res = post_file(PREDICT_URL, files, headers)
    if not res["ok"]:
        return f"Analysis failed: {res['error']}", None, None, None

    data = res["data"]
    pred_label = data.get("prediction", "N/A")
    confidence = data.get("confidence", 0)
    
    return "Analysis completed successfully", pred_label, confidence, data

# === History ===
def fetch_history(token, page):
    if not token:
        return "Authentication required. Please login to view history.", []

    headers = {"Authorization": f"Bearer {token}"}
    params = {"page": int(page)} if page else {"page": 1}
    
    res = get_json(HISTORY_URL, headers=headers, params=params)

    if not res["ok"]:
        return f"Failed to load history: {res['error']}", []

    data = res["data"]
    
    # Handle different response structures
    items = data.get("items", []) if isinstance(data, dict) else data
    
    if not items:
        return "No prediction history found", []
    
    rows = []
    for item in items:
        rows.append([
            item.get("result_id", item.get("id", "N/A")),
            item.get("prediction_label", item.get("prediction", "N/A")),
            f"{float(item.get('confidence_score', item.get('confidence', 0))):.2%}",
            item.get("timestamp", item.get("created_at", "N/A"))
        ])
    
    return f"Loaded {len(rows)} record(s)", rows

# === UI ===
with gr.Blocks() as demo:
    
    token_state = gr.State(value=None)
    
    # Custom CSS for proper contrast and visibility
    gr.HTML("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Force light theme and proper contrast */
        .gradio-container {
            background-color: #f5f7fa !important;
        }
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }
        
        /* Text visibility fixes */
        .gr-box, .gr-form, .gr-input, label, .gr-button {
            color: #1e293b !important;
        }
        
        input, textarea, select {
            color: #1e293b !important;
            background-color: #ffffff !important;
        }
        
        .header-container {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
            padding: 40px 30px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            margin-bottom: 30px;
        }
        
        .header-title {
            margin: 0;
            font-size: 2.5em;
            color: white;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .header-subtitle {
            margin: 12px 0 0 0;
            font-size: 1.15em;
            color: rgba(255,255,255,0.95);
            font-weight: 400;
        }
        
        .section-header {
            text-align: center;
            margin: 35px 0 25px 0;
        }
        
        .section-title {
            color: #1e3a8a;
            margin: 0;
            font-size: 1.75em;
            font-weight: 600;
        }
        
        .section-description {
            color: #475569;
            margin: 8px 0 0 0;
            font-size: 1em;
        }
        
        .divider {
            margin: 40px 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        }
        
        .card {
            background: #ffffff !important;
            padding: 25px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .info-box {
            background: #dbeafe;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            margin: 15px 0;
        }
        
        .info-text {
            margin: 0;
            font-size: 0.95em;
            color: #1e40af !important;
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            background: #ffffff;
            border-radius: 10px;
            margin-top: 40px;
            border: 1px solid #e2e8f0;
        }
        
        .footer-title {
            margin: 0 0 15px 0;
            font-size: 1.1em;
            color: #1e293b;
            font-weight: 600;
        }
        
        .footer-text {
            margin: 8px 0;
            color: #475569;
            font-size: 0.95em;
        }
        
        .credential-badge {
            background: #f1f5f9;
            padding: 4px 12px;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #1e293b;
        }
        
        .meta-info {
            margin: 25px 0 0 0;
            font-size: 0.9em;
            color: #64748b;
        }
        
        /* Ensure all labels are visible */
        label {
            color: #334155 !important;
            font-weight: 500 !important;
            font-size: 0.95em !important;
        }
        
        /* Button styling */
        .gr-button-primary {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
        }
        
        .gr-button-secondary {
            background: #f1f5f9 !important;
            color: #334155 !important;
            border: 1px solid #cbd5e1 !important;
        }
    </style>
    """)
    
    # Header
    gr.HTML("""
    <div class='header-container'>
        <h1 class='header-title'>Breast Cancer Detection System</h1>
        <p class='header-subtitle'>AI-Powered Medical Image Analysis Platform</p>
    </div>
    """)

    # Main Layout
    with gr.Row():
        # Sample Image Column
        with gr.Column(scale=2):
            gr.HTML("<div style='background: white; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;'>")
            gr.HTML("<h3 style='margin-top: 0; color: #1e293b;'>Sample Medical Image</h3>")
            if os.path.exists(SAMPLE_IMAGE_PATH):
                gr.Image(SAMPLE_IMAGE_PATH, label="", show_label=False)
            else:
                gr.HTML("""
                <div style='padding: 60px; background: #f8fafc; border-radius: 8px; text-align: center; border: 2px dashed #cbd5e1;'>
                    <p style='font-size: 1.1em; color: #64748b; margin: 0;'>Sample Image Not Available</p>
                    <p style='font-size: 0.9em; color: #94a3b8; margin: 10px 0 0 0;'>
                        Expected: ml-model/dataset/test/benign/8867_idx5_x551_y501_class0.png
                    </p>
                </div>
                """)
            gr.HTML("</div>")
        
        # Authentication Column
        with gr.Column(scale=1):
            gr.HTML("<div style='background: white; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;'>")
            
            with gr.Accordion("Create New Account", open=False):
                gr.HTML("<div style='padding: 10px 0;'>")
                username = gr.Textbox(label="Username", placeholder="Enter your username")
                email_s = gr.Textbox(label="Email Address", placeholder="your.email@example.com")
                pwd_s = gr.Textbox(label="Password", type="password", placeholder="Minimum 8 characters")
                signup_btn = gr.Button("Create Account", variant="primary")
                signup_out = gr.Textbox(label="", show_label=False, placeholder="Registration status will appear here")
                gr.HTML("</div>")
            
            gr.HTML("<div style='margin: 20px 0;'></div>")
            
            with gr.Accordion("Sign In", open=True):
                gr.HTML("<div style='padding: 10px 0;'>")
                gr.HTML("""
                <div style='background: #dbeafe; padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 15px;'>
                    <p style='margin: 0; font-size: 0.9em; color: #1e40af;'><strong>Test Credentials:</strong> Use the pre-filled credentials below for quick access</p>
                </div>
                """)
                email_l = gr.Textbox(
                    label="Email Address", 
                    placeholder="your.email@example.com", 
                    value="demo@test.local"
                )
                pwd_l = gr.Textbox(
                    label="Password", 
                    type="password", 
                    placeholder="Enter your password", 
                    value="testpass123"
                )
                login_btn = gr.Button("Sign In", variant="primary")
                login_out = gr.Textbox(label="", show_label=False, placeholder="Authentication status will appear here")
                gr.HTML("</div>")
            
            gr.HTML("</div>")

    # Wire up authentication
    signup_btn.click(signup, [username, email_s, pwd_s], signup_out)
    login_btn.click(login, [email_l, pwd_l, token_state], [login_out, token_state])

    # Divider
    gr.HTML("<div class='divider'></div>")

    # Prediction Section
    gr.HTML("""
    <div class='section-header'>
        <h2 class='section-title'>Image Analysis Dashboard</h2>
        <p class='section-description'>Upload a medical image to receive AI-powered diagnostic insights</p>
    </div>
    """)
    
    gr.HTML("<div style='background: white; padding: 25px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 20px;'>")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Upload Medical Image", show_label=True)
            predict_btn = gr.Button("Analyze Image", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            status = gr.Textbox(label="Analysis Status", interactive=False)
            
            with gr.Row():
                pred = gr.Textbox(label="Diagnosis", interactive=False)
                conf = gr.Number(label="Confidence Level", interactive=False)
            
            with gr.Accordion("View Full Response", open=False):
                raw = gr.JSON(label="API Response Data")
    
    gr.HTML("</div>")

    predict_btn.click(predict, [img_in, token_state], [status, pred, conf, raw])

    # Divider
    gr.HTML("<div class='divider'></div>")

    # History Section
    gr.HTML("""
    <div class='section-header'>
        <h2 class='section-title'>Prediction History</h2>
        <p class='section-description'>Review and track your previous diagnostic analyses</p>
    </div>
    """)
    
    gr.HTML("<div style='background: white; padding: 25px; border-radius: 10px; border: 1px solid #e2e8f0;'>")
    
    with gr.Row():
        with gr.Column(scale=1):
            page = gr.Number(label="Page Number", value=1, precision=0)
        with gr.Column(scale=3):
            hist_btn = gr.Button("Load History", variant="secondary")
    
    hist_status = gr.Textbox(label="", show_label=False, placeholder="History status will appear here")
    hist_table = gr.Dataframe(
        headers=["Record ID", "Diagnosis", "Confidence", "Timestamp"],
        label="Historical Records"
    )
    
    gr.HTML("</div>")

    hist_btn.click(fetch_history, [token_state, page], [hist_status, hist_table])

    # Footer
    gr.HTML("""
    <div class='footer'>
        <p class='footer-title'>Quick Start Guide</p>
        <p class='footer-text'>
            <strong>Step 1:</strong> Sign in using 
            <span class='credential-badge'>demo@test.local</span> with password 
            <span class='credential-badge'>testpass123</span>
        </p>
        <p class='footer-text'><strong>Step 2:</strong> Upload a medical image for analysis</p>
        <p class='footer-text'><strong>Step 3:</strong> Review results and access historical data</p>
        <p class='meta-info'>Secure Authentication | HIPAA Compliant | AI-Powered Diagnostics</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)