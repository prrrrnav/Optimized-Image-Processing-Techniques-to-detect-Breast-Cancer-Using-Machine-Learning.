import io, requests, os
import gradio as gr
from PIL import Image

# === CONFIG ===
NODE_BASE = "http://127.0.0.1:3000"

AUTH_SIGNUP = f"{NODE_BASE}/api/auth/signup"
AUTH_LOGIN  = f"{NODE_BASE}/api/auth/login"
PREDICT_URL = f"{NODE_BASE}/predict"
HISTORY_URL = f"{NODE_BASE}/api/history"


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
        return "Account created successfully. Please login."
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
    
    # Premium CSS Design
    gr.HTML("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        
        /* Global Reset & Base */
        * {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .gradio-container {
            background: #000000 !important;
            min-height: 100vh !important;
            padding: 20px !important;
        }
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }
        
        /* Premium Dark Cards */
        .premium-card {
            background: #1a1a1a !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 24px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5),
                        0 0 0 1px rgba(255, 255, 255, 0.05) inset !important;
            padding: 56px 48px !important;
            position: relative !important;
            overflow: hidden !important;
            max-width: 480px !important;
            margin: 0 auto !important;
            width: 100% !important;
        }
        
        /* Auth Container */
        .auth-container {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0 !important;
            padding: 60px 20px !important;
            min-height: calc(100vh - 200px) !important;
        }
        
        /* Form Group */
        .form-group {
            width: 100% !important;
            margin-bottom: 24px !important;
        }
        
        /* Divider */
        .auth-divider {
            width: 100% !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent) !important;
            margin: 40px 0 !important;
        }
        
        /* Enhanced Input Styling */
        .premium-card input[type="text"],
        .premium-card input[type="password"],
        .premium-card textarea {
            width: 100% !important;
            margin-bottom: 20px !important;
        }
        
        /* Button Full Width in Auth Card */
        .premium-card button {
            width: 100% !important;
            margin-top: 8px !important;
        }
        
        /* Header with Dark Design */
        .header-container {
            background: #1a1a1a !important;
            padding: 64px 48px !important;
            border-radius: 28px !important;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
                        0 0 0 1px rgba(255, 255, 255, 0.05) inset !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            margin-bottom: 48px !important;
            text-align: center !important;
        }
        
        .header-title {
            margin: 0 !important;
            font-size: 3.75em !important;
            font-weight: 800 !important;
            letter-spacing: -2px !important;
            color: #ffffff !important;
            position: relative !important;
            line-height: 1.1 !important;
        }
        
        .header-subtitle {
            margin: 20px 0 0 0 !important;
            font-size: 1.25em !important;
            color: rgba(255,255,255,0.7) !important;
            font-weight: 400 !important;
            letter-spacing: 0.2px !important;
            line-height: 1.5 !important;
        }
        
        /* Section Headers */
        .section-header {
            text-align: center !important;
            margin: 60px 0 50px 0 !important;
            padding: 0 20px !important;
        }
        
        .section-title {
            font-size: 2.75em !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            margin: 0 0 12px 0 !important;
            letter-spacing: -1px !important;
            line-height: 1.2 !important;
        }
        
        .section-description {
            color: rgba(255, 255, 255, 0.65) !important;
            margin: 0 !important;
            font-size: 1.05em !important;
            font-weight: 400 !important;
            line-height: 1.6 !important;
        }
        
        /* Premium Buttons */
        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 16px 32px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4),
                        0 0 0 1px rgba(255, 255, 255, 0.1) inset !important;
            position: relative !important;
            overflow: hidden !important;
            font-size: 1.05em !important;
            letter-spacing: 0.3px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .gr-button-primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5),
                        0 0 0 1px rgba(255, 255, 255, 0.15) inset !important;
        }
        
        .gr-button-primary:active {
            transform: translateY(0) !important;
        }
        
        .gr-button-secondary {
            background: #2a2a2a !important;
            color: #ffffff !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            font-weight: 600 !important;
            padding: 12px 28px !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
        }
        
        .gr-button-secondary:hover {
            background: #333333 !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
        }
        
        /* Premium Input Fields */
        input, textarea, select {
            background: #2a2a2a !important;
            border: 2px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            color: #ffffff !important;
            font-size: 0.95em !important;
            transition: all 0.3s ease !important;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.1) !important;
            outline: none !important;
            background: #333333 !important;
        }
        
        /* Labels */
        label {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 0.95em !important;
            margin-bottom: 8px !important;
            letter-spacing: 0.2px !important;
        }
        
        /* Tabs Styling */
        .tab-nav {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 16px !important;
            padding: 8px !important;
            margin-bottom: 30px !important;
        }
        
        .tab-nav button {
            background: transparent !important;
            border: none !important;
            color: rgba(255, 255, 255, 0.7) !important;
            padding: 12px 24px !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .tab-nav button.selected {
            background: rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Info Boxes */
        .info-box {
            background: #2a2a2a !important;
            padding: 20px !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            margin: 20px 0 !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        }
        
        .info-text {
            margin: 0 !important;
            font-size: 0.95em !important;
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        
        /* Footer */
        .footer {
            text-align: center !important;
            padding: 40px !important;
            background: #1a1a1a !important;
            border-radius: 24px !important;
            margin-top: 50px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
        }
        
        .footer-title {
            margin: 0 0 20px 0 !important;
            font-size: 1.3em !important;
            font-weight: 700 !important;
            color: #ffffff !important;
        }
        
        .footer-text {
            margin: 12px 0 !important;
            color: rgba(255, 255, 255, 0.8) !important;
            font-size: 1em !important;
            line-height: 1.6 !important;
        }
        
        .credential-badge {
            background: #333333 !important;
            color: #ffffff !important;
            padding: 6px 16px !important;
            border-radius: 8px !important;
            font-family: 'Courier New', monospace !important;
            font-size: 0.9em !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
            display: inline-block !important;
            margin: 0 4px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        .meta-info {
            margin: 30px 0 0 0 !important;
            font-size: 0.9em !important;
            color: rgba(255, 255, 255, 0.6) !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
        }
        
        /* Dataframe Styling */
        .dataframe {
            background: #1a1a1a !important;
            border-radius: 16px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Accordion Styling */
        .gr-accordion {
            background: #1a1a1a !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            margin: 15px 0 !important;
        }
        
        /* Image Container */
        .image-container {
            background: #1a1a1a !important;
            border-radius: 20px !important;
            padding: 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
        }
        
        /* Status Messages */
        .status-success {
            color: #10b981 !important;
            font-weight: 600 !important;
        }
        
        .status-error {
            color: #ef4444 !important;
            font-weight: 600 !important;
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        /* Text Visibility */
        .gr-box, .gr-form, .gr-input, .gr-textbox {
            color: #ffffff !important;
            background: #2a2a2a !important;
        }
        
        /* Tab Styling */
        .tab-nav button {
            color: rgba(255, 255, 255, 0.7) !important;
        }
        
        .tab-nav button.selected {
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.1) !important;
        }
        
        /* JSON and other components */
        .gr-json {
            background: #1a1a1a !important;
            color: #ffffff !important;
        }
        
        /* Number inputs */
        input[type="number"] {
            background: #2a2a2a !important;
            color: #ffffff !important;
            border: 2px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* All text elements */
        p, h1, h2, h3, h4, h5, h6, span, div {
            color: #ffffff !important;
        }
        
        /* Tables */
        table, th, td {
            color: #ffffff !important;
            background: #1a1a1a !important;
        }
        
        /* Ensure proper contrast for all interactive elements */
        .gr-component {
            color: #ffffff !important;
        }
        
    </style>
    """)
    
    # Premium Header (shown on all pages)
    gr.HTML("""
    <div class='header-container'>
        <h1 class='header-title'>Breast Cancer Detection System</h1>
        <p class='header-subtitle'>AI-Powered Medical Image Analysis Platform</p>
    </div>
    """)

    # Multi-page interface with tabs
    with gr.Tabs() as tabs:
        
        # === PAGE 1: HOME / AUTHENTICATION ===
        with gr.Tab("Home", id="home"):
            gr.HTML("""
            <div class='auth-container'>
                <div class='premium-card'>
                    <h2 style='margin: 0 0 8px 0; font-size: 2em; font-weight: 700; color: #ffffff; text-align: center; letter-spacing: -0.5px;'>Sign In</h2>
                    <p style='margin: 0 0 32px 0; font-size: 0.95em; color: rgba(255, 255, 255, 0.6); text-align: center;'>Enter your credentials to access the platform</p>
            """)
            
            gr.HTML("""
                    <div class='info-box' style='margin-bottom: 24px;'>
                        <p class='info-text' style='margin: 0; font-size: 0.9em;'><strong>Test Credentials:</strong> Use the pre-filled credentials below for quick access</p>
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
            login_btn = gr.Button("Sign In", variant="primary", size="lg")
            login_out = gr.Textbox(label="", show_label=False, placeholder="Authentication status will appear here", interactive=False)
            
            gr.HTML("""
                    <div class='auth-divider'></div>
                    
                    <h3 style='margin: 0 0 8px 0; font-size: 1.5em; font-weight: 600; color: #ffffff; text-align: center;'>Create New Account</h3>
                    <p style='margin: 0 0 24px 0; font-size: 0.9em; color: rgba(255, 255, 255, 0.6); text-align: center;'>Don't have an account? Sign up below</p>
            """)
            
            username = gr.Textbox(label="Username", placeholder="Enter your username")
            email_s = gr.Textbox(label="Email Address", placeholder="your.email@example.com")
            pwd_s = gr.Textbox(label="Password", type="password", placeholder="Minimum 8 characters")
            signup_btn = gr.Button("Create Account", variant="primary", size="lg")
            signup_out = gr.Textbox(label="", show_label=False, placeholder="Registration status will appear here", interactive=False)
            
            gr.HTML("</div></div>")
            
            # Premium Footer for Home page
            gr.HTML("""
            <div class='footer'>
                <p class='footer-title'>Quick Start Guide</p>
                <p class='footer-text'>
                    <strong>Step 1:</strong> Sign in using 
                    <span class='credential-badge'>demo@test.local</span> with password 
                    <span class='credential-badge'>testpass123</span>
                </p>
                <p class='footer-text'><strong>Step 2:</strong> Navigate to "Image Analysis" tab to upload a medical image</p>
                <p class='footer-text'><strong>Step 3:</strong> Check "History" tab to review your previous analyses</p>
                <p class='meta-info'>Secure Authentication | HIPAA Compliant | AI-Powered Diagnostics</p>
            </div>
            """)
            
            # Wire up authentication
            signup_btn.click(signup, [username, email_s, pwd_s], signup_out)
            login_btn.click(login, [email_l, pwd_l, token_state], [login_out, token_state])
            
            # JavaScript for popup notifications and auto-navigation
            gr.HTML("""
            <script>
                // Function to show toast notification
                function showToast(message, type) {
                    // Remove existing toasts
                    const existingToasts = document.querySelectorAll('.toast-notification');
                    existingToasts.forEach(toast => toast.remove());
                    
                    // Create toast element
                    const toast = document.createElement('div');
                    toast.className = 'toast-notification toast-' + type;
                    
                    const icon = type === 'success' ? '✓' : '✕';
                    toast.innerHTML = '<span class="toast-icon">' + icon + '</span><span class="toast-message">' + message + '</span>';
                    
                    document.body.appendChild(toast);
                    
                    // Remove after animation
                    setTimeout(function() {
                        if (toast.parentNode) {
                            toast.remove();
                        }
                    }, 3000);
                }
                
                // Function to switch to Image Analysis tab
                function switchToAnalysisTab() {
                    // Try multiple selectors to find the tab
                    const selectors = [
                        'button:contains("Image Analysis")',
                        'button[data-testid*="tab"]',
                        '.tab-nav button',
                        'button'
                    ];
                    
                    for (let selector of selectors) {
                        const buttons = Array.from(document.querySelectorAll('button'));
                        for (let btn of buttons) {
                            const text = btn.textContent || btn.innerText || '';
                            if (text.trim() === 'Image Analysis' || text.includes('Image Analysis')) {
                                btn.click();
                                return true;
                            }
                        }
                    }
                    
                    // Fallback: try to find tab by index (second tab)
                    const allTabs = document.querySelectorAll('button[role="tab"], .tab-nav button, button');
                    if (allTabs.length >= 2) {
                        allTabs[1].click();
                        return true;
                    }
                    return false;
                }
                
                // Wait for Gradio to be ready
                function initToastSystem() {
                    let lastLoginValue = '';
                    let lastSignupValue = '';
                    
                    function checkStatus() {
                        // Find all hidden status textboxes (both textarea and input)
                        const allElements = document.querySelectorAll('.hidden-status-textbox, textarea.hidden-status-textbox, input.hidden-status-textbox');
                        
                        // Also try finding by placeholder
                        const allTextareas = document.querySelectorAll('textarea, input[type="text"]');
                        
                        let loginOutput = null;
                        let signupOutput = null;
                        
                        // Find login output
                        for (let el of allElements) {
                            const placeholder = (el.getAttribute('placeholder') || '').toLowerCase();
                            if (placeholder.includes('authentication') || placeholder.includes('status')) {
                                loginOutput = el;
                                break;
                            }
                        }
                        
                        // If not found, try all textareas
                        if (!loginOutput) {
                            for (let el of allTextareas) {
                                const placeholder = (el.getAttribute('placeholder') || '').toLowerCase();
                                if (placeholder.includes('authentication')) {
                                    loginOutput = el;
                                    break;
                                }
                            }
                        }
                        
                        // Find signup output
                        for (let el of allElements) {
                            const placeholder = (el.getAttribute('placeholder') || '').toLowerCase();
                            if (placeholder.includes('registration')) {
                                signupOutput = el;
                                break;
                            }
                        }
                        
                        if (!signupOutput) {
                            for (let el of allTextareas) {
                                const placeholder = (el.getAttribute('placeholder') || '').toLowerCase();
                                if (placeholder.includes('registration')) {
                                    signupOutput = el;
                                    break;
                                }
                            }
                        }
                        
                        // Check login status
                        if (loginOutput) {
                            const value = (loginOutput.value || loginOutput.textContent || '').trim();
                            if (value && value !== lastLoginValue) {
                                lastLoginValue = value;
                                
                                if (value.startsWith('SUCCESS:')) {
                                    const message = value.replace('SUCCESS:', '').trim();
                                    showToast(message, 'success');
                                    setTimeout(switchToAnalysisTab, 1000);
                                } else if (value.startsWith('ERROR:')) {
                                    const message = value.replace('ERROR:', '').trim();
                                    showToast(message, 'error');
                                }
                            }
                        }
                        
                        // Check signup status
                        if (signupOutput) {
                            const value = (signupOutput.value || signupOutput.textContent || '').trim();
                            if (value && value !== lastSignupValue) {
                                lastSignupValue = value;
                                
                                if (value.startsWith('SUCCESS:')) {
                                    const message = value.replace('SUCCESS:', '').trim();
                                    showToast(message, 'success');
                                } else if (value.startsWith('ERROR:')) {
                                    const message = value.replace('ERROR:', '').trim();
                                    showToast(message, 'error');
                                }
                            }
                        }
                    }
                    
                    // Poll for changes
                    setInterval(checkStatus, 300);
                    
                    // Also use MutationObserver for immediate detection
                    const observer = new MutationObserver(function(mutations) {
                        checkStatus();
                    });
                    
                    observer.observe(document.body, {
                        childList: true,
                        subtree: true,
                        characterData: true,
                        attributes: true
                    });
                }
                
                // Initialize when DOM is ready
                function waitForGradio() {
                    // Check if Gradio container exists
                    const gradioContainer = document.querySelector('.gradio-container');
                    if (gradioContainer) {
                        initToastSystem();
                    } else {
                        setTimeout(waitForGradio, 100);
                    }
                }
                
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', waitForGradio);
                } else {
                    waitForGradio();
                }
                
                // Also try after delays to catch Gradio's dynamic loading
                setTimeout(waitForGradio, 500);
                setTimeout(waitForGradio, 1500);
                setTimeout(waitForGradio, 3000);
            </script>
            """)

        # === PAGE 2: IMAGE ANALYSIS ===
        with gr.Tab("Image Analysis", id="analysis"):
            gr.HTML("""
            <div class='section-header'>
                <h2 class='section-title'>Image Analysis Dashboard</h2>
                <p class='section-description'>Upload a medical image to receive AI-powered diagnostic insights</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="pil", label="Upload Medical Image", show_label=True, container=True)
                    predict_btn = gr.Button("Analyze Image", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    status = gr.Textbox(label="Analysis Status", interactive=False)
                    
                    with gr.Row():
                        pred = gr.Textbox(label="Diagnosis", interactive=False)
                        conf = gr.Number(label="Confidence Level", interactive=False)
                    
                    with gr.Accordion("View Full Response", open=False):
                        raw = gr.JSON(label="API Response Data")
            
            predict_btn.click(predict, [img_in, token_state], [status, pred, conf, raw])

        # === PAGE 3: HISTORY ===
        with gr.Tab("History", id="history"):
            gr.HTML("""
            <div class='section-header'>
                <h2 class='section-title'>Prediction History</h2>
                <p class='section-description'>Review and track your previous diagnostic analyses</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    page = gr.Number(label="Page Number", value=1, precision=0)
                with gr.Column(scale=3):
                    hist_btn = gr.Button("Load History", variant="secondary")
            
            hist_status = gr.Textbox(label="", show_label=False, placeholder="History status will appear here")
            hist_table = gr.Dataframe(
                headers=["Record ID", "Diagnosis", "Confidence", "Timestamp"],
                label="Historical Records",
                wrap=True
            )
            
            hist_btn.click(fetch_history, [token_state, page], [hist_status, hist_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)