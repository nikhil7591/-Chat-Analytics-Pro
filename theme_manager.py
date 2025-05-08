import streamlit as st

# Define enhanced theme presets with more sophisticated color schemes
THEMES = {
    "Professional": {
        "primaryColor": "#0063B2",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F5F7FA",
        "textColor": "#2C3E50",
        "accentColor": "#9DC6E0",
        "font": "Inter, sans-serif",
        "borderRadius": "6px",
        "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)"
    },
    "Dark Pro": {
        "primaryColor": "#7C4DFF",
        "backgroundColor": "#1E1E2E",
        "secondaryBackgroundColor": "#2D2D3F",
        "textColor": "#E0E0E0",
        "accentColor": "#B39DDB",
        "font": "Inter, sans-serif",
        "borderRadius": "8px",
        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.15)"
    },
    "Midnight Blue": {
        "primaryColor": "#3498DB",
        "backgroundColor": "#0A192F",
        "secondaryBackgroundColor": "#172A45",
        "textColor": "#E6F1FF",
        "accentColor": "#64FFDA",
        "font": "SF Pro Display, sans-serif",
        "borderRadius": "6px",
        "boxShadow": "0 3px 8px rgba(0, 0, 0, 0.3)"
    },
    "Ocean Breeze": {
        "primaryColor": "#00B4D8",
        "backgroundColor": "#F0F8FF",
        "secondaryBackgroundColor": "#E1F5FE",
        "textColor": "#263238",
        "accentColor": "#90E0EF",
        "font": "Nunito, sans-serif",
        "borderRadius": "10px",
        "boxShadow": "0 4px 6px rgba(0, 99, 178, 0.1)"
    },
    "Sunset Horizon": {
        "primaryColor": "#F76B1C",
        "backgroundColor": "#FFFAF0",
        "secondaryBackgroundColor": "#FFF1E6",
        "textColor": "#4A4A4A",
        "accentColor": "#FFC288",
        "font": "Poppins, sans-serif",
        "borderRadius": "8px",
        "boxShadow": "0 3px 10px rgba(247, 107, 28, 0.08)"
    },
    "Minimal Elegance": {
        "primaryColor": "#555B6E",
        "backgroundColor": "#FCFCFC",
        "secondaryBackgroundColor": "#F4F4F8",
        "textColor": "#2B2D42",
        "accentColor": "#BEC1CC",
        "font": "Roboto, sans-serif",
        "borderRadius": "4px",
        "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.05)"
    },
    "Forest Green": {
        "primaryColor": "#2E7D32",
        "backgroundColor": "#F8FBF6",
        "secondaryBackgroundColor": "#E8F5E9",
        "textColor": "#1B5E20",
        "accentColor": "#A5D6A7",
        "font": "Montserrat, sans-serif",
        "borderRadius": "6px",
        "boxShadow": "0 3px 8px rgba(46, 125, 50, 0.08)"
    },
    "Corporate Purple": {
        "primaryColor": "#673AB7",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F5F0FF",
        "textColor": "#37474F",
        "accentColor": "#D1C4E9",
        "font": "Lato, sans-serif",
        "borderRadius": "6px",
        "boxShadow": "0 3px 10px rgba(103, 58, 183, 0.1)"
    }
}

def apply_theme():
    """
    Apply the currently selected theme to the app with enhanced styling
    """
    # Get the current theme name from session state
    theme_name = st.session_state.get("theme", "Professional")
    
    # Get theme colors and properties
    theme = THEMES[theme_name]
    
    # Apply theme using CSS with enhanced styling
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Montserrat:wght@400;500;600&family=Nunito:wght@400;600;700&family=Poppins:wght@400;500;600&family=Roboto:wght@400;500&family=Lato:wght@400;700&display=swap');
        
        /* Global Styles */
        * {{
            transition: all 0.2s ease;
        }}
        
        /* Main App */
        .stApp {{
            background-color: {theme["backgroundColor"]};
            color: {theme["textColor"]};
            font-family: {theme["font"]};
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {theme["secondaryBackgroundColor"]};
            border-right: 1px solid rgba(0,0,0,0.05);
            padding: 1rem 0;
        }}
        
        /* Sidebar Title */
        .sidebar .sidebar-content .block-container h1 {{
            color: {theme["primaryColor"]};
            font-weight: 600;
            font-size: 1.25rem;
            margin-bottom: 1.5rem;
        }}
        
        /* Cards and containers */
        div.stCard {{
            background-color: {theme["secondaryBackgroundColor"]};
            border-radius: {theme["borderRadius"]};
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: {theme["boxShadow"]};
            border: 1px solid rgba(0,0,0,0.03);
        }}
        
        /* Custom metric cards */
        .metric-card {{
            background-color: {theme["secondaryBackgroundColor"]};
            border-left: 4px solid {theme["primaryColor"]};
            padding: 16px;
            border-radius: {theme["borderRadius"]};
            box-shadow: {theme["boxShadow"]};
            transition: transform 0.2s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }}
        
        /* Headers and titles */
        h1 {{
            color: {theme["primaryColor"]};
            font-weight: 600;
            font-size: 2rem;
            margin-bottom: 1rem;
        }}
        
        h2 {{
            color: {theme["primaryColor"]};
            font-weight: 500;
            font-size: 1.5rem;
            margin-bottom: 0.75rem;
        }}
        
        h3, h4, h5, h6 {{
            color: {theme["textColor"]};
            font-weight: 500;
        }}
        
        /* Links */
        a {{
            color: {theme["primaryColor"]};
            text-decoration: none;
            transition: color 0.2s ease;
        }}
        
        a:hover {{
            color: {theme["accentColor"]};
            text-decoration: underline;
        }}
        
        /* Buttons */
        button[data-baseweb="button"] {{
            background-color: {theme["primaryColor"]};
            border-radius: {theme["borderRadius"]};
            transition: all 0.2s ease;
            font-weight: 500;
            border: none;
        }}
        
        button[data-baseweb="button"]:hover {{
            background-color: {theme["primaryColor"]}E6;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        
        /* Dropdown menus */
        div[data-baseweb="select"] {{
            background-color: {theme["secondaryBackgroundColor"]};
            border-radius: {theme["borderRadius"]};
            border: 1px solid rgba(0,0,0,0.1);
        }}
        
        /* Navigation pills */
        button[role="tab"] {{
            background-color: {theme["secondaryBackgroundColor"]};
            color: {theme["textColor"]};
            border-radius: {theme["borderRadius"]};
            border: none;
            margin-right: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }}
        
        button[role="tab"][aria-selected="true"] {{
            background-color: {theme["primaryColor"]}20;
            color: {theme["primaryColor"]};
            border-bottom: 2px solid {theme["primaryColor"]};
        }}
        
        button[role="tab"]:hover {{
            background-color: {theme["primaryColor"]}10;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input {{
            color: {theme["textColor"]};
            border-radius: {theme["borderRadius"]};
            border: 1px solid rgba(0,0,0,0.1);
            padding: 0.5rem;
            background-color: {theme["backgroundColor"]};
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: {theme["primaryColor"]};
            box-shadow: 0 0 0 2px {theme["primaryColor"]}40;
        }}
        
        /* Data frames */
        .stDataFrame {{
            color: {theme["textColor"]};
            border-radius: {theme["borderRadius"]};
            overflow: hidden;
        }}
        
        .stDataFrame td {{
            padding: 0.5rem 1rem !important;
        }}
        
        .stDataFrame thead tr th {{
            background-color: {theme["primaryColor"]}20 !important;
            color: {theme["primaryColor"]} !important;
            font-weight: 600 !important;
        }}
        
        .stDataFrame tbody tr:nth-child(even) {{
            background-color: {theme["secondaryBackgroundColor"]};
        }}
        
        /* Markdown text */
        div[data-testid="stMarkdownContainer"] {{
            color: {theme["textColor"]};
            line-height: 1.6;
        }}
        
        /* Code blocks */
        code {{
            color: {theme["primaryColor"]};
            background-color: {theme["secondaryBackgroundColor"]};
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        /* Plot styling */
        .stPlotlyChart {{
            background-color: {theme["backgroundColor"]};
            border-radius: {theme["borderRadius"]};
            box-shadow: {theme["boxShadow"]};
            padding: 1rem;
            border: 1px solid rgba(0,0,0,0.03);
        }}
        
        /* Progress bars */
        div[data-testid="stDecoration"] {{
            background-color: {theme["primaryColor"]};
        }}
        
        /* Chat messages */
        div[data-testid="stChatMessage"] {{
            background-color: {theme["secondaryBackgroundColor"]};
            color: {theme["textColor"]};
            border-radius: {theme["borderRadius"]};
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: {theme["boxShadow"]};
        }}
        
        /* User chat bubble */
        div[data-testid="stChatMessage"][data-testid="user"] {{
            background-color: {theme["primaryColor"]}20;
            border-left: 3px solid {theme["primaryColor"]};
        }}
        
        /* Expandable sections */
        div[data-testid="stExpander"] {{
            background-color: {theme["secondaryBackgroundColor"]};
            border-radius: {theme["borderRadius"]};
            border: 1px solid rgba(0,0,0,0.05);
            overflow: hidden;
        }}
        
        /* Expander header */
        .streamlit-expanderHeader {{
            background-color: {theme["secondaryBackgroundColor"]};
            color: {theme["primaryColor"]};
            font-weight: 500;
            padding: 0.75rem 1rem;
        }}
        
        /* Expander content */
        .streamlit-expanderContent {{
            background-color: {theme["secondaryBackgroundColor"]};
            padding: 1rem;
            border-top: 1px solid rgba(0,0,0,0.05);
        }}
        
        /* Slider */
        div[data-testid="stSlider"] > div > div > div {{
            background-color: {theme["primaryColor"]};
        }}
        
        /* Alerts and messages */
        div[data-baseweb="notification"] {{
            background-color: {theme["secondaryBackgroundColor"]};
            border-radius: {theme["borderRadius"]};
            border-left: 4px solid {theme["primaryColor"]};
            color: {theme["textColor"]};
            box-shadow: {theme["boxShadow"]};
        }}
        
        /* Success messages */
        div[data-baseweb="notification"][kind="positive"] {{
            border-left-color: #4CAF50;
        }}
        
        /* Warning messages */
        div[data-baseweb="notification"][kind="warning"] {{
            border-left-color: #FF9800;
        }}
        
        /* Error messages */
        div[data-baseweb="notification"][kind="negative"] {{
            border-left-color: #F44336;
        }}
        
        /* File uploader */
        button[data-testid="stFileUploadDropzone"] {{
            border: 2px dashed rgba(0,0,0,0.1);
            border-radius: {theme["borderRadius"]};
            padding: 1rem;
            background-color: {theme["secondaryBackgroundColor"]};
            color: {theme["textColor"]};
            transition: all 0.2s ease;
        }}
        
        button[data-testid="stFileUploadDropzone"]:hover {{
            border-color: {theme["primaryColor"]};
            background-color: {theme["secondaryBackgroundColor"]};
        }}
    </style>
    """, unsafe_allow_html=True)

def theme_selector():
    """
    Create an enhanced theme selection dropdown in the sidebar
    """
    # Initialize session state for theme if it doesn't exist
    if "theme" not in st.session_state:
        st.session_state.theme = "Professional"
    
    st.sidebar.markdown("### 🎨 Appearance")
    
    # Theme selector with enhanced styling
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        selected_theme = st.selectbox(
            "Select Theme",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.theme),
            key="theme_selector"
        )
    
    # Preview color for the selected theme
    with col2:
        st.markdown(
            f"""
            <div style="
                width: 30px;
                height: 30px;
                border-radius: 50%;
                background-color: {THEMES[selected_theme]['primaryColor']};
                margin-top: 25px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            "></div>
            """,
            unsafe_allow_html=True
        )
    
    # Theme description based on selection
    theme_descriptions = {
        "Professional": "Clean and modern look for business applications.",
        "Dark Pro": "Sleek dark theme with purple accents.",
        "Midnight Blue": "Dark blue theme inspired by developer environments.",
        "Ocean Breeze": "Light and refreshing aqua-themed design.",
        "Sunset Horizon": "Warm color palette with orange accents.",
        "Minimal Elegance": "Subtle and refined neutral design.",
        "Forest Green": "Nature-inspired theme with calming green tones.",
        "Corporate Purple": "Professional theme with purple elements."
    }
    
    st.sidebar.caption(theme_descriptions[selected_theme])
    
    # Update theme if changed
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()  # Rerun to apply new theme
        
# Demo usage example
def theme_preview():
    """Optional function to preview theme elements"""
    theme = THEMES[st.session_state.get("theme", "Professional")]
    
    st.markdown(f"""
    <div style="
        background-color: {theme['primaryColor']}; 
        color: white; 
        padding: 1rem; 
        border-radius: {theme['borderRadius']}; 
        margin-bottom: 1rem;
        box-shadow: {theme['boxShadow']};
    ">
        <h2 style="margin:0; color: white;">Current Theme: {st.session_state.get("theme", "Professional")}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Theme Preview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-top:0;">Primary Color</h3>
            <div style="
                width: 100%;
                height: 30px;
                background-color: {theme['primaryColor']};
                border-radius: 4px;
            "></div>
            <p style="margin-bottom:0;">{theme['primaryColor']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-top:0;">Background</h3>
            <div style="
                width: 100%;
                height: 30px;
                background-color: {theme['backgroundColor']};
                border-radius: 4px;
                border: 1px solid rgba(0,0,0,0.1);
            "></div>
            <p style="margin-bottom:0;">{theme['backgroundColor']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin-top:0;">Secondary BG</h3>
            <div style="
                width: 100%;
                height: 30px;
                background-color: {theme['secondaryBackgroundColor']};
                border-radius: 4px;
                border: 1px solid rgba(0,0,0,0.1);
            "></div>
            <p style="margin-bottom:0;">{theme['secondaryBackgroundColor']}</p>
        </div>
        """, unsafe_allow_html=True)

