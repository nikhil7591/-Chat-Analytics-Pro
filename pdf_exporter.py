import io
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd

def create_pdf_report(title, user_df, stats, charts=None):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, title, 0, 1, 'C')
    pdf.ln(10)

    # Stats Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "WhatsApp Chat Analysis Report", 0, 1, 'L')
    pdf.ln(5)
    pdf.set_font('Arial', '', 12)

    def add_stat_line(label, value):
        if value is not None:
            pdf.cell(0, 10, f"{label}: {value}", 0, 1, 'L')

    add_stat_line("Total Messages", stats.get('msg_count'))
    add_stat_line("Total Users", stats.get('user_count'))
    add_stat_line("Date Range", f"{stats.get('days')} days")
    add_stat_line("Media Shared", stats.get('media_cnt'))
    add_stat_line("Words Exchanged", stats.get('word_count'))
    add_stat_line("Links Shared", stats.get('links_cnt'))

    # Most Active Users
    if 'user_counts' in stats and isinstance(stats['user_counts'], pd.Series):
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Most Active Users", 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        for i, (user, count) in enumerate(stats['user_counts'].items()):
            if i >= 5:
                break
            pdf.cell(0, 10, f"{user}: {count} messages", 0, 1, 'L')

    # Sentiment Analysis
    if 'sentiment_counts' in stats and stats['sentiment_counts'] is not None:
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Sentiment Analysis", 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        for sentiment, count in stats['sentiment_counts'].items():
            pdf.cell(0, 10, f"{sentiment}: {count} messages", 0, 1, 'L')

    # Add charts (if any)
    if charts:
        for chart in charts:
            pdf.add_page()
            img_buffer = io.BytesIO()
            chart.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            pdf.image(img_buffer, x=10, y=30, w=180)
            plt.close(chart)

    # Notes
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "Analysis Notes", 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, "This report was generated using WhatsApp Chat Analyzer Pro. "
                         "It includes user engagement, sentiment insights, and basic stats.")

    # Return as bytes
    try:
        return pdf.output(dest='S').encode('latin-1')  # proper PDF byte encoding
    except Exception as e:
        print(f"[PDF ERROR] {e}")
        return b""
    

def generate_report_pdf(df, title, user_name, stats):
    report_stats = {
        'msg_count': stats.get('msg_count', 0),
        'user_count': len(df['User'].unique()),
        'days': df['Date'].nunique(),
        'media_cnt': stats.get('media_cnt', 0),
        'word_count': stats.get('word_count', 0),
        'links_cnt': stats.get('links_cnt', 0),
        'user_counts': df['User'].value_counts() if user_name == 'Everyone' else pd.Series(),
        'sentiment_counts': df['sentiment'].value_counts() if 'sentiment' in df.columns else None
    }

    pdf_title = f"WhatsApp Chat Analysis: {user_name}"
    return create_pdf_report(pdf_title, df, report_stats)
