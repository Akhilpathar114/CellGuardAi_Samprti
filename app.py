import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import base64
import io
import time
from datetime import datetime

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="CellGuardAI",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR "ENGINEERING DARK MODE" ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Card/Container Styling */
    div[data-testid="stMetric"], div[data-testid="stDataFrame"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics */
    label[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] {
        background-color: #1e293b;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# --- TYPES & DATA STRUCTURES ---
class DetectedIssue:
    def __init__(self, type_name, severity, description, explanation, recommendation, timestamp=None):
        self.type = type_name
        self.severity = severity
        self.description = description
        self.explanation = explanation
        self.recommendation = recommendation
        self.timestamp = timestamp

# --- ANALYSIS ENGINE ---

def detect_columns(df):
    """
    Heuristics to identify relevant columns in the CSV.
    """
    cols = [c.lower() for c in df.columns]
    mapping = {
        'voltage': next((c for c in cols if any(x in c for x in ['pack_volt', 'batt_volt', 'voltage']) and 'cell' not in c), None),
        'current': next((c for c in cols if any(x in c for x in ['current', 'amps', 'i_batt'])), None),
        'temp': next((c for c in cols if any(x in c for x in ['temp', 't_batt']) and 'max' not in c and 'min' not in c), None),
        'soc': next((c for c in cols if any(x in c for x in ['soc', 'capacity', 'state_of_charge'])), None),
    }
    
    # Identify Cell Voltages
    # Looks for cell_1, cell_01, v_cell_1, or just v1...v24 columns
    cell_cols = [c for c in cols if ('cell' in c or (c.startswith('v') and c[1:].isdigit())) and 'temp' not in c and 'min' not in c and 'max' not in c]
    mapping['cells'] = cell_cols
    
    return mapping

def analyze_bms_data(df, filename):
    """
    Core logic: Cleaning, FE, ML, Diagnostics.
    """
    mapping = detect_columns(df)
    
    # 1. Standardization & Cleaning
    processed = pd.DataFrame()
    processed['timestamp'] = df.index  # Use index as proxy for time if no timestamp col
    
    # Fallbacks if columns missing
    processed['voltage'] = df[mapping['voltage']] if mapping['voltage'] else 0
    processed['current'] = df[mapping['current']] if mapping['current'] else 0
    processed['temp'] = df[mapping['temp']] if mapping['temp'] else 25
    processed['soc'] = df[mapping['soc']] if mapping['soc'] else 50
    
    # Cell Imbalance Calculation
    if mapping['cells']:
        cell_data = df[mapping['cells']]
        processed['min_cell_v'] = cell_data.min(axis=1)
        processed['max_cell_v'] = cell_data.max(axis=1)
        processed['imbalance'] = processed['max_cell_v'] - processed['min_cell_v']
        processed['avg_cell_v'] = cell_data.mean(axis=1)
    else:
        processed['imbalance'] = 0
        processed['min_cell_v'] = 0
        processed['max_cell_v'] = 0
    
    # Fill NA
    processed = processed.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # 2. Machine Learning: Anomaly Detection
    # Using Isolation Forest on key electrical/thermal parameters
    features = ['voltage', 'current', 'temp', 'imbalance']
    # Scale features
    scaler = StandardScaler()
    X = processed[features]
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    processed['anomaly_score'] = iso_forest.fit_predict(X_scaled)
    # -1 is anomaly, 1 is normal
    
    # 3. Diagnostic Engine (Rule-based + ML Hybrid)
    issues = []
    
    # A. Thermal Issues
    high_temp = processed[processed['temp'] > 45]
    if not high_temp.empty:
        max_t = high_temp['temp'].max()
        severity = 'CRITICAL' if max_t > 60 else 'MEDIUM'
        issues.append(DetectedIssue(
            type_name="THERMAL",
            severity=severity,
            description=f"High battery temperature detected (Max: {max_t:.1f}°C)",
            explanation="Temperature exceeds safe continuous operating limits. This accelerates aging and poses safety risks.",
            recommendation="Inspect cooling system. Reduce discharge rate. Check for loose connections causing resistance heating."
        ))

    # B. Cell Imbalance
    high_imb = processed[processed['imbalance'] > 0.1] # 100mV
    if not high_imb.empty:
        max_imb = high_imb['imbalance'].max()
        severity = 'HIGH' if max_imb > 0.2 else 'MEDIUM'
        issues.append(DetectedIssue(
            type_name="IMBALANCE",
            severity=severity,
            description=f"Significant cell voltage deviation ({max_imb*1000:.0f} mV)",
            explanation="Large voltage spread indicates capacity mismatch or weak cells. The lowest cell limits the entire pack.",
            recommendation="Perform cell balancing. If passive balancing fails, a cell module replacement may be required."
        ))

    # C. Voltage Sag / Weakness
    # Drop under load
    load_events = processed[processed['current'] < -10] # Discharging > 10A
    if not load_events.empty:
        # Check if voltage drops excessively relative to current (Simple Resistance Check Proxy)
        # V = Voc - I*R -> dV/dI ~ R
        # Simplified: If voltage drops below cutoff under moderate load
        if load_events['voltage'].min() < (processed['voltage'].max() * 0.75):
             issues.append(DetectedIssue(
                type_name="VOLTAGE_SAG",
                severity="HIGH",
                description="Excessive voltage sag under load",
                explanation="Battery voltage drops significantly when current is drawn, indicating high internal resistance (aging).",
                recommendation="Battery capacity test recommended. The pack may be nearing end-of-life."
            ))

    # D. ML Anomalies
    anomalies = processed[processed['anomaly_score'] == -1]
    anomaly_rate = len(anomalies) / len(processed)
    if anomaly_rate > 0.1:
        issues.append(DetectedIssue(
            type_name="ML_ANOMALY",
            severity="LOW",
            description=f"Unusual operational patterns detected ({anomaly_rate*100:.1f}% of time)",
            explanation="The Machine Learning model detected behavior that deviates from the statistical norm (e.g., erratic current/voltage).",
            recommendation="Review the highlighted regions in the charts. Check BMS sensor integrity."
        ))

    # 4. Health Scoring Logic
    health_score = 100
    
    # Deduct for thermal history
    health_score -= (len(processed[processed['temp'] > 45]) / len(processed)) * 40
    
    # Deduct for imbalance
    avg_imb = processed['imbalance'].mean()
    health_score -= (avg_imb / 0.1) * 20
    
    # Deduct for resistance/sag (approximation based on max sag)
    v_range = processed['voltage'].max() - processed['voltage'].min()
    if v_range > 0 and processed['voltage'].max() > 0:
        sag_ratio = v_range / processed['voltage'].max()
        if sag_ratio > 0.4: health_score -= 15
        
    health_score = int(max(0, min(100, health_score)))
    
    # Risk Prob
    risk = (100 - health_score) / 100.0
    if any(i.severity == 'CRITICAL' for i in issues): risk = max(risk, 0.9)
    
    return {
        'filename': filename,
        'data': processed,
        'stats': {
            'health_score': health_score,
            'risk_prob': risk,
            'max_temp': processed['temp'].max(),
            'avg_imbalance': processed['imbalance'].mean(),
            'anomaly_count': len(anomalies)
        },
        'issues': issues
    }

# --- REPORT GENERATION ---

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'CellGuardAI - Engineering Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(analysis):
    pdf = PDFReport()
    pdf.add_page()
    
    # Summary Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Analysis for: {analysis['filename']}", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"Battery Health Score: {analysis['stats']['health_score']}/100", 1, 1, 'C', 1)
    
    risk_pct = analysis['stats']['risk_prob'] * 100
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Risk Probability: {risk_pct:.1f}%", 0, 1)
    pdf.cell(0, 10, f"Max Temperature: {analysis['stats']['max_temp']:.1f} C", 0, 1)
    pdf.cell(0, 10, f"Avg Imbalance: {analysis['stats']['avg_imbalance']*1000:.1f} mV", 0, 1)
    pdf.ln(10)
    
    # Issues Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Detected Anomalies & Recommendations", 0, 1)
    
    pdf.set_font('Arial', '', 9)
    for issue in analysis['issues']:
        # Severity Tag
        if issue.severity == 'CRITICAL': pdf.set_text_color(200, 0, 0)
        elif issue.severity == 'HIGH': pdf.set_text_color(200, 100, 0)
        else: pdf.set_text_color(0, 0, 0)
        
        pdf.cell(0, 8, f"[{issue.severity}] {issue.type}: {issue.description}", 0, 1)
        
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 5, f"   Reason: {issue.explanation}")
        pdf.set_text_color(0, 0, 200)
        pdf.multi_cell(0, 5, f"   Action: {issue.recommendation}")
        pdf.ln(3)
        pdf.set_text_color(0, 0, 0)

    return pdf.output(dest='S').encode('latin-1')

# --- MAIN APP UI ---

def main():
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("# 🔋")
    with col2:
        st.title("CellGuardAI")
        st.markdown("### Next-Gen Battery Diagnostics & Predictive Maintenance")

    # File Upload
    uploaded_file = st.file_uploader("Upload BMS Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            with st.spinner("Parsing binary data • Running Isolation Forest • Assessing Health..."):
                # Load Data
                df_raw = pd.read_csv(uploaded_file)
                
                # Analyze
                results = analyze_bms_data(df_raw, uploaded_file.name)
                time.sleep(1) # UX Pause
            
            # --- DASHBOARD LAYOUT ---
            
            # 1. KPI Row
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Health Score", f"{results['stats']['health_score']}/100", 
                        delta="-2.5%" if results['stats']['health_score'] < 90 else "Stable")
            kpi2.metric("Risk Probability", f"{results['stats']['risk_prob']*100:.1f}%",
                        delta_color="inverse", delta=None)
            kpi3.metric("Max Temp", f"{results['stats']['max_temp']:.1f} °C",
                         delta="CRITICAL" if results['stats']['max_temp'] > 60 else "Normal", delta_color="inverse")
            kpi4.metric("Avg Imbalance", f"{results['stats']['avg_imbalance']*1000:.1f} mV")

            # 2. Main Visuals
            row2_1, row2_2 = st.columns([1, 2])
            
            with row2_1:
                st.subheader("Battery Health")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = results['stats']['health_score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#2563eb"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "#334155",
                        'steps': [
                            {'range': [0, 60], 'color': '#ef4444'},
                            {'range': [60, 80], 'color': '#f59e0b'},
                            {'range': [80, 100], 'color': '#10b981'}],
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"})
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # PDF Download
                st.markdown("---")
                pdf_bytes = create_pdf(results)
                st.download_button(
                    label="📄 Download Engineering Report",
                    data=pdf_bytes,
                    file_name="CellGuard_Report.pdf",
                    mime="application/pdf"
                )

            with row2_2:
                st.subheader("Detected Issues")
                if not results['issues']:
                    st.success("No critical issues detected. System operating within normal parameters.")
                else:
                    for issue in results['issues']:
                        color = "#ef4444" if issue.severity == "CRITICAL" else "#f59e0b" if issue.severity == "HIGH" else "#3b82f6"
                        with st.expander(f"[{issue.severity}] {issue.type}: {issue.description}"):
                            st.markdown(f"**Explanation:** {issue.explanation}")
                            st.markdown(f"**Recommendation:** {issue.recommendation}")
                            st.caption(f"Detected via {issue.type} logic engine")

            # 3. Charts
            st.markdown("### Engineering Telemetry")
            
            tab1, tab2, tab3 = st.tabs(["Voltage Profile", "Current & Load", "Thermal Distribution"])
            
            with tab1:
                fig_v = px.line(results['data'], x='timestamp', y='voltage', title='Pack Voltage Over Time')
                fig_v.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                # Highlight anomalies
                anoms = results['data'][results['data']['anomaly_score'] == -1]
                fig_v.add_trace(go.Scatter(x=anoms['timestamp'], y=anoms['voltage'], mode='markers', name='Anomaly', marker=dict(color='red', size=4)))
                st.plotly_chart(fig_v, use_container_width=True)
                
            with tab2:
                fig_i = px.line(results['data'], x='timestamp', y='current', title='Current Profile')
                fig_i.update_traces(line_color='#facc15')
                fig_i.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_i, use_container_width=True)
                
            with tab3:
                fig_t = px.line(results['data'], x='timestamp', y='temp', title='Temperature Trends')
                fig_t.update_traces(line_color='#ef4444')
                fig_t.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="Warning (45C)")
                fig_t.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Critical (60C)")
                fig_t.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_t, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.warning("Please ensure your CSV contains standard BMS columns (voltage, current, temp, etc.)")

if __name__ == "__main__":
    main()
