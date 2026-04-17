import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pharmevo Business Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1rem; }

    .kpi-card {
        background: linear-gradient(135deg, #1a1d2e, #2d2d44);
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #00d4ff;
        margin: 8px 0;
    }
    .kpi-label {
        font-size: 13px;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-delta {
        font-size: 14px;
        color: #00ff88;
        font-weight: 600;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a2d1a, #1a1d2e);
        border-left: 4px solid #00ff88;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        color: #e0e0e0;
        font-size: 14px;
    }
    .warning-box {
        background: linear-gradient(135deg, #2d1a1a, #1a1d2e);
        border-left: 4px solid #ff4444;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        color: #e0e0e0;
        font-size: 14px;
    }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #00d4ff;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }
    div[data-testid="stMetricValue"] {
        color: #00d4ff;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_sales    = pd.read_csv('data/processed/sales_clean.csv')
    df_act      = pd.read_csv('data/processed/activities_clean.csv')
    df_merged   = pd.read_csv('data/processed/merged_analysis.csv')
    df_roi      = pd.read_csv('data/processed/roi_analysis.csv')
    df_returns  = pd.read_csv('data/processed/sales_returns.csv')

    with open('data/processed/kpis.json') as f:
        kpis = json.load(f)

    df_sales['Date']  = pd.to_datetime(df_sales['Date'])
    df_act['Date']    = pd.to_datetime(df_act['Date'])

    return df_sales, df_act, df_merged, df_roi, df_returns, kpis

df_sales, df_act, df_merged, df_roi, df_returns, kpis = load_data()

# ── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/pill.png", width=60)
st.sidebar.title("💊 Pharmevo BI")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Executive Summary",
    "📈 Sales Analysis",
    "💰 Promotional Analysis",
    "🔗 Combined ROI Analysis",
    "🔮 Predictions & Forecast",
    "🚨 Alerts & Opportunities"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Filters")

year_filter = st.sidebar.multiselect(
    "Select Year(s)",
    options=sorted(df_sales['Yr'].unique()),
    default=sorted(df_sales['Yr'].unique())
)

team_filter = st.sidebar.multiselect(
    "Select Team(s)",
    options=sorted(df_sales['TeamName'].unique()),
    default=[]
)

# Apply filters
df_s = df_sales[df_sales['Yr'].isin(year_filter)]
df_a = df_act[df_act['Yr'].isin(year_filter)]

if team_filter:
    df_s = df_s[df_s['TeamName'].isin(team_filter)]
    df_a = df_a[df_a['RequestorTeams'].str.upper().isin(
        [t.upper() for t in team_filter])]

st.sidebar.markdown("---")
st.sidebar.caption("Data: Pharmevo SQL Server\nUpdated: March 2026")

# ════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════
if page == "🏠 Executive Summary":

    st.markdown("""
    <h1 style='color:#00d4ff; margin-bottom:0'>
    💊 Pharmevo Business Intelligence Dashboard
    </h1>
    <p style='color:#a0a0b0; font-size:16px'>
    Sales & Promotional Analytics | 2017–2026 | Powered by SQL Server
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── KPI CARDS ROW 1 ─────────────────────────────────────
    st.markdown("### 📊 Key Performance Indicators")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Total Revenue</div>
            <div class='kpi-value'>PKR 47.8B</div>
            <div class='kpi-delta'>↑ +16.6% YoY</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Total Units Sold</div>
            <div class='kpi-value'>153.1M</div>
            <div class='kpi-delta'>2024–2026</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Promo Investment</div>
            <div class='kpi-value'>PKR 7.67B</div>
            <div class='kpi-delta'>↑ +38.2% in 2025</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Overall ROI</div>
            <div class='kpi-value'>20.3x</div>
            <div class='kpi-delta'>PKR 1 → PKR 20.3</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Doctors Targeted</div>
            <div class='kpi-value'>10,040</div>
            <div class='kpi-delta'>Unique Doctors</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI CARDS ROW 2 ─────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Top Product</div>
            <div class='kpi-value' style='font-size:18px'>X-Plended</div>
            <div class='kpi-delta'>PKR 4.3B Revenue</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Top Team</div>
            <div class='kpi-value' style='font-size:18px'>Challengers</div>
            <div class='kpi-delta'>PKR 6.5B Revenue</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Best ROI Product</div>
            <div class='kpi-value' style='font-size:18px'>Ramipace</div>
            <div class='kpi-delta'>99.7x ROI</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Discount Rate</div>
            <div class='kpi-value'>1.5%</div>
            <div class='kpi-delta'>PKR 749M discounts</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Promo Correlation</div>
            <div class='kpi-value'>0.784</div>
            <div class='kpi-delta'>Strong same-month link</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── REVENUE TREND ────────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Revenue Trend</div>",
                unsafe_allow_html=True)

    monthly = df_s.groupby('Date')['TotalRevenue'].sum().reset_index()
    complete = monthly[monthly['Date'].dt.year < 2026]
    partial  = monthly[monthly['Date'].dt.year >= 2026]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=complete['Date'], y=complete['TotalRevenue']/1e6,
        name='Revenue', line=dict(color='#00d4ff', width=2.5),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.1)',
        mode='lines+markers', marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=partial['Date'], y=partial['TotalRevenue']/1e6,
        name='2026 (Partial)', line=dict(color='#ffa500',
        width=2.5, dash='dash'),
        mode='lines+markers', marker=dict(size=6)
    ))
    fig.update_layout(
        plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
        font_color='white', height=350,
        xaxis=dict(gridcolor='#2d2d44', title='Month'),
        yaxis=dict(gridcolor='#2d2d44',
                   title='Revenue (Million PKR)',
                   tickformat='PKR,.0f'),
        legend=dict(bgcolor='#1a1d2e'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── BOTTOM ROW: Products + Teams ─────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>🏆 Top 10 Products</div>",
                    unsafe_allow_html=True)
        top_p = (df_s.groupby('ProductName')['TotalRevenue']
                 .sum().nlargest(10).reset_index())
        fig2 = px.bar(top_p, x='TotalRevenue', y='ProductName',
                      orientation='h',
                      color='TotalRevenue',
                      color_continuous_scale='Blues')
        fig2.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=350,
            yaxis=dict(autorange='reversed'),
            coloraxis_showscale=False,
            xaxis_title='Revenue (PKR)',
            yaxis_title=''
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>👥 Top 10 Teams</div>",
                    unsafe_allow_html=True)
        top_t = (df_s.groupby('TeamName')['TotalRevenue']
                 .sum().nlargest(10).reset_index())
        fig3 = px.bar(top_t, x='TotalRevenue', y='TeamName',
                      orientation='h',
                      color='TotalRevenue',
                      color_continuous_scale='Greens')
        fig3.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=350,
            yaxis=dict(autorange='reversed'),
            coloraxis_showscale=False,
            xaxis_title='Revenue (PKR)',
            yaxis_title=''
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── KEY INSIGHTS ─────────────────────────────────────────
    st.markdown("<div class='section-header'>💡 Key Insights</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='insight-box'>
        🌟 <b>Revenue grew 16.6%</b> from 2024 to 2025
        (PKR 20.2B → PKR 23.6B) — strongest growth in 3 years
        </div>
        <div class='insight-box'>
        🌟 <b>COVID-19 Impact Proven:</b> 2020 saw a 33.7% drop
        in promo spend followed by full recovery in 2021 (+68.6%)
        </div>
        <div class='insight-box'>
        🌟 <b>Promo spend correlation = 0.784</b> in same month —
        marketing campaigns show immediate sales impact
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='insight-box'>
        🌟 <b>Ramipace ROI = 99.7x</b> — only PKR 4.3M spent
        generates PKR 430M in revenue. Massively underinvested!
        </div>
        <div class='warning-box'>
        ⚠️ <b>Shevit Budget Alert:</b> PKR 29M spent but only
        5.6x ROI — 17x less efficient than Ramipace
        </div>
        <div class='warning-box'>
        ⚠️ <b>2026 data is partial</b> (Jan–Mar only).
        Do not compare directly with full years.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 2: SALES ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📈 Sales Analysis":

    st.markdown("<h2 style='color:#00d4ff'>📈 Sales Deep Analysis</h2>",
                unsafe_allow_html=True)

    # ── YEARLY COMPARISON ────────────────────────────────────
    st.markdown("<div class='section-header'>Year-over-Year Comparison</div>",
                unsafe_allow_html=True)

    yearly = (df_s[df_s['Yr'] < 2026]
              .groupby('Yr').agg(
                  Revenue=('TotalRevenue','sum'),
                  Units=('TotalUnits','sum'),
                  Invoices=('InvoiceCount','sum')
              ).reset_index())

    c1, c2, c3 = st.columns(3)
    figs = [
        ('Revenue (PKR)', 'Revenue', '#00d4ff'),
        ('Units Sold',    'Units',   '#00ff88'),
        ('Invoices',      'Invoices','#ffa500')
    ]
    for col, (title, field, color) in zip([c1,c2,c3], figs):
        with col:
            fig = px.bar(yearly, x='Yr', y=field,
                         title=title,
                         color_discrete_sequence=[color])
            fig.update_layout(
                plot_bgcolor='#1a1d2e',
                paper_bgcolor='#1a1d2e',
                font_color='white', height=280,
                xaxis=dict(gridcolor='#2d2d44',
                           tickmode='array',
                           tickvals=yearly['Yr']),
                yaxis=dict(gridcolor='#2d2d44')
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── PRODUCT ANALYSIS ─────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Product Revenue 2024 vs 2025</div>",
                    unsafe_allow_html=True)
        rev_yr = (df_s[df_s['Yr'].isin([2024,2025])]
                  .groupby(['ProductName','Yr'])['TotalRevenue']
                  .sum().reset_index())
        top15 = (rev_yr.groupby('ProductName')['TotalRevenue']
                 .sum().nlargest(15).index)
        rev_yr = rev_yr[rev_yr['ProductName'].isin(top15)]

        fig = px.bar(rev_yr, x='TotalRevenue', y='ProductName',
                     color='Yr', barmode='group',
                     orientation='h',
                     color_discrete_map={2024:'#00d4ff',2025:'#00ff88'})
        fig.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=500,
            yaxis=dict(autorange='reversed'),
            xaxis_title='Revenue (PKR)', yaxis_title=''
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Fastest Growing Products 2024→2025</div>",
                    unsafe_allow_html=True)

        rev24 = (df_s[df_s['Yr']==2024]
                 .groupby('ProductName')['TotalRevenue'].sum())
        rev25 = (df_s[df_s['Yr']==2025]
                 .groupby('ProductName')['TotalRevenue'].sum())
        growth = pd.DataFrame({'2024':rev24,'2025':rev25}).dropna()
        growth = growth[growth['2024'] > 5000000]
        growth['Growth%'] = ((growth['2025']-growth['2024'])
                              /growth['2024']*100)
        growth = growth.sort_values('Growth%',
                                    ascending=False).head(15).reset_index()

        fig = px.bar(growth, x='Growth%', y='ProductName',
                     orientation='h',
                     color='Growth%',
                     color_continuous_scale='Greens',
                     title='Growth % 2024→2025')
        fig.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=500,
            yaxis=dict(autorange='reversed'),
            coloraxis_showscale=False,
            xaxis_title='Growth %', yaxis_title=''
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── SEASONALITY ──────────────────────────────────────────
    st.markdown("<div class='section-header'>📅 Sales Seasonality Heatmap</div>",
                unsafe_allow_html=True)

    months_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',
                  6:'Jun',7:'Jul',8:'Aug',9:'Sep',
                  10:'Oct',11:'Nov',12:'Dec'}

    heat = (df_s[df_s['Yr'] < 2026]
            .groupby(['Yr','Mo'])['TotalRevenue']
            .sum().reset_index())
    heat['Month'] = heat['Mo'].map(months_map)
    heat_pivot = heat.pivot(index='Yr', columns='Month',
                            values='TotalRevenue')
    month_order = list(months_map.values())
    heat_pivot = heat_pivot.reindex(columns=month_order)

    fig = px.imshow(heat_pivot/1e6,
                    color_continuous_scale='Blues',
                    aspect='auto',
                    labels=dict(color='Revenue (M PKR)'))
    fig.update_layout(
        plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
        font_color='white', height=250,
        xaxis_title='Month', yaxis_title='Year'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Darker = higher revenue. Oct–Dec consistently strongest months.")


# ════════════════════════════════════════════════════════════
# PAGE 3: PROMOTIONAL ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "💰 Promotional Analysis":

    st.markdown("<h2 style='color:#00d4ff'>💰 Promotional Spend Analysis</h2>",
                unsafe_allow_html=True)

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    total_spend_filtered = df_a['TotalAmount'].sum()
    with c1:
        st.metric("Total Promo Spend",
                  f"PKR {total_spend_filtered/1e9:.2f}B")
    with c2:
        st.metric("Total Requests",
                  f"{df_a['RequestCount'].sum():,.0f}")
    with c3:
        st.metric("Avg per Request",
                  f"PKR {total_spend_filtered/df_a['RequestCount'].sum():,.0f}")
    with c4:
        st.metric("Peak Year", "2025",
                  delta="PKR 1.37B (+38.2%)")

    st.markdown("---")

    # ── SPEND TREND ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Promotional Spend Trend (2017–2026)</div>",
                    unsafe_allow_html=True)
        yearly_sp = df_a.groupby('Yr')['TotalAmount'].sum().reset_index()
        fig = px.bar(yearly_sp, x='Yr', y='TotalAmount',
                     color='TotalAmount',
                     color_continuous_scale='Blues')
        fig.add_scatter(x=yearly_sp['Yr'],
                        y=yearly_sp['TotalAmount'],
                        mode='lines+markers',
                        line=dict(color='#00ff88', width=2),
                        name='Trend')
        fig.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=320,
            xaxis=dict(gridcolor='#2d2d44',
                       tickmode='array',
                       tickvals=yearly_sp['Yr']),
            yaxis=dict(gridcolor='#2d2d44',
                       title='Amount (PKR)'),
            coloraxis_showscale=False,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Spend by Activity Type (Top 10)</div>",
                    unsafe_allow_html=True)
        act_sp = (df_a.groupby('ActivityHead')['TotalAmount']
                  .sum().nlargest(10).reset_index())
        fig = px.pie(act_sp, values='TotalAmount',
                     names='ActivityHead',
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TEAM & PRODUCT SPEND ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Top 10 Teams by Promo Spend</div>",
                    unsafe_allow_html=True)
        team_sp = (df_a.groupby('RequestorTeams')['TotalAmount']
                   .sum().nlargest(10).reset_index())
        fig = px.bar(team_sp, x='TotalAmount',
                     y='RequestorTeams',
                     orientation='h',
                     color='TotalAmount',
                     color_continuous_scale='Oranges')
        fig.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=350,
            yaxis=dict(autorange='reversed'),
            coloraxis_showscale=False,
            xaxis_title='Total Spend (PKR)', yaxis_title=''
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Top 10 Products by Promo Investment</div>",
                    unsafe_allow_html=True)
        prod_sp = (df_a.groupby('Product')['TotalAmount']
                   .sum().nlargest(10).reset_index())
        fig = px.bar(prod_sp, x='TotalAmount', y='Product',
                     orientation='h',
                     color='TotalAmount',
                     color_continuous_scale='Purples')
        fig.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=350,
            yaxis=dict(autorange='reversed'),
            coloraxis_showscale=False,
            xaxis_title='Total Spend (PKR)', yaxis_title=''
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── GL HEAD BREAKDOWN ────────────────────────────────────
    st.markdown("<div class='section-header'>Budget Allocation by GL Head</div>",
                unsafe_allow_html=True)
    gl_sp = (df_a.groupby('GLHead')['TotalAmount']
             .sum().nlargest(8).reset_index())
    gl_sp['Pct'] = (gl_sp['TotalAmount'] /
                    gl_sp['TotalAmount'].sum() * 100).round(1)

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type":"bar"},{"type":"pie"}]])
    fig.add_trace(go.Bar(
        x=gl_sp['TotalAmount'], y=gl_sp['GLHead'],
        orientation='h',
        marker_color='#00d4ff', name='Amount'
    ), row=1, col=1)
    fig.add_trace(go.Pie(
        values=gl_sp['TotalAmount'],
        labels=gl_sp['GLHead'],
        name='Share'
    ), row=1, col=2)
    fig.update_layout(
        plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
        font_color='white', height=350, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 4: COMBINED ROI ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "🔗 Combined ROI Analysis":

    st.markdown("<h2 style='color:#00d4ff'>🔗 Combined ROI Analysis</h2>",
                unsafe_allow_html=True)
    st.markdown("*Linking promotional spend (2017–2026) with actual sales (2024–2026)*")

    # Correlation callout
    st.markdown("""
    <div class='insight-box' style='font-size:16px'>
    🔬 <b>Key Finding:</b> Promotional spend and same-month revenue 
    have a <b>0.784 correlation</b> — a STRONG positive relationship.
    Every PKR 1 invested in promotions generates PKR 20.3 in revenue on average.
    </div>""", unsafe_allow_html=True)

    # ── SPEND VS REVENUE ────────────────────────────────────
    st.markdown("<div class='section-header'>Promo Spend vs Revenue (Monthly)</div>",
                unsafe_allow_html=True)

    monthly_sp = df_a.groupby('Date')['TotalAmount'].sum().reset_index()
    monthly_rv = df_s.groupby('Date')['TotalRevenue'].sum().reset_index()
    combo = pd.merge(monthly_sp, monthly_rv, on='Date', how='inner')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=combo['Date'], y=combo['TotalAmount']/1e6,
        name='Promo Spend (M PKR)',
        marker_color='rgba(255,165,0,0.7)'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=combo['Date'], y=combo['TotalRevenue']/1e6,
        name='Revenue (M PKR)',
        line=dict(color='#00d4ff', width=3),
        mode='lines+markers'
    ), secondary_y=True)
    fig.update_layout(
        plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
        font_color='white', height=350,
        hovermode='x unified',
        legend=dict(bgcolor='#1a1d2e')
    )
    fig.update_yaxes(title_text="Promo Spend (M PKR)",
                     gridcolor='#2d2d44', secondary_y=False)
    fig.update_yaxes(title_text="Revenue (M PKR)",
                     gridcolor='#2d2d44', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── ROI SCATTER ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>ROI by Product (Bubble Chart)</div>",
                    unsafe_allow_html=True)
        roi_plot = df_roi[df_roi['TotalPromoSpend'] > 0].copy()
        roi_plot = roi_plot[roi_plot['ROI'] < 200]

        fig = px.scatter(roi_plot,
                         x='TotalPromoSpend',
                         y='TotalRevenue',
                         size='ROI',
                         color='ROI',
                         hover_name='ProductName',
                         color_continuous_scale='RdYlGn',
                         size_max=50,
                         labels={
                             'TotalPromoSpend':'Promo Spend (PKR)',
                             'TotalRevenue':'Revenue (PKR)'
                         })
        fig.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Bigger bubble = higher ROI. Top-left = high ROI low spend = opportunity!")

    with col2:
        st.markdown("<div class='section-header'>Top 15 Products by ROI</div>",
                    unsafe_allow_html=True)
        top_roi = df_roi.nlargest(15,'ROI')
        colors = ['#00ff88' if r > 50 else
                  '#00d4ff' if r > 20 else
                  '#ffa500' for r in top_roi['ROI']]
        fig = go.Figure(go.Bar(
            x=top_roi['ROI'],
            y=top_roi['ProductName'],
            orientation='h',
            marker_color=colors,
            text=[f"{r:.1f}x" for r in top_roi['ROI']],
            textposition='outside'
        ))
        fig.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=400,
            yaxis=dict(autorange='reversed'),
            xaxis_title='ROI (Revenue / Promo Spend)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TEAM ROI TABLE ───────────────────────────────────────
    st.markdown("<div class='section-header'>Team ROI Summary</div>",
                unsafe_allow_html=True)

    team_roi_data = {
        'Team': ['CHALLENGERS','METABOLIZERS','BONE SAVIORS',
                 'LEGENDS','WARRIORS','BRAVO',
                 'WINNERS','TITANS','ALPHA','CHAMPIONS'],
        'Promo Spend (PKR)': [118634409,81731983,133620048,
                               78147367,75495400,44872269,
                               67180420,101703110,61419102,37460330],
        'Revenue (PKR)': [4533914172,2380049006,2323743063,
                          2096411088,1586697007,1521194946,
                          1494926401,1331012729,1107743422,1074352302],
        'ROI': [38.2,29.1,17.4,26.8,21.0,33.9,22.3,13.1,18.0,28.7]
    }
    team_roi_df = pd.DataFrame(team_roi_data)
    team_roi_df['Promo Spend (PKR)'] = team_roi_df['Promo Spend (PKR)'].apply(
        lambda x: f"PKR {x:,.0f}")
    team_roi_df['Revenue (PKR)'] = team_roi_df['Revenue (PKR)'].apply(
        lambda x: f"PKR {x:,.0f}")
    team_roi_df['ROI'] = team_roi_df['ROI'].apply(lambda x: f"{x:.1f}x")
    st.dataframe(team_roi_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# PAGE 5: PREDICTIONS
# ════════════════════════════════════════════════════════════
elif page == "🔮 Predictions & Forecast":

    st.markdown("<h2 style='color:#00d4ff'>🔮 Sales Prediction & Forecast</h2>",
                unsafe_allow_html=True)

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    # ── PREPARE ML DATA ──────────────────────────────────────
    df_ml = df_merged.copy()
    df_ml = df_ml[df_ml['Revenue'] > 0]
    df_ml = df_ml[df_ml['PromoSpend'] > 0]

    # Features
    le_prod = LabelEncoder()
    le_team = LabelEncoder()
    df_ml['Product_enc'] = le_prod.fit_transform(df_ml['ProductName'])
    df_ml['Team_enc']    = le_team.fit_transform(df_ml['TeamName'])
    df_ml['Month_sin']   = np.sin(2*np.pi*df_ml['Mo']/12)
    df_ml['Month_cos']   = np.cos(2*np.pi*df_ml['Mo']/12)

    features = ['PromoSpend','Requests','Product_enc',
                'Team_enc','Mo','Yr','Month_sin','Month_cos']
    X = df_ml[features]
    y = df_ml['Revenue']

    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # ── TRAIN 3 MODELS ───────────────────────────────────────
    models = {
        'Linear Regression'     : LinearRegression(),
        'Random Forest'         : RandomForestRegressor(
            n_estimators=100, random_state=42),
        'Gradient Boosting'     : GradientBoostingRegressor(
            n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            'model'  : model,
            'preds'  : preds,
            'r2'     : r2_score(y_test, preds),
            'mae'    : mean_absolute_error(y_test, preds)
        }

    # ── MODEL COMPARISON ─────────────────────────────────────
    st.markdown("<div class='section-header'>Model Comparison</div>",
                unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    colors_model = ['#ffa500','#00d4ff','#00ff88']
    for col, (name, res), color in zip(
            [c1,c2,c3], results.items(), colors_model):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>{name}</div>
                <div class='kpi-value' style='color:{color}'>
                    R² = {res['r2']:.3f}
                </div>
                <div class='kpi-delta'>
                    MAE = PKR {res['mae']:,.0f}
                </div>
            </div>""", unsafe_allow_html=True)

    # Best model
    best_name = max(results, key=lambda k: results[k]['r2'])
    best_res  = results[best_name]
    best_model= best_res['model']

    st.markdown(f"""
    <div class='insight-box' style='margin-top:15px; font-size:15px'>
    🏆 <b>Best Model: {best_name}</b> — R² = {best_res['r2']:.3f}<br>
    This means the model explains {best_res['r2']*100:.1f}% of 
    variance in sales revenue using promotional data.
    </div>""", unsafe_allow_html=True)

    # ── ACTUAL VS PREDICTED ───────────────────────────────────
    st.markdown("<div class='section-header'>Actual vs Predicted Revenue</div>",
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))),
        y=y_test.values/1e6,
        name='Actual', mode='lines',
        line=dict(color='#00d4ff', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))),
        y=best_res['preds']/1e6,
        name='Predicted', mode='lines',
        line=dict(color='#00ff88', width=2, dash='dot')
    ))
    fig.update_layout(
        plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
        font_color='white', height=320,
        xaxis=dict(gridcolor='#2d2d44', title='Sample Index'),
        yaxis=dict(gridcolor='#2d2d44', title='Revenue (M PKR)'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── FEATURE IMPORTANCE ────────────────────────────────────
    if hasattr(best_model, 'feature_importances_'):
        st.markdown("<div class='section-header'>What Drives Sales? (Feature Importance)</div>",
                    unsafe_allow_html=True)
        fi = pd.DataFrame({
            'Feature'   : features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig = px.bar(fi, x='Importance', y='Feature',
                     orientation='h',
                     color='Importance',
                     color_continuous_scale='Blues')
        fig.update_layout(
            plot_bgcolor='#1a1d2e', paper_bgcolor='#1a1d2e',
            font_color='white', height=300,
            yaxis=dict(autorange='reversed'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── FORECAST SIMULATOR ───────────────────────────────────
    st.markdown("<div class='section-header'>🎯 Revenue Forecast Simulator</div>",
                unsafe_allow_html=True)
    st.markdown("*Predict revenue based on planned promotional spend*")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sim_spend = st.number_input(
            "Promo Spend (PKR)",
            min_value=100000,
            max_value=50000000,
            value=5000000,
            step=500000)
    with col2:
        sim_month = st.selectbox(
            "Month",
            options=list(range(1,13)),
            format_func=lambda x: list(months_map.values())[x-1]
            if 'months_map' in dir() else str(x))
    with col3:
        sim_year = st.selectbox("Year", [2025, 2026])
    with col4:
        sim_requests = st.number_input(
            "No. of Requests", 1, 100, 10)

    if st.button("🔮 Predict Revenue", type="primary"):
        sim_input = pd.DataFrame([{
            'PromoSpend' : sim_spend,
            'Requests'   : sim_requests,
            'Product_enc': 0,
            'Team_enc'   : 0,
            'Mo'         : sim_month,
            'Yr'         : sim_year,
            'Month_sin'  : np.sin(2*np.pi*sim_month/12),
            'Month_cos'  : np.cos(2*np.pi*sim_month/12)
        }])
        predicted = best_model.predict(sim_input)[0]
        roi_sim   = predicted / sim_spend

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Predicted Revenue</div>
                <div class='kpi-value'>PKR {predicted/1e6:.1f}M</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Expected ROI</div>
                <div class='kpi-value'>{roi_sim:.1f}x</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>Model Used</div>
                <div class='kpi-value' style='font-size:16px'>
                {best_name}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 6: ALERTS & OPPORTUNITIES
# ════════════════════════════════════════════════════════════
elif page == "🚨 Alerts & Opportunities":

    st.markdown("<h2 style='color:#00d4ff'>🚨 Alerts & Strategic Opportunities</h2>",
                unsafe_allow_html=True)

    # ── OPPORTUNITIES ─────────────────────────────────────────
    st.markdown("<div class='section-header'>🌟 Hidden Opportunities — Underinvested Products</div>",
                unsafe_allow_html=True)
    st.markdown("*High ROI but low promo spend — these deserve more budget!*")

    opp = df_roi[
        (df_roi['ROI'] > 20) &
        (df_roi['TotalPromoSpend'] < df_roi['TotalPromoSpend'].median())
    ].sort_values('ROI', ascending=False).head(10)

    for _, row in opp.iterrows():
        potential = row['ROI'] * row['TotalPromoSpend'] * 2
        st.markdown(f"""
        <div class='insight-box'>
        🌟 <b>{row['ProductName']}</b> — ROI: <b>{row['ROI']:.1f}x</b><br>
        Current spend: PKR {row['TotalPromoSpend']:,.0f} →
        Revenue: PKR {row['TotalRevenue']:,.0f}<br>
        <i>💡 Doubling spend could generate ~PKR {potential:,.0f} revenue</i>
        </div>""", unsafe_allow_html=True)

    # ── WARNINGS ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>⚠️ Budget Waste Alerts — Low ROI Products</div>",
                unsafe_allow_html=True)
    st.markdown("*High spend but low return — strategy review needed!*")

    waste = df_roi[
        (df_roi['ROI'] < 10) &
        (df_roi['TotalPromoSpend'] > df_roi['TotalPromoSpend'].median())
    ].sort_values('TotalPromoSpend', ascending=False).head(5)

    for _, row in waste.iterrows():
        wasted = row['TotalPromoSpend'] * (1 - row['ROI']/20)
        st.markdown(f"""
        <div class='warning-box'>
        ⚠️ <b>{row['ProductName']}</b> — ROI: <b>{row['ROI']:.1f}x</b>
        (vs 20x company average)<br>
        Spent: PKR {row['TotalPromoSpend']:,.0f} →
        Revenue: PKR {row['TotalRevenue']:,.0f}<br>
        <i>🔍 Consider reviewing promotional strategy for this product</i>
        </div>""", unsafe_allow_html=True)

    # ── STRATEGIC RECOMMENDATIONS ────────────────────────────
    st.markdown("<div class='section-header'>📋 Strategic Recommendations</div>",
                unsafe_allow_html=True)

    recs = [
        ("Reallocate Budget", "Move 20% of Shevit/Ferfer budget to Ramipace/Xcept — could add PKR 500M+ in revenue", "insight-box"),
        ("Scale X-Plended", "Top revenue product at PKR 4.3B with 21.9% growth — increase promo investment", "insight-box"),
        ("Focus on Oct-Dec", "Historically strongest sales months — concentrate promotional events here", "insight-box"),
        ("Team Challengers", "Highest ROI team at 38.2x — replicate their strategy across other teams", "insight-box"),
        ("Finno-Q Alert", "226% growth in 2025 — emerging product needs immediate promotional support", "insight-box"),
        ("COVID Resilience", "2021 recovery showed +68.6% bounce — maintain emergency promo budget reserve", "insight-box"),
    ]
    for title, desc, cls in recs:
        st.markdown(f"""
        <div class='{cls}'>
        <b>{title}:</b> {desc}
        </div>""", unsafe_allow_html=True)

    # ── QUICK WINS TABLE ─────────────────────────────────────
    st.markdown("<div class='section-header'>⚡ Quick Wins Summary Table</div>",
                unsafe_allow_html=True)

    quick = pd.DataFrame({
        'Action'         : ['Increase Ramipace budget 2x',
                            'Increase Xcept budget 2x',
                            'Cut Shevit budget 50%',
                            'Cut Ferfer budget 30%',
                            'Boost Oct-Dec campaigns'],
        'Current Spend'  : ['PKR 4.3M','PKR 5.2M',
                            'PKR 29M','PKR 47M','Varies'],
        'Expected Impact': ['+PKR 430M revenue','+PKR 395M revenue',
                            'Save PKR 14.5M','Save PKR 14.2M',
                            '+15% Q4 revenue'],
        'Priority'       : ['🔴 HIGH','🔴 HIGH',
                            '🟡 MEDIUM','🟡 MEDIUM','🟢 LOW']
    })
    st.dataframe(quick, use_container_width=True, hide_index=True)
