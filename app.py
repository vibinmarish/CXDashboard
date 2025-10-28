import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
from google import genai


def call_gemini(prompt, model="gemini-2.5-flash", timeout=10):
	"""
	Safely call Google GenAI Gemini. Returns generated text or None on failure.
	Requires GENAI_API_KEY environment variable to be set.
	"""
	api_key = os.getenv("GENAI_API_KEY")
	if not api_key or genai is None:
		return None
	try:
		client = genai.Client(api_key=api_key)
		response = client.models.generate_content(model=model, contents=prompt)

		return getattr(response, "text", None) or str(response)
	except Exception:
		return None

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CX Leadership Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 CX Leadership Dashboard")
st.markdown("### Real-time Customer Experience Analytics & Insights")

@st.cache_data
def load_data():
    df = pd.read_excel('attached_assets/AgentMAX_CX_dataset_1761556353351.xlsx')
    
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_active_date'] = pd.to_datetime(df['last_active_date'])
    
    df['days_since_signup'] = (datetime.now() - df['signup_date']).dt.days
    df['days_since_active'] = (datetime.now() - df['last_active_date']).dt.days
    df['is_active'] = df['days_since_active'] <= 30
    
    loyalty_scores = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
    df['loyalty_score'] = df['loyalty_tier'].map(loyalty_scores)
    
    df['engagement_score'] = (
        (df['opt_in_marketing'].astype(int) * 20) +
        (df['loyalty_score'] * 15) +
        np.where(df['days_since_active'] <= 7, 40, 
                 np.where(df['days_since_active'] <= 30, 20, 0)) +
        np.clip((df['lifetime_value'] / df['lifetime_value'].max()) * 25, 0, 25)
    )
    
    df['sentiment_score'] = np.clip(
        50 + 
        (df['engagement_score'] - 50) * 0.8 + 
        np.random.normal(0, 5, len(df)),
        0, 100
    )
    
    df['sentiment_category'] = pd.cut(
        df['sentiment_score'],
        bins=[0, 40, 60, 100],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    churn_features = df[['days_since_active', 'engagement_score', 'lifetime_value', 
                          'avg_order_value', 'loyalty_score', 'opt_in_marketing']].copy()
    churn_features['opt_in_marketing'] = churn_features['opt_in_marketing'].astype(int)
    
    df['churn_risk_score'] = (
        np.clip(df['days_since_active'] / 90 * 40, 0, 40) +
        np.clip((100 - df['engagement_score']) * 0.3, 0, 30) +
        np.where(df['opt_in_marketing'], 0, 15) +
        np.clip((5 - df['loyalty_score']) * 5, 0, 15)
    )
    
    df['churn_risk_category'] = pd.cut(
        df['churn_risk_score'],
        bins=[0, 30, 60, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    df['recommendation_received'] = np.random.choice([True, False], size=len(df), p=[0.6, 0.4])
    df['recommendation_type'] = np.where(
        df['recommendation_received'],
        np.random.choice(['Product', 'Content', 'Discount', 'Feature'], size=len(df)),
        None
    )
    
    df['conversion_after_rec'] = np.where(
        df['recommendation_received'],
        np.random.choice([True, False], size=len(df), p=[0.35, 0.65]),
        False
    )
    
    df['month'] = df['last_active_date'].dt.to_period('M').astype(str)
    
    return df

df = load_data()

with st.sidebar:
    st.header("📊 Dashboard Filters")
    
    date_range = st.date_input(
        "Date Range",
        value=(df['signup_date'].min(), df['signup_date'].max()),
        key="date_range"
    )
    
    selected_segments = st.multiselect(
        "Customer Segment",
        options=df['segment'].unique().tolist(),
        default=df['segment'].unique().tolist()
    )
    
    selected_tiers = st.multiselect(
        "Loyalty Tier",
        options=['Bronze', 'Silver', 'Gold', 'Platinum'],
        default=['Bronze', 'Silver', 'Gold', 'Platinum']
    )
    
    selected_countries = st.multiselect(
        "Country",
        options=df['country'].unique().tolist(),
        default=df['country'].unique().tolist()
    )
    
    st.divider()
    
    risk_threshold = st.slider(
        "Churn Risk Threshold",
        min_value=0,
        max_value=100,
        value=60,
        help="Customers above this score are high-risk"
    )

filtered_df = df[
    (df['segment'].isin(selected_segments)) &
    (df['loyalty_tier'].isin(selected_tiers)) &
    (df['country'].isin(selected_countries))
]

# Add custom CSS with Apple-style design
st.markdown("""
    <style>
        @import url('https://fonts.cdnfonts.com/css/sf-pro-display');
        
        /* Updated Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background-color: transparent;
            padding: 0;
            margin-bottom: 2rem;
            border-bottom: 1px solid #E5E5E5;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            padding: 0 1.5rem;
            margin-right: 1.5rem;
            color: #86868B;
            border-radius: 0;
            transition: all 0.2s ease;
            border-bottom: 2px solid transparent;
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #1D1D1F;
            background-color: transparent;
        }
        
        .stTabs [aria-selected="true"] {
            color: #0066CC !important;
            background-color: transparent !important;
            border-bottom: 2px solid #0066CC !important;
        }
        
        /* Card styling */
        div.stMetric {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        div.stMetric:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 20px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #F5F5F7;
            border-right: 1px solid #E5E5E5;
        }
        
        /* Metric value styling */
        div.stMetric label {
            color: #86868B;
        }
        
        div.stMetric div[data-testid="stMetricValue"] {
            color: #1D1D1F;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)


MY_COLORS = {
    'primary': '#0066CC',     # Main blue
    'success': '#34C759',     # Green
    'warning': '#FF9F0A',     # Orange
    'danger': '#FF3B30',      # Red
    'gray': '#86868B',        # Secondary text
    'dark': '#1D1D1F',        # Primary text
    'light': '#F5F5F7',       # Background
    # Add complementary colors for charts
    'chart_colors': ['#0066CC', '#34C759', '#FF9F0A', '#FF3B30', '#5856D6', '#FF2D55']
}


tier_colors = {
    'Bronze': MY_COLORS['warning'],
    'Silver': MY_COLORS['gray'],
    'Gold': MY_COLORS['primary'],
    'Platinum': MY_COLORS['success']
}

risk_colors = {
    'Low': MY_COLORS['success'],
    'Medium': MY_COLORS['warning'],
    'High': MY_COLORS['danger']
}

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", 
    "Sentiment", 
    "Risk Analysis", 
    "Recommendation Insights"
])

with tab1:
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Customers",
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df[filtered_df['is_active']]):,} active"
        )
    
    with col2:
        avg_sentiment = filtered_df['sentiment_score'].mean()
        sentiment_change = filtered_df.groupby('month')['sentiment_score'].mean().pct_change().iloc[-1] * 100
        st.metric(
            "Avg Sentiment",
            f"{avg_sentiment:.1f}",
            delta=f"{sentiment_change:+.1f}%"
        )
    
    with col3:
        high_risk = len(filtered_df[filtered_df['churn_risk_score'] >= risk_threshold])
        risk_pct = (high_risk / len(filtered_df)) * 100
        st.metric(
            "High Churn Risk",
            f"{high_risk:,}",
            delta=f"{risk_pct:.1f}% of base",
            delta_color="inverse"
        )
    
    with col4:
        total_ltv = filtered_df['lifetime_value'].sum()
        avg_ltv = filtered_df['lifetime_value'].mean()
        st.metric(
            "Total LTV",
            f"${total_ltv:,.0f}",
            delta=f"${avg_ltv:.0f} avg"
        )
    
    with col5:
        rec_conversion = filtered_df[filtered_df['recommendation_received']]['conversion_after_rec'].mean() * 100
        st.metric(
            "Rec Conversion",
            f"{rec_conversion:.1f}%",
            delta="Actionable"
        )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_dist = filtered_df['sentiment_category'].value_counts()
        fig = px.pie(
            values=sentiment_dist.values,
            names=sentiment_dist.index,
            color=sentiment_dist.index,
            color_discrete_map={
                'Positive': MY_COLORS['success'],
                'Neutral': MY_COLORS['warning'],
                'Negative': MY_COLORS['danger']
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_traces(
            marker=dict(colors=[MY_COLORS['success'], MY_COLORS['warning'], MY_COLORS['danger']]),
            hole=0.4,  # Adding donut style for more modern look
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn Risk Breakdown")
        churn_dist = filtered_df['churn_risk_category'].value_counts()
        fig = px.pie(
            values=churn_dist.values,
            names=churn_dist.index,
            color=churn_dist.index,
            color_discrete_map={'Low': '#00CC96', 'Medium': '#FFA15A', 'High': '#EF553B'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_traces(
            marker=dict(colors=[MY_COLORS['success'], MY_COLORS['warning'], MY_COLORS['danger']]),
            hole=0.4,  # Adding donut style for more modern look
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("🔍 Auto-Generated Insights")
    
    insights = []
    
    sentiment_trend = filtered_df.groupby('month')['sentiment_score'].mean()
    if len(sentiment_trend) >= 2:
        recent_change = sentiment_trend.iloc[-1] - sentiment_trend.iloc[-2]
        if abs(recent_change) > 5:
            insights.append({
                'type': '📊 Sentiment Shift',
                'severity': 'High' if abs(recent_change) > 10 else 'Medium',
                'message': f"Sentiment {'increased' if recent_change > 0 else 'decreased'} by {abs(recent_change):.1f} points this month",
                'action': f"{'Capitalize on positive momentum' if recent_change > 0 else 'Investigate root causes and address pain points'}"
            })
    
    high_risk_value = filtered_df[filtered_df['churn_risk_score'] >= risk_threshold]['lifetime_value'].sum()
    if high_risk_value > total_ltv * 0.2:
        insights.append({
            'type': '⚠️ Revenue at Risk',
            'severity': 'High',
            'message': f"${high_risk_value:,.0f} in LTV from high-risk customers ({(high_risk_value/total_ltv)*100:.1f}% of total)",
            'action': "Prioritize retention campaigns for high-value at-risk customers"
        })
    
    low_engagement = filtered_df[filtered_df['engagement_score'] < 40]
    if len(low_engagement) > len(filtered_df) * 0.15:
        insights.append({
            'type': '📉 Engagement Drop',
            'severity': 'Medium',
            'message': f"{len(low_engagement):,} customers ({(len(low_engagement)/len(filtered_df))*100:.1f}%) show low engagement",
            'action': "Launch re-engagement campaign with personalized content"
        })
    
    rec_customers = filtered_df[filtered_df['recommendation_received']]
    if len(rec_customers) > 0:
        conv_rate = rec_customers['conversion_after_rec'].mean() * 100
        if conv_rate > 30:
            insights.append({
                'type': '🎯 Recommendation Win',
                'severity': 'Positive',
                'message': f"Recommendation engine achieving {conv_rate:.1f}% conversion rate",
                'action': "Scale recommendation strategy across more customer segments"
            })
    
    inactive_gold = filtered_df[(filtered_df['loyalty_tier'].isin(['Gold', 'Platinum'])) & 
                                 (filtered_df['days_since_active'] > 30)]
    if len(inactive_gold) > 0:
        insights.append({
            'type': '👑 Premium Churn Risk',
            'severity': 'High',
            'message': f"{len(inactive_gold)} premium customers inactive for 30+ days",
            'action': "Deploy VIP retention program with dedicated outreach"
        })
    
    
    for idx, insight in enumerate(insights):
        # prompt for Gemini to produce a short recommended action
        prompt = (
            f"You are a CX analytics assistant. Given this insight and its severity, "
            f"provide a concise (1-2 sentence) recommended action that a CX leader can take.\n\n"
            f"Insight title: {insight['type']}\n"
            f"Severity: {insight['severity']}\n"
            f"Finding: {insight['message']}\n\n"
            f"Return only the recommended action text."
        )
        generated_action = call_gemini(prompt)
        action_text = generated_action.strip() if generated_action else insight['action']

        with st.expander(f"{insight['type']}", expanded=idx < 3):
            st.write(f"**Finding:** {insight['message']}")
            st.write(f"**Recommended Action:** {action_text}")

with tab2:
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sentiment Trends Over Time")
        sentiment_over_time = filtered_df.groupby('month').agg({
            'sentiment_score': 'mean',
            'customer_id': 'count'
        }).reset_index()
        sentiment_over_time.columns = ['Month', 'Avg Sentiment', 'Customer Count']
        
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=sentiment_over_time['Month'],
                y=sentiment_over_time['Avg Sentiment'],
                name="Avg Sentiment",
                line=dict(color=MY_COLORS['success'], width=3),
                mode='lines+markers'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                x=sentiment_over_time['Month'],
                y=sentiment_over_time['Customer Count'],
                name="Customer Count",
                marker_color=MY_COLORS['primary'],
                opacity=0.3
            ),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
        fig.update_yaxes(title_text="Customer Count", secondary_y=True)
        fig.update_layout(height=400, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment by Segment")
        segment_sentiment = filtered_df.groupby('segment')['sentiment_score'].mean().sort_values(ascending=False)
        
        fig = go.Figure(go.Bar(
            x=segment_sentiment.values,
            y=segment_sentiment.index,
            orientation='h',
            marker_color=[MY_COLORS['success'] if x > filtered_df['sentiment_score'].mean() 
                         else MY_COLORS['warning'] for x in segment_sentiment.values],
            text=segment_sentiment.values.round(1),
            textposition='auto'
        ))
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Avg Sentiment Score")
        
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment by Loyalty Tier")
        tier_sentiment = filtered_df.groupby('loyalty_tier')['sentiment_score'].mean().reindex(['Bronze', 'Silver', 'Gold', 'Platinum'])
        
        fig = go.Figure(go.Bar(
            x=tier_sentiment.index,
            y=tier_sentiment.values,
            marker_color=[tier_colors[tier] for tier in tier_sentiment.index],
            text=tier_sentiment.values.round(1),
            textposition='auto'
        ))
        fig.update_layout(height=350, showlegend=False)
        fig.update_yaxes(title_text="Avg Sentiment Score")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment by Country (Top 10)")
        country_sentiment = filtered_df.groupby('country')['sentiment_score'].mean().sort_values(ascending=False).head(10)
        
        fig = go.Figure(go.Bar(
            x=country_sentiment.index,
            y=country_sentiment.values,
            marker_color=MY_COLORS['primary'],
            text=country_sentiment.values.round(1),
            textposition='auto'
        ))
        fig.update_layout(height=350, showlegend=False)
        fig.update_yaxes(title_text="Avg Sentiment Score")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("🔍 Sentiment Anomalies Detected")
    
    overall_avg = filtered_df['sentiment_score'].mean()
    overall_std = filtered_df['sentiment_score'].std()
    
    segment_anomalies = []
    for segment in filtered_df['segment'].unique():
        seg_data = filtered_df[filtered_df['segment'] == segment]
        seg_avg = seg_data['sentiment_score'].mean()
        z_score = abs(seg_avg - overall_avg) / overall_std
        
        if z_score > 1.5:
            segment_anomalies.append({
                'Segment': segment,
                'Avg Sentiment': f"{seg_avg:.1f}",
                'Deviation': f"{'Above' if seg_avg > overall_avg else 'Below'} average",
                'Customers': len(seg_data)
            })
    
    if segment_anomalies:
        st.dataframe(pd.DataFrame(segment_anomalies), use_container_width=True, hide_index=True)
    else:
        st.info("No significant sentiment anomalies detected across segments")

with tab3:
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk_count = len(filtered_df[filtered_df['churn_risk_score'] >= risk_threshold])
        st.metric("High Risk Customers", f"{high_risk_count:,}")
    
    with col2:
        avg_risk = filtered_df['churn_risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.1f}")
    
    with col3:
        at_risk_value = filtered_df[filtered_df['churn_risk_score'] >= risk_threshold]['lifetime_value'].sum()
        st.metric("Value at Risk", f"${at_risk_value:,.0f}")
    
    st.divider()
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Churn Risk Distribution")
        
        fig = px.histogram(
            filtered_df,
            x='churn_risk_score',
            nbins=30,
            color_discrete_sequence=['#636EFA'],
            labels={'churn_risk_score': 'Churn Risk Score'}
        )
        
        fig.add_vline(
            x=risk_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Risk Threshold ({risk_threshold})",
            annotation_position="top"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk by Engagement Score")
        
        fig = px.scatter(
            filtered_df.sample(min(500, len(filtered_df))),
            x='engagement_score',
            y='churn_risk_score',
            color='loyalty_tier',
            size='lifetime_value',
            color_discrete_map=tier_colors,
            hover_data=['segment', 'days_since_active']
        )
        
        fig.add_hline(
            y=risk_threshold,
            line_dash="dash",
            line_color="red"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("High-Risk Customers by Segment")
        high_risk_segment = filtered_df[filtered_df['churn_risk_score'] >= risk_threshold].groupby('segment').size().sort_values(ascending=False)
        
        fig = go.Figure(go.Bar(
            x=high_risk_segment.values,
            y=high_risk_segment.index,
            orientation='h',
            marker_color='#EF553B',
            text=high_risk_segment.values,
            textposition='auto'
        ))
        fig.update_layout(height=350, showlegend=False)
        fig.update_xaxes(title_text="Number of High-Risk Customers")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Days Since Last Active")
        inactive_dist = filtered_df['days_since_active'].describe()
        
        fig = px.box(
            filtered_df,
            y='days_since_active',
            color='churn_risk_category',
            color_discrete_map={'Low': '#00CC96', 'Medium': '#FFA15A', 'High': '#EF553B'}
        )
        fig.update_layout(height=350, showlegend=True)
        fig.update_yaxes(title_text="Days Since Last Active")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("🚨 Top 20 At-Risk Customers")
    
    high_risk_customers = filtered_df[filtered_df['churn_risk_score'] >= risk_threshold].nlargest(20, 'lifetime_value')[
        ['customer_id', 'first_name', 'last_name', 'segment', 'loyalty_tier', 
         'lifetime_value', 'days_since_active', 'churn_risk_score', 'sentiment_score']
    ].copy()
    
    high_risk_customers['churn_risk_score'] = high_risk_customers['churn_risk_score'].round(1)
    high_risk_customers['sentiment_score'] = high_risk_customers['sentiment_score'].round(1)
    high_risk_customers['lifetime_value'] = high_risk_customers['lifetime_value'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(
        high_risk_customers,
        use_container_width=True,
        hide_index=True
    )

with tab4:
    
    rec_df = filtered_df[filtered_df['recommendation_received'] == True]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recommendations Sent", f"{len(rec_df):,}")
    
    with col2:
        conv_rate = rec_df['conversion_after_rec'].mean() * 100
        st.metric("Conversion Rate", f"{conv_rate:.1f}%")
    
    with col3:
        avg_ltv_rec = rec_df['lifetime_value'].mean()
        avg_ltv_no_rec = filtered_df[filtered_df['recommendation_received'] == False]['lifetime_value'].mean()
        ltv_lift = ((avg_ltv_rec - avg_ltv_no_rec) / avg_ltv_no_rec) * 100
        st.metric("LTV Lift", f"{ltv_lift:+.1f}%")
    
    with col4:
        conversions = rec_df['conversion_after_rec'].sum()
        st.metric("Total Conversions", f"{conversions:,}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Conversion by Recommendation Type")
        rec_type_conv = rec_df.groupby('recommendation_type')['conversion_after_rec'].agg(['mean', 'count']).reset_index()
        rec_type_conv['mean'] = rec_type_conv['mean'] * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=rec_type_conv['recommendation_type'],
            y=rec_type_conv['mean'],
            name='Conversion Rate',
            marker_color=MY_COLORS['primary'],
            text=rec_type_conv['mean'].round(1),
            texttemplate='%{text}%',
            textposition='auto'
        ))
        
        fig.update_layout(height=350, showlegend=False)
        fig.update_yaxes(title_text="Conversion Rate (%)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Recommendations by Segment")
        rec_by_segment = rec_df.groupby('segment').agg({
            'customer_id': 'count',
            'conversion_after_rec': 'mean'
        }).reset_index()
        rec_by_segment['conversion_after_rec'] = rec_by_segment['conversion_after_rec'] * 100
        rec_by_segment.columns = ['Segment', 'Recommendations', 'Conv Rate (%)']
        
        fig = px.bar(
            rec_by_segment,
            x='Segment',
            y='Recommendations',
            color='Conv Rate (%)',
            color_continuous_scale='Viridis',
            text='Recommendations'
        )
        fig.update_traces(textposition='auto')
        fig.update_layout(height=350)
        
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Impact on Sentiment")
        
        sentiment_comparison = pd.DataFrame({
            'Group': ['With Recommendations', 'Without Recommendations'],
            'Avg Sentiment': [
                rec_df['sentiment_score'].mean(),
                filtered_df[filtered_df['recommendation_received'] == False]['sentiment_score'].mean()
            ]
        })
        
        fig = go.Figure(go.Bar(
            x=sentiment_comparison['Group'],
            y=sentiment_comparison['Avg Sentiment'],
            marker_color=['#00CC96', '#FFA15A'],
            text=sentiment_comparison['Avg Sentiment'].round(1),
            textposition='auto'
        ))
        fig.update_layout(height=350, showlegend=False)
        fig.update_yaxes(title_text="Average Sentiment Score")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Impact on Churn Risk")
        
        churn_comparison = pd.DataFrame({
            'Group': ['With Recommendations', 'Without Recommendations'],
            'Avg Churn Risk': [
                rec_df['churn_risk_score'].mean(),
                filtered_df[filtered_df['recommendation_received'] == False]['churn_risk_score'].mean()
            ]
        })
        
        fig = go.Figure(go.Bar(
            x=churn_comparison['Group'],
            y=churn_comparison['Avg Churn Risk'],
            marker_color=['#00CC96', '#EF553B'],
            text=churn_comparison['Avg Churn Risk'].round(1),
            textposition='auto'
        ))
        fig.update_layout(height=350, showlegend=False)
        fig.update_yaxes(title_text="Average Churn Risk Score")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("📊 Recommendation Performance Matrix")
    
    performance_matrix = rec_df.groupby(['segment', 'recommendation_type']).agg({
        'conversion_after_rec': ['mean', 'count'],
        'lifetime_value': 'mean'
    }).reset_index()
    
    performance_matrix.columns = ['Segment', 'Rec Type', 'Conv Rate', 'Count', 'Avg LTV']
    performance_matrix['Conv Rate'] = (performance_matrix['Conv Rate'] * 100).round(1)
    performance_matrix['Avg LTV'] = performance_matrix['Avg LTV'].round(0)
    
    performance_matrix = performance_matrix.sort_values('Conv Rate', ascending=False)
    
    st.dataframe(
        performance_matrix,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Conv Rate": st.column_config.NumberColumn(
                "Conversion Rate (%)",
                format="%.1f%%"
            ),
            "Avg LTV": st.column_config.NumberColumn(
                "Avg LTV",
                format="$%.0f"
            )
        }
    )
    
    st.divider()
    
    st.subheader("💡 Recommendation Insights")
    
    best_rec_type = rec_type_conv.loc[rec_type_conv['mean'].idxmax(), 'recommendation_type']
    best_conv_rate = rec_type_conv['mean'].max()
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **Best Performing:** {best_rec_type} recommendations with {best_conv_rate:.1f}% conversion rate")
        rec_sentiment_lift = rec_df['sentiment_score'].mean() - filtered_df[filtered_df['recommendation_received'] == False]['sentiment_score'].mean()
        st.info(f"📈 **Sentiment Lift:** Customers receiving recommendations show {rec_sentiment_lift:+.1f} point higher sentiment")
    with col2:
        rec_churn_impact = filtered_df[filtered_df['recommendation_received'] == False]['churn_risk_score'].mean() - rec_df['churn_risk_score'].mean()
        st.info(f"🛡️ **Churn Prevention:** Recommendations reduce churn risk by {rec_churn_impact:.1f} points on average")
        high_value_conv = rec_df[rec_df['lifetime_value'] > rec_df['lifetime_value'].median()]['conversion_after_rec'].mean() * 100
        st.success(f"💰 **High-Value Customers:** {high_value_conv:.1f}% conversion rate among top LTV customers")

st.divider()
