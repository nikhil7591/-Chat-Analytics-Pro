import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import time
import functions
import ml_models
import theme_manager 
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from fpdf import FPDF
import io
import pdf_exporter
# Set page config
st.set_page_config(
    page_title="Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)
theme_manager.apply_theme()
# Custom CSS
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom-color: #4CAF50;
    }
    .stTitle {
        font-weight: bold;
        font-size: 24px;
        color: #333;
    }
    .stSubheader {
        font-size: 20px;
        font-weight: 600;
        color: #4CAF50;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title('Chat Analytics Pro')
st.markdown('#### Analyze your WhatsApp chats with ML features')

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/WhatsApp.svg/1200px-WhatsApp.svg.png", width=100)
    st.title("Settings")
    theme_manager.theme_selector()
    # File uploader
    file = st.file_uploader("Upload WhatsApp Chat (.txt)", type=['txt'])
    
    # Instructions
    with st.expander("How to export WhatsApp chat"):
        st.markdown("""
        1. Open WhatsApp chat
        2. Tap on the three dots in the top right
        3. Select 'More' > 'Export chat'
        4. Choose 'Without media'
        5. Save the .txt file
        6. Upload the file here
        """)

# Main content
if file:
    with st.spinner('Processing chat data...'):
        # Progress bar
        progress_bar = st.progress(0)
        
        # Generate DataFrame
        df = functions.generateDataFrame(file)
        progress_bar.progress(20)
        
        # Date format selection
        date_format = st.radio("Select Date Format in your chat export:", ('dd/mm/yyyy', 'mm/dd/yyyy'))
        dayfirst = True if date_format == 'dd/mm/yyyy' else False
        
        # Get users
        users = functions.getUsers(df)
        progress_bar.progress(30)
        
        # Sidebar user selection
        with st.sidebar:
            selected_user = st.selectbox("Select User", users)
            
            # Analysis button
            analyze_button = st.button("Analyze Chat", type="primary")
        
        # If analyze button is clicked
        if analyze_button:
            # Preprocess data
            df = functions.PreProcess(df, dayfirst)
            progress_bar.progress(40)
            
            # Filter for selected user
            if selected_user != "Everyone":
                user_df = df[df['User'] == selected_user]
                title_name = f"{selected_user}'s"
            else:
                user_df = df
                title_name = "Group"
            
            # Get statistics
            filtered_df, media_cnt, deleted_msgs_cnt, links_cnt, word_count, msg_count, avg_msg_length = functions.getStats(user_df)
            progress_bar.progress(50)
            
            # Add sentiment analysis
            filtered_df, user_sentiment, sentiment_counts, sentiment_over_time = functions.sentiment_analysis(filtered_df)
            progress_bar.progress(60)
            
            # Create tabs
            tabs = st.tabs([
                "📊 Overview", 
                "👥 User Analysis", 
                "💬 Message Analysis", 
                "🔍 Topic Analysis",
                "📈 Prediction & Trends",
                "🧠 ML Insights"
            ])
            
            # Overview Tab
            with tabs[0]:
                st.header(f"{title_name} Chat Overview")
                
                # hvhjkljhgcfxd
                st.markdown("## 📥 Download Chat Analysis Report")

                # 📄 CSV Export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name='chat_report.csv',
                    mime='text/csv',
                    key='csv-download'
                )

                # 📊 Excel Export
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Report', index=False)
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name='chat_report.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='excel-download'
                )

                # PDF Export
                stats_dict = {
                    'msg_count': msg_count,
                    'user_count': len(filtered_df['User'].unique()),
                    'days': filtered_df['Date'].nunique(),
                    'media_cnt': media_cnt,
                    'word_count': word_count,
                    'links_cnt': links_cnt,
                    'user_counts': filtered_df['User'].value_counts(),
                    'sentiment_counts': sentiment_counts if 'sentiment' in filtered_df.columns else None,
                }

                try:
                    pdf_bytes = pdf_exporter.generate_report_pdf(
                        filtered_df,
                        "WhatsApp Chat Analysis",
                        selected_user,
                        stats_dict
                    )

                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_bytes,
                        file_name=f"whatsapp_analysis_{selected_user}.pdf",
                        mime="application/pdf",
                        key="pdf-download"
                    )
                except Exception as e:
                    st.error(f"Failed to generate PDF: {str(e)}")

                # Basic statistics in cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Total Messages</h3>
                        <h2>{msg_count:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Total Days</h3>
                        <h2>{filtered_df['Date'].nunique():,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Media Shared</h3>
                        <h2>{media_cnt:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Words Exchanged</h3>
                        <h2>{word_count:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Secondary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Average Message Length</h3>
                        <h2>{avg_msg_length:.1f} chars</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Links Shared</h3>
                        <h2>{links_cnt:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Deleted Messages</h3>
                        <h2>{deleted_msgs_cnt:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Msgs per Day</h3>
                        <h2>{msg_count/filtered_df['Date'].nunique():.1f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Daily Timeline
                st.subheader("Message Trends")
                functions.dailytimeline(filtered_df)
                progress_bar.progress(70)
                
                # Activity Heatmap
                st.subheader("Weekly Activity Heatmap")
                user_heatmap = functions.activity_heatmap(filtered_df)
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(user_heatmap, cmap="YlGnBu", linewidths=0.5, ax=ax)
                ax.set_title("Message Activity by Day and Hour")
                st.pyplot(fig)
                
                # Most Active Days and Months
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Activity by Day of Week")
                    functions.WeekAct(filtered_df)
                
                with col2:
                    st.subheader("Activity by Month")
                    functions.MonthAct(filtered_df)
                
                # Sentiment Analysis Overview
                st.subheader("Message Sentiment Overview")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment Distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#2ecc71', '#3498db', '#e74c3c']
                    sentiment_counts.plot(kind='bar', ax=ax, color=colors)
                    ax.set_title("Message Sentiment Distribution")
                    ax.set_ylabel("Number of Messages")
                    st.pyplot(fig)
                
                with col2:
                    # Sentiment Over Time
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(sentiment_over_time['Date'], sentiment_over_time['sentiment_score'], label='Daily Sentiment')
                    ax.plot(sentiment_over_time['Date'], sentiment_over_time['Rolling_Avg'], color='red', linewidth=2, label='7-day Average')
                    ax.set_title("Sentiment Trend Over Time")
                    ax.set_ylabel("Sentiment Score (-1 to +1)")
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                progress_bar.progress(80)
            
            # User Analysis Tab
            with tabs[1]:
                st.header("User Analysis")
                
                if selected_user == "Everyone":
                    # User Message Distribution
                    st.subheader("User Message Distribution")
                    user_message_counts = filtered_df['User'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # User Message Count Table
                        user_message_pct = (user_message_counts / user_message_counts.sum() * 100).round(2)
                        user_message_df = pd.DataFrame({
                            'User': user_message_counts.index,
                            'Messages': user_message_counts.values,
                            'Percentage': user_message_pct.values
                        })
                        st.dataframe(user_message_df, use_container_width=True)
                    
                    with col2:
                        # User Message Count Chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(user_message_df['User'], user_message_df['Messages'], 
                                     color=sns.color_palette("viridis", len(user_message_df)))
                        ax.set_title("Messages Sent by User")
                        ax.set_xlabel("User")
                        ax.set_ylabel("Number of Messages")
                        plt.xticks(rotation=45, ha='right')
                        
                        # Add data labels
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                                   f'{height:,}', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # User Clustering
                    st.subheader("User Clustering Analysis")
                    user_stats, cluster_descriptions = functions.user_clustering(filtered_df)
                    
                    if user_stats is not None and 'Cluster' in user_stats.columns:
                        # Show clusters
                        st.write("Users have been grouped into different clusters based on messaging patterns:")
                        
                        for i, desc in enumerate(cluster_descriptions):
                            st.write(f"{desc}")
                            cluster_users = user_stats[user_stats['Cluster'] == i]['User'].tolist()
                            st.write(f"Users in this cluster: {', '.join(cluster_users)}")
                        
                        # Plot user clusters
                        st.subheader("User Cluster Visualization")
                        
                        # Combine key user stats for visualization
                        user_cluster_viz = user_stats[['User', 'Message', 'message_length', 'Cluster']].copy()
                        
                        # Create a bubble chart
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Create scatter plot for each cluster
                        for cluster in user_cluster_viz['Cluster'].unique():
                            cluster_data = user_cluster_viz[user_cluster_viz['Cluster'] == cluster]
                            ax.scatter(
                                cluster_data['Message'],
                                cluster_data['message_length'],
                                s=200,
                                alpha=0.7,
                                label=f'Cluster {cluster+1}'
                            )
                        
                        # Add user labels
                        for i, row in user_cluster_viz.iterrows():
                            ax.annotate(
                                row['User'],
                                (row['Message'], row['message_length']),
                                fontsize=9,
                                ha='center',
                                va='center'
                            )
                        
                        ax.set_title("User Clusters by Message Count and Length")
                        ax.set_xlabel("Number of Messages")
                        ax.set_ylabel("Average Message Length")
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                    
                    # User Interaction Network
                    st.subheader("User Interaction Network")
                    user_interactions = ml_models.user_interaction_network(filtered_df)
                    
                    if user_interactions is not None:
                        # Create a network visualization
                        st.write("This network shows how users interact with each other. Stronger connections indicate more frequent interactions.")
                        
                        # Filter for significant interactions
                        significant_interactions = user_interactions[user_interactions['interaction_count'] > 10]
                        
                        # Create network figure
                        try:
                            import networkx as nx
                            
                            # Create graph
                            G = nx.DiGraph()
                            
                            # Add nodes
                            all_users = set(significant_interactions['prev_user'].unique()) | set(significant_interactions['User'].unique())
                            for user in all_users:
                                G.add_node(user)
                            
                            # Add edges
                            for _, row in significant_interactions.iterrows():
                                G.add_edge(
                                    row['prev_user'],
                                    row['User'],
                                    weight=row['interaction_count'],
                                    width=row['interaction_strength']*5
                                )
                            
                            # Draw graph
                            fig, ax = plt.subplots(figsize=(12, 10))
                            pos = nx.spring_layout(G, k=0.3, iterations=50)
                            
                            # Draw nodes
                            nx.draw_networkx_nodes(
                                G, pos,
                                node_size=500,
                                node_color='lightblue',
                                alpha=0.8
                            )
                            
                            # Draw edges
                            edge_widths = [G[u][v]['width'] for u, v in G.edges()]
                            nx.draw_networkx_edges(
                                G, pos,
                                width=edge_widths,
                                alpha=0.5,
                                edge_color='gray',
                                arrows=True,
                                arrowsize=20
                            )
                            
                            # Draw labels
                            nx.draw_networkx_labels(
                                G, pos,
                                font_size=10
                            )
                            
                            plt.axis('off')
                            st.pyplot(fig)
                        except ImportError:
                            st.write("Network visualization requires the networkx package.")
                    else:
                        st.write("Not enough interaction data to create a meaningful network visualization.")
                
                # User Engagement Analysis
                st.subheader("User Engagement Analysis")
                user_engagement = ml_models.user_engagement_analysis(filtered_df)
                
                if user_engagement is not None:
                    # Select top users by engagement score
                    top_users = user_engagement.sort_values('engagement_score', ascending=False).head(10)
                    
                    # Show engagement metrics
                    st.write("User engagement metrics based on message frequency and participation:")
                    
                    # Create engagement chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot engagement score
                    bars = ax.bar(
                        top_users['User'], 
                        top_users['engagement_score'],
                        color=sns.color_palette("viridis", len(top_users))
                    )
                    
                    ax.set_title("User Engagement Scores")
                    ax.set_xlabel("User")
                    ax.set_ylabel("Engagement Score")
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add data labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                               f'{height:.2f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show engagement metrics table
                    st.write("Detailed engagement metrics:")
                    engagement_metrics = user_engagement[['User', 'message_count', 'days_active', 
                                                         'participation_rate', 'msgs_per_day', 'engagement_score']]
                    engagement_metrics = engagement_metrics.sort_values('engagement_score', ascending=False)
                    
                    # Format the table
                    engagement_metrics['participation_rate'] = engagement_metrics['participation_rate'].map(lambda x: f"{x*100:.1f}%")
                    engagement_metrics['msgs_per_day'] = engagement_metrics['msgs_per_day'].map(lambda x: f"{x:.1f}")
                    engagement_metrics['engagement_score'] = engagement_metrics['engagement_score'].map(lambda x: f"{x:.2f}")
                    
                    st.dataframe(engagement_metrics, use_container_width=True)
                
                progress_bar.progress(85)
            
            # Message Analysis Tab
            with tabs[2]:
                st.header("Message Analysis")
                
                # Message Length Distribution
                st.subheader("Message Length Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram of message lengths
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(filtered_df['message_length'].clip(upper=200), bins=20, kde=True, ax=ax)
                    ax.set_title("Distribution of Message Lengths")
                    ax.set_xlabel("Message Length (characters)")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                
                with col2:
                    # Message length stats
                    st.write("Message length statistics:")
                    message_length_stats = pd.DataFrame({
                        'Statistic': ['Average', 'Median', 'Maximum', 'Minimum'],
                        'Value': [
                            f"{filtered_df['message_length'].mean():.1f} chars",
                            f"{filtered_df['message_length'].median():.1f} chars",
                            f"{filtered_df['message_length'].max():.1f} chars",
                            f"{filtered_df['message_length'].min():.1f} chars"
                        ]
                    })
                    st.dataframe(message_length_stats, use_container_width=True)
                    
                    # Message type distribution
                    st.write("Message type distribution:")
                    
                    media_count = media_cnt
                    deleted_count = deleted_msgs_cnt
                    links_count = links_cnt
                    text_count = msg_count - (media_count + deleted_count + links_count)
                    
                    message_types = pd.DataFrame({
                        'Type': ['Text', 'Media', 'Links', 'Deleted'],
                        'Count': [text_count, media_count, links_count, deleted_count]
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
                    ax.pie(
                        message_types['Count'], 
                        labels=message_types['Type'],
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors,
                        explode=[0.05, 0.05, 0.05, 0.05]
                    )
                    ax.set_title("Message Type Distribution")
                    plt.axis('equal')
                    st.pyplot(fig)
                
                # Emoji Analysis
                st.subheader("Emoji Analysis")
                emoji_df = functions.getEmoji(filtered_df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if not emoji_df.empty:
                        # Show emoji table
                        st.dataframe(emoji_df, use_container_width=True)
                    else:
                        st.write("No emojis found in the selected messages.")
                
                with col2:
                    if not emoji_df.empty:
                        # Show emoji pie chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.pie(
                            emoji_df[1].head(10), 
                            labels=emoji_df[0].head(10),
                            autopct='%1.1f%%',
                            startangle=90
                        )
                        ax.set_title("Top 10 Emojis Used")
                        plt.axis('equal')
                        st.pyplot(fig)
                
                # Common Words
                st.subheader("Most Common Words")
                common_words = functions.MostCommonWords(filtered_df)
                
                if common_words is not None:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.bar(common_words[0], common_words[1], color=sns.color_palette("viridis", len(common_words[0])))
                    ax.set_title("Most Frequently Used Words")
                    ax.set_xlabel("Words")
                    ax.set_ylabel("Frequency")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Word Cloud
                st.subheader("Word Cloud")
                word_cloud = functions.create_wordcloud(filtered_df)
                
                if word_cloud is not None:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(word_cloud)
                    ax.axis('off')
                    st.pyplot(fig)
                
                progress_bar.progress(90)
            
            # Topic Analysis Tab
            with tabs[3]:
                st.header("Topic Analysis")
                
                # Topic Classification
                st.subheader("Conversation Topic Classification")
                long_messages, topic_keywords = ml_models.conversation_topic_classifier(filtered_df)
                
                if long_messages is not None and topic_keywords is not None:
                    st.write("The AI has identified these main topics in your conversations:")
                    
                    # Display topics and keywords
                    for topic, keywords in topic_keywords.items():
                        st.write(f"{topic}:** {', '.join(keywords[:5])}")
                    
                    # Show top messages for each topic
                    st.subheader("Sample Messages by Topic")
                    
                    for topic_id in long_messages['topic'].unique():
                        topic_messages = long_messages[long_messages['topic'] == topic_id].head(3)
                        
                        st.write(f"*Topic {topic_id+1}*")
                        for _, msg in topic_messages.iterrows():
                            st.write(f"- {msg['User']}: {msg['Message'][:100]}..." if len(msg['Message']) > 100 else f"- {msg['User']}: {msg['Message']}")
                else:
                    st.write("Not enough data for topic classification. Need more messages with sufficient length.")
                
                # Response Analysis
                st.subheader("Response Pattern Analysis")
                user_response_data = ml_models.user_response_analysis(filtered_df)
                
                if user_response_data is not None:
                    # Filter for significant response pairs
                    significant_responses = user_response_data[user_response_data['response_count'] > 5]
                    
                    if not significant_responses.empty:
                        st.write("Analysis of how quickly users respond to each other:")
                        
                        # Format response time
                        significant_responses['avg_response_time_formatted'] = significant_responses['avg_response_time'].apply(
                            lambda x: f"{int(x//60)}h {int(x%60)}m" if x >= 60 else f"{int(x)}m"
                        )
                        
                        # Format response rate
                        significant_responses['response_rate_pct'] = significant_responses['response_rate'] * 100
                        
                        # Select columns to display
                        display_columns = significant_responses[['sender', 'responder', 'response_count', 
                                                           'avg_response_time_formatted', 'response_rate_pct']]
                        
                        # Rename columns
                        display_columns.columns = ['From', 'To', 'Responses', 'Avg. Response Time', 'Response Rate (%)']
                        
                        # Format percentage
                        display_columns['Response Rate (%)'] = display_columns['Response Rate (%)'].apply(lambda x: f"{x:.1f}%")
                        
                        # Sort by response count
                        display_columns = display_columns.sort_values('Responses', ascending=False)
                        
                        # Show table
                        st.dataframe(display_columns, use_container_width=True)
                        
                        # Show response time visualization
                        st.subheader("Response Time Visualization")
                        
                        try:
                            # Create heatmap data
                            pivot_response_time = significant_responses.pivot(
                                index='sender', 
                                columns='responder', 
                                values='avg_response_time'
                            ).fillna(0)
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(
                                pivot_response_time,
                                annot=True,
                                fmt=".0f",
                                cmap="YlGnBu",
                                ax=ax,
                                cbar_kws={'label': 'Avg. Response Time (minutes)'}
                            )
                            ax.set_title("Average Response Time Between Users (minutes)")
                            plt.tight_layout()
                            st.pyplot(fig)
                        except:
                            st.write("Cannot create response time heatmap with current data.")
                    else:
                        st.write("Not enough response data between users for meaningful analysis.")
                else:
                    st.write("Not enough message data for response pattern analysis.")
                
                # Conversation Intensity
                st.subheader("Conversation Intensity Analysis")
                peak_times, conversation_summary = ml_models.conversation_intensity_analysis(filtered_df)

                if peak_times is not None and conversation_summary is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Conversation summary:")
                        summary_display = pd.DataFrame({
                            'Metric': [
                                'Average Messages per Day',
                                'Busiest Day', 
                                'Messages on Busiest Day',
                                'Max Consecutive Busy Days',
                                'Peak Hour Message Rate (msgs/min)'
                            ],
                            'Value': [
                                f"{conversation_summary['avg_daily_messages']:.1f}",
                                conversation_summary['busiest_day'].strftime('%d %b %Y'),
                                f"{conversation_summary['busiest_day_count']}",
                                f"{conversation_summary['max_consecutive_busy_days']}",
                                f"{conversation_summary['peak_hour_intensity']:.2f}"
                            ]
                        })
                        st.dataframe(summary_display, use_container_width=True)
                    
                    with col2:
                        st.write("Peak conversation times:")
                        if not peak_times.empty:
                            # Convert Date to datetime if it's not already
                            peak_times['Date'] = pd.to_datetime(peak_times['Date'])
                            
                            # Now it's safe to use .dt accessor
                            peak_times['date_formatted'] = peak_times['Date'].dt.strftime('%d %b %Y')
                            peak_display = peak_times[['date_formatted', 'hour', 'message_count', 'intensity']]
                            peak_display.columns = ['Date', 'Hour', 'Messages', 'Msgs/Min']
                            peak_display['Hour'] = peak_display['Hour'].apply(lambda x: f"{x}:00 - {x+1}:00")
                            st.dataframe(peak_display, use_container_width=True)

                progress_bar.progress(95)

            # Prediction & Trends Tab
            with tabs[4]:
                st.header("Prediction & Trends")
                
                # Message Trend Prediction
                st.subheader("Message Frequency Forecast")
                
                try:
                    fig, components_fig, forecast = ml_models.perform_message_trend_prediction(filtered_df)
                    
                    # Show forecast plot
                    st.pyplot(fig)
                    
                    # Show forecast components
                    st.subheader("Forecast Components")
                    st.pyplot(components_fig)
                    
                    # Show forecast table
                    st.subheader("Forecast Data")
                    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
                    forecast_display.columns = ['Date', 'Predicted Messages', 'Lower Bound', 'Upper Bound']
                    forecast_display['Date'] = pd.to_datetime(forecast_display['Date']).dt.strftime('%d %b %Y')
                    forecast_display['Predicted Messages'] = forecast_display['Predicted Messages'].round().astype(int)
                    forecast_display['Lower Bound'] = forecast_display['Lower Bound'].round().astype(int)
                    forecast_display['Upper Bound'] = forecast_display['Upper Bound'].round().astype(int)
                    st.dataframe(forecast_display, use_container_width=True)
                except Exception as e:
                    st.write("Not enough data for meaningful trend prediction. Need more consistent chat history.")
                    st.write(f"Error: {e}")
                
                # Seasonal Analysis
                st.subheader("Seasonal Chat Patterns")
                
                try:
                    seasonal_data = functions.seasonal_analysis(filtered_df)
                    
                    if seasonal_data is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Hour of day seasonality
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.lineplot(x='hour', y='message_count', data=seasonal_data['hourly'], marker='o', ax=ax)
                            ax.set_title("Activity by Hour of Day")
                            ax.set_xlabel("Hour")
                            ax.set_ylabel("Average Messages")
                            ax.set_xticks(range(0, 24, 2))
                            ax.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                        
                        with col2:
                            # Day of week seasonality
                            fig, ax = plt.subplots(figsize=(10, 6))
                            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            sns.barplot(x='day_name', y='message_count', data=seasonal_data['daily'], order=day_order, ax=ax)
                            ax.set_title("Activity by Day of Week")
                            ax.set_xlabel("Day")
                            ax.set_ylabel("Average Messages")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        # Monthly seasonality
                        st.subheader("Monthly Chat Pattern")
                        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                                       'July', 'August', 'September', 'October', 'November', 'December']
                        
                        if 'monthly' in seasonal_data and not seasonal_data['monthly'].empty:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            sns.barplot(x='month_name', y='message_count', data=seasonal_data['monthly'], order=month_order, ax=ax)
                            ax.set_title("Activity by Month")
                            ax.set_xlabel("Month")
                            ax.set_ylabel("Average Messages")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                except Exception as e:
                    st.write("Could not perform seasonal analysis with the current data.")
                    st.write(f"Error: {e}")
                
                # Message Intensity Trends
                st.subheader("Message Intensity Trends")
                
                try:
                    intensity_df = functions.calculate_message_intensity(filtered_df)
                    
                    if intensity_df is not None and not intensity_df.empty:
                        fig = px.line(
                            intensity_df, 
                            x='datetime', 
                            y='intensity',
                            title='Message Intensity Over Time (Messages per Minute)',
                            labels={'datetime': 'Date', 'intensity': 'Messages per Minute'}
                        )
                        
                        fig.update_layout(
                            height=500,
                            xaxis_title='Date',
                            yaxis_title='Messages per Minute',
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add rolling average
                        st.subheader("Long-term Conversation Trends")
                        long_term_df = functions.calculate_long_term_trends(filtered_df)
                        
                        if long_term_df is not None and not long_term_df.empty:
                            fig = go.Figure()
                            
                            # Add daily message count
                            fig.add_trace(go.Scatter(
                                x=long_term_df['date'],
                                y=long_term_df['message_count'],
                                mode='lines',
                                name='Daily Messages',
                                line=dict(color='rgba(0, 119, 182, 0.3)')
                            ))
                            
                            # Add 7-day moving average
                            fig.add_trace(go.Scatter(
                                x=long_term_df['date'],
                                y=long_term_df['7d_rolling_avg'],
                                mode='lines',
                                name='7-day Average',
                                line=dict(color='rgba(0, 119, 182, 1)', width=2)
                            ))
                            
                            # Add 30-day moving average
                            fig.add_trace(go.Scatter(
                                x=long_term_df['date'],
                                y=long_term_df['30d_rolling_avg'],
                                mode='lines',
                                name='30-day Average',
                                line=dict(color='rgba(182, 0, 0, 1)', width=2)
                            ))
                            
                            fig.update_layout(
                                title='Long-term Message Trends',
                                xaxis_title='Date',
                                yaxis_title='Messages',
                                height=500,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.write("Could not calculate message intensity trends with the current data.")
                    st.write(f"Error: {e}")
                
                progress_bar.progress(98)
            
            # ML Insights Tab
            with tabs[5]:
                st.header("ML Insights")

                # Sentiment Analysis
                st.subheader("Detailed Sentiment Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # User sentiment comparison
                    if selected_user == "Everyone" and user_sentiment is not None:
                        try:
                            # First determine what type of data structure user_sentiment is
                            if isinstance(user_sentiment, pd.DataFrame):
                                # It's already a DataFrame
                                if 'message_count' not in user_sentiment.columns:
                                    # Calculate message count
                                    message_counts = filtered_df.groupby('User').size().reset_index(name='message_count')
                                    user_sentiment = user_sentiment.merge(message_counts, on='User', how='left')
                            else:
                                # It's likely a Series with user sentiment averages
                                # Convert to DataFrame with proper column names
                                if hasattr(user_sentiment, 'index') and hasattr(user_sentiment, 'name'):
                                    user_sentiment_df = pd.DataFrame({
                                        'User': user_sentiment.index,
                                        'avg_sentiment': user_sentiment.values
                                    })
                                    
                                    # Calculate message count
                                    message_counts = filtered_df.groupby('User').size().reset_index(name='message_count')
                                    user_sentiment = user_sentiment_df.merge(message_counts, on='User', how='left')
                                else:
                                    st.write("User sentiment data format is not supported.")
                                    user_sentiment = None
                            
                            # Now continue with the filtering if we have valid data
                            if user_sentiment is not None:
                                # Filter for users with enough messages
                                user_sentiment_filtered = user_sentiment[user_sentiment['message_count'] > 10]
                                
                                if not user_sentiment_filtered.empty:
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    # Sort by average sentiment
                                    user_sentiment_filtered = user_sentiment_filtered.sort_values('avg_sentiment')
                                    
                                    # Create bar chart
                                    bars = ax.bar(
                                        user_sentiment_filtered['User'],
                                        user_sentiment_filtered['avg_sentiment'],
                                        color=sns.diverging_palette(220, 20, as_cmap=True)(
                                            (user_sentiment_filtered['avg_sentiment'] + 1) / 2
                                        )
                                    )
                                    
                                    # Add horizontal line at 0
                                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                                    
                                    ax.set_title("Average Sentiment by User")
                                    ax.set_xlabel("User")
                                    ax.set_ylabel("Average Sentiment (-1 to +1)")
                                    plt.xticks(rotation=45, ha='right')
                                    
                                    # Add data labels
                                    for bar in bars:
                                        height = bar.get_height()
                                        ax.text(
                                            bar.get_x() + bar.get_width()/2.,
                                            height + 0.05 if height >= 0 else height - 0.1,
                                            f'{height:.2f}',
                                            ha='center',
                                            va=('bottom' if height >= 0 else 'top')
                                        )
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    st.write("Not enough data for user sentiment comparison.")
                        except Exception as e:
                            st.write(f"Error processing user sentiment data: {str(e)}")
                    else:
                        st.write("Switch to 'Everyone' view to see user sentiment comparison.")

                with col2:
                    # Topic-based sentiment
                    topic_sentiment = ml_models.topic_sentiment_analysis(filtered_df)
                    
                    if topic_sentiment is not None and not topic_sentiment.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Sort by sentiment
                        topic_sentiment = topic_sentiment.sort_values('avg_sentiment')
                        
                        # Create bar chart
                        bars = ax.bar(
                            topic_sentiment['topic'],
                            topic_sentiment['avg_sentiment'],
                            color=sns.diverging_palette(220, 20, as_cmap=True)(
                                (topic_sentiment['avg_sentiment'] + 1) / 2
                            )
                        )
                        
                        # Add horizontal line at 0
                        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                        
                        ax.set_title("Average Sentiment by Topic")
                        ax.set_xlabel("Topic")
                        ax.set_ylabel("Average Sentiment (-1 to +1)")
                        
                        # Add data labels
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width()/2.,
                                height + 0.05 if height >= 0 else height - 0.1,
                                f'{height:.2f}',
                                ha='center',
                                va=('bottom' if height >= 0 else 'top')
                            )
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write("Not enough data for topic-based sentiment analysis.")
                
                # Conversation Tone Analysis
        
                tone_analysis = ml_models.conversation_tone_analysis(filtered_df)

                if tone_analysis is not None:
                    # Display tone distribution
                    st.subheader("Conversation Tone Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Tone distribution pie chart
                        tone_distribution = tone_analysis['tone_distribution']
                        
                        if not tone_distribution.empty:
                            fig, ax = plt.subplots(figsize=(8, 8))
                            
                            # Define colors for each tone
                            colors = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
                            tone_colors = [colors[tone] for tone in tone_distribution['tone']]
                            
                            # Create pie chart
                            ax.pie(
                                tone_distribution['percentage'],
                                labels=tone_distribution['tone'],
                                autopct='%1.1f%%',
                                colors=tone_colors,
                                startangle=90,
                                shadow=True
                            )
                            
                            ax.set_title("Distribution of Conversation Tone")
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                            
                            st.pyplot(fig)
                        else:
                            st.write("Not enough data for tone distribution analysis.")
                    
                    with col2:
                        # Sentiment over time
                        daily_sentiment = tone_analysis['daily_sentiment']
                        
                        if not daily_sentiment.empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Create line plot
                            ax.plot(
                                daily_sentiment['Date'],
                                daily_sentiment['avg_sentiment'],
                                marker='o',
                                linestyle='-',
                                alpha=0.6,
                                label='Daily Sentiment'
                            )
                            
                            # Add rolling average
                            ax.plot(
                                daily_sentiment['Date'],
                                daily_sentiment['rolling_sentiment'],
                                color='red',
                                linewidth=2,
                                label='7-day Rolling Average'
                            )
                            
                            # Add horizontal line at 0
                            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                            
                            # Shade the background based on sentiment ranges
                            ax.axhspan(-1, -0.3, color='#FFCDD2', alpha=0.3)  # Light red for negative
                            ax.axhspan(-0.3, 0.3, color='#FFFDE7', alpha=0.3)  # Light yellow for neutral
                            ax.axhspan(0.3, 1, color='#DCEDC8', alpha=0.3)    # Light green for positive
                            
                            ax.set_title("Sentiment Trend Over Time")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Average Sentiment (-1 to +1)")
                            ax.legend()
                            
                            # Rotate x-axis labels for better readability
                            plt.xticks(rotation=45, ha='right')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.write("Not enough data for sentiment trend analysis.")
                    
                    # Display significant tone shifts
                    significant_shifts = tone_analysis['significant_shifts']
                    
                    if not significant_shifts.empty:
                        st.subheader("Significant Tone Shifts")
                        
                        # Create a dataframe for display
                        display_shifts = significant_shifts[['Date', 'avg_sentiment', 'sentiment_shift', 'shift_direction']].copy()
                        display_shifts['Date'] = display_shifts['Date'].astype(str)
                        display_shifts['avg_sentiment'] = display_shifts['avg_sentiment'].round(2)
                        display_shifts['sentiment_shift'] = display_shifts['sentiment_shift'].round(2)
                        
                        # Rename columns for display
                        display_shifts.columns = ['Date', 'Sentiment', 'Shift Magnitude', 'Direction']
                        
                        # Display the table
                        st.dataframe(display_shifts)
                    else:
                        st.write("No significant tone shifts detected.")
                else:
                    st.write("Not enough data for conversation tone analysis.")

                # 
                # User Style Analysis
                st.subheader("User Communication Style Analysis")
                user_styles = ml_models.user_communication_style(filtered_df)
                
                if user_styles is not None and not user_styles.empty:
                    # Select users with enough data
                    filtered_styles = user_styles[user_styles['message_count'] > 20]
                    
                    if not filtered_styles.empty:
                        # Create radar chart
                        fig = go.Figure()
                        
                        style_metrics = ['formality', 'expressiveness', 'complexity', 'politeness', 'directness']
                        
                        for _, user_row in filtered_styles.iterrows():
                            fig.add_trace(go.Scatterpolar(
                                r=[user_row[m] for m in style_metrics],
                                theta=style_metrics,
                                fill='toself',
                                name=user_row['User']
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="User Communication Style Profiles",
                            showlegend=True,
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show style description
                        st.write("Communication style metrics explained:")
                        style_descriptions = {
                            'formality': 'How formal the language used is (higher = more formal)',
                            'expressiveness': 'Use of emojis, exclamations, and emotional language',
                            'complexity': 'Complexity of language, sentence structure, and vocabulary',
                            'politeness': 'Use of polite phrases, requests, and courteous language',
                            'directness': 'How direct and to-the-point the communication is'
                        }
                        
                        for style, desc in style_descriptions.items():
                            st.write(f"- {style.capitalize()}: {desc}")
                    else:
                        st.write("Not enough messages from each user for meaningful style analysis.")
                else:
                    st.write("Could not perform user style analysis with current data.")
                
                # Key Phrase Extraction
                st.subheader("Key Phrases and Topics")
                key_phrases = ml_models.extract_key_phrases(filtered_df)
                
                if key_phrases is not None and key_phrases:
                    # Create word cloud for key phrases
                    from wordcloud import WordCloud
                    
                    # Create word cloud from phrases with their weights
                    phrase_weights = {' '.join(phrase): weight for phrase, weight in key_phrases.items()}
                    
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis',
                        max_words=100
                    ).generate_from_frequencies(phrase_weights)
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title("Key Topics and Phrases")
                    st.pyplot(fig)
                    
                    # Display top phrases in table format
                    top_phrases = sorted(key_phrases.items(), key=lambda x: x[1], reverse=True)[:20]
                    phrase_df = pd.DataFrame(top_phrases, columns=['Phrase', 'Relevance Score'])
                    phrase_df['Phrase'] = phrase_df['Phrase'].apply(lambda x: ' '.join(x))
                    phrase_df['Relevance Score'] = phrase_df['Relevance Score'].round(3)
                    
                    st.write("Top key phrases from conversations:")
                    st.dataframe(phrase_df, use_container_width=True)
                else:
                    st.write("Not enough data for key phrase extraction.")
                
                progress_bar.progress(100)
            
            # After analysis is done
            st.success("Analysis complete! Explore the tabs to see insights about your WhatsApp chat.")
        else:
            st.info("Click the 'Analyze Chat' button to start the analysis.")

        

else:
    # Show welcome message and instructions
    st.markdown("""
    ### 👋 Welcome to Chat Analytics Pro!
    
    Upload your WhatsApp chat export to get started. This tool provides advanced analytics including:
    
    - 📊 Message statistics and trends
    - 👥 User behavior analysis
    - 💬 Message content analysis
    - 🔍 Topic detection and classification
    - 📈 Prediction and trend forecasting
    - 🧠 AI-powered insights
    
    To get started, export your chat from WhatsApp and upload the .txt file using the sidebar.
    """)
    
    # Show demo images
    col1, col2 = st.columns(2)
    
    with col1:
        # st.image("whatsapp.png", 
        #         caption="Advanced Analytics Dashboard", use_container_width=True)
        image = Image.open("whatsapp.png")
        resized_image = image.resize((600, 400))  # Set width and height as needed
        
        st.image(resized_image, caption="Advanced Analytics Dashboard", use_container_width=False)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71", 
                caption="User Interaction Analysis", use_container_width=True)
    
    # Testimonials
    st.markdown("""
    ### What users are saying:
    
    > "This tool helped me understand group chat dynamics I never noticed before!" - Sarah K.
    
    > "The sentiment analysis feature is surprisingly accurate. Great for tracking team morale." - Mark T.
    
    > "I love the topic classification - finally understood what my family chat is really about!" - Priya M.
    """)
    
    # FAQ
    with st.expander("Frequently Asked Questions"):
        st.markdown("""
        Is my data safe?
        Yes! Your chat data is processed entirely in your browser session and is not stored on any server.
        
        What chat size is recommended?
        The tool works best with chats containing at least 500 messages. For optimal ML analysis, 1000+ messages are recommended.
        
        Can I analyze group chats?
        Yes, both individual and group chats are supported.
        
        Will my media files be analyzed?
        No, only the text content of your chat is analyzed. Export your chat 'Without Media' for faster processing.
        
        Which languages are supported?
        The tool supports chats in any language, though some advanced features work best with English content.
        """)
    
    # Footer
    st.markdown("""
    ---
    Made with ❤ by WhatsApp Chat Analytics Team | [Privacy Policy](https://example.com) | [Terms of Service](https://example.com)
    """)
