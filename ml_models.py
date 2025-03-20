import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st

def perform_message_trend_prediction(df, prediction_days=30):
    """Predict future message trends using Facebook Prophet"""
    # Prepare data for Prophet
    message_counts = df.groupby(pd.Grouper(key='Date', freq='D')).size().reset_index(name='messages')
    
    # Rename columns to match Prophet requirements
    message_counts.columns = ['ds', 'y']
    
    # Create and train Prophet model
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    m.fit(message_counts)
    
    # Create future dataframe for prediction
    future = m.make_future_dataframe(periods=prediction_days)
    
    # Make predictions
    forecast = m.predict(future)
    
    # Create forecast plot
    fig = m.plot(forecast)
    plt.title('Message Frequency Forecast')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    
    # Add confidence interval legend
    plt.plot([], [], 'b-', label='Actual')
    plt.plot([], [], 'r-', label='Predicted')
    plt.fill_between([], [], [], color='#0072B2', alpha=0.2, label='Confidence Interval')
    plt.legend()
    
    # Create components plot (trends, weekly patterns)
    components_fig = m.plot_components(forecast)
    
    return fig, components_fig, forecast

def user_engagement_analysis(df):
    """Analyze user engagement patterns"""
    # Create user engagement metrics
    user_engagement = df.groupby('User').agg({
        'Message': 'count',
        'message_length': ['mean', 'max', 'min', 'std'],
        'Date': pd.Series.nunique  # number of days active
    }).reset_index()
    
    # Flatten multi-level column names
    user_engagement.columns = ['_'.join(col).strip('_') for col in user_engagement.columns.values]
    
    # Rename columns for clarity
    user_engagement = user_engagement.rename(columns={
        'User_': 'User',
        'Message_count': 'message_count',
        'message_length_mean': 'avg_message_length',
        'message_length_max': 'max_message_length',
        'message_length_min': 'min_message_length',
        'message_length_std': 'std_message_length',
        'Date_nunique': 'days_active'
    })
    
    # Calculate messages per day active
    user_engagement['msgs_per_day'] = user_engagement['message_count'] / user_engagement['days_active']
    
    # Calculate total days in chat (from first to last message)
    total_days = (df['Date'].max() - df['Date'].min()).days + 1
    
    # Calculate participation rate (days active / total days)
    user_engagement['participation_rate'] = user_engagement['days_active'] / total_days
    
    # Calculate z-scores for key metrics to identify outliers
    metrics = ['message_count', 'avg_message_length', 'msgs_per_day']
    for metric in metrics:
        user_engagement[f'{metric}_zscore'] = (user_engagement[metric] - user_engagement[metric].mean()) / user_engagement[metric].std()
    
    # Identify highly engaged users (high in messages and participation)
    user_engagement['engagement_score'] = (
        (user_engagement['message_count_zscore'] + 
         user_engagement['participation_rate'] * 2) / 3
    )
    
    return user_engagement


def conversation_topic_classifier(df, n_topics=5):
    """Classify conversations into topics using TF-IDF and clustering"""
    # Get messages longer than 10 characters to avoid noise
    long_messages = df[df['Message'].str.len() > 10]
    
    if len(long_messages) < 10:  # Not enough data
        return None, None
    
    # Convert messages to TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        max_df=0.9
    )
    
    try:
        # Create document-term matrix
        X = vectorizer.fit_transform(long_messages['Message'])
        
        # Apply dimensionality reduction (PCA)
        pca = PCA(n_components=min(10, X.shape[0]-1))
        X_pca = pca.fit_transform(X.toarray())
        
        # Determine optimal number of clusters
        n_clusters_range = range(2, min(n_topics+1, X_pca.shape[0]))
        
        if len(n_clusters_range) < 2:  # Not enough data for multiple clusters
            return None, None
            
        silhouette_scores = []
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_pca)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find optimal number of clusters
        optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
        long_messages['topic'] = kmeans.fit_predict(X_pca)
        
        # Get top words for each topic
        topic_keywords = {}
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        
        for i in range(optimal_n_clusters):
            topic_terms = [terms[ind] for ind in order_centroids[i, :10]]
            topic_keywords[f"Topic {i+1}"] = topic_terms
        
        return long_messages, topic_keywords
    
    except Exception as e:
        print(f"Error in topic classification: {e}")
        return None, None

def user_response_analysis(df):
    """Analyze response patterns between users"""
    if len(df) < 30:  # Not enough data
        return None
    
    # Sort by date and time
    df = df.sort_values(by=['Date', 'Time'])
    
    # Add previous message user and time
    df['prev_user'] = df['User'].shift(1)
    df['prev_time'] = df['Date'].shift(1)
    
    # Calculate response time in minutes
    df['response_time'] = (df['Date'] - df['prev_time']).dt.total_seconds() / 60
    
    # Filter for actual responses (different user from previous)
    responses = df[(df['User'] != df['prev_user']) & (df['response_time'] < 60*24)]  # Limit to 24 hours
    
    if len(responses) < 10:  # Not enough response data
        return None
    
    # Group by user pairs with explicit column names to avoid multi-level issues
    message_counts = responses.groupby(['prev_user', 'User']).size().reset_index(name='response_count')
    
    # Calculate response time metrics separately
    response_times = responses.groupby(['prev_user', 'User'])['response_time'].agg([
        ('avg_response_time', 'mean'),
        ('median_response_time', 'median'),
        ('min_response_time', 'min'),
        ('max_response_time', 'max')
    ]).reset_index()
    
    # Merge the response counts and response times
    user_pairs = message_counts.merge(response_times, on=['prev_user', 'User'])
    
    # Rename columns for clarity
    user_pairs = user_pairs.rename(columns={
        'prev_user': 'sender',
        'User': 'responder'
    })
    
    # Calculate response rates
    total_messages_by_user = df.groupby('User').size().reset_index(name='total_messages')
    
    # Merge with user_pairs
    user_pairs = user_pairs.merge(
        total_messages_by_user.rename(columns={'User': 'sender', 'total_messages': 'sender_messages'}),
        on='sender',
        how='left'
    )
    
    # Calculate response rate
    user_pairs['response_rate'] = user_pairs['response_count'] / user_pairs['sender_messages']
    
    return user_pairs

def conversation_intensity_analysis(df):
    """Analyze conversation intensity patterns"""
    # Create hourly message counts
    hourly_counts = df.groupby([df['Date'].dt.date, df['hour']]).size().reset_index(name='message_count')
    
    # Identify active conversation hours (more than 10 messages)
    active_hours = hourly_counts[hourly_counts['message_count'] > 10]
    
    if len(active_hours) < 5:  # Not enough data
        return None, None
    
    # Calculate conversation intensity (messages per minute)
    active_hours['intensity'] = active_hours['message_count'] / 60
    
    # Find peak conversation times
    peak_times = active_hours.sort_values('message_count', ascending=False).head(5)
    
    # Calculate average daily message count
    daily_counts = df.groupby(df['Date'].dt.date).size()
    avg_daily_messages = daily_counts.mean()
    busiest_day = daily_counts.idxmax()
    busiest_day_count = daily_counts.max()
    
    # Calculate consecutive busy days (days with more than average messages)
    busy_days = daily_counts[daily_counts > avg_daily_messages].index
    if len(busy_days) > 1:
        busy_days = pd.Series(busy_days).sort_values()
        busy_days_diff = busy_days.diff().dt.days
        max_consecutive = 1
        current_consecutive = 1
        
        for diff in busy_days_diff:
            if diff == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
    else:
        max_consecutive = len(busy_days)
    
    # Create summary stats
    summary = {
        'avg_daily_messages': avg_daily_messages,
        'busiest_day': busiest_day,
        'busiest_day_count': busiest_day_count,
        'max_consecutive_busy_days': max_consecutive,
        'peak_hour_intensity': peak_times['intensity'].max() if not peak_times.empty else 0
    }
    
    return peak_times, summary

def communication_pattern_prediction(df, selected_user=None):
    """Predict communication patterns for a specific user or everyone"""
    if selected_user and selected_user != "Everyone":
        user_df = df[df['User'] == selected_user]
    else:
        user_df = df
    
    if len(user_df) < 50:  # Not enough data
        return None, None
    
    # Aggregate data by day
    daily_messages = user_df.groupby(user_df['Date'].dt.date).size().reset_index(name='message_count')
    daily_messages['day_of_week'] = pd.to_datetime(daily_messages['Date']).dt.dayofweek
    
    # Calculate rolling averages
    daily_messages['rolling_avg'] = daily_messages['message_count'].rolling(window=7, min_periods=1).mean()
    
    # Prepare features
    X = pd.get_dummies(daily_messages['day_of_week'], prefix='day')
    X['week_num'] = (pd.to_datetime(daily_messages['Date']) - pd.to_datetime(daily_messages['Date']).min()).dt.days // 7
    
    # Target variable
    y = daily_messages['message_count']
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.fit(X_train, y_train).predict(X_test)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create prediction for next week
        last_date = pd.to_datetime(daily_messages['Date']).max()
        next_week_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
        
        next_week_df = pd.DataFrame({
            'Date': next_week_dates,
            'day_of_week': [date.dayofweek for date in next_week_dates]
        })
        
        next_week_X = pd.get_dummies(next_week_df['day_of_week'], prefix='day')
        
        # Handle missing columns
        for col in X.columns:
            if col not in next_week_X.columns and col != 'week_num':
                next_week_X[col] = 0
        
        next_week_X['week_num'] = (X['week_num'].max() + 1)
        
        # Make sure columns are in the same order
        next_week_X = next_week_X[X.columns]
        
        # Predict
        next_week_pred = model.predict(next_week_X)
        
        # Combine with dates
        next_week_df['predicted_messages'] = next_week_pred
        
        return next_week_df, feature_importance
    
    except Exception as e:
        print(f"Error in communication pattern prediction: {e}")
        return None, None

def user_interaction_network(df):
    """Create a user interaction network analysis"""
    if len(df) < 50:  # Not enough data
        return None
    
    # Sort by date and time
    df = df.sort_values(by=['Date', 'Time'])
    
    # Add previous message user
    df['prev_user'] = df['User'].shift(1)
    
    # Filter for actual interactions (different user from previous)
    interactions = df[df['User'] != df['prev_user']].copy()
    
    if len(interactions) < 10:  # Not enough interaction data
        return None
    
    # Count interactions between users
    user_interactions = interactions.groupby(['prev_user', 'User']).size().reset_index(name='interaction_count')
    
    # Remove self-interactions and None values
    user_interactions = user_interactions[
        (user_interactions['prev_user'] != user_interactions['User']) & 
        (~user_interactions['prev_user'].isna()) &
        (~user_interactions['User'].isna())
    ]
    
    # Calculate normalized interaction strength
    max_interactions = user_interactions['interaction_count'].max()
    user_interactions['interaction_strength'] = user_interactions['interaction_count'] / max_interactions
    
    # Calculate total interactions for each user
    user_total_interactions = pd.concat([
        user_interactions.groupby('User')['interaction_count'].sum().reset_index(name='total_interactions'),
        user_interactions.groupby('prev_user')['interaction_count'].sum().reset_index().rename(columns={'prev_user': 'User', 'interaction_count': 'initiated_interactions'})
    ], axis=0).groupby('User').sum().reset_index()
    
    # Merge with user_interactions
    user_interactions = user_interactions.merge(
        user_total_interactions,
        left_on='User',
        right_on='User',
        how='left'
    )
    
    user_interactions = user_interactions.merge(
        user_total_interactions,
        left_on='prev_user',
        right_on='User',
        how='left',
        suffixes=('', '_initiator')
    ).rename(columns={'User_initiator': 'init_user'})
    
    # Calculate interaction percentage
    user_interactions['interaction_percentage'] = user_interactions['interaction_count'] / user_interactions['total_interactions'] * 100
    
    return user_interactions

def generate_chat_analytics_report(df, selected_user=None):
    """Generate comprehensive chat analytics report"""
    if selected_user and selected_user != "Everyone":
        user_df = df[df['User'] == selected_user]
        title = f"Chat Analytics Report for {selected_user}"
    else:
        user_df = df
        title = "Chat Analytics Report for All Users"
    
    # Basic statistics
    total_messages = len(user_df)
    total_days = (user_df['Date'].max() - user_df['Date'].min()).days + 1
    avg_messages_per_day = round(total_messages / total_days, 2)
    
    # Message length statistics
    avg_message_length = round(user_df['message_length'].mean(), 2)
    
    # Time statistics
    active_hours = user_df.groupby('hour').size().reset_index(name='message_count')
    peak_hour = active_hours.loc[active_hours['message_count'].idxmax(), 'hour']
    
    # Day statistics
    active_days = user_df.groupby('day').size().reset_index(name='message_count')
    peak_day = active_days.loc[active_days['message_count'].idxmax(), 'day']
    
    # Sentiment statistics
    avg_sentiment = round(user_df['sentiment_score'].mean(), 2)
    sentiment_distribution = user_df['sentiment'].value_counts()
    
    # Package all statistics
    report = {
        'title': title,
        'total_messages': total_messages,
        'total_days': total_days,
        'avg_messages_per_day': avg_messages_per_day,
        'avg_message_length': avg_message_length,
        'peak_hour': peak_hour,
        'peak_day': peak_day,
        'avg_sentiment': avg_sentiment,
        'sentiment_distribution': sentiment_distribution
    }
    
    return report

def topic_sentiment_analysis(df):
    """Analyze sentiment by topic in messages"""
    # Check if we have enough data
    if len(df) < 20:  # Not enough data
        return None
    
    # First get topic classifications
    topic_messages, _ = conversation_topic_classifier(df)
    
    if topic_messages is None:
        return None
    
    # Make sure sentiment_score column exists
    if 'sentiment_score' not in topic_messages.columns:
        return None
    
    # Group by topic and calculate average sentiment
    topic_sentiment = topic_messages.groupby('topic').agg({
        'sentiment_score': 'mean',
        'Message': 'count'
    }).reset_index()
    
    # Rename columns
    topic_sentiment = topic_sentiment.rename(columns={
        'sentiment_score': 'avg_sentiment',
        'Message': 'message_count'
    })
    
    # Filter for topics with enough messages for reliable sentiment
    topic_sentiment = topic_sentiment[topic_sentiment['message_count'] >= 5]
    
    # If we have no topics with enough messages, return None
    if topic_sentiment.empty:
        return None
    
    # Convert topic numbers to topic labels
    topic_sentiment['topic'] = topic_sentiment['topic'].apply(lambda x: f'Topic {x+1}')
    
    return topic_sentiment

def conversation_tone_analysis(df):
    """Analyze the tone of conversations over time"""
    # Check if we have enough data and required columns
    if len(df) < 20 or 'sentiment_score' not in df.columns:
        return None
    
    # Group by date and calculate average sentiment
    daily_sentiment = df.groupby(df['Date'].dt.date).agg({
        'sentiment_score': 'mean',
        'Message': 'count'
    }).reset_index()
    
    # Rename columns
    daily_sentiment = daily_sentiment.rename(columns={
        'sentiment_score': 'avg_sentiment',
        'Message': 'message_count'
    })
    
    # Calculate rolling average (7-day window)
    daily_sentiment['rolling_sentiment'] = daily_sentiment['avg_sentiment'].rolling(window=7, min_periods=1).mean()
    
    # Identify tone shifts (days where sentiment changes significantly)
    daily_sentiment['sentiment_shift'] = daily_sentiment['avg_sentiment'].diff()
    
    # Classify days by tone
    daily_sentiment['tone'] = pd.cut(
        daily_sentiment['avg_sentiment'],
        bins=[-1, -0.3, 0.3, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    # Calculate tone distribution
    tone_distribution = daily_sentiment['tone'].value_counts().reset_index()
    tone_distribution.columns = ['tone', 'count']
    
    # Calculate percentage of each tone
    total_days = tone_distribution['count'].sum()
    tone_distribution['percentage'] = tone_distribution['count'] / total_days * 100
    
    # Identify significant tone shifts
    threshold = daily_sentiment['sentiment_shift'].std() * 1.5
    significant_shifts = daily_sentiment[abs(daily_sentiment['sentiment_shift']) > threshold].copy()
    
    # Add shift direction
    significant_shifts['shift_direction'] = significant_shifts['sentiment_shift'].apply(
        lambda x: 'Positive' if x > 0 else 'Negative'
    )
    
    # Filter for days with enough messages
    daily_sentiment = daily_sentiment[daily_sentiment['message_count'] >= 5]
    
    # Return results
    return {
        'daily_sentiment': daily_sentiment,
        'tone_distribution': tone_distribution,
        'significant_shifts': significant_shifts
    }

def user_communication_style(df):
    """Analyze communication style patterns for each user"""
    if len(df) < 30:  # Not enough data
        return None
    
    # Initialize user style dataframe
    users = df['User'].unique()
    
    # Create a dictionary to store user styles
    user_styles = []
    
    for user in users:
        user_df = df[df['User'] == user]
        
        if len(user_df) < 10:  # Skip users with too few messages
            continue
            
        # Calculate message length statistics
        avg_length = user_df['message_length'].mean()
        max_length = user_df['message_length'].max()
        
        # Calculate message frequency
        user_days = user_df['Date'].dt.date.nunique()
        msgs_per_day = len(user_df) / user_days if user_days > 0 else 0
        
        # Calculate average sentiment (if available)
        avg_sentiment = user_df['sentiment_score'].mean() if 'sentiment_score' in user_df.columns else 0
        
        # Calculate response time (if applicable)
        response_times = df[df['prev_user'] == user]['response_time'] if 'response_time' in df.columns else None
        avg_response_time = response_times.mean() if response_times is not None and len(response_times) > 0 else None
        
        # Calculate style metrics
        
        # 1. Formality score (based on message length, punctuation, etc.)
        # Check for formal language markers
        formal_markers = ['please', 'thank you', 'regards', 'sincerely', 'would you', 'could you']
        informal_markers = ['lol', 'haha', 'yeah', 'nah', 'btw', 'gonna', 'wanna']
        
        formality_score = 0.5  # Default neutral score
        
        # Count formal/informal markers
        formal_count = sum(user_df['Message'].str.lower().str.contains(marker, regex=False).sum() 
                           for marker in formal_markers)
        informal_count = sum(user_df['Message'].str.lower().str.contains(marker, regex=False).sum() 
                             for marker in informal_markers)
        
        # Adjust formality based on markers
        total_markers = formal_count + informal_count
        if total_markers > 0:
            formality_score = min(1.0, max(0.0, 0.5 + 0.5 * (formal_count - informal_count) / total_markers))
        
        # Adjust formality based on message length (longer messages tend to be more formal)
        if avg_length > 50:
            formality_score = min(1.0, formality_score + 0.1)
        elif avg_length < 15:
            formality_score = max(0.0, formality_score - 0.1)
        
        # 2. Expressiveness (based on emoji usage, exclamations, etc.)
        expressiveness_score = 0.5  # Default neutral score
        
        # Count emojis if available
        emoji_rate = user_df['emoji_count'].mean() if 'emoji_count' in user_df.columns else 0
        
        # Count exclamation marks
        exclamation_rate = user_df['Message'].str.count('!').mean()
        
        # Count question marks
        question_rate = user_df['Message'].str.count('\\?').mean()
        
        # Adjust expressiveness based on emoji and punctuation usage
        expressiveness_score = min(1.0, max(0.0, 0.5 + 0.2 * emoji_rate + 0.1 * exclamation_rate + 0.05 * question_rate))
        
        # 3. Complexity (based on message length, vocabulary diversity, etc.)
        complexity_score = 0.5  # Default neutral score
        
        # Adjust complexity based on message length
        if avg_length > 50:
            complexity_score += 0.2
        elif avg_length > 30:
            complexity_score += 0.1
        elif avg_length < 15:
            complexity_score -= 0.1
            
        # Calculate vocabulary diversity (if messages are long enough)
        if avg_length > 20:
            # Get all words
            all_words = ' '.join(user_df['Message']).lower().split()
            # Calculate unique word ratio
            if len(all_words) > 0:
                unique_ratio = len(set(all_words)) / len(all_words)
                complexity_score += min(0.3, unique_ratio * 0.5)
        
        complexity_score = min(1.0, max(0.0, complexity_score))
        
        # 4. Politeness (based on thank you, please, etc.)
        politeness_score = 0.5  # Default neutral score
        
        polite_markers = ['please', 'thank', 'thanks', 'appreciate', 'sorry', 'excuse']
        polite_count = sum(user_df['Message'].str.lower().str.contains(marker, regex=False).sum() 
                           for marker in polite_markers)
        
        # Adjust politeness based on polite marker frequency
        politeness_score += min(0.4, polite_count / len(user_df) * 0.8)
        politeness_score = min(1.0, max(0.0, politeness_score))
        
        # 5. Directness (based on message length, question frequency, etc.)
        directness_score = 0.5  # Default neutral score
        
        # Shorter messages tend to be more direct
        if avg_length < 15:
            directness_score += 0.2
        elif avg_length > 50:
            directness_score -= 0.1
            
        # Questions are usually less direct
        if question_rate > 0.5:
            directness_score -= 0.1
            
        # Check for directive language
        directive_markers = ['do this', 'please do', 'need to', 'should', 'must', 'have to']
        directive_count = sum(user_df['Message'].str.lower().str.contains(marker, regex=False).sum() 
                              for marker in directive_markers)
        
        # Adjust directness based on directive marker frequency
        directness_score += min(0.3, directive_count / len(user_df) * 0.6)
        directness_score = min(1.0, max(0.0, directness_score))
        
        # Determine overall style traits
        style_traits = []
        
        # Message length style
        if avg_length < 15:
            length_style = "Concise"
        elif avg_length < 40:
            length_style = "Moderate"
        else:
            length_style = "Detailed"
        style_traits.append(length_style)
        
        # Frequency style
        if msgs_per_day < 3:
            freq_style = "Infrequent"
        elif msgs_per_day < 10:
            freq_style = "Regular"
        else:
            freq_style = "Frequent"
        style_traits.append(freq_style)
        
        # Sentiment style (if available)
        if 'sentiment_score' in user_df.columns:
            if avg_sentiment < -0.2:
                sent_style = "Negative"
            elif avg_sentiment > 0.2:
                sent_style = "Positive"
            else:
                sent_style = "Neutral"
            style_traits.append(sent_style)
        
        # Store user style information
        user_styles.append({
            'User': user,
            'message_count': len(user_df),
            'avg_message_length': avg_length,
            'max_message_length': max_length,
            'msgs_per_day': msgs_per_day,
            'avg_sentiment': avg_sentiment if 'sentiment_score' in user_df.columns else None,
            'avg_response_time': avg_response_time,
            'formality': formality_score,
            'expressiveness': expressiveness_score,
            'complexity': complexity_score,
            'politeness': politeness_score,
            'directness': directness_score,
            'primary_style': " & ".join(style_traits[:2]),  # Main style descriptor
            'style_traits': style_traits
        })
    
    # Convert to dataframe
    user_styles_df = pd.DataFrame(user_styles)
    
    return user_styles_df

def extract_key_phrases(df):
    """Extract key phrases and topics from chat messages"""
    
    # Check if we have enough data
    if len(df) < 20:  # Not enough data
        return None
    
    # Get messages longer than 5 characters to avoid noise
    long_messages = df[df['Message'].str.len() > 5]
    
    if len(long_messages) < 10:  # Not enough data
        return None
    
    try:
        # Convert messages to one text corpus
        all_text = ' '.join(long_messages['Message'].tolist())
        
        # Basic preprocessing
        # Remove URLs
        import re
        all_text = re.sub(r'http\S+', '', all_text)
        
        # Convert to lowercase
        all_text = all_text.lower()
        
        # Tokenize the text
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Tokenize by sentence, then by word
        sentences = sent_tokenize(all_text)
        word_frequencies = {}
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            
            for word in words:
                if word.lower() not in stop_words and word.isalnum() and len(word) > 2:
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        
        # Normalize word frequencies
        max_frequency = max(word_frequencies.values()) if word_frequencies else 1
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
        
        # Extract n-grams (phrases)
        try:
            from nltk.util import ngrams
        except:
            nltk.download('util', quiet=True)
            from nltk.util import ngrams
            
        phrases = {}
        
        # Process each message separately
        for message in long_messages['Message']:
            # Clean and tokenize
            clean_msg = re.sub(r'http\S+', '', message.lower())
            tokens = word_tokenize(clean_msg)
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words 
                              and word.isalnum() and len(word) > 2]
            
            # Extract bigrams and trigrams
            if len(filtered_tokens) >= 2:
                message_bigrams = list(ngrams(filtered_tokens, 2))
                for gram in message_bigrams:
                    if gram in phrases:
                        phrases[gram] += 1
                    else:
                        phrases[gram] = 1
            
            if len(filtered_tokens) >= 3:
                message_trigrams = list(ngrams(filtered_tokens, 3))
                for gram in message_trigrams:
                    if gram in phrases:
                        phrases[gram] += 1
                    else:
                        phrases[gram] = 1
        
        # Normalize phrase frequencies
        max_phrase_freq = max(phrases.values()) if phrases else 1
        for phrase in phrases:
            phrases[phrase] = phrases[phrase] / max_phrase_freq
        
        # Filter out less frequent phrases
        min_freq = 0.1  # Minimum normalized frequency threshold
        phrases = {phrase: score for phrase, score in phrases.items() if score >= min_freq}
        
        # Try to apply TextRank or similar algorithm if we can
        # This uses a simplified scoring approach instead of full TextRank
        # but gives reasonably good results for key phrase extraction
        
        # Score phrases by word importance
        for phrase in list(phrases.keys()):
            phrase_words = phrase
            phrase_score = sum(word_frequencies.get(word, 0) for word in phrase_words)
            phrases[phrase] = phrase_score / len(phrase_words)
        
        # Return phrases sorted by score
        return phrases
        
    except Exception as e:
        print(f"Error in key phrase extraction: {e}")
        return None


