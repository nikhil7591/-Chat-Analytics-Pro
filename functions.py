import re
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import urlextract
import emoji
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords




# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def generateDataFrame(file):
    """Generate DataFrame from WhatsApp chat export file"""
    data = file.read().decode("utf-8")
    data = data.replace('\u202f', ' ')
    data = data.replace('\n', ' ')
    dt_format = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:AM\s|PM\s|am\s|pm\s)?-\s'
    msgs = re.split(dt_format, data)[1:]
    date_times = re.findall(dt_format, data)
    date = []
    time = []
    for dt in date_times:
        date.append(re.search('\d{1,2}/\d{1,2}/\d{2,4}', dt).group())
        time.append(re.search('\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?', dt).group())
    users = []
    message = []
    for m in msgs:
        s = re.split('([\w\W]+?):\s', m)
        if (len(s) < 3):
            users.append("Notifications")
            message.append(s[0])
        else:
            users.append(s[1])
            message.append(s[2])
    df = pd.DataFrame(list(zip(date, time, users, message)), columns=["Date", "Time(U)", "User", "Message"])
    return df


def getUsers(df):
    """Extract unique users from DataFrame"""
    users = df['User'].unique().tolist()
    users.sort()
    users.remove('Notifications')
    users.insert(0, 'Everyone')
    return users


def PreProcess(df, dayfirst):
    """Preprocess DataFrame with date and time features"""
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=dayfirst)
    df['Time'] = pd.to_datetime(df['Time(U)']).dt.time
    df['year'] = df['Date'].apply(lambda x: int(str(x)[:4]))
    df['month'] = df['Date'].apply(lambda x: int(str(x)[5:7]))
    df['date'] = df['Date'].apply(lambda x: int(str(x)[8:10]))
    df['day'] = df['Date'].apply(lambda x: x.day_name())
    df['hour'] = df['Time'].apply(lambda x: int(str(x)[:2]))
    df['month_name'] = df['Date'].apply(lambda x: x.month_name())
    
    # Add day_type feature: Weekday or Weekend
    df['day_type'] = df['day'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')
    
    # Add time_of_day feature
    df['time_of_day'] = df['hour'].apply(categorize_time_of_day)
    
    return df


def categorize_time_of_day(hour):
    """Categorize hour into time of day"""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'


def getStats(df):
    """Get statistics from DataFrame"""
    media = df[df['Message'] == "<Media omitted> "]
    media_cnt = media.shape[0]
    df.drop(media.index, inplace=True)
    
    deleted_msgs = df[df['Message'] == "This message was deleted "]
    deleted_msgs_cnt = deleted_msgs.shape[0]
    df.drop(deleted_msgs.index, inplace=True)
    
    temp = df[df['User'] == 'Notifications']
    df.drop(temp.index, inplace=True)
    
    # Extract links
    extractor = urlextract.URLExtract()
    links = []
    for msg in df['Message']:
        x = extractor.find_urls(msg)
        if x:
            links.extend(x)
    links_cnt = len(links)
    
    # Count words
    word_list = []
    for msg in df['Message']:
        word_list.extend(msg.split())
    word_count = len(word_list)
    
    msg_count = df.shape[0]
    
    # Calculate average message length
    df['message_length'] = df['Message'].apply(len)
    avg_msg_length = round(df['message_length'].mean(), 2)
    
    return df, media_cnt, deleted_msgs_cnt, links_cnt, word_count, msg_count, avg_msg_length


def getEmoji(df):
    """Extract and count emojis from messages"""
    emojis = []
    for message in df['Message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))


def getMonthlyTimeline(df):
    """Get monthly timeline of messages"""
    df.columns = df.columns.str.strip()
    df=df.reset_index()
    timeline = df.groupby(['year', 'month']).count()['Message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(str(timeline['month'][i]) + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline


def MostCommonWords(df, top_n=20):
    """Get most common words used in chat"""
    stop_words_list = set(stopwords.words('english'))
    
    try:
        f = open('stop_hinglish.txt')
        custom_stop_words = f.read()
        f.close()
        
        # Combine both stopword lists
        for word in custom_stop_words.split():
            stop_words_list.add(word.lower())
    except:
        pass
    
    words = []
    for message in df['Message']:
        for word in message.lower().split():
            if word not in stop_words_list and len(word) > 2:
                words.append(word)
    
    return pd.DataFrame(Counter(words).most_common(top_n))


def dailytimeline(df):
    """Plot daily timeline of messages"""
    df['taarek'] = df['Date']
    daily_timeline = df.groupby('taarek').count()['Message'].reset_index()
    
    # Calculate 7-day rolling average
    daily_timeline['Rolling_Average'] = daily_timeline['Message'].rolling(window=7).mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_timeline['taarek'], daily_timeline['Message'], label='Daily Messages', alpha=0.7)
    ax.plot(daily_timeline['taarek'], daily_timeline['Rolling_Average'], 
            color='red', linewidth=2, label='7-day Rolling Average')
    ax.set_ylabel("Messages Sent")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.title('Daily Timeline with Trend Analysis')
    st.pyplot(fig)


def WeekAct(df):
    """Plot weekly activity"""
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day'].value_counts().reindex(weekday_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(day_counts.index, day_counts.values, color=sns.color_palette("viridis", 7))
    ax.set_xlabel("Days of Week")
    ax.set_ylabel("Messages Sent")
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)


def MonthAct(df):
    """Plot monthly activity"""
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['month_name'].value_counts().reindex(month_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(month_counts.index, month_counts.values, color=sns.color_palette("magma", 12))
    ax.set_xlabel("Months")
    ax.set_ylabel("Messages Sent")
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


def activity_heatmap(df):
    """Create heatmap of activity by day and hour"""
    period = []
    for hour in df[['day', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    user_heatmap = df.pivot_table(index='day', columns='period', values='Message', 
                                 aggfunc='count').fillna(0)
    
    # Reindex to ensure correct day order
    user_heatmap = user_heatmap.reindex(day_order)
    
    return user_heatmap


def create_wordcloud(df):
    """Create and return wordcloud from messages"""
    stop_words_list = set(stopwords.words('english'))
    
    try:
        f = open('stop_hinglish.txt', 'r')
        custom_stop_words = f.read()
        f.close()
        
        # Combine both stopword lists
        for word in custom_stop_words.split():
            stop_words_list.add(word.lower())
    except:
        pass
    
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words_list and len(word) > 2:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=800, height=400, min_font_size=10, background_color='white',
                  colormap='viridis', contour_width=1, contour_color='steelblue')
    df['Message'] = df['Message'].apply(remove_stop_words)
    df_wc = wc.generate(df['Message'].str.cat(sep=" "))
    return df_wc


def sentiment_analysis(df):
    """Perform sentiment analysis on messages"""
    # Function to get sentiment score using TextBlob
    def get_sentiment(text):
        analysis = TextBlob(text)
        # Return polarity score: -1 to 1 (negative to positive)
        return analysis.sentiment.polarity
    
    # Apply sentiment analysis to each message
    df['sentiment_score'] = df['Message'].apply(get_sentiment)
    
    # Categorize sentiment into positive, neutral, or negative
    def categorize_sentiment(score):
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment'] = df['sentiment_score'].apply(categorize_sentiment)
    
    # Group by user and calculate average sentiment
    user_sentiment = df.groupby('User')['sentiment_score'].mean().sort_values(ascending=False)
    
    # Count occurrences of each sentiment category overall
    sentiment_counts = df['sentiment'].value_counts()
    
    # Sentiment over time
    sentiment_over_time = df.groupby(['Date'])['sentiment_score'].mean().reset_index()
    sentiment_over_time['Rolling_Avg'] = sentiment_over_time['sentiment_score'].rolling(window=7).mean()
    
    return df, user_sentiment, sentiment_counts, sentiment_over_time


def topic_modeling(df, num_topics=5):
    """Perform topic modeling on messages"""
    # Combine all messages by user
    user_messages = df.groupby('User')['Message'].apply(' '.join).reset_index()
    
    # Create a document-term matrix
    vectorizer = CountVectorizer(
        stop_words='english', 
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.9  # Ignore terms that appear in more than 90% of documents
    )
    
    # If less than 5 users, adjust to the number of users
    if len(user_messages) < num_topics:
        num_topics = max(2, len(user_messages) - 1)
    
    try:
        dtm = vectorizer.fit_transform(user_messages['Message'])
        feature_names = vectorizer.get_feature_names_out()
        
        # Apply LDA
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=20
        )
        
        lda_output = lda_model.fit_transform(dtm)
        
        # Get the top words for each topic
        topic_words = {}
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words[f"Topic {topic_idx+1}"] = top_words
        
        # Get dominant topic for each user
        user_messages['Dominant_Topic'] = lda_output.argmax(axis=1) + 1
        user_topic_df = pd.concat([user_messages['User'], 
                                  pd.DataFrame(lda_output, 
                                             columns=[f'Topic_{i+1}' for i in range(num_topics)])], 
                                 axis=1)
        
        return topic_words, user_topic_df
    except:
        return None, None


def user_clustering(df):
    """Perform clustering to identify user groups"""
    # Create features for each user
    user_stats = df.groupby('User').agg({
        'Message': 'count',
        'message_length': 'mean',
        'sentiment_score': 'mean',
        'hour': 'mean'
    }).reset_index()
    
    # Add feature for percentage of messages sent on weekends
    weekend_pct = df[df['day_type'] == 'Weekend'].groupby('User').size() / df.groupby('User').size()
    user_stats['weekend_pct'] = weekend_pct
    
    # Fill NaN values
    user_stats = user_stats.fillna(0)
    
    # For clustering, exclude the 'User' column and scale the features
    features = user_stats.drop('User', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Determine optimal number of clusters
    n_clusters_range = range(2, min(6, len(user_stats) - 1)) if len(user_stats) > 3 else [2]
    silhouette_scores = []
    
    try:
        for n_clusters in n_clusters_range:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(scaled_features)
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Choose the number of clusters with the highest silhouette score
        optimal_clusters = n_clusters_range[silhouette_scores.index(max(silhouette_scores))]
        
        # Perform final clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        user_stats['Cluster'] = kmeans.fit_predict(scaled_features)
        
        # Get cluster centers and interpret
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Create descriptions for clusters
        cluster_descriptions = []
        for i, center in enumerate(centers):
            desc = f"Cluster {i+1}: "
            if center[0] > features['Message'].mean():
                desc += "Frequent messengers"
            else:
                desc += "Occasional messengers"
            
            if center[1] > features['message_length'].mean():
                desc += ", verbose"
            else:
                desc += ", concise"
            
            if center[2] > features['sentiment_score'].mean():
                desc += ", positive tone"
            elif center[2] < features['sentiment_score'].mean():
                desc += ", negative tone"
            else:
                desc += ", neutral tone"
            
            if center[3] < 12:
                desc += ", morning preference"
            elif center[3] < 17:
                desc += ", afternoon preference"
            elif center[3] < 21:
                desc += ", evening preference"
            else:
                desc += ", night preference"
            
            if center[4] > 0.5:
                desc += ", weekend active"
            else:
                desc += ", weekday active"
            
            cluster_descriptions.append(desc)
        
        return user_stats, cluster_descriptions
    except:
        # If clustering fails, return original user_stats without clustering
        user_stats['Cluster'] = 0
        return user_stats, ["Unable to perform meaningful clustering"]


def get_user_personality(df, user):
    """Generate a simple personality analysis based on user's messaging patterns"""
    if user == "Everyone":
        return None
    
    user_df = df[df['User'] == user]
    
    if len(user_df) < 10:  # Not enough data
        return None
    
    personality = {}
    
    # Message frequency
    msg_per_day = len(user_df) / len(user_df['Date'].dt.date.unique())
    if msg_per_day > 20:
        personality['chattiness'] = "Very talkative"
    elif msg_per_day > 10:
        personality['chattiness'] = "Talkative"
    elif msg_per_day > 5:
        personality['chattiness'] = "Moderately talkative"
    else:
        personality['chattiness'] = "Reserved"
    
    # Message length
    avg_length = user_df['message_length'].mean()
    if avg_length > 100:
        personality['verbosity'] = "Very detailed communicator"
    elif avg_length > 50:
        personality['verbosity'] = "Detailed communicator"
    elif avg_length > 20:
        personality['verbosity'] = "Concise communicator"
    else:
        personality['verbosity'] = "Brief communicator"
    
    # Response time (if available)
    # This is complex and requires determining conversation threads
    
    # Time patterns
    hour_counts = user_df['hour'].value_counts()
    max_hour = hour_counts.idxmax()
    if 5 <= max_hour < 12:
        personality['active_time'] = "Morning person"
    elif 12 <= max_hour < 17:
        personality['active_time'] = "Afternoon person"
    elif 17 <= max_hour < 21:
        personality['active_time'] = "Evening person"
    else:
        personality['active_time'] = "Night owl"
    
    # Weekday vs Weekend
    day_type_counts = user_df['day_type'].value_counts()
    if 'Weekend' in day_type_counts and 'Weekday' in day_type_counts:
        weekend_ratio = day_type_counts['Weekend'] / (day_type_counts['Weekend'] + day_type_counts['Weekday'])
        if weekend_ratio > 0.6:
            personality['day_preference'] = "Weekend chatter"
        elif weekend_ratio < 0.2:
            personality['day_preference'] = "Weekday chatter"
        else:
            personality['day_preference'] = "Consistent throughout the week"
    
    # Sentiment
    avg_sentiment = user_df['sentiment_score'].mean()
    if avg_sentiment > 0.2:
        personality['tone'] = "Very positive"
    elif avg_sentiment > 0.05:
        personality['tone'] = "Positive"
    elif avg_sentiment > -0.05:
        personality['tone'] = "Neutral"
    elif avg_sentiment > -0.2:
        personality['tone'] = "Negative"
    else:
        personality['tone'] = "Very negative"
    
    # Emoji usage
    emoji_count = sum(len([c for c in msg if c in emoji.EMOJI_DATA]) for msg in user_df['Message'])
    emoji_per_msg = emoji_count / len(user_df)
    if emoji_per_msg > 2:
        personality['expressiveness'] = "Very expressive with emojis"
    elif emoji_per_msg > 1:
        personality['expressiveness'] = "Expressive with emojis"
    elif emoji_per_msg > 0.5:
        personality['expressiveness'] = "Occasionally uses emojis"
    elif emoji_per_msg > 0:
        personality['expressiveness'] = "Rarely uses emojis"
    else:
        personality['expressiveness'] = "Doesn't use emojis"
    
    return personality


def conversation_pattern_analysis(df):
    """Analyze conversation patterns"""
    # Messages by time of day
    time_of_day_counts = df['time_of_day'].value_counts()
    
    # Messages by day type (weekend vs weekday)
    day_type_counts = df['day_type'].value_counts()
    
    # Calculate average response time (simplified)
    df['prev_timestamp'] = df['Date'].shift()
    df['prev_user'] = df['User'].shift()
    
    # Filter for actual responses (different user from previous message)
    response_df = df[(df['User'] != df['prev_user']) & (~df['prev_user'].isna())]
    response_df['response_time_min'] = (response_df['Date'] - response_df['prev_timestamp']).dt.total_seconds() / 60
    
    # Filter out unreasonable response times (e.g., > 24 hours)
    response_df = response_df[response_df['response_time_min'] < 24*60]
    
    avg_response_time = response_df['response_time_min'].mean() if len(response_df) > 0 else None
    
    return time_of_day_counts, day_type_counts, avg_response_time


def predict_user_activity(df, user=None):
    """Simple prediction model for user activity patterns"""
    if user and user != "Everyone":
        df = df[df['User'] == user]
    
    if len(df) < 30:  # Not enough data for meaningful predictions
        return None, None
    
    # Group by hour and day to get message frequency
    hourly_activity = df.groupby(['day', 'hour']).size().reset_index(name='message_count')
    
    # Create hour of week feature (0-167)
    day_map = {
        'Monday': 0,
        'Tuesday': 24,
        'Wednesday': 48,
        'Thursday': 72,
        'Friday': 96,
        'Saturday': 120,
        'Sunday': 144
    }
    
    hourly_activity['hour_of_week'] = hourly_activity['day'].map(day_map) + hourly_activity['hour']
    
    # Find top 5 active hours
    top_hours = hourly_activity.sort_values('message_count', ascending=False).head(5)
    
    # Describe top hours in natural language
    predictions = []
    for _, row in top_hours.iterrows():
        day = row['day']
        hour = row['hour']
        count = row['message_count']
        
        # Format time
        if hour < 12:
            time_str = f"{hour} AM"
        elif hour == 12:
            time_str = "12 PM"
        else:
            time_str = f"{hour-12} PM"
        
        predictions.append(f"{day} at {time_str} ({count} messages)")
    
    # Create a heatmap data for visualization
    heatmap_data = df.groupby(['day', 'hour']).size().unstack(fill_value=0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    return predictions, heatmap_data



def seasonal_analysis(df):
    """
    Analyze seasonal patterns in chat activity by hour, day, and month.
    
    Parameters:
    df (pandas.DataFrame): Preprocessed chat dataframe
    
    Returns:
    dict: Dictionary containing hourly, daily, and monthly DataFrames
    """
    import pandas as pd
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Check if dataframe is empty or missing required columns
    required_columns = ['hour', 'day', 'month_name', 'Message']
    if df_copy.empty or not all(col in df_copy.columns for col in required_columns):
        return None
    
    # Hourly analysis
    hourly_data = df_copy.groupby('hour').count()['Message'].reset_index()
    hourly_data.columns = ['hour', 'message_count']
    
    # Daily analysis - group by day of week
    if 'day' in df_copy.columns:
        daily_data = df_copy.groupby('day').count()['Message'].reset_index()
        daily_data.columns = ['day_name', 'message_count']
    else:
        # If day column is missing, create empty dataframe
        daily_data = pd.DataFrame(columns=['day_name', 'message_count'])
    
    # Monthly analysis
    monthly_data = df_copy.groupby('month_name').count()['Message'].reset_index()
    monthly_data.columns = ['month_name', 'message_count']
    
    # Create dictionary with all seasonal data
    seasonal_data = {
        'hourly': hourly_data,
        'daily': daily_data,
        'monthly': monthly_data
    }
    
    return seasonal_data


def calculate_message_intensity(df):
    """
    Calculate message intensity as messages per minute over time.
    
    Parameters:
    df (pandas.DataFrame): Preprocessed chat dataframe
    
    Returns:
    pandas.DataFrame: DataFrame with datetime and intensity columns
    """
    import pandas as pd
    from datetime import timedelta
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Check if dataframe is empty or missing Date column
    if df_copy.empty or 'Date' not in df_copy.columns or 'Time' not in df_copy.columns:
        return pd.DataFrame()
    
    # Make sure we have datetime information
    try:
        # Create a combined datetime column if it doesn't exist
        if 'datetime' not in df_copy.columns:
            # Ensure Date is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_copy['Date']):
                df_copy['Date'] = pd.to_datetime(df_copy['Date'])
                
            # Create datetime by combining date and time
            df_copy['datetime'] = pd.to_datetime(
                df_copy['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
                df_copy['Time'].astype(str)
            )
    except:
        # If datetime creation fails, return empty dataframe
        return pd.DataFrame()
    
    # Sort by datetime
    df_copy = df_copy.sort_values('datetime')
    
    # Calculate message intensity (messages per minute)
    # Group by minute and count messages
    df_copy['minute'] = df_copy['datetime'].dt.floor('min')
    minute_counts = df_copy.groupby('minute').size().reset_index(name='count')
    minute_counts.columns = ['datetime', 'intensity']
    
    return minute_counts


def calculate_long_term_trends(df):
    """
    Calculate long-term message trends including daily counts and rolling averages.
    
    Parameters:
    df (pandas.DataFrame): Preprocessed chat dataframe
    
    Returns:
    pandas.DataFrame: DataFrame with date, message_count, and rolling averages
    """
    import pandas as pd
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Check if dataframe is empty or missing Date column
    if df_copy.empty or 'Date' not in df_copy.columns:
        return pd.DataFrame()
    
    # Make sure Date is datetime type
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_copy['Date']):
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    except:
        return pd.DataFrame()
    
    # Extract just the date portion
    df_copy['date'] = df_copy['Date'].dt.date
    
    # Group by date and count messages
    daily_counts = df_copy.groupby('date').size().reset_index(name='message_count')
    
    # Convert date column to datetime for sorting and calculations
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    # Sort by date
    daily_counts = daily_counts.sort_values('date')
    
    # Calculate rolling averages
    daily_counts['7d_rolling_avg'] = daily_counts['message_count'].rolling(window=7, min_periods=1).mean()
    daily_counts['30d_rolling_avg'] = daily_counts['message_count'].rolling(window=30, min_periods=1).mean()
    
    return daily_counts
