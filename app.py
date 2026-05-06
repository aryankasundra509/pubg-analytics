import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px

st.set_page_config(
    page_title="PUBG Analytics",
    page_icon="🎮",
    layout="wide")

st.title("🎮 PUBG Analytics Dashboard")
st.text("ML-Powered Predictions & Insights based on 4.4 Million Real PUBG Matches")

@st.cache_resource #Model load karke sab resources ko rkhega apne pass
def load_models():
    rf_model = joblib.load('rf_survival_model.pkl')
    kmeans = joblib.load('kmeans_full_model.pkl')
    scaler = joblib.load('scaler_full.pkl')
    with open('model_config.json') as f:
        config = json.load(f)
    return rf_model, kmeans, scaler, config

rf_model, kmeans, scaler, config = load_models()

tab1, tab2, tab3 = st.tabs([
    "🎯 Survival Predictor",
    "👥 Player Segmentation", 
    "📊 EDA & Insights"
])

with tab1:
    st.title("🎯 PUBG Survival Predictor")
    st.markdown("### Enter your match stats — let's see your chances of reaching Top 10!")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        kills = st.slider("Kills", 0, 50, 2)
        damage = st.number_input("Damage Dealt", 0, 6000, 150)
        heals = st.slider("Heals Used", 0, 30, 2)
        boosts = st.slider("Boosts Used", 0, 20, 1)
        walk = st.number_input("Walk Distance (m)", 0, 10000, 1000, 100)
        weapons = st.slider("Weapons Acquired", 0, 20, 4)
        assists = st.slider("Assists", 0, 20, 0)
        dbnos = st.slider("Knock Outs (DBNOs)", 0, 20, 0)

    with col2:
        headshot_kills = st.slider("Headshot Kills", 0, 30, 0)
        kill_streaks = st.slider("Kill Streaks", 0, 20, 0)
        longest_kill = st.number_input("Longest Kill (m)", 0, 1000, 50)
        ride_distance = st.number_input("Ride Distance (m)", 0, 20000, 0)
        revives = st.slider("Revives", 0, 10, 0)
        team_kills = st.slider("Team Kills", 0, 5, 0)
        match_type = st.selectbox("Match Type", ["Solo", "Duo", "Squad", "Other"])
        
    match_map = {"Solo": 0, "Duo": 1, "Squad": 2, "Other": 3}
    match_encoded = match_map[match_type]

    if st.button("🎮 Predict!", use_container_width=True):
        
        error = False
        
        if headshot_kills > kills:
            st.error("❌ Headshot Kills cannot be more than total Kills!")
            error = True
        
        if not error:
            input_data = np.array([[assists, boosts, damage, dbnos, headshot_kills, heals, kills, kill_streaks,
                                    longest_kill, match_encoded, revives, ride_distance, team_kills, walk, weapons]])

            prob = rf_model.predict_proba(input_data)[0][1]
            threshold = config['threshold']
            prediction = int(prob >= threshold)

            st.divider()

            if prediction == 1:
                color = "#00CC44"
                emoji = "🏆"
                title = "TOP 10!"
                desc = f"Your survival probability is {prob*100:.1f}% — Chicken Dinner is within reach!"
            else:
                color = "#FF4B4B"
                emoji = "💀"
                title = "ELIMINATED"
                desc = f"Your survival probability is {prob*100:.1f}% — You need to improve your stats!"

            st.markdown(f"""
                <div style="
                    background-color: {color}22;
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 20px 30px;
                    text-align: center;
                    width: 45%;
                    margin: 0 auto;
                ">
                    <div style="font-size: 48px;">{emoji}</div>
                    <div style="
                        font-size: 42px;
                        font-weight: bold;
                        color: {color};
                        margin: 0px 0;
                    ">{title}</div>
                    <div style="
                        font-size: 24px;
                        color: #CCCCCC;
                    ">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

            st.divider()
            st.markdown("### 💡 Tips to Improve:")

            tips = []
            if walk < 1500:
                tips.append("📍 Move more — your walk distance is too low. Stay with the zone!")
            if boosts < 2:
                tips.append("⚡ Use more boosts — Energy drinks and painkillers are key to late game survival!")
            if heals < 2:
                tips.append("💊 Use more heals — Keep your HP up during fights!")
            if weapons < 3:
                tips.append("🔫 Loot more weapons — Having backup guns saves lives!")
            if kills == 0 and damage < 100:
                tips.append("⚔️ Engage more — You need to deal damage to improve your placement!")
            if team_kills > 0:
                tips.append("🚫 Stop team killing — it hurts your squad's chances!")

            if tips:
                for tip in tips:
                    st.markdown(f"- {tip}")
            else:
                st.markdown("✅ Great game! Keep playing like this and Chicken Dinner is yours!")


with tab2:
    st.title("👥 PUBG Player Segmentation")
    st.markdown("### Enter your match stats — find out if you are a Noob, Average or Pro player!")

    col1, col2 = st.columns([1, 1])

    with col1:
        s_kills = st.slider("Kills", 0, 50, 2, key="seg_kills")
        s_damage = st.number_input("Damage Dealt", 0, 6000, 150, key="seg_damage")
        s_headshot = st.slider("Headshot Kills", 0, 30, 0, key="seg_headshot")
        s_killstreaks = st.slider("Kill Streaks", 0, 20, 0, key="seg_ks")
        s_longest = st.number_input("Longest Kill (m)", 0, 1000, 50, key="seg_long")
        s_roadkills = st.slider("Road Kills", 0, 10, 0, key="seg_rk")

    with col2:
        s_heals = st.slider("Heals Used", 0, 30, 2, key="seg_heals")
        s_boosts = st.slider("Boosts Used", 0, 20, 1, key="seg_boosts")
        s_weapons = st.slider("Weapons Acquired", 0, 20, 4, key="seg_weapons")
        s_walk = st.number_input("Walk Distance (m)", 0, 10000, 1000, key="seg_walk")
        s_ride = st.number_input("Ride Distance (m)", 0, 20000, 0, key="seg_ride")

    if st.button("🔍 Find My Player Type!", use_container_width=True):

        # Input array — same order jaise segmentation_features thi
        seg_input = np.array([[s_kills, s_damage, s_headshot,
                                s_killstreaks, s_longest, s_roadkills,
                                s_heals, s_boosts, s_weapons,
                                s_walk, s_ride]])

        # Scale karo — kmeans ko scaled data chahiye
        seg_scaled = scaler.transform(seg_input)

        # Predict karo
        cluster = kmeans.predict(seg_scaled)[0]

        # Cluster to player type
        cluster_map = {0: 'Average', 1: 'Pro', 2: 'Noob'}
        player_type = cluster_map[cluster]

        st.divider()

        # Player type card — CSS se banaya
        if player_type == 'Pro':
            color = "#00CC44"
            emoji = "🏆"
            desc = "You dominate the battlefield. High kills, high damage, excellent survival instinct."
        elif player_type == 'Average':
            color = "#FFA500"
            emoji = "⚔️"
            desc = "You hold your own in fights. Work on your movement and boost usage to reach Pro level."
        else:
            color = "#FF4B4B"
            emoji = "🎮"
            desc = "Everyone starts somewhere! Focus on survival first — loot, heal, and move with the zone."

        st.markdown(f"""
            <div style="
                background-color: {color}22;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 30px;
                text-align: center;
                width: 45%;
                margin: 0 auto;
            ">
                <div style="font-size: 48px;">{emoji}</div>
                <div style="
                    font-size: 40px;
                    font-weight: bold;
                    color: {color};
                    margin: 0px 0;
                ">{player_type.upper()}</div>
                <div style="
                    font-size: 18px;
                    color: #CCCCCC;
                ">{desc}</div>
            </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 💡 How to improve:")

        tips = []

        if player_type == 'Noob':
            tips.append("📍 Focus on survival — move with the zone, don't rush fights")
            tips.append("💊 Use heals and boosts regularly")
            tips.append("🔫 Loot properly — get at least 2 good weapons before fighting")

        elif player_type == 'Average':
            tips.append("⚡ Use more boosts in late game — they give you an edge")
            tips.append("⚔️ Take more fights — your damage dealt needs to improve")
            tips.append("📍 Improve your positioning — walk more, cover more ground")

        else:
            tips.append("✅ You are already performing at a high level!")
            tips.append("🏆 Focus on consistency — maintain these stats every match")
            tips.append("🎯 Work on your longest kill — sniping can give extra edge")

        for tip in tips:
            st.markdown(f"- {tip}")

with tab3:
    st.title("📊 EDA & Insights")
    st.markdown("### Data Analysis from 4.4 Million PUBG Matches")

    @st.cache_data
    def load_eda_data():
        winner_stats = pd.read_csv('winner_stats.csv', index_col=0)
        match_stats = pd.read_csv('match_stats.csv', index_col=0)
        corr_matrix = pd.read_csv('corr_matrix.csv', index_col=0)
        fi_df = pd.read_csv('feature_importance.csv')

        return winner_stats, match_stats, corr_matrix, fi_df

    winner_stats, match_stats, corr_matrix, fi_df = load_eda_data()

    st.subheader("🏆 Winners vs Losers — Feature Comparison")

    features_show = ['walkDistance', 'boosts', 'heals', 'kills', 'damageDealt', 'weaponsAcquired']

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, feature in enumerate(features_show):
        with cols[i % 3]:
            fig = px.bar(
                x=['Loser', 'Winner'],
                y=[winner_stats.loc[0, feature],winner_stats.loc[1, feature]],
                color=['Loser', 'Winner'],
                color_discrete_map={
                    'Loser': '#FF4B4B',
                    'Winner': '#00CC44'},
                title=feature,
                labels={'x': '', 'y': 'Average'})
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Graph 2 — Feature Importance ----
    st.subheader("🎯 Feature Importance — What matters most?")

    fi_sorted = fi_df.sort_values('Importance', ascending=True)
    fig2 = px.bar(fi_sorted,
                  x='Importance', y='Feature',
                  orientation='h',
                  color='Importance',
                  color_continuous_scale='Greens')
    
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ---- Graph 3 — Match Type Analysis ----
    st.subheader("🎮 Match Type Analysis")

    match_stats.index = ['Solo', 'Duo', 'Squad', 'Other']

    col1, col2 = st.columns(2)

    with col1:
        fig3 = px.bar(match_stats,
                      x=match_stats.index, y='kills',
                      title='Average Kills by Match Type',
                      color=match_stats.index,
                      labels={'y': 'Avg Kills', 'x': ''})
        
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.bar(match_stats,
                      x=match_stats.index, y='walkDistance',
                      title='Average Walk Distance by Match Type',
                      color=match_stats.index,
                      labels={'y': 'Avg Walk Distance', 'x': ''})
        
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ---- Graph 4 — Correlation Heatmap ----
    st.subheader("🔥 Feature Correlation Heatmap")

    import plotly.graph_objects as go

    fig5 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdYlGn',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12}))

    fig5.update_layout(title='Correlation Heatmap')
    st.plotly_chart(fig5, use_container_width=True)