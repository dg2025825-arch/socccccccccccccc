import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# 페이지 기본 설정 (브라우저 탭 제목 + 아이콘)
# ------------------------------------------------------------
st.set_page_config(
    page_title="선수 유형 나누기",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ 선수 유형 나누기")
st.write("EA FC25 상위 100명 선수의 능력치를 활용해 K-평균 군집분석으로 선수 유형을 나눠봅니다.")

# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/eafc25_top100.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

# 능력치 영문 -> 한글 매핑
STAT_MAP = {
    "pace": "속도",
    "shooting": "슈팅",
    "passing": "패스",
    "dribbling": "드리블",
    "defending": "수비",
    "physic": "몸싸움"
}
STAT_COLS_EN = list(STAT_MAP.keys())
STAT_COLS_KO = list(STAT_MAP.values())

# 한글 컬럼 추가 (원본은 유지)
df_display = df.copy()
for en, ko in STAT_MAP.items():
    df_display[ko] = df_display[en]

# 포지션 대분류 함수
def classify_position(pos_str):
    first_pos = str(pos_str).split(",")[0].strip()
    forward = ["ST", "CF", "LW", "RW"]
    midfielder = ["CM", "CDM", "CAM", "LM", "RM"]
    defender = ["CB", "LB", "RB", "LWB", "RWB"]
    if first_pos in forward:
        return "공격수"
    elif first_pos in midfielder:
        return "미드필더"
    elif first_pos in defender:
        return "수비수"
    else:
        return "기타"

df_display["포지션대분류"] = df_display["positions"].apply(classify_position)

# ------------------------------------------------------------
# 사이드바: 능력치 선택 & 묶음 수 선택
# ------------------------------------------------------------
st.sidebar.header("⚙️ 분석 설정")

selected_stats_ko = st.sidebar.multiselect(
    "묶는 데 사용할 능력치를 선택하세요 (2개 이상)",
    options=STAT_COLS_KO,
    default=STAT_COLS_KO
)

if len(selected_stats_ko) < 2:
    st.warning("⚠️ 능력치를 2개 이상 선택해야 분석이 가능합니다.")
    st.stop()

selected_stats_en = [en for en, ko in STAT_MAP.items() if ko in selected_stats_ko]

n_clusters = st.sidebar.slider(
    "묶음(군집) 수를 선택하세요",
    min_value=2, max_value=6, value=3, step=1
)

RANDOM_STATE = 42  # 난수 고정

# ------------------------------------------------------------
# 표준화 및 K-평균 군집분석
# ------------------------------------------------------------
X = df_display[selected_stats_ko].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)

df_display["raw_cluster"] = raw_labels

# 슈팅 평균이 큰 묶음부터 ㉮, ㉯, ㉰... 순서로 재배정
cluster_shooting_mean = df_display.groupby("raw_cluster")["슈팅"].mean().sort_values(ascending=False)
CIRCLE_LABELS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳"]
label_mapping = {raw_c: CIRCLE_LABELS[i] for i, raw_c in enumerate(cluster_shooting_mean.index)}
df_display["묶음"] = df_display["raw_cluster"].map(label_mapping)

cluster_order = [label_mapping[c] for c in cluster_shooting_mean.index]

# ------------------------------------------------------------
# 2차원 산점도
# ------------------------------------------------------------
st.header("📊 2차원 산점도")

col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("가로축 능력치", options=selected_stats_ko, index=0, key="2d_x")
with col2:
    y_axis = st.selectbox("세로축 능력치", options=selected_stats_ko, index=min(1, len(selected_stats_ko)-1), key="2d_y")

fig_2d = px.scatter(
    df_display,
    x=x_axis, y=y_axis,
    color="묶음",
    category_orders={"묶음": cluster_order},
    hover_name="name_ko",
    hover_data={x_axis: True, y_axis: True, "묶음": True},
    title=f"{x_axis} vs {y_axis} 산점도",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig_2d.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
fig_2d.update_layout(height=500)
st.plotly_chart(fig_2d, use_container_width=True)

# ------------------------------------------------------------
# 3차원 산점도
# ------------------------------------------------------------
st.header("🧊 3차원 산점도")

if len(selected_stats_ko) < 3:
    st.info("ℹ️ 선택한 능력치가 3개 미만이라 3차원 산점도를 그릴 수 없습니다. 능력치를 3개 이상 선택해 주세요.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_3d = st.selectbox("X축 능력치", options=selected_stats_ko, index=0, key="3d_x")
    with col4:
        y_3d = st.selectbox("Y축 능력치", options=selected_stats_ko, index=min(1, len(selected_stats_ko)-1), key="3d_y")
    with col5:
        z_3d = st.selectbox("Z축 능력치", options=selected_stats_ko, index=min(2, len(selected_stats_ko)-1), key="3d_z")

    fig_3d = px.scatter_3d(
        df_display,
        x=x_3d, y=y_3d, z=z_3d,
        color="묶음",
        category_orders={"묶음": cluster_order},
        hover_name="name_ko",
        title=f"{x_3d} · {y_3d} · {z_3d} 3차원 산점도",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_3d.update_traces(marker=dict(size=4))
    fig_3d.update_layout(height=650)
    st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------------------------
# 묶음별 인원 & 능력치 평균 표
# ------------------------------------------------------------
st.header("📋 묶음별 요약")

cluster_summary = df_display.groupby("묶음")[STAT_COLS_KO].mean().round(2)
cluster_counts = df_display.groupby("묶음").size().rename("인원수")
cluster_summary = cluster_counts.to_frame().join(cluster_summary)
cluster_summary = cluster_summary.reindex(cluster_order)

st.dataframe(cluster_summary, use_container_width=True)

# ------------------------------------------------------------
# 묶음별 자동 해석 (전체 평균과 비교)
# ------------------------------------------------------------
st.header("🔎 묶음별 특징 살펴보기")
st.write("전체 선수 평균과 비교했을 때, 각 묶음이 어떤 능력치에서 눈에 띄는지 정리했습니다.")

overall_mean = df_display[STAT_COLS_KO].mean()

def describe_cluster(row, overall_mean, threshold=3.0):
    """전체 평균 대비 차이가 threshold 이상인 능력치를 강점/약점으로 뽑아 문장으로 만듦"""
    diffs = row[STAT_COLS_KO] - overall_mean
    strengths = diffs[diffs >= threshold].sort_values(ascending=False)
    weaknesses = diffs[diffs <= -threshold].sort_values()

    parts = []
    if len(strengths) > 0:
        strength_names = ", ".join(strengths.index.tolist())
        parts.append(f"**{strength_names}** 능력치가 전체 평균보다 뚜렷하게 높습니다")
    if len(weaknesses) > 0:
        weakness_names = ", ".join(weaknesses.index.tolist())
        parts.append(f"**{weakness_names}** 능력치는 전체 평균보다 낮은 편입니다")

    if not parts:
        return "전체 평균과 큰 차이 없이 고른 능력치를 가진 묶음입니다."
    return " / ".join(parts) + "."

for c_label in cluster_order:
    row = cluster_summary.loc[c_label]
    n_players = int(row["인원수"])
    desc = describe_cluster(row, overall_mean)

    cluster_players = df_display[df_display["묶음"] == c_label]
    top_player = cluster_players.sort_values("overall", ascending=False).iloc[0]
    pos_dist = cluster_players["포지션대분류"].value_counts()
    main_pos = pos_dist.idxmax()
    main_pos_ratio = round(pos_dist.max() / n_players * 100, 1)

    with st.expander(f"{c_label} 묶음 해석 보기 (인원 {n_players}명)", expanded=True):
        st.markdown(f"- {desc}")
        st.markdown(f"- 이 묶음에서 종합 능력치가 가장 높은 선수는 **{top_player['name_ko']}** (overall {top_player['overall']})입니다.")
        st.markdown(f"- 포지션 대분류 기준으로 **{main_pos}**의 비중이 가장 높습니다 (약 {main_pos_ratio}%).")

# ------------------------------------------------------------
# 묶음별 종합 능력치 상위 5명
# ------------------------------------------------------------
st.header("🏆 묶음별 종합 능력치(overall) 상위 5명")

cols = st.columns(len(cluster_order))
for i, c_label in enumerate(cluster_order):
    with cols[i]:
        st.subheader(f"{c_label} 묶음")
        top5 = (
            df_display[df_display["묶음"] == c_label]
            .sort_values("overall", ascending=False)
            .head(5)[["name_ko", "overall"]]
            .reset_index(drop=True)
        )
        top5.index = top5.index + 1
        st.table(top5)

# ------------------------------------------------------------
# 포지션 교차표
# ------------------------------------------------------------
st.header("🔀 묶음 × 포지션 교차표")
st.write("포지션 정보는 군집분석에 사용하지 않고, 결과 해석을 위해서만 참고합니다.")

cross_tab = pd.crosstab(df_display["묶음"], df_display["포지션대분류"])
cross_tab = cross_tab.reindex(cluster_order)
st.dataframe(cross_tab, use_container_width=True)

# ------------------------------------------------------------
# 선수별 설명 보기
# ------------------------------------------------------------
st.header("🔍 선수별 설명 보기")
st.write("선수를 한 명 선택하면, 그 선수의 능력치가 소속된 묶음 안에서 어떤 특징을 가지는지 자동으로 설명해 드립니다.")

player_name_selected = st.selectbox(
    "설명을 보고 싶은 선수를 선택하세요",
    options=sorted(df_display["name_ko"].unique())
)

player_row = df_display[df_display["name_ko"] == player_name_selected].iloc[0]
player_cluster = player_row["묶음"]
cluster_mates = df_display[df_display["묶음"] == player_cluster]
cluster_mean_for_player = cluster_mates[STAT_COLS_KO].mean()

# 선수 개인 능력치가 소속 묶음 평균 대비 높은지 낮은지 비교
diffs_player = player_row[STAT_COLS_KO] - cluster_mean_for_player
player_strengths = diffs_player[diffs_player >= 3.0].sort_values(ascending=False)
player_weaknesses = diffs_player[diffs_player <= -3.0].sort_values()

with st.container():
    st.subheader(f"{player_name_selected} 선수 정보")

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("소속 묶음", player_cluster)
    with info_col2:
        st.metric("종합 능력치(overall)", int(player_row["overall"]))
    with info_col3:
        st.metric("나이", int(player_row["age"]))

    st.write(f"**소속 클럽**: {player_row['club']}  |  **포지션**: {player_row['positions']}  |  **키**: {int(player_row['height_cm'])}cm")

    # 능력치 표 (선수 개인 vs 소속 묶음 평균)
    compare_df = pd.DataFrame({
        "이 선수": player_row[STAT_COLS_KO],
        f"{player_cluster} 묶음 평균": cluster_mean_for_player.round(2)
    })
    st.table(compare_df)

    # 자동 설명 문장
    desc_parts = []
    if len(player_strengths) > 0:
        strength_names = ", ".join(player_strengths.index.tolist())
        desc_parts.append(f"같은 묶음({player_cluster}) 선수들과 비교했을 때 **{strength_names}** 능력치가 특히 높은 편입니다")
    if len(player_weaknesses) > 0:
        weakness_names = ", ".join(player_weaknesses.index.tolist())
        desc_parts.append(f"**{weakness_names}** 능력치는 같은 묶음 평균보다 낮은 편입니다")

    if desc_parts:
        st.markdown("💬 " + " / ".join(desc_parts) + ".")
    else:
        st.markdown(f"💬 이 선수는 소속 묶음({player_cluster})의 평균적인 능력치 분포와 비슷한 특징을 보입니다.")

# ------------------------------------------------------------
# 마케팅 관점 생각 정리하기 (학생 작성 공간)
# ------------------------------------------------------------
st.header("💡 마케팅팀 관점에서 생각해보기")
st.write(
    "위 분석 결과(능력치 평균, 상위 선수, 포지션 분포 등)를 참고해서, "
    "여러분이 마케팅팀 인턴이라면 어떤 묶음에 자원을 더 투자하고 싶은지 정리해 봅시다."
)

recommended_cluster = st.selectbox(
    "가장 추천하고 싶은 묶음을 골라 보세요",
    options=cluster_order
)

reason_text = st.text_area(
    f"'{recommended_cluster}' 묶음을 추천하는 이유를 한 가지 적어 주세요",
    placeholder="예: 이 묶음은 슈팅과 드리블이 모두 높아 공격 포인트를 만들기 좋고, 상위 선수들의 인지도도 높아 마케팅 효과가 클 것 같습니다.",
    height=100
)

if reason_text:
    st.success(f"✅ '{recommended_cluster}' 묶음을 추천했고, 이유를 정리했습니다.")

# ------------------------------------------------------------
# 엘보우 방법 (묶음 수에 따른 SSE)
# ------------------------------------------------------------
st.header("📉 엘보우 방법으로 적절한 묶음 수 확인하기")
st.write("현재 선택한 능력치를 기준으로, 묶음 수를 1개부터 7개까지 바꿔가며 군집 내 거리 제곱합(SSE, 관성)을 계산합니다.")

k_range = list(range(1, 8))
sse_list = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    sse_list.append(km.inertia_)

fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(
    x=k_range, y=sse_list,
    mode="lines+markers",
    line=dict(color="royalblue", width=3),
    marker=dict(size=9, color="royalblue"),
    name="SSE"
))

fig_elbow.add_vline(
    x=n_clusters,
    line_width=2,
    line_dash="dash",
    line_color="crimson",
    annotation_text=f"현재 묶음 수: {n_clusters}",
    annotation_position="top"
)

fig_elbow.update_layout(
    title="묶음 수에 따른 SSE(군집 내 거리 제곱합)",
    xaxis_title="묶음 수 (k)",
    yaxis_title="SSE",
    height=500,
    xaxis=dict(tickmode="linear", dtick=1)
)
st.plotly_chart(fig_elbow, use_container_width=True)

sse_df = pd.DataFrame({
    "묶음 수": k_range,
    "SSE": [round(v, 2) for v in sse_list]
})
sse_diff = [None] + [round(sse_list[i-1] - sse_list[i], 2) for i in range(1, len(sse_list))]
sse_df["직전 대비 감소량"] = sse_diff

st.dataframe(sse_df.set_index("묶음 수"), use_container_width=True)

# ------------------------------------------------------------
# 실루엣 점수
# ------------------------------------------------------------
st.header("🌟 실루엣 점수로 묶음 수 평가하기")
st.write("실루엣 점수는 -1에서 1 사이 값으로, 1에 가까울수록 군집이 잘 나뉘었다는 뜻입니다. (묶음 수가 2 이상일 때만 계산 가능)")

sil_k_range = list(range(2, 8))
sil_scores = []

for k in sil_k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_k)
    sil_scores.append(score)

fig_sil = go.Figure()
fig_sil.add_trace(go.Scatter(
    x=sil_k_range, y=sil_scores,
    mode="lines+markers",
    line=dict(color="seagreen", width=3),
    marker=dict(size=9, color="seagreen"),
    name="실루엣 점수"
))

fig_sil.add_vline(
    x=n_clusters,
    line_width=2,
    line_dash="dash",
    line_color="crimson",
    annotation_text=f"현재 묶음 수: {n_clusters}",
    annotation_position="top"
)

fig_sil.update_layout(
    title="묶음 수에 따른 실루엣 점수",
    xaxis_title="묶음 수 (k)",
    yaxis_title="실루엣 점수",
    height=500,
    xaxis=dict(tickmode="linear", dtick=1)
)
st.plotly_chart(fig_sil, use_container_width=True)

sil_df = pd.DataFrame({
    "묶음 수": sil_k_range,
    "실루엣 점수": [round(s, 4) for s in sil_scores]
})
st.dataframe(sil_df.set_index("묶음 수"), use_container_width=True)

# ------------------------------------------------------------
# 마무리
# ------------------------------------------------------------
st.markdown("---")
st.caption("데이터 출처: EA FC25 Top 100 players (greatsong/modudata)")
