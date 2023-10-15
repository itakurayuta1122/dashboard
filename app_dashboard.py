from datetime import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from japanmap import picture
import streamlit as st
import altair as alt
from vega_datasets import data

pagelist_kashika = ["販売系ダッシュボード","製造系ダッシュボード","医療系ダッシュボード"]


selector2=st.sidebar.radio(':blue[ダッシュボード]',pagelist_kashika)
if selector2 == "販売系ダッシュボード":
    st.title(':blue[販売系ダッシュボード]')
if selector2 == "製造系ダッシュボード":
    st.title(':blue[製造系ダッシュボード]')
    # carsデータセットの読み込み
    df = data.cars()

    # 定量データ項目のリスト
    item_list = [
        col for col in df.columns if df[col].dtype in ['float64', 'int64']]

    # 製造地域のリスト
    origin_list = list(df['Origin'].unique())

    # 西暦列の作成
    df['YYYY'] = df['Year'].apply(lambda x: x.year)
    min_year = df['YYYY'].min().item()
    max_year = df['YYYY'].max().item()

    # サイドバー
    st.sidebar.title("Dashboard of Cars Dataset")
    st.sidebar.markdown('###')
    st.sidebar.markdown("### *Settings*")
    start_year, end_year = st.sidebar.slider(
        "Period",
        min_value=min_year, max_value=max_year,
        value=(min_year, max_year))

    st.sidebar.markdown('###')
    origins = st.sidebar.multiselect('Origins', origin_list,
                                    default=origin_list)
    st.sidebar.markdown('###')
    item1 = st.sidebar.selectbox('Item 1', item_list, index=0)
    item2 = st.sidebar.selectbox('Item 2', item_list, index=3)

    df_rng = df[(df['YYYY'] >= start_year) & (df['YYYY'] <= end_year)]
    source = df_rng[df_rng['Origin'].isin(origins)]

    # コンテンツ
    base = alt.Chart(source).properties(height=300)

    bar = base.mark_bar().encode(
        x=alt.X('count(Origin):Q', title='Number of Records'),
        y=alt.Y('Origin:N', title='Origin'),
        color=alt.Color('Origin:N', legend=None)
    )

    point = base.mark_circle(size=50).encode(
        x=alt.X(item1 + ':Q', title=item1),
        y=alt.Y(item2 + ':Q', title=item2),
        color=alt.Color('Origin:N', title='',
                        legend=alt.Legend(orient='bottom-left'))
    )

    line1 = base.mark_line(size=5).encode(
        x=alt.X('yearmonth(Year):T', title='Date'),
        y=alt.Y('mean(' + item1 + '):Q', title=item1),
        color=alt.Color('Origin:N', title='',
                        legend=alt.Legend(orient='bottom-left'))
    )

    line2 = base.mark_line(size=5).encode(
        x=alt.X('yearmonth(Year):T', title='Date'),
        y=alt.Y('mean(' + item2 + '):Q', title=item2),
        color=alt.Color('Origin:N', title='',
                        legend=alt.Legend(orient='bottom-left'))
    )

    # レイアウト (コンテンツ)
    left_column, right_column = st.columns(2)

    left_column.markdown(
        '**Number of Records (' + str(start_year) + '-' + str(end_year) + ')**')
    left_column.altair_chart(bar, use_container_width=True)

    right_column.markdown(
        '**Scatter Plot of _' + item1 + '_ and _' + item2 + '_**')
    right_column.altair_chart(point, use_container_width=True)

    left_column.markdown('**_' + item1 + '_ (Monthly Average)**')
    left_column.altair_chart(line1, use_container_width=True)

    right_column.markdown('**_' + item2 + '_ (Monthly Average)**')
    right_column.altair_chart(line2, use_container_width=True)


if selector2 == "医療系ダッシュボード":
    st.title(':blue[医療系ダッシュボード]')
    #オープンデータのURL 使用していないものもある
    newly_confirmed_cases_daily = "https://covid19.mhlw.go.jp/public/opendata/newly_confirmed_cases_daily.csv"
    requiring_inpatient_care_etc_daily = "https://covid19.mhlw.go.jp/public/opendata/requiring_inpatient_care_etc_daily.csv"
    deaths_cumulative_daily = "https://covid19.mhlw.go.jp/public/opendata/deaths_cumulative_daily.csv"
    severe_cases_daily = "https://covid19.mhlw.go.jp/public/opendata/severe_cases_daily.csv"
    population = "https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032110815&fileKind=0"

    #オープンデータを読み込みデータフレームを作成
    df01 = pd.read_csv(newly_confirmed_cases_daily, index_col="Date")
    df01.index = pd.DatetimeIndex(df01.index).date
    df02 = pd.read_csv(severe_cases_daily, index_col="Date")
    df02.index = pd.DatetimeIndex(df02.index).date
    df03 = pd.read_csv(deaths_cumulative_daily, index_col="Date")
    df03.index = pd.DatetimeIndex(df03.index).date

    prefecture_slection = st.multiselect("都道府県を選択してください", df01.columns, default="ALL")

    min_period = df01.index.min()
    start_period = st.slider("開始日付",df01.index.min(), df01.index.max(),value=df01.index.min())
    end_period = st.slider("終了日付",df01.index.min(), df01.index.max(), df01.index.max())

    df02["Fukushima"] = pd.to_numeric(df02["Fukushima"],errors="coerce")
    df02["Saitama"] = pd.to_numeric(df02["Saitama"],errors="coerce")
    df02["Chiba"] = pd.to_numeric(df02["Chiba"],errors="coerce")
    df02["Ehime"] = pd.to_numeric(df02["Ehime"],errors="coerce")
    df02=df02.fillna(0)
    #移動平均の設定
    moving_average = st.slider("移動平均の日数",1, 30,value=7)
    df01 = df01.rolling(moving_average, min_periods=1).mean().round(1)
    df02 = df02.rolling(moving_average, min_periods=1).mean().round(1)

    fig= make_subplots(specs=[[{"secondary_y": True}]])
    pre = prefecture_slection

    for idx, prefecture in enumerate(pre):
        fig.add_trace(go.Bar(x=df01.index, y=df01[prefecture],name=(prefecture + "_新規陽性者数")))
        fig.add_trace(go.Bar(x=df02.index, y=df02[prefecture], name=(prefecture + "_重症者数")))
        fig.update_layout(barmode='overlay', xaxis=dict(range=(start_period, end_period)))
        fig.add_trace(go.Scatter(x=df03.index, y=df03[prefecture], name=(prefecture + "_累計死者数")))
                    
    st.plotly_chart(fig, use_container_width=True)

    df_today = df01.iloc[-1]
    df_today =df_today.rename({"Hokkaido":"北海道", 
                            "Aomori":"青森","Akita":"秋田", "Iwate":"岩手", "Miyagi":"宮城","Yamagata":"山形", "Fukushima":"福島", 
                            "Ibaraki":"茨城", "Tochigi":"栃木", "Gunma":"群馬", "Saitama":"埼玉", "Chiba":"千葉", "Tokyo":"東京", "Kanagawa":"神奈川",
                            "Niigata":"新潟", "Toyama":"富山", "Ishikawa":"石川","Fukui":"福井", "Yamanashi":"山梨", "Nagano":"長野", 
                            "Gifu":"岐阜","Shizuoka":"静岡", "Aichi":"愛知", "Mie":"三重",
                            "Shiga":"滋賀", "Kyoto":"京都", "Osaka":"大阪","Hyogo":"兵庫", "Nara":"奈良", "Wakayama":"和歌山", 
                            "Tottori":"鳥取","Shimane":"島根", "Okayama":"岡山", "Hiroshima":"広島", "Yamaguchi":"山口",
                            "Kagawa":"香川", "Tokushima":"徳島","Ehime":"愛媛", "Kochi":"高知", 
                            "Fukuoka":"福岡", "Saga":"佐賀", "Nagasaki":"長崎", "Kumamoto":"熊本", "Oita":"大分", "Miyazaki":"宮崎", "Kagoshima":"鹿児島", "Okinawa":"沖縄"})

    df_today = df_today[1:]

    df_population = pd.read_excel(population, header = 5, skipfooter=1)
    df_population = df_population.rename(columns = {"人":"人口"})
    df_population = df_population[["都道府県名","人口"]].iloc[1:, :]
    df_population["陽性患者数"] = df_today.values
    df_population["人口あたりの陽性患者数"] = df_population["陽性患者数"] / df_population["人口"]

    #人口あたり新規陽性者総数
    cmap = plt.get_cmap('Blues')
    norm = plt.Normalize(vmin=df_population["人口あたりの陽性患者数"].min(), vmax=df_population["人口あたりの陽性患者数"].max())
    fcol = lambda x: '#' + bytes(cmap(norm(x), bytes=True)[:3]).hex()
    fig,ax = plt.subplots(figsize=(4,4))
    plt.colorbar(plt.cm.ScalarMappable(norm, cmap),ax=ax)
    plt.imshow(picture(df_population["人口あたりの陽性患者数"].apply(fcol)))
    st.pyplot(fig)