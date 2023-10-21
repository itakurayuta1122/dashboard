import hydralit as hy
from datetime import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from japanmap import picture
import streamlit as st
import altair as alt
from vega_datasets import data
from datetime import date
import plotly.express as px

app = hy.HydraApp(title='Simple Multi-Page App')

@app.addapp()
def my_home():
    #hy.info('Hello from app1')
    st.title(':blue[販売系ダッシュボード]')
    # エクセルの読み込み
    df = pd.read_excel("注文履歴.xlsx", sheet_name="注文履歴", header=0, usecols="A:G")
    df = df.dropna()  # 空白データがある行を除外
    df[["単価", "数量", "金額"]] = df[["単価", "数量", "金額"]].astype(int)  # 金額や数量を整数型に変換
    df["月"] = df["購入日"].dt.month.astype(str)  # "月"の列を追加
    df["購入日|部署"] = df["購入日"].astype(str).str.cat(df["部署"], sep="|")  # "購入日|部署" 列を追加

    view_columns = ['購入日', '部署', '品名', '単価', '数量', '金額']


    # 現在の年月を取得
    today = date.today()
    this_year = today.year
    this_month = today.month
    this_year = 2022  # サンプルCSVをそのまま使用する場合はこの行のコメントを解除してください
    this_month = 9  # サンプルCSVをそのまま使用する場合はこの行のコメントを解除してください

    st.title(f"{this_year}年{this_month}月")

    # 4カラム表示
    col1, col2, col3, col4 = st.columns(4)
    # 今年の購入回数
    this_year_counts = df.loc[df["購入日"].dt.year == this_year, "購入日|部署"].nunique()
    col1.metric("📝今年の購入回数", f"{this_year_counts}回")
    # 今年の購入額
    this_year_purchase = df.loc[df["購入日"].dt.year == this_year, "金額"].sum()
    col2.metric("💰今年の購入額", f"{this_year_purchase}円")
    # 今月の購入回数
    this_month_counts = df.loc[df["購入日"].dt.month == this_month, "購入日|部署"].nunique()
    col3.metric("📝今月の購入回数", f"{this_month_counts}回")
    # 今月の購入額
    this_month_purchase = df.loc[df["購入日"].dt.month == this_month, "金額"].sum()
    col4.metric("💰今月の購入額", f"{this_month_purchase}円")


    # 3カラム表示 (1:2:2)
    col1, col2, col3 = st.columns([1, 2, 2])
    # 購入数TOP10
    many_df = df.groupby(by="品名").sum(numeric_only=True).sort_values(by="数量", ascending=False).reset_index()
    col1.subheader("購入数TOP10")
    col1.table(many_df[["品名", "単価", "数量", "金額"]].iloc[:10])
    # 部署別購入金額
    department_group_df = df.groupby(["部署", "月"]).sum(numeric_only=True)
    fig = px.bar(department_group_df.reset_index(), x="金額", y="部署", color="月", orientation="h")
    col2.subheader("部署別購入金額")
    col2.plotly_chart(fig, use_container_width=True)
    # 直近3件の購入
    recent_df = df[df["購入日|部署"].isin(sorted(df["購入日|部署"].unique())[-3:])]
    recent_df["購入日"] = recent_df["購入日"].dt.strftime("%Y-%m-%d")
    col3.subheader("直近3件の購入")
    col3.table(recent_df[view_columns])

    # 月ごとの購入金額推移
    month_group_df = df.groupby(["月", "部署"]).sum(numeric_only=True)
    fig = px.bar(month_group_df.reset_index(), x="月", y="金額", color="部署", title="月別購入金額")
    st.plotly_chart(fig, use_container_width=True)


    # 詳細表示
    with st.expander("詳細データ"):
        # 表示する期間の入力
        min_date = df["購入日"].min().date()
        max_date = df["購入日"].max().date()
        start_date, end_date = st.slider(
            "表示する期間を入力",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY/MM/DD")

        col1, col2 = st.columns(2)

        # 表示する部署の選択
        departments = df["部署"].unique()
        select_departments = col1.multiselect("表示部署", options=departments, default=departments)

        df["購入日"] = df["購入日"].apply(lambda x: x.date())
        detail_df = df[(start_date <= df["購入日"]) & (df["購入日"] <= end_date) & (df["部署"].isin(select_departments))]

        productname_group_df = detail_df.groupby(["品名", "部署"]).sum(numeric_only=True)
        view_h = len(productname_group_df)*15
        fig = px.bar(productname_group_df.reset_index(), x="金額", y="品名", color="部署", orientation="h", title="購入品別購入金額", height=view_h+300, width=600)
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        col1.plotly_chart(fig, use_container_width=True)

        col2.subheader("購入一覧")
        col2.dataframe(detail_df[view_columns], height=view_h+200)


@app.addapp()
def app2():
    #hy.info('Hello from app 2')
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
    st.title("Dashboard of Cars Dataset")
    st.markdown('###')
    st.markdown("### *Settings*")
    start_year, end_year = st.slider(
        "Period",
        min_value=min_year, max_value=max_year,
        value=(min_year, max_year))

    st.markdown('###')
    origins = st.multiselect('Origins', origin_list,
                                    default=origin_list)
    st.markdown('###')
    item1 = st.selectbox('Item 1', item_list, index=0)
    item2 = st.selectbox('Item 2', item_list, index=3)

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


@app.addapp(title='医療系')
def app3():
    hy.info('Hello from app 3, A.K.A, The Best 🥰')
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

#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()