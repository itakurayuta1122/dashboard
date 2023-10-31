import csv
import streamlit as st
from st_on_hover_tabs import on_hover_tabs
import streamlit_float
import pandas as pd
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import date
from datetime import datetime as dt
import numpy as np
import xgboost as xgb
import catboost as cb
import lightgbm as lgbm
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from catboost import CatBoost
from catboost import Pool
import time

if 'random_state_num' not in st.session_state:
    st.session_state.random_state_num = 0
if 'start_count' not in st.session_state: 
    st.session_state.start_count = 1 #countがsession_stateに追加されていない場合，0で初期化

if 'state_const_num' not in st.session_state:
    st.session_state.state_const_num = ''

st.set_page_config(layout="wide")

streamlit_float.float_init(include_unstable_primary=False)

st.markdown('<style>' + open("style.css").read() + '</style>', unsafe_allow_html=True)



with st.sidebar:
    tabs = on_hover_tabs(tabName=['MI', '-------', '-------'], 
                         iconName=['home', 'search', 'settings'], default_choice=0)

if tabs =='MI':
    tab1, tab2, tab3, tab4 = st.tabs(["複数目的変数最適化", "bbb", "ccc", "ddd"])
    append_list = []
    with tab1:
        test_data_pred = False
        depl_test_x = ""
        depl_test_y = ""      
        target_count = 0
        count = 0
        alg_list = []
        st.info('複数目的変数最適化')
        reveth_num = 2
        df = pd.read_csv("boston.csv", header=0)
        df_columns_name = df.columns    
        df_test = pd.read_csv("tes.csv", header=0)
        df_test = df_test.set_index('index')
        random_state_select_count = []
        generate_count = st.session_state.start_count * 10
        generate_int = 0
        for i in range(0,generate_count):
           if i == 0:
               random_state_select_count.append('')
           random_state_select_count.append(generate_int)
           generate_int += 1
        #state_const_num = st.selectbox('random_stateを選択',random_state_select_count,key=1)
        if st.session_state.state_const_num == '':
            reveth_num = 11
        else:
            random_state_num = state_const_num
        df_autoscaled = (df - df.mean(axis=0)) / df.std(axis=0, ddof=1)
        df_autoscaled = df
        target_list = st.multiselect('目的変数を選択',df_columns_name)
        st.write("state_const_num")
        st.write(st.session_state.state_const_num)
        if st.session_state.state_const_num != '':
            for y in target_list:
                st.write(y)
                alg_list.append(st.radio('アルゴリズム', ("xgb","lgbm","cat","gpr","gbr","rf","lasso","ridge","lr","ex","knr","dt","el"), horizontal=True, key=count))
                count += 1
            #st.dataframe(alg_list)
            #検証用アルゴリズムの確認
            test_data_pred = st.button('test_data_pred')

        start_button = st.button('start')
        if (start_button or test_data_pred) and target_list != '':
            st.session_state.start_count += 1
            if st.session_state.start_count >= 2:
                st.session_state.state_const_num = st.selectbox('random_stateを選択',random_state_select_count,key=2)
            for i in range(0,generate_count):
                if i == 0:
                    random_state_select_count.append('')
                random_state_select_count.append(generate_int)
                generate_int += 1
            st.write(st.session_state.start_count)
            if test_data_pred:
                df_test_pred = pd.read_csv("boston_test.csv", header=0)
            for i in range(1,reveth_num):
                r2_max_sum = []
                for target in target_list:
                    alg_text = ""
                    if len(alg_list) > 0:
                        alg_text = alg_list[target_count]
                    append_list = []
                    df_autoscaled["target"] = df_autoscaled[target]
                    df_select_target = df_autoscaled.drop(target_list,axis=1)
                    #df_select_target = df_autoscaled.drop(target,axis=1)
                    features = [c for c in df_select_target.columns if c != '' "target"]   
                    train, test = train_test_split(df_select_target, test_size = 0.1, random_state=st.session_state.random_state_num)
                    
                    #学習用データ
                    X_train = train[features]
                    y_train = train["target"].values
                    
                    #検証用データ
                    X_test = test[features]
                    X_test_depl = X_test
                    X_test_depl = X_test_depl.reset_index()
                    del X_test_depl["index"]
                    if target_list.index(target) == 0:
                        depl_test_x = X_test_depl
                        if test_data_pred:
                            depl_test_x = df_test_pred
                        #for i in range(1,len(target_list)):
                            #del depl_test_x[target_list[i]]
                    y_test = test["target"].values
        
                    #Lasso用スケールしてないデータ
                    df["target"] = df[target]
                    #df_select_target_anscale = df.drop(target_list,axis=1)
                    df_select_target_anscale = df.drop(target,axis=1)   
                    train, test = train_test_split(df_select_target_anscale, test_size = 0.1, random_state=st.session_state.random_state_num)
                    features_anscale = [c for c in df_select_target_anscale.columns if c != '' "target"]
                    #学習用データ
                    X_train_anscale = train[features_anscale]
                    y_train_anscale = train["target"].values
                    #検証用データ
                    X_test_anscale = test[features_anscale]
                    y_test_anscale = test["target"].values
        
        
        
        
                    #xgboost予測開始
                    # データ形式の変換
                    dtrain = xgb.DMatrix(X_train, y_train)
                    dtest = xgb.DMatrix(X_test, y_test)
                    
                    # パラメータ設定
                    # regression: 回帰, squarederror: 二乗誤差
                    params = {"objective": "reg:squarederror"}
                    
                    # 学習
                    xgb_r = xgb.train(
                        params = params,
                        dtrain = dtrain,
                        evals = [(dtrain, "train"), (dtest, "test")],
                    )
                
                    y_train_preds = xgb_r.predict(dtrain)
                    #st.write(target)
                    #st.write("xgboost_target_train:")
                    r2s = r2_score(y_train, y_train_preds)
                    test_preds = xgb_r.predict(dtest)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "xgb":
                        if test_data_pred:
                            dtest = xgb.DMatrix(df_test_pred)
                            test_preds = xgb_r.predict(dtest)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #xgboost予測終了
                    #lightgbm予測開始
                    # データ形式の変換		
                    train_set = lgbm.Dataset(X_train, y_train)
                    test_set = lgbm.Dataset(X_test, y_test)
                    
                    # パラメータ設定
                    params = {"objective": "regression", # 回帰
                              "metric": "rmse",          # 平均二乗誤差の平方根
                              "verbosity": -1}           # warningなどを出力しない
                    
                    # 学習
                    model_lgbm = lgbm.train(
                        params = params,
                        train_set = train_set,
                        valid_sets = [train_set, test_set],
                    )
                
                    preds_lgbm_train = model_lgbm.predict(X_train)
                    #st.write(target)
                    #st.write("lightgbm_target_train:")
                    r2s = r2_score(y_train, preds_lgbm_train)
                    test_preds = model_lgbm.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    #lightgbm予測終了

                    if alg_text == "lgbm":
                        if test_data_pred:
                            test_preds = model_lgbm.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #catboost予測開始
                    # データセットを生成する
                    train_pool = Pool(X_train, y_train)
                    test_pool = Pool(X_test, y_test)
                    
                    #catboostのハイパーパラメータを設定する
                    params = {'loss_function': 'RMSE',
                             'num_boost_round': 1000,
                             'early_stopping_rounds': 10,
                             }
                    
                    # 上記のパラメータでモデルを学習する
                    model_cat = CatBoost(params)
                    model_cat.fit(train_pool, eval_set=[test_pool], use_best_model=True)
                    
                    y_pred_train = model_cat.predict(train_pool)
                    #st.write(target)
                    #st.write("catboost_target_train:")
                    r2s = r2_score(y_train, y_pred_train)
                    test_preds = model_cat.predict(test_pool)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)

                    if alg_text == "cat":
                        if test_data_pred:
                            dtest = Pool(df_test_pred)
                            test_preds = model_cat.predict(dtest)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #catboost予測終了
                    #GaussianProcessReg予測開始
                    kernel = DotProduct() + WhiteKernel()
                    gpr = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
                    gpr_pred_train_y = gpr.predict(X_train)
                    r2s = r2_score(y_train, gpr_pred_train_y)
                    test_preds = gpr.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "gpr":
                        if test_data_pred:
                            test_preds = gpr.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #GaussianProcessReg予測終了
                    #GradientBoostingReg予測開始
                    gbr = GradientBoostingRegressor().fit(X_train,y_train)
                    gbr_pred_tarin_y = gbr.predict(X_train)
                    r2s = r2_score(y_train, gbr_pred_tarin_y)
                    test_preds = gbr.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "gbr":
                        if test_data_pred:
                            test_preds = gbr.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #GradientBoostingReg予測終了
                    #RandomForest予測
                    rfr = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None)
                    rfr.fit(X_train, y_train)
                    
                    test_pool = Pool(X_test, y_test)
                    
                    # 学習データの件数を指定する
                    train_size = len(X_train)
                    test_size = df.shape[0] - train_size
                    
                    # 学習させたモデルを使ってテストデータに対する予測を出力する
                    pred_random_train_y = rfr.predict(X_train)
                    r2s = r2_score(y_train, pred_random_train_y)
                    test_preds = rfr.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "rf":
                        if test_data_pred:
                            test_preds = rfr.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #RandomForest予測終了
                    #Lasso予測開始
                    lasso = Lasso()
                    lasso.fit(X_train, y_train)
                    lasso_pred_train_y = lasso.predict(X_train)
                    r2s = r2_score(y_train, lasso_pred_train_y )
                    test_preds = lasso.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "lasso":
                        if test_data_pred:
                            test_preds = lasso.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #Lasso予測終了
                    #ridge予測開始
                    ridge = Ridge().fit(X_train, y_train)
                    ridge_pred_train_y = ridge.predict(X_train)
                    #st.write(target)
                    #st.write("Ridge_target_train:")
                    r2s = r2_score(y_train, ridge_pred_train_y)
                    test_preds = ridge.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "ridge":
                        if test_data_pred:
                            test_preds = ridge.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #ridge予測終了
                    #Linear予測開始
                    reg_lr = LinearRegression()
                    # 訓練データにモデルを適用する
                    reg_lr.fit(X_train,y_train)
                    reg_lr_train_y = reg_lr.predict(X_train)
                    #st.write(target)
                    #st.write("Linear_target_train:")
                    r2s = r2_score(y_train, reg_lr_train_y)
                    test_preds = reg_lr.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "lr":
                        if test_data_pred:
                            test_preds = reg_lr.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #Linear予測終了
                    #Extratree予測開始
                    exreg = ExtraTreesRegressor(n_estimators=100).fit(X_train, y_train)
                    exreg_pred_train_y = exreg.predict(X_train)
                    #st.write(target)
                    #st.write("ExtraTrees_target_train:")
                    r2s = r2_score(y_train, exreg_pred_train_y)
                    test_preds = exreg.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "ex":
                        if test_data_pred:
                            test_preds = exreg.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)

                    #Extratree予測終了
                    #KNeighbors予測開始
                    knr = KNeighborsRegressor()
                    knr.fit(X_train, y_train)
                    knr_pred_train_y = knr.predict(X_train)
                    r2s = r2_score(y_train, knr_pred_train_y)
                    test_preds = knr.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "knr":
                        if test_data_pred:
                            test_preds = knr.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #KNeighbors予測終了
                    #DecisionTree予測開始
                    dtreg = DecisionTreeRegressor()
                    dtreg = dtreg.fit(X_train, y_train)
                    dtreg_pred_train_y = dtreg.predict(X_train)
                    #st.write(target)
                    #st.write("DecisionTree_target_train:")
                    r2s = r2_score(y_train, dtreg_pred_train_y)
                    test_preds = dtreg.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "dt":
                        if test_data_pred:
                            test_preds = dtreg.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #DecisionTree予測終了
                    #ElasticNet予測開始
                    elregr = ElasticNet()
                    elregr.fit(X_train, y_train)
                    elregr_pred_train_y = elregr.predict(X_train)
                    r2s = r2_score(y_train, elregr_pred_train_y)
                    test_preds = elregr.predict(X_test)
                    r2s = round(r2_score(y_test, test_preds),2)
                    append_list.append(r2s)
                    if alg_text == "el":
                        if test_data_pred:
                            test_preds = elregr.predict(df_test_pred)
                        test_preds = pd.DataFrame(test_preds)
                        test_preds[target] = test_preds[0]
                        del test_preds[0]
                        X_test_depl = pd.concat([X_test_depl,test_preds], axis=1)
                        if test_data_pred:
                            X_test_depl = pd.concat([df_test_pred,test_preds], axis=1)
                        depl_test_x = pd.concat([depl_test_x,test_preds], axis=1)
                    #ElasticNet予測終了
                    df_appand = pd.DataFrame(append_list)
                    df_test[target] = append_list
                    r2_max_sum.append(max(append_list))
                    target_count += 1
                    if st.session_state.state_const_num != '':
                        st.write(X_test_depl)
                if st.session_state.state_const_num != '':
                    st.write(depl_test_x)
                st.write("r2_max_sum")
                st.write(sum(r2_max_sum))
                st.write("random_state")
                st.write(st.session_state.random_state_num)
                st.session_state.random_state_num = st.session_state.random_state_num + 1
                st.table(df_test.style.format(precision=2).highlight_max(axis=0))

    with tab2:
        st.header("A dog")
        st.write('🐶')
    with tab3:
        st.header("A fox")
        st.write('🦊')
    with tab4:
        st.header('A hamster')
        st.write('🐹')
elif tabs == '実験別検索ページ':
    df_number = pd.read_csv(r"C:\Users\WDAGUtilityAccount\Desktop\excel\number.csv")
    df_number = df_number.to_numpy().tolist()
    df_number = sum(df_number, [])
    number = st.selectbox('実験番号',df_number)
    result_path = r"C:\Users\WDAGUtilityAccount\Desktop\excel\result\2222"+number+".csv"
    result_path = result_path.replace("2222",'')
    df_number_final = pd.read_csv(result_path,encoding='shift_jis')
    st.write(df_number_final)
elif tabs == 'Economy':
    st.header('A hamster')