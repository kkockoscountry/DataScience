import io

import streamlit as st
# from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.optimize import minimize
from dataframe import explain, pretreat, by_year, by_month, by_count1, by_count2, by_count3, by_count4, by_ratio, by_desc, by_channel, by_cat, learning, learning2

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('유튜브 인기동영상')

mnu = st.sidebar.selectbox('메뉴', options=['데이터 설명', '데이터 분석', '가중치 분석'])

if mnu == '데이터 설명':
    explain()
if mnu == '데이터 분석':
    with st.sidebar:
        choose1 = option_menu('Option', ["데이터 전처리", "날짜별 분석", "좋아요, 댓글, 조회수 분석", '채널 분석', '설명란 분석', '카테고리 분석'], default_index=0, key=1)

    if choose1 == '데이터 전처리':
        pretreat()

    if choose1 == '날짜별 분석':
        with st.sidebar:
            choose2 = option_menu('Detail Option', ["년도별 분석", "월별 분석"], default_index=0, key=2)

        if choose2 == '년도별 분석':
            by_year()
        
        if choose2 == '월별 분석':
            by_month()

    if choose1 == '좋아요, 댓글, 조회수 분석':
        with st.sidebar:
            choose3 = option_menu('Detail Option', ["좋아요와 댓글이 모두 0이 아닌 영상", "좋아요가 0인 영상", 
                                                "댓글이 0인 영상", "좋아요와 댓글이 모두 0인 영상", "비율에 따른 분석"], default_index=0, key=3)
            
        if choose3 == "좋아요와 댓글이 모두 0이 아닌 영상":
            by_count1()
        
        if choose3 == "좋아요가 0인 영상":
            by_count2()
        
        if choose3 == "댓글이 0인 영상":
            by_count3()
        
        if choose3 == "좋아요와 댓글이 모두 0인 영상":
            by_count4()

        if choose3 == "비율에 따른 분석":
            by_ratio()
    
    if choose1 == '채널 분석':
        by_channel()
    
    if choose1 == '설명란 분석':
        by_desc()

    if choose1 == '카테고리 분석':
        by_cat()
if mnu == '가중치 분석':
    with st.sidebar:
        choose4 = option_menu('Option', ['임의로 구한 초기 가중치', '초기 가중치 설정'], default_index=0, key=4)

    if choose4 == '임의로 구한 초기 가중치':
        learning()

    if choose4 == '초기 가중치 설정':
        learning2()
        
