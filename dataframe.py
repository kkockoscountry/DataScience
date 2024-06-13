import io

import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.optimize import minimize

st.set_option('deprecation.showPyplotGlobalUse', False)

data_path = ''
df_kr = pd.read_csv(data_path + 'KR_youtube_trending_data.csv')
global a

a = io.StringIO()
df_kr.info(buf=a)

df_kr.drop(['video_id', 'channelId', 'dislikes', 'thumbnail_link', 'tags', 'comments_disabled', 'ratings_disabled'], axis=1, inplace=True)
df_kr['publishedAt'] = df_kr['publishedAt'].str.slice(stop=10)
df_kr['publishedAt'] = df_kr['publishedAt'].astype('datetime64[ns]')
df_kr['publishedAt'] = pd.to_datetime(df_kr['publishedAt'], format='YYYY-MM-DD')
df_kr['trending_date'] = df_kr['trending_date'].str.slice(stop=10)
df_kr['trending_date'] = df_kr['trending_date'].astype('datetime64[ns]')
df_kr['trending_date'] = pd.to_datetime(df_kr['trending_date'], format='YYYY-MM-DD')
df_kr['taken_time'] = df_kr['trending_date'] - df_kr['publishedAt']
df_kr['taken_time'] = df_kr['taken_time'].astype('str')
df_kr['taken_time'] = df_kr['taken_time'].str.slice(stop=-4)
df_kr['taken_time'] = df_kr['taken_time'].astype('int')
df_kr['publishedAt'] = df_kr['publishedAt'].astype('str')
df_kr['trending_date'] = df_kr['trending_date'].astype('str')
df_kr['description'] = np.where(df_kr['description'].notna(), 'yes', df_kr['description'])
df_kr['description'] = np.where(df_kr['description'].isna(), 'no', df_kr['description'])

df_kr_bycount = df_kr.copy(deep=True)
df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)
df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
df_kr_bycount.reset_index(drop=True, inplace=True)
df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount1 = df_kr_bycount.copy(deep=True)
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != 0]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != 0]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != 0]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != -np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != -np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != -np.inf]
df_kr_bycount1 = df_kr_bycount1.drop(columns=[col for col in df_kr_bycount1.columns if col not in ['mean_day_likes', 'mean_day_comment_count', 'mean_day_view_count']])

def explain():
    st.write("2020년 8월 부터 2024년 4월 까지의 11개국의 유튜브 인기동영상에 대한 데이터이다.")
    st.write("데이터의 출처: https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset")
    st.subheader('column이 의미하는 항목')
    st.write("데이터에는 16개의 column이 존재한다.")
    st.write("""* video_id : 영상의 id
* title : 영상의 제목
* publishedAt : 영상 업로든 날짜
* channelId : 업로드 채널의 id
* channelTitle : 업로드 채널의 이름
* categoryId : 직접 설정하거나, 자동 설정된 카테고리의 id
* trending_date : 인기동영상에 선정된 날짜
* tags : 설정한 태그
* view_count : 조회수
* likes : 좋아요 수
* dislikes : 싫어요 수
* comment_count : 댓글 수
* thunmbnail_link : 섬네일 사진 링크
* comments_disabled : 댓글 작성 가능여부
* ratings_disabled : 좋아요 싫어요 공개 여부
* description : 설명란""")
    st.subheader('category id가 의미하는 항목')
    st.write("""* 1 -  Film & Animation
* 2 - Autos & Vehicles
* 10 - Music
* 15 - Pets & Animals
* 17 - Sports
* 18 - Short Movies
* 19 - Travel & Events
* 20 - Gaming
* 21 - Videoblogging
* 22 - People & Blogs
* 23 - Comedy
* 24 - Entertainment
* 25 - News & Politics
* 26 - Howto & Style
* 27 - Education
* 28 - Science & Technology
* 29 - Nonprofits & Activism
* 30 - Movies
* 31 - Anime/Animation
* 32 - Action/Adventure
* 33 - Classics
* 34 - Comedy
* 35 - Documentary
* 36 - Drama
* 37 - Family
* 38 - Foreign
* 39 - Horror
* 40 - Sci-Fi/Fantasy
* 41 - Thriller
* 42 - Shorts
* 43 - Shows
* 44 - Trailers""")

def pretreat(a=a):
    st.subheader('라이브러리 불러오기')
    st.code('''import matplotlib.pyplot as plt
import pandas as pd
import numpy as np''')
    st.subheader('데이터 읽어오기')
    st.code('''df_kr = pd.read_csv('KR_youtube_trending_data.csv')''')
    st.subheader('전처리 과정')
    st.code('''df_kr.info()''')
    buffer = a
    st.text(buffer.getvalue())
    st.write('영상의 id와 채널의 id는 결국 영상의 제목과 채널의 제목으로 대체 가능하므로 삭제한다.')
    st.write('현재 유튜브는 싫어요 수가 나타나지 않기 때문에 컬럼을 삭제한다.')
    st.write('섬네일 링크는 필요하지 않기 때문에 삭제한다.')
    st.write('tag의 경우 영상과 관련 여부없이 원하는 만큼 업로더가 지정이 가능해 데이터와 관계없다 판단하여 삭제한다.')
    st.write('댓글이 막힌 여부의 경우 댓글 수가 0임을 통해 확인 가능하므로 삭제한다.')
    st.write('마찬가지로 좋아요 여부가 막힌 경우 좋아요 수가 0임을 통해 확인 가능하므로 삭제한다.')
    with st.expander('see code'):
        st.code('''df_kr.drop(['video_id', 'channelId', 'dislikes', 'thumbnail_link', 'tags', 'comments_disabled', 'ratings_disabled'], axis=1, inplace=True)''')
    st.write('업로드 날짜와 인기 동영상 날짜의 변수를 datetime "YYYY-MM-DD"로 변환한다.')
    with st.expander('see code'):
        st.code('''df_kr['publishedAt'] = df_kr['publishedAt'].str.slice(stop=10)
df_kr['publishedAt'] = df_kr['publishedAt'].astype('datetime64[ns]')
df_kr['publishedAt'] = pd.to_datetime(df_kr['publishedAt'], format='YYYY-MM-DD')
df_kr['trending_date'] = df_kr['trending_date'].str.slice(stop=10)
df_kr['trending_date'] = df_kr['trending_date'].astype('datetime64[ns]')
df_kr['trending_date'] = pd.to_datetime(df_kr['trending_date'], format='YYYY-MM-DD')''')
    st.write('영상이 업로드 되고 인기 동영상이 되는데 걸린 기간에 대한 taken_time column을 추가한다.')
    st.write('timedate형태로 나타나기 때문에 숫자만으로 인덱싱 해준다.')
    with st.expander('see code'):
        st.code('''df_kr['taken_time'] = df_kr['trending_date'] - df_kr['publishedAt']
df_kr['taken_time'] = df_kr['taken_time'].astype('str')
df_kr['taken_time'] = df_kr['taken_time'].str.slice(stop=-4)
df_kr['taken_time'] = df_kr['taken_time'].astype('int')''')
    st.write('업로드 날짜와 인기 동영상 날짜의 변수를 다시 문자열로 변환한다.')
    with st.expander('see code'):
        st.code('''df_kr['publishedAt'] = df_kr['publishedAt'].astype('str')
df_kr['trending_date'] = df_kr['trending_date'].astype('str')''')
    st.write('설명란의 경우 존재하는 경우와 그렇지 않은 경우만을 나누어 분석하기 위해 null은 "no"으로 그렇지 않으면 "yes"으로 변환한다.')
    with st.expander('see code'):
        st.code('''df_kr['description'] = np.where(df_kr['description'].notna(), 'yes', df_kr['description'])
df_kr['description'] = np.where(df_kr['description'].isna(), 'no', df_kr['description'])''')
    st.code('''df_kr.info()''')
    buffer.truncate(0)
    buffer = io.StringIO()
    df_kr.info(buf=buffer)
    st.text(buffer.getvalue())

def by_year():
    df_kr_bytime = df_kr.copy(deep=True)
    df_kr_bytime = df_kr.groupby(['title', 'channelTitle', 'publishedAt'])['taken_time'].apply(list).reset_index()
    df_kr_bytime = df_kr_bytime.sort_values(by='publishedAt')
    df_kr_bytime.reset_index(drop=True, inplace=True)
    df_kr_bytime['trending_number'] = df_kr_bytime['taken_time'].apply(len)
    df_kr_bytime['first_trending'] = df_kr_bytime['taken_time'].apply(min)
    df_kr_bytime['last_trending'] = df_kr_bytime['taken_time'].apply(max)
    st.write('영상 제목과 채널의 이름을 그룹으로 모든 시간을 리스트로 묵는다.')
    st.write('업로드 날짜를 기준으로 오름차순 정렬한다.')
    with st.expander('see code'):
        st.code('''df_kr_bytime = df_kr.copy(deep=True)
df_kr_bytime = df_kr.groupby(['title', 'channelTitle', 'publishedAt'])['taken_time'].apply(list).reset_index()
df_kr_bytime = df_kr_bytime.sort_values(by='publishedAt')
df_kr_bytime.reset_index(drop=True, inplace=True)''')
    st.write('며칠동안 인기동영상이었는지를 나타내는 trending_number 컬럼을 추가한다.')
    with st.expander('see code'):
        st.code('''df_kr_bytime['trending_number'] = df_kr_bytime['taken_time'].apply(len)''')
    st.write('처음 인기동영상이 되는데 걸린 시간을 나타내는 first_trending과 마지막 인기 동영상이 되는데 걸린 시간인 last_trending 컬럼을 추가한다.')
    with st.expander('see code'):
        st.code('''df_kr_bytime['first_trending'] = df_kr_bytime['taken_time'].apply(min)
df_kr_bytime['last_trending'] = df_kr_bytime['taken_time'].apply(max)''')
    st.write('2020년부터 2024년까지 년도별로 데이터를 나눈다.')
    with st.expander('see code'):
        st.code('''df_kr_bytime_2020 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2020']
df_kr_bytime_2021 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2021']
df_kr_bytime_2022 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2022']
df_kr_bytime_2023 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2023']
df_kr_bytime_2024 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2024']''')
    st.write('년도별 막대 그래프를 그린다.')
    with st.expander('see code'):
        st.code('''years = ['2020', '2021', '2022', '2023', '2024', 'all']
values = [[df_kr_bytime_2020['trending_number'].mean().round(2),
          df_kr_bytime_2021['trending_number'].mean().round(2),
          df_kr_bytime_2022['trending_number'].mean().round(2),
          df_kr_bytime_2023['trending_number'].mean().round(2),
          df_kr_bytime_2024['trending_number'].mean().round(2),
          df_kr_bytime['trending_number'].mean().round(2)],
           [df_kr_bytime_2020['first_trending'].mean().round(2),
            df_kr_bytime_2021['first_trending'].mean().round(2),
            df_kr_bytime_2022['first_trending'].mean().round(2),
            df_kr_bytime_2023['first_trending'].mean().round(2),
            df_kr_bytime_2024['first_trending'].mean().round(2),
            df_kr_bytime['first_trending'].mean().round(2)],
             [df_kr_bytime_2020['last_trending'].mean().round(2),
              df_kr_bytime_2021['last_trending'].mean().round(2),
              df_kr_bytime_2022['last_trending'].mean().round(2),
              df_kr_bytime_2023['last_trending'].mean().round(2),
              df_kr_bytime_2024['last_trending'].mean().round(2),
              df_kr_bytime['last_trending'].mean().round(2)],
               [df_kr_bytime_2020['trending_number'].max(),
                df_kr_bytime_2021['trending_number'].max(),
                df_kr_bytime_2022['trending_number'].max(),
                df_kr_bytime_2023['trending_number'].max(),
                df_kr_bytime_2024['trending_number'].max(),
                df_kr_bytime['trending_number'].max()],
                 [df_kr_bytime_2020['first_trending'].max(),
                  df_kr_bytime_2021['first_trending'].max(),
                  df_kr_bytime_2022['first_trending'].max(),
                  df_kr_bytime_2023['first_trending'].max(),
                  df_kr_bytime_2024['first_trending'].max(),
                  df_kr_bytime['first_trending'].max()],
                   [df_kr_bytime_2020['last_trending'].max(),
                    df_kr_bytime_2021['last_trending'].max(),
                    df_kr_bytime_2022['last_trending'].max(),
                    df_kr_bytime_2023['last_trending'].max(),
                    df_kr_bytime_2024['last_trending'].max(),
                    df_kr_bytime['last_trending'].max()]]

title = ['trending_number', 'first_trending', 'last_trending']

fig, axs = plt.subplots(2, 3, figsize=(10.5, 7))

for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.bar(years, values[i * 3 + j])

        for bar, value in zip(ax.patches, values[i * 3 + j]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, str(value), ha='center', va='bottom')

        ax.set_title(title[j])
        ax.set_xlabel('Years')
        if i == 0:
          ax.set_ylabel('mean value')
        elif i ==1:
          ax.set_ylabel('max value')
        else:
          ax.set_ylabel('min value')

plt.tight_layout()
plt.show()''')
    df_kr_bytime_2020 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2020']
    df_kr_bytime_2021 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2021']
    df_kr_bytime_2022 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2022']
    df_kr_bytime_2023 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2023'] 
    df_kr_bytime_2024 = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] == '2024']
    years = ['2020', '2021', '2022', '2023', '2024', 'all']
    values = [[df_kr_bytime_2020['trending_number'].mean().round(2),
            df_kr_bytime_2021['trending_number'].mean().round(2),
            df_kr_bytime_2022['trending_number'].mean().round(2),
            df_kr_bytime_2023['trending_number'].mean().round(2),
            df_kr_bytime_2024['trending_number'].mean().round(2),
            df_kr_bytime['trending_number'].mean().round(2)],
            [df_kr_bytime_2020['first_trending'].mean().round(2),
                df_kr_bytime_2021['first_trending'].mean().round(2),
                df_kr_bytime_2022['first_trending'].mean().round(2),
                df_kr_bytime_2023['first_trending'].mean().round(2),
                df_kr_bytime_2024['first_trending'].mean().round(2),
                df_kr_bytime['first_trending'].mean().round(2)],
                [df_kr_bytime_2020['last_trending'].mean().round(2),
                df_kr_bytime_2021['last_trending'].mean().round(2),
                df_kr_bytime_2022['last_trending'].mean().round(2),
                df_kr_bytime_2023['last_trending'].mean().round(2),
                df_kr_bytime_2024['last_trending'].mean().round(2),
                df_kr_bytime['last_trending'].mean().round(2)],
                [df_kr_bytime_2020['trending_number'].max(),
                    df_kr_bytime_2021['trending_number'].max(),
                    df_kr_bytime_2022['trending_number'].max(),
                    df_kr_bytime_2023['trending_number'].max(),
                    df_kr_bytime_2024['trending_number'].max(),
                    df_kr_bytime['trending_number'].max()],
                    [df_kr_bytime_2020['first_trending'].max(),
                    df_kr_bytime_2021['first_trending'].max(),
                    df_kr_bytime_2022['first_trending'].max(),
                    df_kr_bytime_2023['first_trending'].max(),
                    df_kr_bytime_2024['first_trending'].max(),
                    df_kr_bytime['first_trending'].max()],
                    [df_kr_bytime_2020['last_trending'].max(),
                        df_kr_bytime_2021['last_trending'].max(),
                        df_kr_bytime_2022['last_trending'].max(),
                        df_kr_bytime_2023['last_trending'].max(),
                        df_kr_bytime_2024['last_trending'].max(),
                        df_kr_bytime['last_trending'].max()]]

    title = ['trending_number', 'first_trending', 'last_trending']

    fig, axs = plt.subplots(2, 3, figsize=(10.5, 7))

    for i in range(2):
        for j in range(3):
            ax = axs[i, j]
            ax.bar(years, values[i * 3 + j])

            for bar, value in zip(ax.patches, values[i * 3 + j]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, str(value), ha='center', va='bottom')

            ax.set_title(title[j])
            ax.set_xlabel('Years')
            if i == 0:
                ax.set_ylabel('mean value')
            elif i ==1:
                ax.set_ylabel('max value')
            else:
                ax.set_ylabel('min value')

    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('평균적인 인기동영상으로 변환되는 횟수는 년도가 지날수록 증가하는 추세이다.')
    st.write('평균적인 처음 인기동영상으로 변환되는데 걸리는 시간은 느려지는 추세이다.')
    st.write('평균적인 마지막으로 인기동영상으로 변환되는데 걸리는 시간또한 느려지는 추세이다.')
    st.write('1년 전체의 데이터가 아닌 2020년과 2024년을 제외하면 더 명확히 나타난다.')
    st.write('최대값에 대한 데이터는 평균적인 데이터와 유의미한 관계가 없기에 분석의 의미가 없다고 생각된다.')

def by_month():
    df_kr_bytime = df_kr.copy(deep=True)
    df_kr_bytime = df_kr.groupby(['title', 'channelTitle', 'publishedAt'])['taken_time'].apply(list).reset_index()
    df_kr_bytime = df_kr_bytime.sort_values(by='publishedAt')
    df_kr_bytime.reset_index(drop=True, inplace=True)
    df_kr_bytime['trending_number'] = df_kr_bytime['taken_time'].apply(len)
    df_kr_bytime['first_trending'] = df_kr_bytime['taken_time'].apply(min)
    df_kr_bytime['last_trending'] = df_kr_bytime['taken_time'].apply(max)
    st.write('영상 제목과 채널의 이름을 그룹으로 모든 시간을 리스트로 묵는다.')
    st.write('업로드 날짜를 기준으로 오름차순 정렬한다.')
    with st.expander('see code'):
        st.code('''df_kr_bytime = df_kr.copy(deep=True)
df_kr_bytime = df_kr.groupby(['title', 'channelTitle', 'publishedAt'])['taken_time'].apply(list).reset_index()
df_kr_bytime = df_kr_bytime.sort_values(by='publishedAt')
df_kr_bytime.reset_index(drop=True, inplace=True)''')
    st.write('몇일동안 인기동영상이었는지를 나타내는 trending_number 컬럼을 추가한다.')
    with st.expander('see code'):
        st.code('''df_kr_bytime['trending_number'] = df_kr_bytime['taken_time'].apply(len)''')
    st.write('처음 인기동영상이 되는데 걸린 시간을 나타내는 first_trending과 마지막 인기 동영상이 되는데 걸린 시간인 last_trending 컬럼을 추가한다.')
    with st.expander('see code'):
        st.code('''df_kr_bytime['first_trending'] = df_kr_bytime['taken_time'].apply(min)
df_kr_bytime['last_trending'] = df_kr_bytime['taken_time'].apply(max)''')
    st.write('2020년과 2024년의 경우 비어있는 월이 존재하므로 배제하기 위해 제거한다.')
    st.write('2021년부터 2023년까지 월별로 데이터를 나눈다.')
    with st.expander('see code'):
        st.code('''df_kr_bytime = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] != '2020']
df_kr_bytime = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] != '2024']
df_kr_bytime_Jan = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '01']
df_kr_bytime_Feb = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '02']
df_kr_bytime_Mar = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '03']
df_kr_bytime_Apr = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '04']
df_kr_bytime_May = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '05']
df_kr_bytime_Jun = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '06']
df_kr_bytime_Jul = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '07']
df_kr_bytime_Aug = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '08']
df_kr_bytime_Sep = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '09']
df_kr_bytime_Oct = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '10']
df_kr_bytime_Nob = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '11']
df_kr_bytime_Dec = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '12']''')
    st.write('월별로 인기동영상이 된 횟수를 모두 더해 그래프를 그린다.')
    with st.expander('see code'):
        st.code('''months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
data = [df_kr_bytime_Jan['trending_number'].sum(),
        df_kr_bytime_Feb['trending_number'].sum(),
        df_kr_bytime_Mar['trending_number'].sum(),
        df_kr_bytime_Apr['trending_number'].sum(),
        df_kr_bytime_May['trending_number'].sum(),
        df_kr_bytime_Jun['trending_number'].sum(),
        df_kr_bytime_Jul['trending_number'].sum(),
        df_kr_bytime_Aug['trending_number'].sum(),
        df_kr_bytime_Sep['trending_number'].sum(),
        df_kr_bytime_Oct['trending_number'].sum(),
        df_kr_bytime_Nob['trending_number'].sum(),
        df_kr_bytime_Dec['trending_number'].sum()]

df_month = pd.DataFrame({'months': months, 'data': data})
df_month = df_month.sort_values(by='data', ascending=False)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].bar(months, data)
axs[1].bar(df_month['months'], df_month['data'])

plt.tight_layout()
plt.show()''')
    df_kr_bytime = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] != '2020']
    df_kr_bytime = df_kr_bytime[df_kr_bytime['publishedAt'].str[:4] != '2024']
    df_kr_bytime_Jan = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '01']
    df_kr_bytime_Feb = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '02']
    df_kr_bytime_Mar = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '03']
    df_kr_bytime_Apr = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '04']
    df_kr_bytime_May = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '05']
    df_kr_bytime_Jun = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '06']
    df_kr_bytime_Jul = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '07']
    df_kr_bytime_Aug = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '08']
    df_kr_bytime_Sep = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '09']
    df_kr_bytime_Oct = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '10']
    df_kr_bytime_Nob = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '11']
    df_kr_bytime_Dec = df_kr_bytime[df_kr_bytime['publishedAt'].str[5:7] == '12']
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    data = [df_kr_bytime_Jan['trending_number'].sum(),
            df_kr_bytime_Feb['trending_number'].sum(),
            df_kr_bytime_Mar['trending_number'].sum(),
            df_kr_bytime_Apr['trending_number'].sum(),
            df_kr_bytime_May['trending_number'].sum(),
            df_kr_bytime_Jun['trending_number'].sum(),
            df_kr_bytime_Jul['trending_number'].sum(),
            df_kr_bytime_Aug['trending_number'].sum(),
            df_kr_bytime_Sep['trending_number'].sum(),
            df_kr_bytime_Oct['trending_number'].sum(),
            df_kr_bytime_Nob['trending_number'].sum(),
            df_kr_bytime_Dec['trending_number'].sum()]

    df_month = pd.DataFrame({'months': months, 'data': data})
    df_month = df_month.sort_values(by='data', ascending=False)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(months, data)
    axs[1].bar(df_month['months'], df_month['data'])

    plt.tight_layout()
    plt.show()
    st.pyplot()

    st.write('월별에 따라 막대그래프를 그리면 월별로 거의 차이가 나지 않는다.')
    st.write('내림차순으로 정렬해서 나타내면 6월이 가장 인기동영상이 많고, 2월이 가장 인기동영상이 적다.')
    st.write('7월의 인기동영상으로 선정된 수는 19,271회이다.')
    st.write('2월의 인기동영상으로 선정된 수는 17,034회이다.')
    st.write('2월이 인기동영상이 가장 적은 이유는 2월은 29일이기에 다른 달 보다 기간이 짧기 때문이라 예상된다.')
    st.write('마찬가지의 이유로 30일인 달보다 31일인 1,3, 5, 7, 8, 10, 12월에 인기동영상이 많게 나타난다.')
    st.write('이중 6월이 특이하게 30일이지만 31일인 월들 사이에서 2등을 차지하고 있다.')
    st.write('월의 일수로 나눈 값을 막대그래프로 나타낸다.')
    with st.expander('see code'):
        st.code('''months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
data = [df_kr_bytime_Jan['trending_number'].sum()/31,
        df_kr_bytime_Feb['trending_number'].sum()/29,
        df_kr_bytime_Mar['trending_number'].sum()/31,
        df_kr_bytime_Apr['trending_number'].sum()/30,
        df_kr_bytime_May['trending_number'].sum()/31,
        df_kr_bytime_Jun['trending_number'].sum()/30,
        df_kr_bytime_Jul['trending_number'].sum()/31,
        df_kr_bytime_Aug['trending_number'].sum()/31,
        df_kr_bytime_Sep['trending_number'].sum()/30,
        df_kr_bytime_Oct['trending_number'].sum()/31,
        df_kr_bytime_Nob['trending_number'].sum()/30,
        df_kr_bytime_Dec['trending_number'].sum()/31]

df_month = pd.DataFrame({'months': months, 'data': data})
df_month = df_month.sort_values(by='data', ascending=False)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].bar(months, data)
axs[1].bar(df_month['months'], df_month['data'])

plt.tight_layout()
plt.show()''')
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    data = [df_kr_bytime_Jan['trending_number'].sum()/31,
            df_kr_bytime_Feb['trending_number'].sum()/29,
            df_kr_bytime_Mar['trending_number'].sum()/31,
            df_kr_bytime_Apr['trending_number'].sum()/30,
            df_kr_bytime_May['trending_number'].sum()/31,
            df_kr_bytime_Jun['trending_number'].sum()/30,
            df_kr_bytime_Jul['trending_number'].sum()/31,
            df_kr_bytime_Aug['trending_number'].sum()/31,
            df_kr_bytime_Sep['trending_number'].sum()/30,
            df_kr_bytime_Oct['trending_number'].sum()/31,
            df_kr_bytime_Nob['trending_number'].sum()/30,
            df_kr_bytime_Dec['trending_number'].sum()/31]

    df_month = pd.DataFrame({'months': months, 'data': data})
    df_month = df_month.sort_values(by='data', ascending=False)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(months, data)
    axs[1].bar(df_month['months'], df_month['data'])

    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('일별로 나누었을 경우 6, 7월에 집중적으로 인기동영상을 노리는 것이 좋다는 분석이 나온다.')

def by_count1():
    st.write('좋아요, 댓글, 조회수를 인기동영상이 되는데 걸린 시간으로 나누어 1일에 필요한 좋아요, 댓글, 조회수를 구한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount = df_kr.copy(deep=True)
df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)''')
    st.write('영상 제목과 채널의 이름을 그룹으로 좋아요, 댓글, 조회수를 각각 리스트로 묵는다.')
    st.write('업로드 날짜를 기준으로 오름차순 정렬한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
df_kr_bycount.reset_index(drop=True, inplace=True)''')
    st.write('각 영상의 일일 좋아요, 댓글, 조회수의 평균을 나타내는 컬럼을 추가한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)''')
        
    df_kr_bycount = df_kr.copy(deep=True)
    df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
    df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
    df_kr_bycount.reset_index(drop=True, inplace=True)
    df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)
    
    st.write('df를 복사한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount1 = df_kr_bycount.copy(deep=True)''')
    st.write('일일 좋아요, 댓글, 조회수가 0, inf, -inf인 데이터를 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != 0]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != 0]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != 0]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != -np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != -np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != -np.inf]''')
    st.write('일일 좋아요, 댓글, 조회수 리스트와 걸린시간은 필요없으므로 열을 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount1.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time'], axis=1, inplace=True)''')
    st.write('최소, 최대, 평균값에 대해 출력한다.')
    with st.expander('see code'):
        st.code('''x = ['mean', 'min', 'max']
y1 = [df_kr_bycount1['mean_day_likes'].mean().round(2),
      df_kr_bycount1['mean_day_likes'].min().round(2),
      df_kr_bycount1['mean_day_likes'].max().round(2)]
y2 = [df_kr_bycount1['mean_day_comment_count'].mean().round(2),
      df_kr_bycount1['mean_day_comment_count'].min().round(2),
      df_kr_bycount1['mean_day_comment_count'].max().round(2)]
y3 = [df_kr_bycount1['mean_day_view_count'].mean().round(2),
      df_kr_bycount1['mean_day_view_count'].min().round(2),
      df_kr_bycount1['mean_day_view_count'].max().round(2)]

print('mean_day_likes', y1)
print('mean_day_comment_count', y2)
print('mean_day_view_count', y3)''')
    st.write('구간을 30개로 히스토그램을 그린다.')
    df_kr_bycount1 = df_kr_bycount.copy(deep=True)
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != 0]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != 0]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != 0]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != -np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != -np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != -np.inf]
    df_kr_bycount1.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time'], axis=1, inplace=True)
    x = ['mean', 'min', 'max']
    y1 = [df_kr_bycount1['mean_day_likes'].mean().round(2),
        df_kr_bycount1['mean_day_likes'].min().round(2),
        df_kr_bycount1['mean_day_likes'].max().round(2)]
    y2 = [df_kr_bycount1['mean_day_comment_count'].mean().round(2),
        df_kr_bycount1['mean_day_comment_count'].min().round(2),
        df_kr_bycount1['mean_day_comment_count'].max().round(2)]
    y3 = [df_kr_bycount1['mean_day_view_count'].mean().round(2),
        df_kr_bycount1['mean_day_view_count'].min().round(2),
        df_kr_bycount1['mean_day_view_count'].max().round(2)]
    st.write('mean_day_likes', y1)
    st.write('mean_day_comment_count', y2)
    st.write('mean_day_view_count', y3)
    with st.expander('see code'):
        st.code('''fig, axs = plt.subplots(1, 3, figsize=(10.5, 4))

counts0, bins0, patches0 = axs[0].hist(df_kr_bycount1['mean_day_likes'], bins=30, edgecolor='black')
axs[0].set_title('mean_day_likes')

counts1, bins1, patches1 = axs[1].hist(df_kr_bycount1['mean_day_comment_count'], bins=30, edgecolor='black')
axs[1].set_title('mean_day_comment_count')

counts2, bins2, patches2 = axs[2].hist(df_kr_bycount1['mean_day_view_count'], bins=30, edgecolor='black')
axs[2].set_title('mean_day_view_count')

plt.tight_layout()
plt.show()''')
    
    
    
    
    
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4))

    counts0, bins0, patches0 = axs[0].hist(df_kr_bycount1['mean_day_likes'], bins=30, edgecolor='black')
    axs[0].set_title('mean_day_likes')

    counts1, bins1, patches1 = axs[1].hist(df_kr_bycount1['mean_day_comment_count'], bins=30, edgecolor='black')
    axs[1].set_title('mean_day_comment_count')

    counts2, bins2, patches2 = axs[2].hist(df_kr_bycount1['mean_day_view_count'], bins=30, edgecolor='black')
    axs[2].set_title('mean_day_view_count')

    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('평균, 최소, 최대를 막대 그래프로 그린다.')
    with st.expander('see code'):
        st.code('''fig, axs = plt.subplots(1, 3, figsize=(10.5, 4))

axs[0].bar(x, y1)
axs[0].set_title('mean_day_likes')

axs[1].bar(x, y2)
axs[1].set_title('mean_day_comment_count')

axs[2].bar(x, y3)
axs[2].set_title('mean_day_view_count')

plt.tight_layout()
plt.show()''')
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4))

    axs[0].bar(x, y1)
    axs[0].set_title('mean_day_likes')

    axs[1].bar(x, y2)
    axs[1].set_title('mean_day_comment_count')

    axs[2].bar(x, y3)
    axs[2].set_title('mean_day_view_count')

    plt.tight_layout()
    plt.show()
    st.pyplot()
    
    st.write('히스토그램의 가장 많이 분포하는 범위로 한정한다.')
    with st.expander('see code'):
        st.code('''max_bin_index = np.argmax(counts0)
max_bin_range = (bins0[max_bin_index], bins0[max_bin_index + 1])

df_kr_bycount10 = df_kr_bycount1.copy(deep=True)
df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_likes'] >= bins0[max_bin_index]]
df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_likes'] < bins0[max_bin_index + 1]]
print(df_kr_bycount10['mean_day_likes'].mean())

max_bin_index = np.argmax(counts1)
max_bin_range = (bins1[max_bin_index], bins1[max_bin_index + 1])

df_kr_bycount10 = df_kr_bycount1.copy(deep=True)
df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_comment_count'] >= bins1[max_bin_index]]
df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_comment_count'] < bins1[max_bin_index + 1]]
print(df_kr_bycount10['mean_day_comment_count'].mean())

max_bin_index = np.argmax(counts2)
max_bin_range = (bins2[max_bin_index], bins2[max_bin_index + 1])

df_kr_bycount10 = df_kr_bycount1.copy(deep=True)
df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_view_count'] >= bins2[max_bin_index]]
df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_view_count'] < bins2[max_bin_index + 1]]
print(df_kr_bycount10['mean_day_view_count'].mean())''')
    max_bin_index = np.argmax(counts0)
    max_bin_range = (bins0[max_bin_index], bins0[max_bin_index + 1])

    df_kr_bycount10 = df_kr_bycount1.copy(deep=True)
    df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_likes'] >= bins0[max_bin_index]]
    df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_likes'] < bins0[max_bin_index + 1]]
    st.write("mean_day_likes", df_kr_bycount10['mean_day_likes'].mean())

    max_bin_index = np.argmax(counts1)
    max_bin_range = (bins1[max_bin_index], bins1[max_bin_index + 1])

    df_kr_bycount10 = df_kr_bycount1.copy(deep=True)
    df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_comment_count'] >= bins1[max_bin_index]]
    df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_comment_count'] < bins1[max_bin_index + 1]]
    st.write("mean_day_comment_count", df_kr_bycount10['mean_day_comment_count'].mean())

    max_bin_index = np.argmax(counts2)
    max_bin_range = (bins2[max_bin_index], bins2[max_bin_index + 1])

    df_kr_bycount10 = df_kr_bycount1.copy(deep=True)
    df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_view_count'] >= bins2[max_bin_index]]
    df_kr_bycount10 = df_kr_bycount10[df_kr_bycount10['mean_day_view_count'] < bins2[max_bin_index + 1]]
    st.write("mean_day_view_count", df_kr_bycount10['mean_day_view_count'].mean())
    st.write('''히스토그램을 통해 분포를 보았을때 최소값 주변의 분포가 압도적으로 많다.\n
최소값과 최대값은 평균과 비교했을 때 유의미한 결과를 주기에는 그 차이가 너무 크다.\n
히스토그램에 가장 많이 분포하는 범위로 한정했을때\n
평균적으로 인기동영상이 되는데 하루에 필요한 좋아요는 6,288개\n
평균적으로 인기동영상이 되는데 하루에 필요한 댓글은 742개\n
평균적으로 인기동영상이 되는데 하루에 필요한 조회수는 180,652회\n
모든 값을 평균냈을 때\n
평균적으로 인기동영상이 되는데 하루에 필요한 좋아요는 14,261개\n
평균적으로 인기동영상이 되는데 하루에 필요한 댓글은 1,225개\n
평균적으로 인기동영상이 되는데 하루에 필요한 조회수는 311,800회\n
히스토그램을 관찰했을 때 가장 많이 분포하는 범위의 표본 수가 압도적으로 많이 존재한다.\n
평균이상의 값을 가지는 표본수가 일부에 불과하다.\n
일부의 평균이상의 값을 가진 표본이 총 평균과 범위를 한정지은 표본의 평균의 약 2배의 차이를 만들어낸다.\n
이런 이유로 적어도 인기동영상이 되는데 하루에 필요한 수는 범위를 한정지은 표본을 따르는 것이 옳다고 판단한다.\n''')

def by_count2():
    st.write('좋아요, 댓글, 조회수를 인기동영상이 되는데 걸린 시간으로 나누어 1일에 필요한 좋아요, 댓글, 조회수를 구한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount = df_kr.copy(deep=True)
df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)''')
    st.write('영상 제목과 채널의 이름을 그룹으로 좋아요, 댓글, 조회수를 각각 리스트로 묵는다.')
    st.write('업로드 날짜를 기준으로 오름차순 정렬한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
df_kr_bycount.reset_index(drop=True, inplace=True)''')
    st.write('각 영상의 일일 좋아요, 댓글, 조회수의 평균을 나타내는 컬럼을 추가한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)''')
        
    df_kr_bycount = df_kr.copy(deep=True)
    df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
    df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
    df_kr_bycount.reset_index(drop=True, inplace=True)
    df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)

    st.write('df를 복사한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount2 = df_kr_bycount.copy(deep=True)''')
    st.write('좋아요가 0인 데이터만 가져온다.')
    st.write('댓글, 조회수가 0, inf, -inf인 데이터를 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_likes'] == 0]
df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_comment_count'] != 0]
df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_view_count'] != 0]
df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_likes'] != np.inf]
df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_comment_count'] != np.inf]
df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_view_count'] != np.inf]
df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_likes'] != -np.inf]
df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_comment_count'] != -np.inf]
df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_view_count'] != -np.inf]''')
    st.write('일일 좋아요, 댓글, 조회수 리스트와 걸린시간은 필요없으므로 열을 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount2.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time'], axis=1, inplace=True)''')
    st.write('최소, 최대, 평균값에 대해 출력한다.')
    with st.expander('see code'):
        st.code('''x = ['mean', 'min', 'max']
y1 = [df_kr_bycount2['mean_day_comment_count'].mean().round(2),
      df_kr_bycount2['mean_day_comment_count'].min().round(2),
      df_kr_bycount2['mean_day_comment_count'].max().round(2)]
y2 = [df_kr_bycount2['mean_day_view_count'].mean().round(2),
      df_kr_bycount2['mean_day_view_count'].min().round(2),
      df_kr_bycount2['mean_day_view_count'].max().round(2)]

print('mean_day_comment_count', y1)
print('mean_day_view_count', y2)''')
    df_kr_bycount2 = df_kr_bycount.copy(deep=True)
    df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_likes'] == 0]
    df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_comment_count'] != 0]
    df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_view_count'] != 0]
    df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_likes'] != np.inf]
    df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_comment_count'] != np.inf]
    df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_view_count'] != np.inf]
    df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_likes'] != -np.inf]
    df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_comment_count'] != -np.inf]
    df_kr_bycount2 = df_kr_bycount2[df_kr_bycount2['mean_day_view_count'] != -np.inf]
    df_kr_bycount2.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time'], axis=1, inplace=True)
    x = ['mean', 'min', 'max']
    y1 = [df_kr_bycount2['mean_day_comment_count'].mean().round(2),
        df_kr_bycount2['mean_day_comment_count'].min().round(2),
        df_kr_bycount2['mean_day_comment_count'].max().round(2)]
    y2 = [df_kr_bycount2['mean_day_view_count'].mean().round(2),
        df_kr_bycount2['mean_day_view_count'].min().round(2),
        df_kr_bycount2['mean_day_view_count'].max().round(2)]
    st.write('mean_day_comment_count', y1)
    st.write('mean_day_view_count', y2)
    st.write('구간을 50개로 히스토그램을 그린다.')
    with st.expander('see code'):
        st.code('''fig, axs = plt.subplots(1, 2, figsize=(7, 4))

counts1, bins1, patches1 = axs[0].hist(df_kr_bycount2['mean_day_comment_count'], bins=50, edgecolor='black')
axs[0].set_title('mean_day_comment_count')

counts2, bins2, patches2 = axs[1].hist(df_kr_bycount2['mean_day_view_count'], bins=50, edgecolor='black')
axs[1].set_title('mean_day_view_count')

plt.tight_layout()
plt.show()''')
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    counts1, bins1, patches1 = axs[0].hist(df_kr_bycount2['mean_day_comment_count'], bins=50, edgecolor='black')
    axs[0].set_title('mean_day_comment_count')

    counts2, bins2, patches2 = axs[1].hist(df_kr_bycount2['mean_day_view_count'], bins=50, edgecolor='black')
    axs[1].set_title('mean_day_view_count')

    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('평균, 최소, 최대를 막대 그래프로 그린다.')
    with st.expander('see code'):
        st.code('''fig, axs = plt.subplots(1, 2, figsize=(7, 4))

axs[0].bar(x, y1)
axs[0].set_title('mean_day_comment_count')

axs[1].bar(x, y2)
axs[1].set_title('mean_day_view_count')


plt.tight_layout()
plt.show()''')
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    axs[0].bar(x, y1)
    axs[0].set_title('mean_day_comment_count')

    axs[1].bar(x, y2)
    axs[1].set_title('mean_day_view_count')


    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('히스토그램의 가장 많이 분포하는 범위로 한정한다.')
    with st.expander('see code'):
        st.code('''max_bin_index = np.argsort(counts1)[-10:][::-1]
max_bin_range = [(bins1[i], bins1[i + 1]) for i in max_bin_index]
a=0
b=0
for i in max_bin_index:
  df_kr_bycount20 = df_kr_bycount2.copy(deep=True)
  df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_comment_count'] >= bins1[i]]
  df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_comment_count'] < bins1[i + 1]]
  a += df_kr_bycount20['mean_day_comment_count'].sum()
  b += counts1[i]
print(a/b)

max_bin_index = np.argsort(counts2)[-10:][::-1]
max_bin_range = [(bins2[i], bins2[i + 1]) for i in max_bin_index]
a=0
b=0
for i in max_bin_index:
  df_kr_bycount20 = df_kr_bycount2.copy(deep=True)
  df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_view_count'] >= bins2[i]]
  df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_view_count'] < bins2[i + 1]]
  a += df_kr_bycount20['mean_day_view_count'].sum()
  b += counts2[i]
print(a/b)

max_bin_index = np.argmax(counts1)
max_bin_range = (bins1[max_bin_index], bins1[max_bin_index + 1])

df_kr_bycount20 = df_kr_bycount2.copy(deep=True)
df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_comment_count'] >= bins1[max_bin_index]]
df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_comment_count'] < bins1[max_bin_index + 1]]
print(df_kr_bycount20['mean_day_comment_count'].mean())

max_bin_index = np.argmax(counts2)
max_bin_range = (bins2[max_bin_index], bins2[max_bin_index + 1])

df_kr_bycount20 = df_kr_bycount2.copy(deep=True)
df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_view_count'] >= bins2[max_bin_index]]
df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_view_count'] < bins2[max_bin_index + 1]]
print(df_kr_bycount20['mean_day_view_count'].mean())''')
    max_bin_index = np.argsort(counts1)[-10:][::-1]
    max_bin_range = [(bins1[i], bins1[i + 1]) for i in max_bin_index]
    a=0
    b=0
    for i in max_bin_index:
        df_kr_bycount20 = df_kr_bycount2.copy(deep=True)
        df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_comment_count'] >= bins1[i]]
        df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_comment_count'] < bins1[i + 1]]
        a += df_kr_bycount20['mean_day_comment_count'].sum()
        b += counts1[i]
    st.write('mean_day_comment_count', a/b)

    max_bin_index = np.argsort(counts2)[-10:][::-1]
    max_bin_range = [(bins2[i], bins2[i + 1]) for i in max_bin_index]
    a=0
    b=0
    for i in max_bin_index:
        df_kr_bycount20 = df_kr_bycount2.copy(deep=True)
        df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_view_count'] >= bins2[i]]
        df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_view_count'] < bins2[i + 1]]
        a += df_kr_bycount20['mean_day_view_count'].sum()
        b += counts2[i]
    st.write('mean_day_view_count', a/b)

    max_bin_index = np.argmax(counts1)
    max_bin_range = (bins1[max_bin_index], bins1[max_bin_index + 1])

    df_kr_bycount20 = df_kr_bycount2.copy(deep=True)
    df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_comment_count'] >= bins1[max_bin_index]]
    df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_comment_count'] < bins1[max_bin_index + 1]]
    st.write('mean_day_comment_count', df_kr_bycount20['mean_day_comment_count'].mean())

    max_bin_index = np.argmax(counts2)
    max_bin_range = (bins2[max_bin_index], bins2[max_bin_index + 1])

    df_kr_bycount20 = df_kr_bycount2.copy(deep=True)
    df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_view_count'] >= bins2[max_bin_index]]
    df_kr_bycount20 = df_kr_bycount20[df_kr_bycount20['mean_day_view_count'] < bins2[max_bin_index + 1]]
    st.write('mean_day_view_count', df_kr_bycount20['mean_day_view_count'].mean())
    st.write('''표본이 425개로 적다.\n
히스토그램을 통해 분포를 보았을 때 최소값 주변의 분포가 압도적으로 많다.\n
최소값과 최대값은 평균과 비교했을 때 유의미한 결과를 주기에는 그 차이가 너무 크다.\n
히스토그램에 가장 많이 분포하는 10개의 범위로 한정했을 때\n
평균적으로 인기동영상이 되는데 하루에 필요한 댓글은 599개\n
평균적으로 인기동영상이 되는데 하루에 필요한 조회수는 299,810회\n
히스토그램에 가장 많이 분포하는 범위로 한정했을 때\n
평균적으로 인기동영상이 되는데 하루에 필요한 댓글은 135개\n
평균적으로 인기동영상이 되는데 하루에 필요한 조회수는 119,602회\n
모든값을 평균냈을 때\n
평균적으로 인기동영상이 되는데 하루에 필요한 댓글은 1,105개\n
평균적으로 인기동영상이 되는데 하루에 필요한 조회수는 380,218회\n
기본적으로 표본의 개수가 적다.\n
히스토그램을 관찰했을 때 가장 많이 분포하는 범위의 표본 수가 그렇지 않은 범위에 비해 많이 존재한다.\n
평균이상의 값을 가지는 표본수가 일부에 불과하다.\n
일부의 평균이상의 값을 가진 표본이 총 평균과 범위를 한정지은 표본의 평균의 약 2~10배의 차이를 만들어낸다.\n
이런 이유로 적어도 인기동영상이 되는데 하루에 필요한 수는 10개의 범위로 한정지은 표본을 따르는 것이 옳다고 판단한다.\n''')

def by_count3():
    st.write('좋아요, 댓글, 조회수를 인기동영상이 되는데 걸린 시간으로 나누어 1일에 필요한 좋아요, 댓글, 조회수를 구한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount = df_kr.copy(deep=True)
df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)''')
    st.write('영상 제목과 채널의 이름을 그룹으로 좋아요, 댓글, 조회수를 각각 리스트로 묵는다.')
    st.write('업로드 날짜를 기준으로 오름차순 정렬한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
df_kr_bycount.reset_index(drop=True, inplace=True)''')
    st.write('각 영상의 일일 좋아요, 댓글, 조회수의 평균을 나타내는 컬럼을 추가한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)''')
        
    df_kr_bycount = df_kr.copy(deep=True)
    df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
    df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
    df_kr_bycount.reset_index(drop=True, inplace=True)
    df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)

    st.write('df를 복사한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount3 = df_kr_bycount.copy(deep=True)''')
    st.write('좋아요가 0인 데이터만 가져온다.')
    st.write('댓글, 조회수가 0, inf, -inf인 데이터를 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_likes'] != 0]
df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_comment_count'] == 0]
df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_view_count'] != 0]
df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_likes'] != np.inf]
df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_comment_count'] != np.inf]
df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_view_count'] != np.inf]
df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_likes'] != -np.inf]
df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_comment_count'] != -np.inf]
df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_view_count'] != -np.inf]''')
    st.write('일일 좋아요, 댓글, 조회수 리스트와 걸린시간은 필요없으므로 열을 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount3.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time'], axis=1, inplace=True)''')
    st.write('최소, 최대, 평균값에 대해 출력한다.')
    with st.expander('see code'):
        st.code('''x = ['mean', 'min', 'max']
y1 = [df_kr_bycount3['mean_day_likes'].mean().round(2),
      df_kr_bycount3['mean_day_likes'].min().round(2),
      df_kr_bycount3['mean_day_likes'].max().round(2)]
y2 = [df_kr_bycount3['mean_day_view_count'].mean().round(2),
      df_kr_bycount3['mean_day_view_count'].min().round(2),
      df_kr_bycount3['mean_day_view_count'].max().round(2)]

print('mean_day_likes', y1)
print('mean_day_view_count', y2)''')
    df_kr_bycount3 = df_kr_bycount.copy(deep=True)
    df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_likes'] != 0]
    df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_comment_count'] == 0]
    df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_view_count'] != 0]
    df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_likes'] != np.inf]
    df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_comment_count'] != np.inf]
    df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_view_count'] != np.inf]
    df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_likes'] != -np.inf]
    df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_comment_count'] != -np.inf]
    df_kr_bycount3 = df_kr_bycount3[df_kr_bycount3['mean_day_view_count'] != -np.inf]
    df_kr_bycount3.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time'], axis=1, inplace=True)
    x = ['mean', 'min', 'max']
    y1 = [df_kr_bycount3['mean_day_likes'].mean().round(2),
        df_kr_bycount3['mean_day_likes'].min().round(2),
        df_kr_bycount3['mean_day_likes'].max().round(2)]
    y2 = [df_kr_bycount3['mean_day_view_count'].mean().round(2),
        df_kr_bycount3['mean_day_view_count'].min().round(2),
        df_kr_bycount3['mean_day_view_count'].max().round(2)]
    st.write('mean_day_likes', y1)
    st.write('mean_day_view_count', y2)
    st.write('구간을 50개로 히스토그램을 그린다.')
    with st.expander('see code'):
        st.code('''fig, axs = plt.subplots(1, 2, figsize=(7, 4))

counts0, bins0, patches0 = axs[0].hist(df_kr_bycount3['mean_day_likes'], bins=50, edgecolor='black')
axs[0].set_title('mean_day_comment_count')

counts2, bins2, patches2 = axs[1].hist(df_kr_bycount3['mean_day_view_count'], bins=50, edgecolor='black')
axs[1].set_title('mean_day_view_count')

plt.tight_layout()
plt.show()''')
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    counts0, bins0, patches0 = axs[0].hist(df_kr_bycount3['mean_day_likes'], bins=50, edgecolor='black')
    axs[0].set_title('mean_day_comment_count')

    counts2, bins2, patches2 = axs[1].hist(df_kr_bycount3['mean_day_view_count'], bins=50, edgecolor='black')
    axs[1].set_title('mean_day_view_count')

    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('평균, 최소, 최대를 막대 그래프로 그린다.')
    with st.expander('see code'):
        st.code('''fig, axs = plt.subplots(1, 2, figsize=(7, 4))

axs[0].bar(x, y1)
axs[0].set_title('mean_day_likes')

axs[1].bar(x, y2)
axs[1].set_title('mean_day_view_count')


plt.tight_layout()
plt.show()''')
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    axs[0].bar(x, y1)
    axs[0].set_title('mean_day_likes')

    axs[1].bar(x, y2)
    axs[1].set_title('mean_day_view_count')


    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('히스토그램의 가장 많이 분포하는 범위로 한정한다.')
    with st.expander('see code'):
        st.code('''max_bin_index = np.argsort(counts0)[-10:][::-1]
max_bin_range = [(bins0[i], bins0[i + 1]) for i in max_bin_index]
a=0
b=0
for i in max_bin_index:
  df_kr_bycount30 = df_kr_bycount3.copy(deep=True)
  df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_likes'] >= bins0[i]]
  df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_likes'] < bins0[i + 1]]
  a += df_kr_bycount30['mean_day_likes'].sum()
  b += counts0[i]
print(a/b)

max_bin_index = np.argsort(counts2)[-10:][::-1]
max_bin_range = [(bins2[i], bins2[i + 1]) for i in max_bin_index]
a=0
b=0
for i in max_bin_index:
  df_kr_bycount30 = df_kr_bycount3.copy(deep=True)
  df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_view_count'] >= bins2[i]]
  df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_view_count'] < bins2[i + 1]]
  a += df_kr_bycount30['mean_day_view_count'].sum()
  b += counts2[i]
print(a/b)

max_bin_index = np.argmax(counts0)
max_bin_range = (bins0[max_bin_index], bins0[max_bin_index + 1])

df_kr_bycount30 = df_kr_bycount3.copy(deep=True)
df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_likes'] >= bins0[max_bin_index]]
df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_likes'] < bins0[max_bin_index + 1]]
print(df_kr_bycount30['mean_day_likes'].mean())

max_bin_index = np.argmax(counts2)
max_bin_range = (bins2[max_bin_index], bins2[max_bin_index + 1])

df_kr_bycount30 = df_kr_bycount3.copy(deep=True)
df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_view_count'] >= bins2[max_bin_index]]
df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_view_count'] < bins2[max_bin_index + 1]]
print(df_kr_bycount30['mean_day_view_count'].mean())''')
    max_bin_index = np.argsort(counts0)[-10:][::-1]
    max_bin_range = [(bins0[i], bins0[i + 1]) for i in max_bin_index]
    a=0
    b=0
    for i in max_bin_index:
        df_kr_bycount30 = df_kr_bycount3.copy(deep=True)
        df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_likes'] >= bins0[i]]
        df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_likes'] < bins0[i + 1]]
        a += df_kr_bycount30['mean_day_likes'].sum()
        b += counts0[i]
    st.write('mean_day_likes', a/b)

    max_bin_index = np.argsort(counts2)[-10:][::-1]
    max_bin_range = [(bins2[i], bins2[i + 1]) for i in max_bin_index]
    a=0
    b=0
    for i in max_bin_index:
        df_kr_bycount30 = df_kr_bycount3.copy(deep=True)
        df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_view_count'] >= bins2[i]]
        df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_view_count'] < bins2[i + 1]]
        a += df_kr_bycount30['mean_day_view_count'].sum()
        b += counts2[i]
    print('mean_day_view_count', a/b)

    max_bin_index = np.argmax(counts0)
    max_bin_range = (bins0[max_bin_index], bins0[max_bin_index + 1])

    df_kr_bycount30 = df_kr_bycount3.copy(deep=True)
    df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_likes'] >= bins0[max_bin_index]]
    df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_likes'] < bins0[max_bin_index + 1]]
    print('mean_day_likes', df_kr_bycount30['mean_day_likes'].mean())

    max_bin_index = np.argmax(counts2)
    max_bin_range = (bins2[max_bin_index], bins2[max_bin_index + 1])

    df_kr_bycount30 = df_kr_bycount3.copy(deep=True)
    df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_view_count'] >= bins2[max_bin_index]]
    df_kr_bycount30 = df_kr_bycount30[df_kr_bycount30['mean_day_view_count'] < bins2[max_bin_index + 1]]
    print('mean_day_view_count', df_kr_bycount30['mean_day_view_count'].mean())
    st.write('''표본이 285개로 적다.\n
히스토그램을 통해 분포를 보았을때 최소값 주변의 분포가 압도적으로 많다.\n
최소값과 최대값은 평균과 비교했을 때 유의미한 결과를 주기에는 그 차이가 너무 크다.\n
히스토그램에 가장 많이 분포하는 10개의 범위로 한정했을때\n
평균적으로 인기동영상이 되는데 하루에 필요한 좋아요는 5920개\n
평균적으로 인기동영상이 되는데 하루에 필요한 조회수는 298,743회\n
히스토그램에 가장 많이 분포하는 범위로 한정했을때\n
평균적으로 인기동영상이 되는데 하루에 필요한 좋아요는 1827개\n
평균적으로 인기동영상이 되는데 하루에 필요한 조회수는 101,400회\n
모든값을 평균냈을때.\n
평균적으로 인기동영상이 되는데 하루에 필요한 좋아요는 12,543개\n
평균적으로 인기동영상이 되는데 하루에 필요한 조회수는 415,646회\n
기본적으로 표본의 개수가 적다.\n
히스토그램을 관찰했을때 가장 많이 분포하는 범위의 표본 수가 그렇지 않은 범위에 비해 많이 존재한다.\n
평균이상의 값을 가지는 표본수가 일부에 불과하다.\n
일부의 평균이상의 값을 가진 표본이 총 평균과 범위를 한정지은 표본의 평균의 약 2~3배의 차이를 만들어낸다.\n
이런 이유로 적어도 인기동영상이 되는데 하루에 필요한 수는 10개의 범위로 한정지은 표본을 따르는 것이 옳다고 판단한다.\n''')

def by_count4():
    st.write('좋아요, 댓글, 조회수를 인기동영상이 되는데 걸린 시간으로 나누어 1일에 필요한 좋아요, 댓글, 조회수를 구한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount = df_kr.copy(deep=True)
df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)''')
    st.write('영상 제목과 채널의 이름을 그룹으로 좋아요, 댓글, 조회수를 각각 리스트로 묵는다.')
    st.write('업로드 날짜를 기준으로 오름차순 정렬한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
df_kr_bycount.reset_index(drop=True, inplace=True)''')
    st.write('각 영상의 일일 좋아요, 댓글, 조회수의 평균을 나타내는 컬럼을 추가한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)''')
        
    df_kr_bycount = df_kr.copy(deep=True)
    df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
    df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
    df_kr_bycount.reset_index(drop=True, inplace=True)
    df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)

    st.write('df를 복사한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount3 = df_kr_bycount.copy(deep=True)''')
    st.write('좋아요와 댓글이 0인 데이터만 가져온다.')
    st.write('조회수가 0, inf, -inf인 데이터를 제거해준다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_likes'] == 0]
df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_comment_count'] == 0]
df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_view_count'] != 0]
df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_likes'] != np.inf]
df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_comment_count'] != np.inf]
df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_view_count'] != np.inf]
df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_likes'] != -np.inf]
df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_comment_count'] != -np.inf]
df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_view_count'] != -np.inf]''')
    st.write('일일 좋아요, 댓글, 조회수 리스트와 걸린시간은 필요없으므로 열을 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount4.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time'], axis=1, inplace=True)''')
    st.write('# 최소, 최대, 평균값에 대해 출력한다.')
    with st.expander('see code'):
        st.code('''x = ['mean', 'min', 'max']
y1 = [df_kr_bycount4['mean_day_view_count'].mean().round(2),
      df_kr_bycount4['mean_day_view_count'].min().round(2),
      df_kr_bycount4['mean_day_view_count'].max().round(2)]

print('mean_day_view_count', y1)''')
    df_kr_bycount4 = df_kr_bycount.copy(deep=True)
    df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_likes'] == 0]
    df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_comment_count'] == 0]
    df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_view_count'] != 0]
    df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_likes'] != np.inf]
    df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_comment_count'] != np.inf]
    df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_view_count'] != np.inf]
    df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_likes'] != -np.inf]
    df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_comment_count'] != -np.inf]
    df_kr_bycount4 = df_kr_bycount4[df_kr_bycount4['mean_day_view_count'] != -np.inf]
    df_kr_bycount4.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time'], axis=1, inplace=True)
    x = ['mean', 'min', 'max']
    y1 = [df_kr_bycount4['mean_day_view_count'].mean().round(2),
        df_kr_bycount4['mean_day_view_count'].min().round(2),
        df_kr_bycount4['mean_day_view_count'].max().round(2)]

    st.write('mean_day_view_count', y1)
    st.write('막대그래프를 그린다.')
    with st.expander('see code'):
        st.code('''plt.figure(figsize=(3.5, 4))

plt.bar(x, y1)
plt.title('mean_day_view_count')

plt.show()''')
    plt.figure(figsize=(3.5, 4))

    plt.bar(x, y1)
    plt.title('mean_day_view_count')

    plt.show()
    st.pyplot()
    st.write('''표본이 18개로 사실상 거의 존재하지 않는다.\n
최소값과 최대값은 평균과 비교했을 때 유의미한 결과를 주기에는 그 차이가 너무 크다.\n
평균적으로 인기동영상이 되는데 하루에 필요한 조회수는 2,057,550회\n
기본적으로 표본이 거의 없다.\n
좋아요와 댓글이 둘다 없기에 일일 조회수 만으로 인기동영상이 되기 위해서는 그렇지 않은 경우의 5,6배의 일일 조회수가 요구된다.\n
하지만 표본이 거의 없기에 이 경우의 분석은 의미가 없다 판단된다.\n''')

def by_ratio():
    st.write('좋아요, 댓글, 조회수를 인기동영상이 되는데 걸린 시간으로 나누어 1일에 필요한 좋아요, 댓글, 조회수를 구한다.')
    with st.expander('see code'):
        st.code('''df_kr_byratio = df_kr.copy(deep=True)
df_kr_byratio['day_likes'] = (df_kr['likes'] / df_kr['taken_time'])
df_kr_byratio['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time'])
df_kr_byratio['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time'])''')
    st.write('좋아요, 댓글, 조회수의 비율을 저장한다. %로 계산한다.')
    with st.expander('see code'):
        st.code('''df_kr_byratio['likes/view'] = (df_kr_byratio['day_likes'] / df_kr_byratio['day_view_count'] * 100)
df_kr_byratio['comment/view'] = (df_kr_byratio['day_comment_count'] / df_kr_byratio['day_view_count'] * 100)
df_kr_byratio['likes/comment'] = (df_kr_byratio['day_likes'] / df_kr_byratio['day_comment_count'] * 100)''')
    st.write('영상 제목과 채널의 이름을 그룹으로 좋아요, 댓글, 조회수의 비율과 조회수를 각각 리스트로 묵는다.')
    st.write('업로드 날짜를 기준으로 오름차순 정렬한다.')
    with st.expander('see code'):
        st.code('''df_kr_byratio = df_kr_byratio.groupby(['title', 'channelTitle', 'publishedAt']).agg({'likes/view': list, 'comment/view': list, 'likes/comment': list, 'day_view_count': list}).reset_index()
df_kr_byratio = df_kr_byratio.sort_values(by='publishedAt')
df_kr_byratio.reset_index(drop=True, inplace=True)''')
    st.write('각 평균을 나타내는 컬럼을 추가한다.')
    with st.expander('see code'):
        st.code('''df_kr_byratio['mean_likes/view'] = df_kr_byratio['likes/view'].apply(lambda x: sum(x) / len(x))
df_kr_byratio['mean_comment/view'] = df_kr_byratio['comment/view'].apply(lambda x: sum(x) / len(x))
df_kr_byratio['mean_likes/comment'] = df_kr_byratio['likes/comment'].apply(lambda x: sum(x) / len(x))
df_kr_byratio['mean_day_view_count'] = df_kr_byratio['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)''')
    st.write('비율과 일일 조회수에 대한 리스트는 필요없으므로 열을 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_byratio.drop(['likes/view', 'comment/view', 'likes/comment', 'day_view_count'], axis=1, inplace=True)''')
    st.write('일일 좋아요, 댓글, 조회수의 비율이 NaN, 0, inf, -inf인 데이터를 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_byratio = df_kr_byratio.dropna()
df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_likes/comment'] != 0]
df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_likes/comment'] != np.inf]
df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_likes/view'] != 0]
df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_likes/view'] != np.inf]
df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_comment/view'] != 0]
df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_comment/view'] != np.inf]''')
    st.write('구간을 100개로 히스토그램을 생성한다.')
    with st.expander('see code'):
        st.code('''fig, axs = plt.subplots(1, 3, figsize=(14, 4))
data1 = df_kr_byratio['mean_likes/view']
data2 = df_kr_byratio['mean_comment/view']
data3 = df_kr_byratio['mean_likes/comment']
axs[0].hist(data1, bins=100, edgecolor='black')
axs[0].set_title('mean_likes/view')

axs[1].hist(data2, bins=100, edgecolor='black')
axs[1].set_title('mean_comment/view')

axs[2].hist(data3, bins=100, edgecolor='black')
axs[2].set_title('mean_comment/view')

plt.tight_layout()
plt.show()''')
    df_kr_byratio = df_kr.copy(deep=True)
    df_kr_byratio['day_likes'] = (df_kr['likes'] / df_kr['taken_time'])
    df_kr_byratio['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time'])
    df_kr_byratio['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time'])
    df_kr_byratio['likes/view'] = (df_kr_byratio['day_likes'] / df_kr_byratio['day_view_count'] * 100)
    df_kr_byratio['comment/view'] = (df_kr_byratio['day_comment_count'] / df_kr_byratio['day_view_count'] * 100)
    df_kr_byratio['likes/comment'] = (df_kr_byratio['day_likes'] / df_kr_byratio['day_comment_count'] * 100)
    df_kr_byratio = df_kr_byratio.groupby(['title', 'channelTitle', 'publishedAt']).agg({'likes/view': list, 'comment/view': list, 'likes/comment': list, 'day_view_count': list}).reset_index()
    df_kr_byratio = df_kr_byratio.sort_values(by='publishedAt')
    df_kr_byratio.reset_index(drop=True, inplace=True)
    df_kr_byratio['mean_likes/view'] = df_kr_byratio['likes/view'].apply(lambda x: sum(x) / len(x))
    df_kr_byratio['mean_comment/view'] = df_kr_byratio['comment/view'].apply(lambda x: sum(x) / len(x))
    df_kr_byratio['mean_likes/comment'] = df_kr_byratio['likes/comment'].apply(lambda x: sum(x) / len(x))
    df_kr_byratio['mean_day_view_count'] = df_kr_byratio['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_byratio.drop(['likes/view', 'comment/view', 'likes/comment', 'day_view_count'], axis=1, inplace=True)
    df_kr_byratio = df_kr_byratio.dropna()
    df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_likes/comment'] != 0]
    df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_likes/comment'] != np.inf]
    df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_likes/view'] != 0]
    df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_likes/view'] != np.inf]
    df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_comment/view'] != 0]
    df_kr_byratio = df_kr_byratio[df_kr_byratio['mean_comment/view'] != np.inf]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    data1 = df_kr_byratio['mean_likes/view']
    data2 = df_kr_byratio['mean_comment/view']
    data3 = df_kr_byratio['mean_likes/comment']
    axs[0].hist(data1, bins=100, edgecolor='black')
    axs[0].set_title('mean_likes/view')

    axs[1].hist(data2, bins=100, edgecolor='black')
    axs[1].set_title('mean_comment/view')

    axs[2].hist(data3, bins=100, edgecolor='black')
    axs[2].set_title('mean_comment/view')

    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('비율의 평균값을 출력한다.')
    print('mean_likes/view', df_kr_byratio['mean_likes/view'].mean())
    print('mean_comment/view', df_kr_byratio['mean_comment/view'].mean())
    print('mean_likes/comment', df_kr_byratio['mean_likes/comment'].mean())
    st.write('''좋아요와 조회수의 평균 비율은 3%\n
댓글과 조회수의 평균 비율은 0.2%\n''')

def by_desc():
    st.write('설명란은 대부분의 채널이 영상과 관련없는 이메일, sns 정보등을 함께 적는 경우가 존재한다.')
    st.write('설명란의 내용은 인기동영상과 관계없다 가정하고 존재 여부만을 분석한다.')
    st.write('df를 복사한다.')
    with st.expander('see code'):
        st.code('''df_kr_bydesc = df_kr.copy(deep=True)''')
    st.write('영상 제목과 설명란 여부만 남기고 모든 열을 삭제합니다.')
    with st.expander('see code'):
        st.code('''df_kr_bydesc = df_kr_bydesc.groupby(['title', 'description']).filter(lambda x: len(x) > 1)
df_kr_bydesc = df_kr_bydesc.drop(columns=[col for col in df_kr_bydesc.columns if col not in ['title', 'description']])
df_kr_bydesc.reset_index(drop=True, inplace=True)''')
    st.write('설명란 여부에 대한 원형 그래프를 그린다.')
    with st.expander('see code'):
        st.code('''counts = df_kr_bydesc['description'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
plt.title('description')
plt.axis('equal')
plt.show()''')
    
    df_kr_bydesc = df_kr.copy(deep=True)
    df_kr_bydesc = df_kr_bydesc.groupby(['title', 'description']).filter(lambda x: len(x) > 1)
    df_kr_bydesc = df_kr_bydesc.drop(columns=[col for col in df_kr_bydesc.columns if col not in ['title', 'description']])
    df_kr_bydesc.reset_index(drop=True, inplace=True)
    counts = df_kr_bydesc['description'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=70, colors=['lightblue', 'lightgreen'])
    plt.title('description')
    plt.axis('equal')
    plt.show()
    st.pyplot()
    st.write('1.8% 의 영상만이 설명란이 존재하지 않았다.')

def by_channel():
    st.write('df를 복사한다.')
    with st.expander('see code'):
        st.code('''df_kr_bychannel = df_kr.copy(deep=True)''')
    st.write('영상 제목과 채널의 이름을 그룹으로 모든 시간을 리스트로 묵는다.')
    st.write('업로드 날짜를 기준으로 오름차순 정렬한다.')
    with st.expander('see code'):
        st.code('''df_kr_bychannel = df_kr_bychannel.groupby(['title', 'channelTitle', 'publishedAt'])['taken_time'].apply(list).reset_index()
df_kr_bychannel = df_kr_bychannel.sort_values(by='publishedAt')
df_kr_bychannel.reset_index(drop=True, inplace=True)''')
    st.write('채널의 이름을 그룹으로 모든 영상제목을 리스트로 묵는다.')
    with st.expander('see code'):
        st.code('''df_kr_bychannel = df_kr_bychannel.groupby(['channelTitle'])['title'].apply(list).reset_index()
df_kr_bychannel.reset_index(drop=True, inplace=True)''')
    st.write('체널에 따른 영상개수에 대한 열을 추가한다.')
    st.write('영상개수를 기준으로 내림차순 정렬다.')
    with st.expander('see code'):
        st.code('''df_kr_bychannel['number_of_trending_videos'] = df_kr_bychannel['title'].apply(len)
df_kr_bychannel = df_kr_bychannel.sort_values(by='number_of_trending_videos', ascending=False)
df_kr_bychannel.reset_index(drop=True, inplace=True)''')
    st.write('필요없는 영상제목에 대한 열은 삭제한다.')
    with st.expander('see code'):
        st.code('''df_kr_bychannel.drop(['title'], axis=1, inplace=True)''')
    st.code('df_kr_bychannel.info()')
    df_kr_bychannel = df_kr.copy(deep=True)
    df_kr_bychannel = df_kr_bychannel.groupby(['title', 'channelTitle', 'publishedAt'])['taken_time'].apply(list).reset_index()
    df_kr_bychannel = df_kr_bychannel.sort_values(by='publishedAt')
    df_kr_bychannel.reset_index(drop=True, inplace=True)
    df_kr_bychannel = df_kr_bychannel.groupby(['channelTitle'])['title'].apply(list).reset_index()
    df_kr_bychannel.reset_index(drop=True, inplace=True)
    df_kr_bychannel['number_of_trending_videos'] = df_kr_bychannel['title'].apply(len)
    df_kr_bychannel = df_kr_bychannel.sort_values(by='number_of_trending_videos', ascending=False)
    df_kr_bychannel.reset_index(drop=True, inplace=True)
    buffer = io.StringIO()
    df_kr_bychannel.info(buf=buffer)
    st.text(buffer.getvalue())
    buffer.truncate(0)
    buffer = io.StringIO()
    st.code('df_kr_bychannel.head()')
    st.text(df_kr_bychannel.head())
    st.write('영상이 한자릿수 개, 두자릿수 개, 세자릿수 개에 따라 나눈다.')
    with st.expander('see code'):
        st.code('''df_kr_bychannel1 = df_kr_bychannel.copy(deep=True)
df_kr_bychannel1 = df_kr_bychannel1[df_kr_bychannel1['number_of_trending_videos'] < 10]

df_kr_bychannel2 = df_kr_bychannel.copy(deep=True)
df_kr_bychannel2 = df_kr_bychannel2[df_kr_bychannel2['number_of_trending_videos'] < 100]
df_kr_bychannel2 = df_kr_bychannel2[df_kr_bychannel2['number_of_trending_videos'] >= 10]

df_kr_bychannel3 = df_kr_bychannel.copy(deep=True)
df_kr_bychannel3 = df_kr_bychannel3[df_kr_bychannel3['number_of_trending_videos'] >= 100]''')
    df_kr_bychannel1 = df_kr_bychannel.copy(deep=True)
    df_kr_bychannel1 = df_kr_bychannel1[df_kr_bychannel1['number_of_trending_videos'] < 10]

    df_kr_bychannel2 = df_kr_bychannel.copy(deep=True)
    df_kr_bychannel2 = df_kr_bychannel2[df_kr_bychannel2['number_of_trending_videos'] < 100]
    df_kr_bychannel2 = df_kr_bychannel2[df_kr_bychannel2['number_of_trending_videos'] >= 10]

    df_kr_bychannel3 = df_kr_bychannel.copy(deep=True)
    df_kr_bychannel3 = df_kr_bychannel3[df_kr_bychannel3['number_of_trending_videos'] >= 100]
    st.code('df_kr_bychannel1.info()')
    buffer.truncate(0)
    buffer = io.StringIO()
    df_kr_bychannel1.info(buf=buffer)
    st.text(buffer.getvalue())
    st.code('df_kr_bychannel2.info()')
    buffer.truncate(0)
    buffer = io.StringIO()
    df_kr_bychannel2.info(buf=buffer)
    st.text(buffer.getvalue())
    st.code('df_kr_bychannel3.info()')
    buffer.truncate(0)
    buffer = io.StringIO()
    df_kr_bychannel3.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write('''인기동영상을 보유한 채널의 수는 3,897개이다.\n
1개 이상 9개 이하의 채널의 수는 3,149개이다.\n
10개 이상 99개 이하의 채널의 수는 730개이다.\n
100개 이상의 채널의 수는 18개이다.''')
    st.write('각 동영상 개수에 따라 채널의 수를 출력한다.')
    with st.expander('see code'):
        st.code('''for i in range(0, 200):
  a = (df_kr_bychannel['number_of_trending_videos'] == i).sum()
  if a:
    print(i, '개를 보유한 채널의 수', a)''')
    with st.expander('show detail'):
        for i in range(0, 200):
            a = (df_kr_bychannel['number_of_trending_videos'] == i).sum()
            if a:
                st.write(i, '개를 보유한 채널의 수', a)

def by_cat():
    st.write('df를 복사한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycat = df_kr.copy(deep=True)''')
    st.write('영상 제목과 카테고리id만 남기고 모든 열을 삭제한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycat = df_kr_bycat.groupby(['title', 'categoryId']).filter(lambda x: len(x) > 1)
df_kr_bycat = df_kr_bycat.drop(columns=[col for col in df_kr_bycat.columns if col not in ['title', 'categoryId']])
df_kr_bycat.reset_index(drop=True, inplace=True)''')
    st.write('카테고리 종류의 빈도에 따라 내림차순 정렬한다.')
    st.write('영상 제목을 삭제하고 카테고리 id를 index로 설정한다.')
    with st.expander('see code'):
        st.code('''value_counts = df_kr_bycat['categoryId'].value_counts()
df_kr_bycat['categoryId_count'] = df_kr_bycat['categoryId'].map(value_counts)
df_kr_bycat_sorted = df_kr_bycat.sort_values(by=['categoryId_count'], ascending=[False])
df_kr_bycat_sorted = df_kr_bycat_sorted.drop(columns=['title'])
df_kr_bycat_sorted = df_kr_bycat_sorted.drop_duplicates(subset=['categoryId', 'categoryId_count'])
df_kr_bycat_sorted = df_kr_bycat_sorted.set_index('categoryId')''')
    st.write('카테고리에 대한 원형 그래프를 그린다.')
    st.write('상위 9개만 표시하고 나머지는 예외로 나타낸다.')
    with st.expander('see code'):
        st.code('''top_n = 9
top_data = df_kr_bycat_sorted['categoryId_count'][:top_n]
other_data = df_kr_bycat_sorted['categoryId_count'][top_n:].sum()
top_labels = df_kr_bycat_sorted.index[:top_n]
other_label = 'Other'

final_data = np.append(top_data, other_data)
final_labels = np.append(top_labels, other_label)

plt.figure(figsize=(6, 6))
plt.pie(final_data, labels=final_labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
plt.title('Category ID')
plt.axis('equal') 
plt.show()''')
    df_kr_bycat = df_kr.copy(deep=True)
    df_kr_bycat = df_kr_bycat.groupby(['title', 'categoryId']).filter(lambda x: len(x) > 1)
    df_kr_bycat = df_kr_bycat.drop(columns=[col for col in df_kr_bycat.columns if col not in ['title', 'categoryId']])
    df_kr_bycat.reset_index(drop=True, inplace=True)
    value_counts = df_kr_bycat['categoryId'].value_counts()
    df_kr_bycat['categoryId_count'] = df_kr_bycat['categoryId'].map(value_counts)
    df_kr_bycat_sorted = df_kr_bycat.sort_values(by=['categoryId_count'], ascending=[False])
    df_kr_bycat_sorted = df_kr_bycat_sorted.drop(columns=['title'])
    df_kr_bycat_sorted = df_kr_bycat_sorted.drop_duplicates(subset=['categoryId', 'categoryId_count'])
    df_kr_bycat_sorted = df_kr_bycat_sorted.set_index('categoryId')
    top_n = 9
    top_data = df_kr_bycat_sorted['categoryId_count'][:top_n]
    other_data = df_kr_bycat_sorted['categoryId_count'][top_n:].sum()
    top_labels = df_kr_bycat_sorted.index[:top_n]
    other_label = 'Other'

    final_data = np.append(top_data, other_data)
    final_labels = np.append(top_labels, other_label)

    plt.figure(figsize=(6, 6))
    plt.pie(final_data, labels=final_labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
    plt.title('Category ID')
    plt.axis('equal') 
    plt.show()
    st.pyplot()
    st.write('''카테고리의 경우 업로더가 직접 설정하는 경우도 있지만 유튜브의 인공지능이 자체적으로 설정하는 경우도 존재한다.\n
카테고리의 id 24와 22 두개가 절반 이상을 차지하고 있다.\n
24는 Entertainment 22는 People & Blogs를 의미하는 부분에서 대부분의 즐거움을 주는 영상은 Entertainment, 사람이 나오는 대부분의 영상은 People & Blogs으로 설정될 수 있다고 예상할 수 있다.\n
나머지 상위 id는 10, 17, 23, 26, 20, 25, 1순으로 차지하고 있다.\n
''')

def learning():
    st.write('''좋아요와 조회수의 평균 비율은 3%\n
댓글과 조회수의 평균 비율은 0.2%\n
이를 바탕으로 조회수에 1의 가중치를 주고 좋아요에는 약 100/3인 33의 가중치, 댓글에는 약 100/0.2인 500의 가중치를 준다.\n
전체 가중치의 합이 1이 되도록 534로 나누어 준다.\n''')
    st.write('df를 복사한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycountx = df_kr_bycount.copy(deep=True)''')
    st.write('일일 좋아요, 댓글, 조회수가 0, inf, -inf인 데이터를 제거해준다.')
    with st.expander('see code'):
        st.code('''df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != 0]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != 0]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != 0]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != -np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != -np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != -np.inf]''')
    st.write('가중치 적용한 열과 일부 열만 남기고 제거한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycountx['weighted_value'] = df_kr_bycountx['mean_day_likes'] * 33 + df_kr_bycountx['mean_day_comment_count'] * 500 + df_kr_bycountx['mean_day_view_count']
df_kr_bycountx['weighted_value'] = df_kr_bycountx['weighted_value'] / 531   
df_kr_bycountx.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time', 'mean_day_likes', 'mean_day_comment_count', 'mean_day_view_count'], axis=1, inplace=True)''')
    st.write('구간을 50개로 히스토그램을 그린다.')
    with st.expander('see code'):
        st.code('''data = df_kr_bycountx['weighted_value']
counts, bins, patches = plt.hist(data, bins=100, edgecolor='black')
plt.tight_layout()
plt.show()''')
    df_kr_bycount = df_kr.copy(deep=True)
    df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
    df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
    df_kr_bycount.reset_index(drop=True, inplace=True)
    df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycountx = df_kr_bycount.copy(deep=True)
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != 0]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != 0]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != 0]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != -np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != -np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != -np.inf]
    df_kr_bycountx['weighted_value'] = df_kr_bycountx['mean_day_likes'] * 33 + df_kr_bycountx['mean_day_comment_count'] * 500 + df_kr_bycountx['mean_day_view_count']
    df_kr_bycountx['weighted_value'] = df_kr_bycountx['weighted_value'] / 531
    df_kr_bycountx.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time', 'mean_day_likes', 'mean_day_comment_count', 'mean_day_view_count'], axis=1, inplace=True)
    data = df_kr_bycountx['weighted_value']
    counts, bins, patches = plt.hist(data, bins=100, edgecolor='black')
    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('평균값과 표준편차를 출력한다.')
    with st.expander('see code'):
        st.code('''print('평균', df_kr_bycountx['weighted_value'].mean())
print('표준편차', df_kr_bycountx['weighted_value'].std())''')
    st.write('평균', df_kr_bycountx['weighted_value'].mean())
    st.write('표준편차', df_kr_bycountx['weighted_value'].std())
    st.write('변동계수 (표준편차 / 평균) :', df_kr_bycountx['weighted_value'].std() / df_kr_bycountx['weighted_value'].mean())
    st.write('히스토그램의 가장 많이 분포하는 범위로 한정한다.')
    with st.expander('see code'):
        st.code('''max_bin_index = np.argmax(counts)
max_bin_range = (bins[max_bin_index], bins[max_bin_index + 1])
df_kr_bycountx1 = df_kr_bycountx.copy(deep=True)
df_kr_bycountx1 = df_kr_bycountx1[df_kr_bycountx1['weighted_value'] >= bins[max_bin_index]]
df_kr_bycountx1 = df_kr_bycountx1[df_kr_bycountx1['weighted_value'] < bins[max_bin_index + 1]]
print('평균', df_kr_bycountx1['weighted_value'].mean())
print('표준편차', df_kr_bycountx1['weighted_value'].std())''')
    
    max_bin_index = np.argmax(counts)
    max_bin_range = (bins[max_bin_index], bins[max_bin_index + 1])
    df_kr_bycountx1 = df_kr_bycountx.copy(deep=True)
    df_kr_bycountx1 = df_kr_bycountx1[df_kr_bycountx1['weighted_value'] >= bins[max_bin_index]]
    df_kr_bycountx1 = df_kr_bycountx1[df_kr_bycountx1['weighted_value'] < bins[max_bin_index + 1]]
    st.write('평균', df_kr_bycountx1['weighted_value'].mean())
    st.write('표준편차', df_kr_bycountx1['weighted_value'].std())
    st.write('변동계수 (표준편차 / 평균) :', df_kr_bycountx1['weighted_value'].std() / df_kr_bycountx1['weighted_value'].mean())
    st.write('''비율에 따라 가중치를 주어 히스토그램을 생성했다.\n
히스토그램의 평균\n
모든 가중치를 적용한 값의 평균은 2627이다.\n
모든 가중치를 적용한 값의 표준편차는 12,188이다.\n
히스토그램에 가장 많이 분포하는 범위로 한정했을때 가중치를 적용한 값의 평균은 1,098\n
히스토그램에 가장 많이 분포하는 범위로 한정했을때 가중치를 적용한 값의 표준편차는 1,123이다.\n''')
    st.write('좋아요, 댓글, 조회수의 일일 수에 대한 전처리 과정을 거친다.')
    with st.expander('see code'):
        st.code('''df_kr_bycount = df_kr.copy(deep=True)
df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)
df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
df_kr_bycount.reset_index(drop=True, inplace=True)
df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)
df_kr_bycount1 = df_kr_bycount.copy(deep=True)
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != 0]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != 0]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != 0]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != -np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != -np.inf]
df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != -np.inf]
df_kr_bycount1 = df_kr_bycount1.drop(columns=[col for col in df_kr_bycount1.columns if col not in ['mean_day_likes', 'mean_day_comment_count', 'mean_day_view_count']])
df = df_kr_bycount1.rename(columns={'mean_day_likes': 'l', 'mean_day_comment_count': 'c', 'mean_day_view_count': 'v'})
''')
    st.write('가중치를 최적화 알고리즘을 통해 구해본다.')
    with st.expander('see code'):
        st.code('''def objective(w):
    weighted_sum = df['l']*w[0] + df['c']*w[1] + df['v']*w[2]
    return np.std(weighted_sum)
def constraint(w):
    return w - 1.e-10
initial_guess = [0.01, 0.01, 0.01]
constraints = ({'type': 'ineq', 'fun': constraint})
result = minimize(objective, initial_guess, method='COBYLA', constraints=constraints)
print("최적 가중치:", result.x)
a = result.x[0]
b = result.x[1]
c = result.x[2]''')
    df_kr_bycount = df_kr.copy(deep=True)
    df_kr_bycount['day_likes'] = (df_kr['likes'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_comment_count'] = (df_kr['comment_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount['day_view_count'] = (df_kr['view_count'] / df_kr['taken_time']).round(2)
    df_kr_bycount = df_kr_bycount.groupby(['title', 'channelTitle', 'publishedAt']).agg({'day_likes': list, 'day_comment_count': list, 'day_view_count': list, 'taken_time': list}).reset_index()
    df_kr_bycount = df_kr_bycount.sort_values(by='publishedAt')
    df_kr_bycount.reset_index(drop=True, inplace=True)
    df_kr_bycount['mean_day_likes'] = df_kr_bycount['day_likes'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_comment_count'] = df_kr_bycount['day_comment_count'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount['mean_day_view_count'] = df_kr_bycount['day_view_count'].apply(lambda x: sum(x) / len(x)).round(2)
    df_kr_bycount1 = df_kr_bycount.copy(deep=True)
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != 0]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != 0]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != 0]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_likes'] != -np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_comment_count'] != -np.inf]
    df_kr_bycount1 = df_kr_bycount1[df_kr_bycount1['mean_day_view_count'] != -np.inf]
    df_kr_bycount1 = df_kr_bycount1.drop(columns=[col for col in df_kr_bycount1.columns if col not in ['mean_day_likes', 'mean_day_comment_count', 'mean_day_view_count']])
    df = df_kr_bycount1.rename(columns={'mean_day_likes': 'l', 'mean_day_comment_count': 'c', 'mean_day_view_count': 'v'})

    df = df_kr_bycount1.rename(columns={'mean_day_likes': 'l', 'mean_day_comment_count': 'c', 'mean_day_view_count': 'v'})
    def objective(w):
        weighted_sum = df['l']*w[0] + df['c']*w[1] + df['v']*w[2]
        return np.std(weighted_sum)
    def constraint(w):
        return w - 1.e-10
    initial_guess = [0.01, 0.01, 0.01]
    constraints = ({'type': 'ineq', 'fun': constraint})
    result = minimize(objective, initial_guess, method='COBYLA', constraints=constraints)
    print("최적 가중치:", result.x)
    a = result.x[0]
    b = result.x[1]
    c = result.x[2]
    st.write('히스토그램을 그리고 평균과 표준편차를 출력한다.')
    with st.expander('see code'):
        st.code('''df_kr_bycountx = df_kr_bycount.copy(deep=True)
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != 0]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != 0]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != 0]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != -np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != -np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != -np.inf]
df_kr_bycountx['weighted_value'] = df_kr_bycountx['mean_day_likes'] * a + df_kr_bycountx['mean_day_comment_count'] * b + df_kr_bycountx['mean_day_view_count'] * c
df_kr_bycountx['weighted_value'] = df_kr_bycountx['weighted_value'] / (a+b+c)
df_kr_bycountx.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time', 'mean_day_likes', 'mean_day_comment_count', 'mean_day_view_count'], axis=1, inplace=True)
data = df_kr_bycountx['weighted_value']
counts, bins, patches = plt.hist(data, bins=100, edgecolor='black')
plt.tight_layout()
plt.show()
print(df_kr_bycountx['weighted_value'].mean())
print(df_kr_bycountx['weighted_value'].std())
max_bin_index = np.argmax(counts)
max_bin_range = (bins[max_bin_index], bins[max_bin_index + 1])
df_kr_bycountx1 = df_kr_bycountx.copy(deep=True)
df_kr_bycountx1 = df_kr_bycountx1[df_kr_bycountx1['weighted_value'] >= bins[max_bin_index]]
df_kr_bycountx1 = df_kr_bycountx1[df_kr_bycountx1['weighted_value'] < bins[max_bin_index + 1]]
print(df_kr_bycountx1['weighted_value'].mean())
print(df_kr_bycountx1['weighted_value'].std())''')
    df_kr_bycountx = df_kr_bycount.copy(deep=True)
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != 0]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != 0]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != 0]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != -np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != -np.inf]
    df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != -np.inf]
    df_kr_bycountx['weighted_value'] = df_kr_bycountx['mean_day_likes'] * a + df_kr_bycountx['mean_day_comment_count'] * b + df_kr_bycountx['mean_day_view_count'] * c
    df_kr_bycountx['weighted_value'] = df_kr_bycountx['weighted_value'] / (a+b+c)
    df_kr_bycountx.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time', 'mean_day_likes', 'mean_day_comment_count', 'mean_day_view_count'], axis=1, inplace=True)
    data = df_kr_bycountx['weighted_value']
    counts, bins, patches = plt.hist(data, bins=100, edgecolor='black')
    plt.tight_layout()
    plt.show()
    st.pyplot()
    st.write('평균', df_kr_bycountx['weighted_value'].mean())
    st.write('표준 편차', df_kr_bycountx['weighted_value'].std())
    st.write('변동계수 (표준편차 / 평균) :', df_kr_bycountx['weighted_value'].std() / df_kr_bycountx['weighted_value'].mean())

def learning2():
    st.subheader('선형회귀를 통해 최적값을 구하기에 시간이 소요될 수 있습니다.')
    st.markdown('* 초기 가중치를 조절해서 입력하면 최적값을 출력해줍니다.')
    st.write('초기 가중치 설정')
    with st.expander('see code'):
        st.code('''df = df_kr_bycount1.rename(columns={'mean_day_likes': 'l', 'mean_day_comment_count': 'c', 'mean_day_view_count': 'v'})
def objective(w):
    weighted_sum = df['l']*w[0] + df['c']*w[1] + df['v']*w[2]
    return np.std(weighted_sum)
def constraint(w):
    return w
w1 = input()
w2 = input()
w3 = input()
initial_guess = [w1, w2, w3]
constraints = ({'type': 'ineq', 'fun': constraint})
result = minimize(objective, initial_guess, method='COBYLA', constraints=constraints)
print("최적 가중치:", result.x)
a = result.x[0]
b = result.x[1]
c = result.x[2]
df_kr_bycountx = df_kr_bycount.copy(deep=True)
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != 0]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != 0]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != 0]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != -np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != -np.inf]
df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != -np.inf]
df_kr_bycountx['weighted_value'] = df_kr_bycountx['mean_day_likes'] * a + df_kr_bycountx['mean_day_comment_count'] * b + df_kr_bycountx['mean_day_view_count'] * c
df_kr_bycountx['weighted_value'] = df_kr_bycountx['weighted_value'] / (a+b+c)
df_kr_bycountx.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time', 'mean_day_likes', 'mean_day_comment_count', 'mean_day_view_count'], axis=1, inplace=True)
print('평균', df_kr_bycountx['weighted_value'].mean())
print('표준편차', df_kr_bycountx['weighted_value'].std())''')
    df = df_kr_bycount1.rename(columns={'mean_day_likes': 'l', 'mean_day_comment_count': 'c', 'mean_day_view_count': 'v'})
    def objective(w):
        weighted_sum = df['l']*w[0] + df['c']*w[1] + df['v']*w[2]
        return np.std(weighted_sum)
    def constraint(w):
        return w
    w1 = st.number_input('w1', min_value=1.e-30, max_value=1.0)
    w2 = st.number_input('w2', min_value=1.e-30, max_value=1.0)
    w3 = st.number_input('w3', min_value=1.e-30, max_value=1.0)
    if st.button("apply"):
        initial_guess = [w1, w2, w3]
        constraints = ({'type': 'ineq', 'fun': constraint})
        result = minimize(objective, initial_guess, method='COBYLA', constraints=constraints)
        st.write('초기 가중치','w1:', w1, 'w2:', w2, 'w3:', w3)
        a = result.x[0]
        b = result.x[1]
        c = result.x[2]
        df_kr_bycountx = df_kr_bycount.copy(deep=True)
        df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != 0]
        df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != 0]
        df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != 0]
        df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != np.inf]
        df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != np.inf]
        df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != np.inf]
        df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_likes'] != -np.inf]
        df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_comment_count'] != -np.inf]
        df_kr_bycountx = df_kr_bycountx[df_kr_bycountx['mean_day_view_count'] != -np.inf]
        df_kr_bycountx['weighted_value'] = df_kr_bycountx['mean_day_likes'] * a + df_kr_bycountx['mean_day_comment_count'] * b + df_kr_bycountx['mean_day_view_count'] * c
        df_kr_bycountx['weighted_value'] = df_kr_bycountx['weighted_value'] / (a+b+c)
        df_kr_bycountx.drop(['day_likes', 'day_comment_count', 'day_view_count', 'taken_time', 'mean_day_likes', 'mean_day_comment_count', 'mean_day_view_count'], axis=1, inplace=True)
        if c == 0:
            st.write('최적 가중치가 0이 되었습니다.')
            st.write('다른 초기값으로 시도해주세요')
        else:
            st.write('w1:', a, 'w2:', b, 'w3:', c)
            st.write('w1 : w2 : w3 =', a/c, ':', b/c, ':', 1)
            st.write('평균', df_kr_bycountx['weighted_value'].mean())
            st.write('표준편차', df_kr_bycountx['weighted_value'].std())
            st.write('변동계수 (표준편차 / 평균) :', df_kr_bycountx['weighted_value'].std() / df_kr_bycountx['weighted_value'].mean())
    st.write('가중치를 33:500:1로 설정했을 때 퍙군은 2,627 표준편차는 12,188이 나왔다.')
    st.write('변동계수 (표준편차 / 평균) 값은 4.64가 나왔다.')
    st.write('변동 계수는 평균 값과의 비율을 통해 데이터의 변동성을 평가한다.')
    st.write('변동 계수가 작다는 것은 데이터가 평균 주변에 많이 모여 있음을 나타낸다')
    st.write('변동 계수가 크다는 것은 데이터가 평균에서 크게 분산되어 있음을 나타낸다')
