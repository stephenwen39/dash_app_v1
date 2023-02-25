import datetime
import pandas as pd
import numpy as np
from dash import Dash, dcc, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from chat_downloader import ChatDownloader

'''第三方程式開始'''
import json
import time
from typing import Generator

import requests
from typing_extensions import Literal

type_property_map = {
    "videos": "videoRenderer",
    "streams": "videoRenderer",
    "shorts": "reelItemRenderer"
}

def get_channel(
    channel_id: str = None,
    channel_url: str = None,
    limit: int = None,
    sleep: int = 1,
    sort_by: Literal["newest", "oldest", "popular"] = "newest",
    content_type: Literal["videos", "shorts", "streams"] = "videos",
) -> Generator[dict, None, None]:

    sort_by_map = {"newest": "dd", "oldest": "da", "popular": "p"}    
    url = "{url}/{content_type}?view=0&sort={sort_by}&flow=grid".format(
        url=channel_url or f"https://www.youtube.com/channel/{channel_id}",
        content_type=content_type,
        sort_by=sort_by_map[sort_by],
    )
    api_endpoint = "https://www.youtube.com/youtubei/v1/browse"
    videos = get_videos(url, api_endpoint, type_property_map[content_type], limit, sleep)
    for video in videos:
        if 'lengthText' in video: # 這行是我改寫的，不抓尚未直播的影片
            yield video


def get_playlist(
    playlist_id: str, limit: int = None, sleep: int = 1
) -> Generator[dict, None, None]:

    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    api_endpoint = "https://www.youtube.com/youtubei/v1/browse"
    videos = get_videos(url, api_endpoint, "playlistVideoRenderer", limit, sleep)
    for video in videos:
        yield video


def get_search(
    query: str,
    limit: int = None,
    sleep: int = 1,
    sort_by: Literal["relevance", "upload_date", "view_count", "rating"] = "relevance",
    results_type: Literal["video", "channel", "playlist", "movie"] = "video",
) -> Generator[dict, None, None]:

    sort_by_map = {
        "relevance": "A",
        "upload_date": "I",
        "view_count": "M",
        "rating": "E",
    }

    results_type_map = {
        "video": ["B", "videoRenderer"],
        "channel": ["C", "channelRenderer"],
        "playlist": ["D", "playlistRenderer"],
        "movie": ["E", "videoRenderer"],
    }

    param_string = f"CA{sort_by_map[sort_by]}SAhA{results_type_map[results_type][0]}"
    url = f"https://www.youtube.com/results?search_query={query}&sp={param_string}"
    api_endpoint = "https://www.youtube.com/youtubei/v1/search"
    videos = get_videos(
        url, api_endpoint, results_type_map[results_type][1], limit, sleep
    )
    for video in videos:
        yield video


def get_videos(
    url: str, api_endpoint: str, selector: str, limit: int, sleep: int
) -> Generator[dict, None, None]:
    session = requests.Session()
    session.headers[
        "User-Agent"
    ] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
    is_first = True
    quit = False
    count = 0
    while True:
        if is_first:
            html = get_initial_data(session, url)
            client = json.loads(
                get_json_from_html(html, "INNERTUBE_CONTEXT", 2, '"}},') + '"}}'
            )["client"]
            api_key = get_json_from_html(html, "innertubeApiKey", 3)
            session.headers["X-YouTube-Client-Name"] = "1"
            session.headers["X-YouTube-Client-Version"] = client["clientVersion"]
            data = json.loads(
                get_json_from_html(html, "var ytInitialData = ", 0, "};") + "}"
            )
            next_data = get_next_data(data)
            is_first = False
        else:
            data = get_ajax_data(session, api_endpoint, api_key, next_data, client)
            next_data = get_next_data(data)
        for result in get_videos_items(data, selector):
            try:
                count += 1
                yield result
                if count == limit:
                    quit = True
                    break
            except GeneratorExit:
                quit = True
                break

        if not next_data or quit:
            break

        time.sleep(sleep)

    session.close()


def get_initial_data(session: requests.Session, url: str) -> str:
    session.cookies.set("CONSENT", "YES+cb", domain=".youtube.com")
    response = session.get(url)

    html = response.text
    return html


def get_ajax_data(
    session: requests.Session,
    api_endpoint: str,
    api_key: str,
    next_data: dict,
    client: dict,
) -> dict:
    data = {
        "context": {"clickTracking": next_data["click_params"], "client": client},
        "continuation": next_data["token"],
    }
    response = session.post(api_endpoint, params={"key": api_key}, json=data)
    return response.json()


def get_json_from_html(html: str, key: str, num_chars: int = 2, stop: str = '"') -> str:
    pos_begin = html.find(key) + len(key) + num_chars
    pos_end = html.find(stop, pos_begin)
    return html[pos_begin:pos_end]


def get_next_data(data: dict) -> dict:
    raw_next_data = next(search_dict(data, "continuationEndpoint"), None)
    if not raw_next_data:
        return None
    next_data = {
        "token": raw_next_data["continuationCommand"]["token"],
        "click_params": {"clickTrackingParams": raw_next_data["clickTrackingParams"]},
    }

    return next_data


def search_dict(partial: dict, search_key: str) -> Generator[dict, None, None]:
    stack = [partial]
    while stack:
        current_item = stack.pop(0)
        if isinstance(current_item, dict):
            for key, value in current_item.items():
                if key == search_key:
                    yield value
                else:
                    stack.append(value)
        elif isinstance(current_item, list):
            for value in current_item:
                stack.append(value)


def get_videos_items(data: dict, selector: str) -> Generator[dict, None, None]:
    return search_dict(data, selector)
    
'''第三方程式結束'''
'''資料用開始'''
class message_analysor(object):
    def __init__(self, url):
        self.message_df = pd.DataFrame(columns=['raw_data'])
        
        self.url = url
        self.chat = ChatDownloader().get_chat(self.url)
        self.test = []
        for message in self.chat:                        # iterate over messages
            self.test.append(self.chat.format(message))
        self.message_df['raw_data'] = self.test
    
    def main(self):
        self.Time_processer()
        self.Identity_name_chat_processer()
        return self.message_df

    def Time_processer(self):
        def time_string_add(x):
            if x[-1] == ' ':
                x = x.split(' ', 1)[0]
            if len(x) == 4: # 1:21
                return '00:0'+x
            elif len(x) == 5: # 17:31
                return '00:'+x
            elif len(x) == 7: # 1:21:21
                return '0'+x
            else:
                return x
        # 解決時間正負號，階段4
        self.message_df['is_time_positive'] = \
        self.message_df.raw_data.apply(lambda x: False if x[0] == '-' else True)
        
        # 丟棄時間負號，階段5
        self.message_df['raw_data'] = \
        self.message_df.raw_data.apply(lambda x: x if x[0] != '-' else x[1:])
        # 擷取小時、分鐘、秒數
        self.message_df['time'] = \
        self.message_df.raw_data.apply(lambda x: x.split('|')[0])
        # 補齊符合datetime的時間字串格式
        self.message_df['time'] = \
        self.message_df.time.apply(time_string_add)
        # 先轉換time成datetime才可以再轉換為time
        self.message_df['time'] = pd.to_datetime(self.message_df['time'])
        # 轉換datetime為time
        self.message_df['time'] = self.message_df.time.dt.time
        # 階段7
        self.message_df['raw_data'] = \
        self.message_df.raw_data.apply(lambda x: x.split('|', 1)[1])

        return self
    
    def Identity_name_chat_processer(self):
        # 如果接下來[0]是'('表示他有一定的身份，為類別1，否則就是others，為類別2
        # 類別1的前三個字會透露他的身份，'Mod','Mem','Own','New'，所以分流處理
        # 分流使用def實作並交由apply套用處理
        # 分流確認身份後，再確認身份時間
        # 如果是other就直接給予other身份就好
        def Verified_filter(x):
            if x[2:5] == 'Ver':
                return True
            else:
                return False
        def Verified_deleter(x):
            if x[2:5] == 'Ver': #Verified
                if x[11] == ',':
                    ans = x.split(',', 1)[1]
                    ans = ans.split(' ', 1)[1]
                    return ' (' + ans
                else:
                    return x.split(')', 1)[1]
            return x
        def identity_filter(x):
            # 多考量一個verified, 只要帳號有勾勾就是有，跟會員與否無關
            # if 後面的 and是為了確保不會有user id長得像「(帽子)北海市長孔文舉」
            # 因為一般來說()裡面是要識別其身份
            if x[1] == '(' and x[2:5] in ('Mem', 'Mod', 'New', 'Own'):
                if x[2:5] == 'Mem':
                    return 'Member'
                elif x[2:5] == 'Mod':
                    return 'Moderator'
                elif x[2:5] == 'New':
                    return 'New member'
                elif x[2:5] == 'Own':
                    return 'Owner'
                else:
                    return 'Unknow'
            else:
                return 'other'
        
        def identity_deleter(x):
            if x[1] == '(' and x[2:5] in ('Mem', 'Mod', 'New', 'Own'):
                if x[2:5] == 'Mem':
                    return x.split('(', 2)[2]
                elif x[2:5] == 'Mod': # 假設若多重身份Mod會在第一個
                    if x[11] == ',':
                        return x.split('(', 2)[2]
                    return x.split('(', 1)[1]
                elif x[2:5] == 'New': # 只有單括號
                    return x.split(')', 1)[1]
                elif x[2:5] == 'Own':
                    return x.split('(', 1)[1]
                else:
                    value1 = x.split(')', 1) # 為了防止姓名或留言裡面有()
                    if len(value1[0].split('(', 3)) == 4: # 表有三個括號
                        value = x.split('(', 3) # 這是防呆機制
                        return value[3]
                    elif len(value1[0].split('(', 2)) == 3: # 表有兩個括號
                        value = x.split('(', 2)
                        return value[2]
                    else:
                        value = x.split('(', 1)
                        return value[1]
            else:
                return x # 不會有括號
        def time_name_filter(x):
            try:
                num = int(x[0])
            except:
                return '0' # x.split(' ', 1)[1] # name
            value = x.split(')', 2)
            return value[0]
        def time_deleter(x):
            try:
                num = int(x[0])
            except:
                try:
                    return x.split(' ', 1)[1] # name
                except:
                    print('error time_deleter:',x)
                    return x.split(' ', 1)[0] # name
            value = x.split(')', 2)
            return value[2]
        # 建立Verified身份
        self.message_df['Verified_or_not'] = \
        self.message_df.raw_data.apply(Verified_filter)
        # 移除Verified 在raw_data中的身份
        self.message_df['raw_data'] = \
        self.message_df.raw_data.apply(Verified_deleter)
        # 建立身份欄位
        self.message_df['user_identity'] = \
        self.message_df.raw_data.apply(identity_filter)
        # 丟棄身份
        self.message_df['raw_data'] = \
        self.message_df.raw_data.apply(identity_deleter)
        # 建立訊息欄位
        self.message_df['message'] = \
        self.message_df.raw_data.apply(lambda x: x.split(':', 1)[1])
        # 丟棄訊息欄位的第一個空白
        self.message_df['message'] = \
        self.message_df.message.apply(lambda x: x.split(' ', 1)[1])
        # 丟棄訊息
        self.message_df['raw_data'] = \
        self.message_df.raw_data.apply(lambda x: x.split(':', 1)[0])
        # 建立身份時間欄位
        self.message_df['identity_time'] = \
        self.message_df.raw_data.apply(time_name_filter)
        # 丟棄身份時間
        self.message_df['raw_data'] = \
        self.message_df.raw_data.apply(time_deleter)
        # 改成user_id
        self.message_df = self.message_df.rename({'raw_data': 'user_id'}, axis=1)
        
        return self
'''
# 下面是示範code
# url = 'https://www.youtube.com/watch?v=0m_Z8FSuBDQ'
url = 'https://www.youtube.com/watch?v=uyXXovyo4lo&ab_channel=P.LEAGUE'
# url = 'https://www.youtube.com/watch?v=pQi2A8ndsYg&ab_channel=USAGIHIMECLUB.%E5%85%94%E5%A7%AC'
# url = 'https://www.youtube.com/watch?v=aHbiwZbmkuQ'
# url = 'https://www.youtube.com/watch?v=rKZoa7LNcYk&ab_channel=e04Ch.'
a = message_analysor(url)

df = a.main()
df.head(30)
'''

# 12/16的內容
class user_analysor(object):
    def __init__(self, df):
        self.df = df

    def main(self):
        # 計算發訊息數量
        df_temp_1 = self.message_counter()
        # 把使用者的訊息都集合成一個list
        df_temp_2 = self.message_lister()
        # 把使用者的時間都集合成一個list
        df_temp_3 = self.timelister()
        # 上述結果merge
        df_result_1 = df_temp_1.merge(df_temp_2, on='user_id', how='inner')
        df_result_2 = df_result_1.merge(df_temp_3, on='user_id', how='left')
        # 改message_x, message_y欄位名字
        df_result_2.rename(columns = {'message_x':'message_count', 'message_y':'message_list', 'time':'time_list'}, inplace = True)
        # merge最後的其他不需計算的欄位
        self.df = self.df.drop_duplicates(['user_id'])
        df_result_2 = df_result_2.merge(self.df[['user_id', 'user_identity', 'identity_time', 'Verified_or_not']], on='user_id', how='left')
        return df_result_2

    def message_counter(self):
        return self.df.groupby(['user_id'])[['message']].count().reset_index()
    
    def message_lister(self):
        return self.df.groupby(['user_id'])['message'].apply(list).reset_index()

    def timelister(self):
        return self.df.groupby(['user_id'])['time'].apply(list).reset_index()
'''
a = user_analysor(df)

df_user = a.main()
df_user.tail(5)
'''

class live_info(object):
    def __init__(self, df):
        self.df = df
    
    def main(self):
        # 截斷時間，01:01:50 -> 01:01:00，之後groupby會比較密集
        self.groupby_minutes()
        # 基於時間count使用者量
        df_temp_1 = self.user_counter()
        # 基於時間創造該分鐘數的留言使用者list
        df_temp_2 = self.user_lister()
        # 基於時間創造該分鐘數的留言list & message_cnt
        df_temp_3 = self.message_lister()
        df_temp_3['message_cnt'] = self.message_counter(df_temp_3)
        
        # merge by time
        df_temp_1 = df_temp_1.merge(df_temp_2, on='time', how='inner')
        df_temp_1 = df_temp_1.merge(df_temp_3, on='time', how='inner')
        # 補齊空白時間，之後畫圖比較好畫
        df_result = self.complete_blank_time(df_temp_1)
        # rename
        df_result.rename(columns = {'user_id_x':'user_cnt', 
                                    'user_id_y':'users_list',
                                    'message':'message_list'}, inplace = True)
        return df_result
    
    def groupby_minutes(self):
        def time_trunc(x):
            return str(x)[0:6]+'00'

        self.df['time'] = self.df.time.apply(time_trunc)
        self.df['time'] = pd.to_datetime(self.df['time'],format= '%H:%M:%S' ).dt.time

    def user_counter(self):
        return self.df.groupby(['time'])['user_id'].nunique().reset_index()

    def user_lister(self):
        return self.df.groupby(['time'])['user_id'].apply(list).reset_index()

    def message_lister(self):
        return self.df.groupby(['time'])['message'].apply(list).reset_index()

    def message_counter(self, df_temp_3):
        return df_temp_3.message.apply(lambda x: len(x))
    
    def complete_blank_time(self, df):
        '''填補空白時間段的方式是：
        1.確定最大時間 end_time，並且建立以分鐘為間隔的時間list
        2.跑for loop，如果該分鐘數沒有出現在df中，就在df最末端插入該分鐘數
        3.全部補齊後，根據分鐘數排序 ASC
        4.最後用reset_index跟drop index來重新設定索引，完成
        '''
        end_time = str(df.iloc[-1]['time'])[0:6]+'00'
        time_values = pd.date_range("00:00", end_time, freq="1min").time
        count = 0
        for i in df['time']:
            while time_values[count] < i:
                df.loc[-1] = [time_values[count], 0, [], [], 0] #
                count += 1
                df.index = df.index + 1
            count += 1
        df = df.sort_values(by='time')
        df = df.reset_index()
        return df.drop(['index'], axis=1)
'''
# 下面是測試code
a = live_info(df) # df from message_analysor from Pre_processor_2.0.py
df2 = a.main()
df2.tail(10)
'''

class get_streams_from_channel(object):
    def __init__(self, channel_url, limit=20):
        """
        channel_url: 頻道首頁連結
        limit: 想要抓的直播url數量，預設20
        """
        self.channel_url = channel_url
        # 下面排除尚未直播的影片url
        test_limit = 5 # test_limit先抓5部近期直播，看下方test會回傳多少直播
        test_list = [] # test_list儲存回傳直播數量，回傳如果只有3，表示有2部尚未開始的直播
        while len(test_list) == 0: 
            test = get_channel(channel_url=self.channel_url,
                            content_type='streams',
                            limit=test_limit)
            # 由於get_channel本身沒有跳過「尚未直播」的直播的功能，因此手動修改內部程式碼
            # get_channel經過我自己的修改，達到可以跳過「即將直播」的直播
            # change the code inside of get_channel, so that I can skip "coming soon" stream
            for i in test:
                test_list.append(i['videoId'])
            test_limit += 1 # 如果尚未開始的直播超過5部，就再+1探索真實數量
        # 下面的(test_limit - len(test_list))就是尚未開始的直播數量，要加到limit上
        real_limit = limit + (test_limit - len(test_list) - 1)
        # 排除完成
        self.videos = get_channel(channel_url=self.channel_url,
                     content_type='streams',
                     limit=real_limit)
        self.url_list = []
    
    def main(self):
        for i in self.videos:
            self.url_list.append('https://www.youtube.com/watch?v=' + i['videoId'])
        return self.url_list
'''
# 下面是示範code
get_streams_from_channel(channel_url='https://www.youtube.com/@nyoro0606tw',
                         limit=5).main()
# 會return近五個已經直播的url
'''

class compare_different_live(object):
    def __init__(self, url_list):
        # 因為之後可能會進行跨直播主的比較，用dict比較好改變key name來辨識直播主
        self.url_list = url_list
        self.url_dict = dict(zip([i for i in range(0, len(self.url_list))], self.url_list))
        self.result_dict = {}
        
    def main_analysis(self):
        for i in self.url_dict:
            print('url', i, 'processing...')
            basic_df = message_analysor(self.url_dict[i]).main()
            user_df = user_analysor(basic_df).main()
            live_time_line_df = live_info(basic_df).main()
            self.result_dict[i] = {'basic_df': basic_df,
                                   'user_df': user_df,
                                   'live_time_line_df': live_time_line_df}
        return self.result_dict

    def user_message_data_merge(self):
        '''
        This function provide user's messages count from different lives.
        remember to run main_analysis() first
        '''
        # 建立基本資料集，下面的for loop再插入其他資料集
        # 'user_identity', 'identity_time'
        user_df = self.result_dict[0]['user_df'][['user_id', 
                                                  'message_count', 
                                                  'user_identity', 
                                                  'identity_time',
                                                  'message_list',
                                                  'time_list']]
        user_df.insert(len(user_df.columns), "from", [0] * len(user_df), True)
        for i in self.result_dict:
            if i != 0:
                df = self.result_dict[i]['user_df'][['user_id', 
                                                     'message_count', 
                                                     'user_identity', 
                                                     'identity_time',
                                                     'message_list',
                                                     'time_list']]
                df.insert(len(df.columns), "from", [i] * len(df), True)
                user_df = user_df.append(df)
        # 最後的資料清理，一般來說不會用到
        user_df = user_df.fillna(0)
        # 使用user id排序
        user_df = user_df.sort_values(by='user_id')
        user_df = user_df.reset_index()
        user_df = user_df.drop(user_df.columns[0], axis=1)
        return user_df

    def live_info_data_merge(self):
        live_df = self.result_dict[0]['live_time_line_df']
        live_df.insert(len(live_df.columns), "from", [0] * len(live_df), True)
        for i in self.result_dict:
            if i != 0:
                df = self.result_dict[i]['live_time_line_df']
                df.insert(len(df.columns), "from", [i] * len(df), True)
                live_df = live_df.append(df)
        live_df = live_df.sort_values(by=['from','time'])
        live_df = live_df.reset_index()
        live_df = live_df.drop(live_df.columns[0], axis=1)
        return live_df
        
    def user_identity_info(self, user_df):
        user_df = user_df.drop_duplicates(['user_id'])
        user_identity_df = user_df.groupby(['user_identity', 'from'])\
        ['user_id'].count().reset_index()
        user_identity_df.rename(columns = {'user_id':'user_cnt'}, inplace = True)
        user_identity_df = user_identity_df.sort_values(by=['from', 'user_identity'])
        # 整理index
        user_identity_df = user_identity_df.reset_index()
        user_identity_df = user_identity_df.drop(user_identity_df.columns[0],axis=1)
        return user_identity_df

    def user_participation_rate(self, user_df):
        '''每個「至少參與1次的觀眾」的近limit場直播中，參加的比率
        比如limit=5，至少參與1次的觀眾數量為100，rate就是這100名觀眾中，平均每人參與5場中的幾場
        '''
        user_df = user_df.groupby(['user_id'])['from'].count().reset_index()
        user_df.rename(columns = {'from':'participation_cnt'}, inplace = True)
        rate = user_df.participation_cnt.sum() / (len(self.url_list) * len(user_df))
        return (user_df, rate)

    #def user_message_info(self, user_df):
        #f
'''
# 下面是示範code
url_list = ['https://www.youtube.com/watch?v=0m_Z8FSuBDQ',
            'https://www.youtube.com/watch?v=3L5sbxIGmAg&t=1s',
            'https://www.youtube.com/watch?v=T6yGWzUA29o']
ans = compare_different_live(url_list).main_analysis()
# call df of url #0: ans[0]['user_df']
'''

'''資料用結束'''

'''程式邏輯開始'''
app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# 選擇影片比較數量
mytitle = dcc.Markdown(children='# 直播留言熱度比較')
mygraph = dcc.Graph(figure={})
dropdown = dcc.Dropdown(options=[1, 2, 3, 4],
                       value=2,
                       clearable=False)

app.layout = dbc.Container([
    dbc.Row([dbc.Col([mytitle], width=6)], justify='center'),
    dbc.Row([dbc.Col([mygraph], width=10, style={"height": "100%"})], style={"height": "80%"}, justify='center'),
    dbc.Row([dbc.Col([dropdown], width=4)], justify='center')
], fluid=True)

@app.callback(
    Output(mygraph, component_property='figure'),
    Input(dropdown, component_property='value')
)
def updating_graph(input_):
    '''fig = go.Figure()
    paras_dict = {}
    width_dict = {}
    
    for i in range(1, 5):
        paras_dict['a'+str(i)] = '#9D9D9D'
        width_dict['w'+str(i)] = 2
    for i in range(4):
        if input_ == type_list[i]:
            paras_dict['a'+str(i+1)] = '#CE0000'
            width_dict['w'+str(i+1)] = 4
            break
    for i in range(4):
        fig.add_trace(go.Scatter(x=[2019, 2020, 2021], y=[df['value'][0+i],
                                                     df['value'][4+i],
                                                     df['value'][8+i]], name=type_list[i],
                     line=dict(color=paras_dict['a'+str(i+1)], width=width_dict['w'+str(i+1)])))
    fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 2019,
            dtick = 1
        ),
        margin=dict(b=10, l=10, t=10, r=10),
        height=250,
        plot_bgcolor= '#F0F0F0'
    )'''
    '''新版本'''
    limit = input_
    url_list = get_streams_from_channel(channel_url='https://www.youtube.com/@Ubye',
                             limit=limit).main()
    ans = compare_different_live(url_list)
    ans_main = ans.main_analysis()
    ans_merge_df = ans.user_message_data_merge()
    ans_live_merge_df = ans.live_info_data_merge()
    
    x = ans_live_merge_df['time'] 
    y = ans_live_merge_df['from'] 
    size = ans_live_merge_df['message_cnt'] 
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            opacity=0.99,
            mode='markers',
            marker=dict(
                size=size,  # 設置傘點大小
                sizemode='diameter',  # 指定傘點大小的模式（diameter 或 area）
                sizeref=0.8,  # 設置傘點大小的縮放因子
                sizemin=1,  # 設置傘點最小大小
            )
        )
    )
    return fig

'''程式邏輯結束'''
if __name__ == '__main__':
    app.run_server()
