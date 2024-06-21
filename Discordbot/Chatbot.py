import discord
import datetime
import asyncio
from icrawler.builtin import BingImageCrawler
import openai
import glob
import re
import os
import tweepy

#Twitter認証
api_key = 'please api key'
api_key_secret = 'please secret api key'
access_token = 'please access token'
access_token_secret = 'please secret access token'
bearer_token = 'please bearer token'

TW_client = tweepy.Client(
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    bearer_token=bearer_token
)


dt = datetime.datetime.today()

#discordアクセストークン
TOKEN = 'please token'

#discordの開発者管理(web)から権限を有効にする必要があります。(でないとエラーが出るかもしれないです)
client = discord.Client(intents=discord.Intents.all())

#OpenAIのキー設定
#個人でも入力する必要があります。（やり方はWebへ）
openai.organization = "prease organization"
openai.api_key      = "prease api key"


async def greeting():
    greet = client.get_channel(prease ch ID)
    await greet.send('**マスターのノートPCが起動しました。**')
    await greet.send(f'{dt.year}年{dt.month}月{dt.day}日 {dt.hour}時{dt.minute}分{dt.second}秒')
    print(f'{dt.year}年{dt.month}月{dt.day}日 {dt.hour}時{dt.minute}分{dt.second}秒')

@client.event
async def on_ready():
    
    # 認識しているサーバーをlist型で取得し、その要素の数を 変数:guild_count に格納しています。
    guild_count = len(client.guilds)
    # 関数:lenは、引数に指定したオブジェクトの長さや要素の数を取得します。
    
    game = discord.Game(f'管理')
    # f文字列(フォーマット済み文字列リテラル)は、Python3.6からの機能です。
    
    # BOTのステータスを変更する
    await client.change_presence(status=discord.Status.online, activity=game)
    # パラメーターの status でステータス状況(オンライン, 退席中など)を変更できます。

    await greeting();
    
    print('ログインしました')


def crawler_function(data):
    # 画像のダウンロード
    crawler = BingImageCrawler(downloader_threads=5,storage={'root_dir': 'prease dir'})
    filters = dict(
        color='color',
        date='pastyear'
    )
    crawler.crawl(keyword=data, filters=filters, offset=0, max_num=10)


def Ask_ChatGPT(message):
    
    # 応答設定
    completion = openai.ChatCompletion.create(
        model    = "gpt-3.5-turbo",     # モデルを選択
        messages = [{
                    "role": "system",
                    "content": "丁寧に答えてください"
                },
                {
                 "role":"user",       # 役割
                 "content":message,   # メッセージ 
                 }],
    
        max_tokens  = 300,             # 生成する文章の最大単語数
        n           = 1,                # いくつの返答を生成するか
        stop        = None,             # 指定した単語が出現した場合、文章生成を打ち切る
        temperature = 0.5,              # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
    )
    
    # 応答
    response = completion.choices[0].message.content
    print(response)
    return response

def remove_glob(pathname, recursive=True):
    for p in glob.glob(pathname, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)
    

@client.event
async def on_message(message):
    # メッセージ送信者がBotだった場合は無視する
    if message.author.bot:
        return
    if message.content.find('!#') != -1:
        remove_glob("please path(imgの保存先)")
        s = message.content[2:]
        crawler_function(s)
        await message.channel.send(message.content[3:]+'の情報を表示するYo！')
        files = glob.glob("please path(imgの保存先)")
        for c_img in files:
            img=c_img
            await message.channel.send(file=discord.File(img))
        return
    
    if message.content.find('!Image') != -1:
        s = message.content[6:]
        response = openai.Image.create(
            prompt=s,
            n=1,
            size="512x512"
        )
        image_url = response['data'][0]['url']
        await message.channel.send(image_url)
        return

    if message.content == '!help':
        await message.channel.send('!# [見つけてほしい画像]\n')
        await message.channel.send('!Image [生成してほしい画像]\n')
        await message.channel.send('!help')
        return

    if message.content.find('!Tweet') != -1:
        s = message.content[6:]+' -RT'
        await message.channel.send('情報を表示します。')
        tweets = TW_client.search_recent_tweets(query=s,  # 検索ワード
                                         max_results=10  # 取得件数
                                         )
        if tweets is not None:
                for tweet in tweets[0]:
                    await message.channel.send(tweet)
                    await message.channel.send('*-*-*-*-*-*-*-*-*-*-*')
        return
    
    if message.content.find('!h') != -1:
        greet = client.get_channel(message.channel.id)
        liver_id = judge_liver(message.content[3:])
        if liver_id!=0:
            await message.channel.send(message.content[3:]+'の情報を表示する余！')
            tweets = TW_client.search_recent_tweets(query='from:'+liver_id+' -RT',  # 検索ワード
                                             max_results=10  # 取得件数
                                             )
            if tweets is not None:
                    for tweet in tweets[0]:
                        await greet.send(tweet)
                        await greet.send('----------')
        else:
            await greet.send(file=discord.File('prease another_img path'))
        return
    
    res = Ask_ChatGPT(message.content)
    print(message.content)
    print(res)
    await message.channel.send(res)
    return


def judge_liver(x):
    liver_dec = {"そらちゃん":"tokino_sora" , "ロボ子さん":"robocosan" , "みこち":"sakuramiko35" , "すいちゃん":"suisei_hosimati" ,
                 "まつりちゃん":"natsuiromatsuri" , "メルちゃん":"yozoramel" , "はあちゃま":"akaihaato" , "むきろぜちゃん":"akirosentahl" , "ふぶちゃん":"shirakamifubuki" ,
                 "AZKi":"AZKi_VDiVA" ,
                 "":"" , "":"" , "":"" , "":"" , "":"" ,
                 "":"" , "":"" , "":"" , "":"" , "":"" ,
                 "":"" , "":"" , "":"" , "":"" , "":"" ,
                 "":"" , "":"" , "":"" , "":"" , "":"" ,
                 }
    if x in liver_dec.keys():
        s = liver_dec[x]
        return s
    else:
        return 0
    
'''
@client.event
async def on_voice_state_update(member, before, after):
 
    # チャンネルへの入室ステータスが変更されたとき（ミュートON、OFFに反応しないように分岐）
    if before.channel != after.channel:
        # 通知メッセージを書き込むテキストチャンネル（チャンネルIDを指定）
        botRoom = client.get_channel()
 
        # 入退室を監視する対象のボイスチャンネル（チャンネルIDを指定）
        announceChannelIds = [,,]
 
        # 退室通知
        if before.channel is not None and before.channel.id in announceChannelIds:
            print(' '+after.channel.name +' : ' + member.name)
        #    await botRoom.send("**" + before.channel.name + "** から、__" + member.name + "__  が抜けたよ")

        # 入室通知
        if after.channel is not None and after.channel.id in announceChannelIds:
            print('join '+after.channel.name +' : ' + member.name)
            await botRoom.send("**" + after.channel.name + "** に、__" + member.name + "__  が参加したよ！")
 

#入室通知
'''

client.run(TOKEN)
