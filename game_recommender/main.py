from flask import Flask,render_template,request
import numpy as np
import pandas as pd

app = Flask(__name__)

df_all = pd.read_csv("static/media/game_with_pro.csv")
df_gamer = df_all.iloc[:,3:11]
df_casual = df_all.iloc[:,11:19]

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclid_sim(v1, v2):
    return sum([(i-j)**2 for i,j in zip(v1,v2)])

def original_recommender_func(input_value, attribute, is_reverse=False):
    if attribute == "ガチ勢":
        df_recommend = df_gamer.copy()
        df_recommend['cos_sim'] = df_gamer.apply(lambda x : cos_sim(list(input_value.values()), list(x)[:9]), axis=1)
        df_recommend['euclid_sim'] = df_gamer.apply(lambda x : 1/euclid_sim(list(input_value.values()), list(x)[:9]), axis=1)
    else:
        df_recommend = df_casual.copy()
        df_recommend['cos_sim'] = df_casual.apply(lambda x : cos_sim(list(input_value.values()), list(x)[:9]), axis=1)
        df_recommend['euclid_sim'] = df_casual.apply(lambda x : euclid_sim(list(input_value.values()), list(x)[:9]), axis=1)

    min_sim, max_sim = min(df_recommend['euclid_sim']), max(df_recommend['euclid_sim'])
    df_recommend['euclid_sim'] = df_recommend['euclid_sim'].map(lambda x : (x-min_sim)/(max_sim-min_sim))
    df_recommend['cos_sim'] = df_recommend['cos_sim']*10 + df_recommend['euclid_sim']/10
    min_sim, max_sim = min(df_recommend['cos_sim']), max(df_recommend['cos_sim'])
    df_recommend['cos_sim'] = df_recommend['cos_sim'].map(lambda x : "{:.5f}".format((x-min_sim)/(max_sim-min_sim)))
    
    df_recommend = pd.merge(df_recommend[['cos_sim']].reset_index(), 
                            df_all[['ゲームタイトル','ジャンル','タグ','詳細URL']].reset_index(), 
                            on='index', 
                            how='inner'
                        ).drop("index", axis=1).rename(columns={"cos_sim":"おすすめ度", "euclid_sim":"誤差の小ささ"})
    df_recommend = df_recommend.sort_values("おすすめ度", ascending=is_reverse).head(10)

    header = df_recommend.columns
    record = df_recommend.values.tolist()
    return header, record

def cos_recommender_func(input_value, attribute, is_reverse=False):
    if attribute == "ガチ勢":
        df_recommend = df_gamer.copy()
        df_recommend['cos_sim'] = df_gamer.apply(lambda x : "{:.5f}".format(cos_sim(list(input_value.values()), list(x)[:9])), axis=1)
    else:
        df_recommend = df_casual.copy()
        df_recommend['cos_sim'] = df_casual.apply(lambda x : "{:.5f}".format(cos_sim(list(input_value.values()), list(x)[:9])), axis=1)
    
    df_recommend = pd.merge(df_recommend[['cos_sim']].reset_index(), 
                            df_all[['ゲームタイトル','ジャンル','タグ','詳細URL']].reset_index(), 
                            on='index', 
                            how='inner'
                        ).drop("index", axis=1).rename(columns={"cos_sim":"おすすめ度"})
    df_recommend = df_recommend.sort_values("おすすめ度", ascending=is_reverse).head(10)

    header = df_recommend.columns
    record = df_recommend.values.tolist()
    return header, record

def euclid_recommender_func(input_value, attribute, ascending=True):
    if attribute == "ガチ勢":
        df_recommend = df_gamer.copy()
        df_recommend['euclid_sim'] = df_gamer.apply(lambda x : "{:.5f}".format(euclid_sim(list(input_value.values()), list(x)[:9])), axis=1)
    else:
        df_recommend = df_casual.copy()
        df_recommend['euclid_sim'] = df_casual.apply(lambda x : "{:.5f}".format(euclid_sim(list(input_value.values()), list(x)[:9])), axis=1)
    
    df_recommend = pd.merge(df_recommend[['euclid_sim']].reset_index(), 
                            df_all[['ゲームタイトル','ジャンル','タグ','詳細URL']].reset_index(), 
                            on='index', 
                            how='inner'
                        ).drop("index", axis=1).rename(columns={"euclid_sim":"誤差の小ささ"})
    df_recommend = df_recommend.sort_values("誤差の小ささ", ascending=ascending).head(10)

    header = df_recommend.columns
    record = df_recommend.values.tolist()
    return header, record


# Home(game recommender(original))
@app.route("/")
@app.route("/original")
def original():
    return render_template("original.html")

@app.route("/original", methods=["post"])
def original_post():
    recommender = request.form["recommender"]
    attribute = request.form["attribute"]
    volume = float(request.form["volume"])
    freedom = float(request.form["freedom"])
    social = float(request.form["social"])
    BGM = float(request.form["BGM"])
    graphic = float(request.form["graphic"])
    empathy = float(request.form["empathy"])
    difficulty = float(request.form["difficulty"])
    value = float(request.form["value"])

    input_value = {
        "ゲームの価値":value / 10
        ,"コンテンツ量":volume / 5
        ,"自由度":freedom / 5
        ,"ソーシャリティ":social / 5
        ,"BGM":BGM / 5
        ,"グラフィック":graphic / 5
        ,"感情移入":empathy / 5
        ,"ゲーム性・敷居の低さ":difficulty / 5
    }

    if recommender == "original":
        header, record = original_recommender_func(input_value, attribute, is_reverse=False)
    elif recommender == "cos":
        header, record = cos_recommender_func(input_value, attribute, is_reverse=False)
    else:
        header, record = euclid_recommender_func(input_value, attribute, ascending=True)

    return render_template("original.html", recommender=recommender, attribute=attribute, volume=volume, freedom=freedom, social=social, BGM=BGM, graphic=graphic, empathy=empathy, difficulty=difficulty, value=value, header=header, record=record, recommend_flag=True)

# game recommender(original) reverse
@app.route("/original_reverse")
def original_reverse():
    return render_template("original_reverse.html")

@app.route("/original_reverse", methods=["post"])
def original_reverse_post():
    recommender = request.form["recommender"]
    attribute = request.form["attribute"]
    volume = float(request.form["volume"])
    freedom = float(request.form["freedom"])
    social = float(request.form["social"])
    BGM = float(request.form["BGM"])
    graphic = float(request.form["graphic"])
    empathy = float(request.form["empathy"])
    difficulty = float(request.form["difficulty"])
    value = float(request.form["value"])

    input_value = {
        "ゲームの価値":value / 10
        ,"コンテンツ量":volume / 5
        ,"自由度":freedom / 5
        ,"ソーシャリティ":social / 5
        ,"BGM":BGM / 5
        ,"グラフィック":graphic / 5
        ,"感情移入":empathy / 5
        ,"ゲーム性・敷居の低さ":difficulty / 5
    }

    if recommender == "original":
        header, record = original_recommender_func(input_value, attribute, is_reverse=True)
    elif recommender == "cos":
        header, record = cos_recommender_func(input_value, attribute, is_reverse=True)
    else:
        header, record = euclid_recommender_func(input_value, attribute, ascending=False)

    return render_template("original_reverse.html", recommender=recommender, attribute=attribute, volume=volume, freedom=freedom, social=social, BGM=BGM, graphic=graphic, empathy=empathy, difficulty=difficulty, value=value, header=header, record=record, recommend_flag=True)


# game recommender(cos)
@app.route("/cos")
def cos_recommender():
    return render_template("cos.html")

@app.route("/cos", methods=["post"])
def cos_recommender_post():
    attribute = request.form["attribute"]
    volume = float(request.form["volume"])
    freedom = float(request.form["freedom"])
    social = float(request.form["social"])
    BGM = float(request.form["BGM"])
    graphic = float(request.form["graphic"])
    empathy = float(request.form["empathy"])
    difficulty = float(request.form["difficulty"])
    value = float(request.form["value"])

    input_value = {
        "ゲームの価値":value / 10
        ,"コンテンツ量":volume / 5
        ,"自由度":freedom / 5
        ,"ソーシャリティ":social / 5
        ,"BGM":BGM / 5
        ,"グラフィック":graphic / 5
        ,"感情移入":empathy / 5
        ,"ゲーム性・敷居の低さ":difficulty / 5
    }

    header, record = cos_recommender_func(input_value, attribute, is_reverse=False)

    return render_template("cos.html", attribute=attribute, volume=volume, freedom=freedom, social=social, BGM=BGM, graphic=graphic, empathy=empathy, difficulty=difficulty, value=value, header=header, record=record, recommend_flag=True)


# game recommender(cos) reverse
@app.route("/cos_reverse")
def cos_recommender_reverse():
    return render_template("cos_reverse.html")

@app.route("/cos_reverse", methods=["post"])
def cos_recommender_reverse_post():
    attribute = request.form["attribute"]
    volume = float(request.form["volume"])
    freedom = float(request.form["freedom"])
    social = float(request.form["social"])
    BGM = float(request.form["BGM"])
    graphic = float(request.form["graphic"])
    empathy = float(request.form["empathy"])
    difficulty = float(request.form["difficulty"])
    value = float(request.form["value"])

    input_value = {
        "ゲームの価値":value / 10
        ,"コンテンツ量":volume / 5
        ,"自由度":freedom / 5
        ,"ソーシャリティ":social / 5
        ,"BGM":BGM / 5
        ,"グラフィック":graphic / 5
        ,"感情移入":empathy / 5
        ,"ゲーム性・敷居の低さ":difficulty / 5
    }

    header, record = cos_recommender_func(input_value, attribute, is_reverse=True)

    return render_template("cos_reverse.html", attribute=attribute, volume=volume, freedom=freedom, social=social, BGM=BGM, graphic=graphic, empathy=empathy, difficulty=difficulty, value=value, header=header, record=record, recommend_flag=True)


# game recommender(euclid)
@app.route("/euclid")
def euclid_recommender():
    return render_template("euclid.html")

@app.route("/euclid", methods=["post"])
def euclid_recommender_post():
    attribute = request.form["attribute"]
    volume = float(request.form["volume"])
    freedom = float(request.form["freedom"])
    social = float(request.form["social"])
    BGM = float(request.form["BGM"])
    graphic = float(request.form["graphic"])
    empathy = float(request.form["empathy"])
    difficulty = float(request.form["difficulty"])
    value = float(request.form["value"])

    input_value = {
        "ゲームの価値":value / 10
        ,"コンテンツ量":volume / 5
        ,"自由度":freedom / 5
        ,"ソーシャリティ":social / 5
        ,"BGM":BGM / 5
        ,"グラフィック":graphic / 5
        ,"感情移入":empathy / 5
        ,"ゲーム性・敷居の低さ":difficulty / 5
    }

    header, record = euclid_recommender_func(input_value, attribute, ascending=True)

    return render_template("euclid.html", attribute=attribute, volume=volume, freedom=freedom, social=social, BGM=BGM, graphic=graphic, empathy=empathy, difficulty=difficulty, value=value, header=header, record=record, recommend_flag=True)


# game recommender(euclid) reverse
@app.route("/euclid_reverse")
def euclid_recommender_reverse():
    return render_template("euclid_reverse.html")

@app.route("/euclid_reverse", methods=["post"])
def euclid_recommender_reverse_post():
    attribute = request.form["attribute"]
    volume = float(request.form["volume"])
    freedom = float(request.form["freedom"])
    social = float(request.form["social"])
    BGM = float(request.form["BGM"])
    graphic = float(request.form["graphic"])
    empathy = float(request.form["empathy"])
    difficulty = float(request.form["difficulty"])
    value = float(request.form["value"])

    input_value = {
        "ゲームの価値":value / 10
        ,"コンテンツ量":volume / 5
        ,"自由度":freedom / 5
        ,"ソーシャリティ":social / 5
        ,"BGM":BGM / 5
        ,"グラフィック":graphic / 5
        ,"感情移入":empathy / 5
        ,"ゲーム性・敷居の低さ":difficulty / 5
    }

    header, record = euclid_recommender_func(input_value, attribute, ascending=False)

    return render_template("euclid_reverse.html", attribute=attribute, volume=volume, freedom=freedom, social=social, BGM=BGM, graphic=graphic, empathy=empathy, difficulty=difficulty, value=value, header=header, record=record, recommend_flag=True)


# About
@app.route("/about")
def about():
    return render_template("about.html")

# Contact
@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)