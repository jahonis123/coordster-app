from flask import Flask, render_template, url_for, request
app = Flask(__name__)
from collections import Counter
import pandas as pd
import numpy as np
import requests
import sklearn
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import Counter
from collections import defaultdict
from sklearn.decomposition import NMF

client_id = '5fb4faafba07456399a7ad48a61e0dc1'
client_secret = 'd26d6dd18ea04f28b76039dbb60017af'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id,client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

artist_diction = pd.read_pickle('https://storage.googleapis.com/coordster_bucket/artist_diction.pkl')

artist_tags = pd.read_pickle('https://storage.googleapis.com/coordster_bucket/artist_tags.pkl')

similar_artists = pd.read_pickle('https://storage.googleapis.com/coordster_bucket/similar_artists.pkl')

song_list = pd.read_pickle('https://storage.googleapis.com/coordster_bucket/song_list.pkl')

song_df_mini = pd.read_pickle('https://storage.googleapis.com/coordster_bucket/song_df_mini.pkl')

random_forest_reg = pd.read_pickle('https://storage.googleapis.com/coordster_bucket/random_forest_reg.pkl')

model_dict = pd.read_pickle('https://storage.googleapis.com/coordster_bucket/model_dict.pkl')

def vectorizer_counter(model, vectorizer,categorical):
    c10 = Counter()
    for item in vectorizer.get_feature_names():
        for thing in categorical:
            if thing == item:
                c10[item] += 1
        else:
            c10[item] += 0
    biggest_test = np.array(list(c10.values()))
    return pd.DataFrame(NMF.transform(model,biggest_test.reshape(1,-1)))

def dimension_reduced_predictor(artist,artist1):
    global predictor
    try:
        artist_genre = artist_diction[artist]['genres']
    except:
        artist_genre = []
    try:
        artist1_genre = artist_diction[artist1]['genres']
    except:
        artist1_genre = []

    try:
        artist_followers = artist_diction[artist]['followers']
    except:
        artist_followers = 0
    try:
        artist1_followers = artist_diction[artist1]['followers']
    except:
        artist1_followers = 0
    try:
        artist_similar = similar_artists[artist]
    except:
        artist_similar = []
    try:
        artist1_similar = similar_artists[artist1]
    except:
        artist1_similar = []
    try:
        artist_pop = artist_diction[artist]['popularity']
    except:
        artist_pop = 0
    try:
        artist1_pop = artist_diction[artist1]['popularity']
    except:
        artist1_pop = 0
    try:
        artist_tag = artist_tags[artist]
    except:
        artist_tag = []
    try:
        artist1_tag = artist_tags[artist1]
    except:
        artist1_tag = []

    total_genre = artist_genre+artist1_genre
    artists = [artist] + [artist1]
    total_followers = (artist_followers+artist1_followers)/2
    total_similar = artist_similar+artist1_similar
    total_pop = (artist_pop + artist1_pop)/2
    total_tag = artist_tag+artist1_tag

    input_genre = vectorizer_counter(model_dict['genre_model'],model_dict['genre_vectorizer'],total_genre)
    input_artists = vectorizer_counter(model_dict['artist_model'],model_dict['artist_vectorizer'],artists)
    pd.DataFrame([total_followers])
    input_similar = vectorizer_counter(model_dict['sim_art_model'],model_dict['sim_art_vectorizer'],total_similar)
    pd.DataFrame([total_pop])
    input_tag = vectorizer_counter(model_dict['tag_model'],model_dict['tag_vectorizer'],total_tag)

    predictor = input_genre.join(input_artists,
                     rsuffix='x').join(pd.DataFrame([total_followers]),
                                       rsuffix='f').join(input_similar,
                                                         rsuffix='s').join(pd.DataFrame([total_pop]),
                                                                           rsuffix='p').join(input_tag,rsuffix='t')

    return random_forest_reg.predict(predictor)[0]

dimension_reduced_predictor('The Weeknd','Johnny Cash')

def collab_filter_lookup(art_name,art_dictionary,dictionary,song_list):
    
    """Uses the Spotify API, a dictionary of similar artists, and the artist URI
    dictionary to use a collaborative filtering approach to find good artist matches"""

    lnames = []
    for song in song_list:
        if art_name in song:
            lnames = lnames + song

    tracks = []
    for num in range(4):
        try:
            for item in sp.artist_albums(art_dictionary[art_name]['uri'],offset=num)['items']:
                if item['uri'] not in tracks:
                    tracks.append(item['uri'])
        except:
            print('Artist not found!')


    songs = {}
    names = []
    for track in tracks:
        for song in sp.album_tracks(track)['items']:
            if song['name'] not in names:
                names.append(song['name'])
            name_list = []
            for name in song['artists']:
                name_list.append(name['name'])
            songs[song['name']] = name_list
    
    c = Counter()
    c1 = Counter()
    
    for item in songs.values():
        for i in item:
            if i != art_name:
                c[i] +=1
                
    for name in c.most_common(50):
        try:
            for artist in dictionary[name[0]]:
                if artist not in lnames:
                    c1[artist] += 1
        except:
            print(name[0])
                    
    for item in c:
        if item in c1:
            c1.pop(item)
    
    return c1.most_common(10)

len_artists = len(artist_diction)
len_songs = len(song_list)

@app.route('/', methods=['GET','POST'])
@app.route("/home")
def home():
    if request.method == 'POST':
        artist = request.form.get('artist')
        artists = collab_filter_lookup(artist,artist_diction,similar_artists,song_list)
        top_score = dimension_reduced_predictor(artist,artists[0][0]).round(1)
        return render_template('home2.html',artist=artist,artists = artists,artist_diction=artist_diction,len_artists=len_artists,len_songs=len_songs,
        top_score=top_score)
        
    if request.method == 'GET':
        return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html',title="About")


if __name__ == '__main__':
    app.run(debug=True)


