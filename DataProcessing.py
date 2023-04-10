import numpy as np

import chart_studio.plotly as py
from plotly.graph_objs import Figure, Layout, Data, Scatter3d, Mesh3d, Scatter
from plotly.graph_objs.scatter3d import Marker
from plotly.graph_objs.scatter import Marker as Marker2D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import colour
import alphashape as ap

import pandas as pd

import spotipy as sp

import time
import tqdm

# The username to authenticate with
# Data from other users can still be processed
AUTH_USERNAME = 'Jack5225x'
# A file that contains the client id and secret in the format:
# Client ID: <id>
# Client Secret: <secret>
AUTH_CREDS_FILE = 'SpotifyCredentials.txt'
# Where to redirect after authentication; if unsure
# just leave as localhost
AUTH_REDIRECT_URI = 'http://localhost'
AUTH_SCOPE = 'user-library-read user-modify-playback-state user-read-currently-playing user-read-playback-state'

def authenticate(username=AUTH_USERNAME, credsFile=AUTH_CREDS_FILE, redirectURI=AUTH_REDIRECT_URI, scope=AUTH_SCOPE):
    """
    Generate a spotipy instance using provided authentication information.

    Parameters
    ----------
    username : str
        The spotify username for authentication.

    credsFile : str
        Path to a file that contains the client ID and secret
        in the format:
        <credsFile>
        Client ID: <id>
        Client Secret: <secret>

    redirectURI : str
        The web address to redirect to during authentication, specifically
        while generating the cached credentials. This does not need
        to be a valid URL; you just need to copy the url it opens
        and provide it back to python (Spotipy will give instructions
        for this if it is necessary). localhost is easist choice.

    Returns
    -------
    spotify : spotipy.Spotify
        Spotify instance that can be used to search for songs,
        users, and playlists.
    """
    clientID = None
    clientSecret = None

    with open(credsFile) as cFile:
        clientID = cFile.readline().split(':')[1].strip()
        clientSecret = cFile.readline().split(':')[1].strip()

    if clientID and clientSecret:
        token = sp.util.prompt_for_user_token(username,
                                              scope=scope,
                                              client_id=clientID,
                                              client_secret=clientSecret,
                                              redirect_uri=redirectURI)

        return sp.Spotify(auth=token)

    else:
        raise Exception(f'Invalid creds file: {credsFile}!')

def fetchPublicPlaylists(username, spotify=None):
    """
    Fetch a list of the playlists that are public
    for a given user.

    Note that playlists may have to be added to a user's
    profile, not just public, to show up here. In Spotify,
    this can be done by clicking on the options for a playlist
    and using the "Add to Profile" option.

    Parameters
    ----------
    username : str
        Valid Spotify username.

    spotify : spotipy.Spotify or None
       Spotify instance used to do the search. If not
       provided (or None) a new one will be created
       using authenticate() with default options.

    Returns
    -------
    playlists : pandas.DataFrame
        Data frame containing playlist information for
        user. Includes the following columns:
            ---------------------------------------
            Playlist name (name)
            Playlist ID (id)
            Number of tracks (tracks:total)
            Playlist owner name (owner:display_name)
        
    """
    if not spotify:
        spotify = authenticate()

    # This queries spotify for a list of tracks
    playlists = spotify.user_playlists(username)["items"]

    # Use pandas dataframe to keep track of everything
    data = pd.DataFrame()

    # You could just autogenerate a pandas dataframe from the provided
    # dictionary, but I don't care about most of those items, so I just
    # pick a few that are relevant
    # Available ones are:
    # ['collaborative', 'description', 'external_urls', 'href', 'id', 'images',
    # 'name', 'owner', 'primary_color', 'public', 'snapshot_id', 'tracks', 'type', 'uri']
    # Notes:
    # - uri and id are pretty much the same: uri can be generated as f"spotify:playlist:{id}"

    # Also, some of the dictionary values are dictionaries themselves, and
    # we just want a element of those; I've set up the colon notation to
    # access the element of an element
    desiredFields = ['name', 'uri', 'tracks:total', 'owner:display_name']

    for f in desiredFields:
        # Access sub elements if necessary
        if ':' in f:
            # Split the field by ':' and take the first half as a key for
            # the first dictionary, and the second half as a key for the
            # second dictionary.
            dataArr = [p[f.split(':')[0]][f.split(':')[1]] for p in playlists]
        else:
            # Otherwise, just access the dictionary as you'd expect
            dataArr = [p[f] for p in playlists]

        # Add it to the data frame
        data[f] = dataArr

    return data

def parseTrackData(tracks, spotify=None):
    """
    """ 
    # Generate a data table of the track information
    # Same deal as in fetchPublicPlaylists above; see there
    # for more information
    data = pd.DataFrame()

    if len(tracks) == 0:
        return data

    # Available fields this time are:
    # ['album', 'artists', 'available_markets', 'disc_number', 'duration_ms',
    # 'episode', 'explicit', 'external_ids', 'external_urls', 'href', 'id',
    # 'is_local', 'name', 'popularity', 'preview_url', 'track', 'track_number',
    # 'type', 'uri']

    # 'artists' will be a list of artists, so we need to be able to handle that
    desiredFields = ['name', 'artists:name', 'uri', 'duration_ms']
    for f in desiredFields:
        # Access sub elements if necessary
        if ':' in f:
            # This is for the multiple artists
            # Check if the element of the dictionary is a list
            if isinstance(tracks[0][f.split(':')[0]], list):
                # If it is a list, we have to take the sub element of each
                # dictionary in the list, then join them together as a comma-
                # separated list of artists.
                dataArr = [', '.join([a[f.split(':')[1]] for a in p[f.split(':')[0]]]) for p in tracks]
            else:
                # If it is not a list, it is just the same sub element accessing
                # as in the fetchPublicPlaylists() method above.
                dataArr = [p[f.split(':')[0]][f.split(':')[1]] for p in tracks]
        else:
            dataArr = [p[f] for p in tracks]

        data[f] = dataArr

    # We also want to fetch the audio features of the data, which is stored separately
    # from the other information in spotify
    # For this, we have the following data available:
    # ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    # 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type',
    # 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature']
    desiredFields = ['danceability', 'energy', 'key',
                     'loudness', 'mode',
                     'speechiness',
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','time_signature']

    # Sidenote: there is also the function Spotify.audio_analysis() which will give
    # a *very* in depth audio anaylsis of the song, but for now, just the summary
    # results from audio_features() is sufficient.

    # We can only request 100 ids at once, so we may have to break our list up a bit
    #print(data.shape)
    if data.shape[0] > 100:
        featuresArr = []
        # Break the data into several lists that are no longer than 100 items
        for i in range(int(np.ceil(data.shape[0] / 100))):
            # Concatenate the results from the multiple lists into a single list
            featuresArr += spotify.audio_features(data["uri"][i*100:min((i+1)*100, data.shape[0])])
    else:
        # If less than 100 items, we can just directly request information
        featuresArr = spotify.audio_features(data["uri"])

    for f in desiredFields:
        # No sub elements here, so can just simply assign values
        # Not sure why some elements end up as none, but I guess we
        # can just ignore them.
        dataArr = [feat[f] if feat is not None else None for feat in featuresArr]

        data[f] = dataArr

    return data

def fetchPlaylistTracks(playlistID, spotify=None):
    """
    Fetch track information for each track on a playlist.

    Note that only tracks for which there are audio feature analysis
    available are returned; it is possible that this information is not
    present for all songs, and therefore some may be missing.

    Parameters
    ----------
    playlistID : str
        A valid Spotify playlist id.

    spotify : spotipy.Spotify or None
       Spotify instance used to do the search. If not
       provided (or None) a new one will be created
       using authenticate() with default options.

    Returns
    -------
    tracks : pandas.DataFrame
        Data frame containing audio features and metadata
        for all tracks in the playlist. Includes the 
        following columns:
            -----------------------------------
            Name (name)
            Artists (artists:name)
            ID (id)
            Duration (duration_ms)
            Danceability (danceability)
            Energy (energy)
            #Key (key)
            Loudness (loudness)
            Mode (mode)
            Speechiness (speechiness)
            Acousticness (acousticness)
            Instrumentalness (instrumentalness)
            Liveness (liveness)
            Valence (valence)
            #Tempo (tempo)
            #Time signature (time_signature)

    """
    if not spotify:
        spotify = authenticate()

    # Query spotify for the tracks on the given playlist
    # This returns some extra information that I don't care
    # about, so I just throw away everything except the 'tracks'
    # element.
    # You also can only fetch 100 items at once, so we have to split it up
    allTracks = []
    i = 0

    while True:
        # Break the data into several lists that are no longer than 100 items
        tracks = spotify.playlist_items(playlistID, limit=100, offset=i*100)["items"]

        if len(tracks) == 0:
            break

        # Remove some structure that makes it easier to process lower down
        tracks = [t["track"] for t in tracks]

        # And add to the running list
        allTracks += tracks
        i += 1

    return parseTrackData(allTracks, spotify)


def mergePlaylistTracks(playlistIDs, spotify=None, normalize=True, removeNan=False):
    """
    Preprocess a collection of playlists to create a big list of aural
    properties of tracks, to be passed to LDA or PCA.

    Parameters
    ----------
    playlistIDs : list(str)
        A list of playlist IDs, for which all of the tracks will be
        merged together into a single list.

    spotify : spotipy.Spotify or None
       Spotify instance used to do the search. If not
       provided (or None) a new one will be created
       using authenticate() with default options.

    Returns
    -------
    phaseSpaceCoords : pandas.DataFrame
        Each phase space coordinate (audio feature) of every track from all playlists.

    coordinateIdentities : list(str)
        ID of the playlist to which each phase space point belongs.

    audioFeatures : list(str)
        Human readable names of the audio features that generate the phase space.
    """
    if not spotify:
        spotify = authenticate()

    # Gather all of the tracks from every playlist
    phaseSpaceCoords = pd.DataFrame()

    # This is to keep track of what playlist each track was originally a part of
    # (which is required for LDA decomposition
    # For a method that doesn't seed the process with this information, see
    # computePCADecomp()
    coordinateIdentities = []

    for p in playlistIDs:
        # Our method defined above to get some playlist information
        trackData = fetchPlaylistTracks(p, spotify)
        # Put everything into a big list
        phaseSpaceCoords = pd.concat([phaseSpaceCoords, trackData])
        # Record which playlist these tracks came from
        # Note that we aren't actually associated each set of
        # coordinates with an identity, but just relying on the
        # ordering to connect these two. eg. coordinateIdentities[200]
        # will give the identity for the point phaseSpaceCoords[200].
        coordinateIdentities += [p] * trackData.shape[0]

    # Make sure we haven't missed anything
    assert len(coordinateIdentities) == phaseSpaceCoords.shape[0], f'Should be same amount of coordinates and identities (reality {len(coordinateIdentities)} vs. {phaseSpaceCoords.shape[0]})'
    
    # Now we remove data that shouldn't affect the dimensionality
    # reduction, like the name, id, etc. These are nice to have for
    # user readability, but for the calculation we do not want them.
    # It is easier to specify which columns will be removed as opposed
    # to which will be kept
    removeColumns = ['name', 'uri', 'artists:name', 'duration_ms']

    for k in removeColumns:
        del phaseSpaceCoords[k]

    # Convert to arrays, and save the feature names
    featureNames = phaseSpaceCoords.columns
    phaseSpaceCoords = np.array(phaseSpaceCoords)
    coordinateIdentities = np.array(coordinateIdentities)

    # Now we have a full list of every song, including corresponding
    # membership to certain playlists.
    
    if removeNan:
        badIndices = np.unique(np.where(np.isnan(phaseSpaceCoords))[0])
        goodIndices = [i for i in range(len(phaseSpaceCoords)) if i not in badIndices]
        phaseSpaceCoords = phaseSpaceCoords[goodIndices]
        coordinateIdentities = coordinateIdentities[goodIndices]

    if not normalize:
        return phaseSpaceCoords, coordinateIdentities, featureNames 

    # The last thing we want to do is to scale the data in a uniform way
    # This is a common preprocessing technique when doing LDA or PCA
    scaler = StandardScaler(copy=False)
    scaledPhaseSpaceCoords = scaler.fit(phaseSpaceCoords).transform(phaseSpaceCoords)

    return scaledPhaseSpaceCoords, coordinateIdentities, featureNames


def computeLDA(phaseSpaceCoords, coordinateIdentities, nComponents=2):
    """
    Reduce dimensionality of the phase space via linear
    discriminant analysis.

    Note that LDA requires the identities to maximize the
    separation between separate communities. For a method that
    does not require this information a priori, see computePCA().

    Parameters
    ----------
    phaseSpaceCoords : pandas.DataFrame
        Coordinates in phase space (audio feature space) for all tracks
        to be transformed into a lower dimensional space.

    coordinateIdentities : list(obj)
        ID of the community to which each point belongs. Can be
        any object type, so long as, for any two points i,j in the
        same community:
            coordinateIdentities[i] == coordinateIdentities[j]

    nComponents : int
        Desired dimensionality of the resultant space. 2 or 3
        .are best, as they can be visualized.

    Returns
    -------
    ldaCoords : numpy.ndarray
        Coordinates for each point in phaseSpaceCoords in the
        lower dimensional space.
    """
    lda = LinearDiscriminantAnalysis(n_components=nComponents)
    ldaCoords = lda.fit(phaseSpaceCoords, coordinateIdentities).transform(phaseSpaceCoords)

    return ldaCoords

def computePCA(phaseSpaceCoords, nComponents=2):
    """
    Reduce dimensionality of the phase space via
    principal component analysis.

    Parameters
    ----------
    phaseSpaceCoords : pandas.DataFrame
        Coordinates in phase space (audio feature space) for all tracks
        to be transformed into a lower dimensional space.

    nComponents : int
        Desired dimensionality of the resultant space. 2 or 3
        are best, as they can be visualized.

    Returns
    -------
    pcaCoords : numpy.ndarray
        Coordinates for each point in phaseSpaceCoords in the
        lower dimensional space.

    pcaComponents : numpy.ndarray
        Projection of the new axes onto each original axis, which
        can be used to interpret axes in the lower dimensional space.
    """

    pca = PCA(n_components=nComponents)
    pcaCoords = pca.fit(phaseSpaceCoords).transform(phaseSpaceCoords)

    return pcaCoords, pca.components_

def interpretDecomposedCoordinates(projections, originalLabels, includeCount=None, includeThreshold=.4):
    """
    Create a text-interpretation of the decomposed coordinates
    acquired via PCA.

    Parameters
    ----------
    projections : numpy.ndarray[N, M]
        Projection (scalar) of M original axes onto N decomposed axes.

    originalLabels : list(str)[M]
        List of M original axis labels.

    includeCount : int or None
        Number of original labels to compose the new labels of.
        If set to None, will be decided based on `includeThreshold`.

    includeThreshold : float
        Magnitude of projection for an original eigenvector to be
        included in the new interpretation. Only used if `includeCount`
        is None.

    Returns
    -------
    labels : list(str)[N]
        List of N constructed labels.
    """

    # N and M from docstring above
    # N is number of new bases
    # M is number of old bases
    N, M = projections.shape

    arrowChars = {1: '(+)', -1: '(-)'}
    #arrowChars = {1: '🠕', -1: '🠗'}
    #arrowChars = {1: '$\\uparrow$', -1: '$\\downarrow$'}
    #arrowChars = {1: '🠖', -1: '🠔'}
    labels = [None] * N

    # For each new basis vector, identify what the most relevant
    # original components are
    for i in range(N):
        contributions = np.abs(projections[i])

        if includeCount is not None:
            # Sorted from largest to smallest 
            # eg. first element is the index with the
            # largest contribution to this new eigenvector
            order = np.argsort(contributions)[::-1]
            includedIndices = order[:includeCount]
        else:
            # Grab the index of every element that is greater
            # than the threshold
            includedIndices = np.where(contributions > includeThreshold)[0]

        includedLabels = [f'{arrowChars[np.sign(projections[i][o])]} {originalLabels[o]}' for o in includedIndices]
        labels[i] = ' '.join(includedLabels).strip()

    return labels


def decompositionHull(decomposedCoords, coordinateIdentities, humanReadableIdentities, alpha=0.1, title=None, figureKwargs={}):
    """
    """

    uniqueIdentities = np.unique(coordinateIdentities)
    nDim = np.shape(decomposedCoords)[1]

    if humanReadableIdentities is not None:
        labels = [humanReadableIdentities[id] for id in uniqueIdentities]
    else:
        labels = uniqueIdentities

    layout = Layout(title='' if not title else title, width=800, height=550)

    if nDim == 2:
        
        data = pd.DataFrame()

        data["x"] = decomposedCoords[:,0]
        data["y"] = decomposedCoords[:,1]
        data["label"] = [humanReadableIdentities[id] for id in coordinateIdentities]
        
        data = []
        
        for s in range(len(uniqueIdentities)):
            indices = np.where(np.array(coordinateIdentities) == uniqueIdentities[s])[0]
            hull = np.array(ap.alphashape(decomposedCoords[indices], alpha).exterior.coords)
            color = str(colour.Color(pick_for=labels[s]))
            
            scatter = Scatter(x=decomposedCoords[indices,0], y=decomposedCoords[indices,1],
                              mode='markers', marker=Marker2D(color=color, size=8), name=labels[s], showlegend=False)
            polygon = Scatter(x=hull[:,0], y=hull[:,1], fill='toself', fillcolor=color, opacity=.2,
                              mode='markers', marker=Marker2D(color=color, size=8), showlegend=True, name=labels[s])

            data.append(scatter)
            data.append(polygon)
        
        fig = Figure(data, layout=layout)

        fig.update_layout(showlegend=True)
        
        return fig

    elif nDim == 3:
       
        dataArr = []

        for s in range(len(uniqueIdentities)):
            indices = np.where(np.array(coordinateIdentities) == uniqueIdentities[s])[0]

            color = str(colour.Color(pick_for=labels[s]))
            scatter = Scatter3d(mode='markers', marker=Marker(color=color, size=2), name=labels[s], showlegend=True,
                                x=decomposedCoords[indices,0], y=decomposedCoords[indices,1], z=decomposedCoords[indices,2])

            mesh = Mesh3d(alphahull=alpha, color=color, opacity=0.3, name=labels[s], showlegend=False,
                                x=decomposedCoords[indices,0], y=decomposedCoords[indices,1], z=decomposedCoords[indices,2])
   
            dataArr.append(scatter)
            dataArr.append(mesh)

        # Now construct the figure with all of the data information
        fig = Figure(data=dataArr, layout=layout)
        fig.update_layout(showlegend=True)

        #fig.show()

        return fig
    else:
        raise Exception(f'Unsupported number of dimensions: {nDim}. Only 2 and 3 are acceptable.')


def fetchAllSavedSongs(spotify, batchSize=50):
    """
    """
    # You can't (to my knowledge) just ask for all of the
    # songs at once, so we have to just keep querying until
    # the result comes back empty
    data = pd.DataFrame()
    i = 0

    while True:
        # Query tracks in the interval [batchSize*i, batchSize*(i+1)]
        retrievedTracks = spotify.current_user_saved_tracks(limit=batchSize, offset=batchSize*i)
       
        # If there's nothing left, we are done
        if len(retrievedTracks["items"]) == 0:
            break

        # Remove some structure that makes it easier to process lower down
        tracks = [t["track"] for t in retrievedTracks["items"]]
        partialData = parseTrackData(tracks, spotify)

        data = pd.concat([data, partialData], axis=0)
        i += 1

    return data

def fetchShuffleOrder(spotify, contextURI=None, tracks=pd.DataFrame(), numTracks=None, returnIndices=False, debug=False):
    """
    """

    # Enable shuffle
    spotify.shuffle(True)
    
    if contextURI is None and len(tracks) == 0:
        # If no context (playlist) is given and trackList are None, shuffle
        # through the current user's library.
        tracks = fetchAllSavedSongs(spotify)

        # Fetch all URIs
        uriList = [spotify.track(tracks.iloc[i]["id"])["uri"] for i in range(len(tracks))]

        # Calculate total number of tracks
        # (It's possible not all of them will be played)
        totalNumTracks = len(uriList)

        # Start playback
        spotify.start_playback(uris=uriList)

        # Wait a bit, since otherwise we will get the previous queue information
        # Generally this formula for the wait time seems to work well enough.
        # If you find that the first element of the order always ends up as
        # -1, then it means you need to wait longer.
        time.sleep(.01*totalNumTracks)

    elif contextURI is not None and len(tracks) == 0:
        # If are given a context (playlist, album, etc.) we can directly
        # start playback from there
        spotify.start_playback(context_uri=contextURI)

        # Calculate the total number of tracks
        playlistInfo = spotify.playlist(contextURI)

        totalNumTracks = playlistInfo["tracks"]["total"]

        # We still need the URIs to convert to index in the playlist later on
        uriList = [t["track"]["uri"] for t in playlistInfo["tracks"]["items"]]

        # We don't need to wait when playing directly from a playlist,
        # since the shuffle information is likely cached somewhere
        time.sleep(1)

    elif contextURI is None and len(tracks) > 0:
        # Otherwise, we can fetch all of the tracks and play them
        uriList = [spotify.track(tracks.iloc[i]["id"])["uri"] for i in range(len(tracks))]

        totalNumTracks = len(uriList)

        spotify.start_playback(uris=uriList)

        # Wait a bit, since otherwise we will get the previous queue information
        # Generally this formula for the wait time seems to work well enough.
        # If you find that the first element of the order always ends up as
        # -1, then it means you need to wait longer.
        time.sleep(.01*len(tracks))

    if numTracks is None:
        numTracks = totalNumTracks

    if debug:
        print('Finished shuffling, starting playback...')
  
    # Start with the currently playing song, since it won't show up
    # in the queue
    trackOrder = [spotify.queue()["currently_playing"]["uri"]]

    while True:
        # Grab the queue and convert to the song IDs
        queue = spotify.queue()["queue"]
        newIDs = [q["uri"] for q in queue]

        # Add to the running list
        trackOrder += newIDs
        #print([q["name"] for q in queue])
        if debug:
            print(f'Skipped through {len(trackOrder)}/{numTracks} tracks.')

        # Break if we have enough items
        # Note that sometimes the queued items
        # overshoot the actual number; we correct for
        # this below.
        if numTracks - len(trackOrder) <= 0:
            break

        # Skip X tracks ahead
        for i in range(len(queue)):
            spotify.next_track()
            time.sleep(.3)

        # Not sure if you actually need to sleep here, but I don't
        # want to overload the server with requests...
        time.sleep(10)

    # Optionally convert to indices in the playlist
    if returnIndices:
        conversionDict = dict(zip(uriList, np.arange(totalNumTracks)))
        trackOrder = np.array([conversionDict.get(t, -1) for t in trackOrder])
   
    # Take only the specified number of items
    trackOrder = trackOrder[:numTracks]

    # There seems to be a bug with spotipy (or maybe the upstream api)
    # where if your queue has less than a certain number of items,
    # it starts the repeating the few items several times. The way to
    # handle this is to look for the first element in our list, and
    # if it shows up twice, we cut off just before there.
    firstItemIndex = np.where(trackOrder == trackOrder[0])[0]
    if len(firstItemIndex) > 1:
        trackOrder = trackOrder[:firstItemIndex[1]]

    return trackOrder
