from flask import Flask, render_template, request, Markup

import threading
import numpy as np

import DataProcessing as da

spotify = da.authenticate()

app = Flask('SpotifyWebStats')

songRef = ''

previousFig = None

@app.route('/', methods=('GET', 'POST'))
def home():

    if request.method == 'POST':
        # If we have a POST, it was presumably because
        # we are redirected from searching a username, 
        # so we should show some results
        username = request.form.get("username", None)

        if username:
            # If we have a valid username, we need to prepare
            # a dictionary to pass information back to display
            # for the user
            data = {}
            data["username"] = username
            data["song-result"] = songRef
          
            # It's possible that the username will not exist,
            # so we need to wrap in try/except
            playlists = None
            try:
                playlists = da.fetchPublicPlaylists(username)
            except Exception as e:
                print(e)
                # If this failed, that means the given username
                # doesn't exist
                # This invalid page is just a generic error page, so
                # we have to define the error message
                data["error_msg"] = f'I don\'t think the username {username} exists... \n Maybe try another one :)'

                return render_template('invalid.html', data=data)

            # If we've made it here, we have properly identified
            # some playlists to analyze

            # Save the names of playlists to display to the user
            data["playlists"] = list(playlists["name"])

            # If options have been changed and this is recalculating
            # there will be some options we have to read in
            reducedDims = int(request.form.get('dims', '3')) # '2' or '3'
            method = request.form.get('decomp_method', 'pca') # 'pca' or 'lda'

            if int(request.form.get('update_page', '0')):
                activePlaylistIndices = np.array([i for i in range(playlists.shape[0]) if playlists["name"][i] in request.form.keys()])
            else:
                data["figure_markup"] = 'Select the desired playlist and options on the left, and then press \'Update\'.'

                return render_template('results.html', data=data)
                #activePlaylistIndices = np.arange(playlists.shape[0])

            if len(activePlaylistIndices) == 0:
                data["error_msg"] = f'No playlists were selected!'

                return render_template('invalid.html', data=data)


            if len(activePlaylistIndices) <= reducedDims and method == 'lda':
                data["error_msg"] = f'LDA requires at least 4 playlists!'

                return render_template('invalid.html', data=data)

            #print(playlists.iloc[activePlaylistIndices])

            # DEBUG
            #data["figure_markup"] = 'Debug'
            #return render_template('results.html', data=data)

            # Merge all of the tracks into a single phase space of audio features
            # NOTE: I used to use the "id" field, but it seems that has been replaced with "uri"
            phaseSpace, identities, featureNames = da.mergePlaylistTracks(playlists.iloc[activePlaylistIndices]["uri"])

            # DEBUG: just start with 2d PCA
            if method == 'pca':
                decompCoords, pcaComponents = da.computePCA(phaseSpace, reducedDims)

                # Interpret the decomposed axes
                interpretedLabels = da.interpretDecomposedCoordinates(pcaComponents, featureNames)

            elif method == 'lda':  
                decompCoords = da.computeLDA(phaseSpace, identities, reducedDims)


            # By default, the identities array uses the uri, but the
            # names are much more readable, so we want to convert to names
            idToName = dict(zip(playlists["uri"], playlists["name"]))

            # Generate the figure
            fig = da.decompositionHull(decompCoords, identities, idToName, 0.05)
           
            if method == 'pca':
                if reducedDims == 2:
                    fig.update_xaxes(title=interpretedLabels[0])
                    fig.update_yaxes(title=interpretedLabels[1])

                else:
                    fig.update_layout(scene=dict(
                                      xaxis_title=interpretedLabels[0],
                                      yaxis_title=interpretedLabels[1],
                                      zaxis_title=interpretedLabels[2]))

            fig.update_layout(showlegend=True,
                              margin={'t':10, 'b':10, 'l':10, 'r':10},
                              plot_bgcolor="#FFFFFF",
                              paper_bgcolor="#FEFEFE")
            # Make it look nice
            #plt.xlabel(interpretedLabels[0])
            #plt.ylabel(interpretedLabels[1])
            #plt.legend(fontsize=12)

            # Now convert to an interactive html embed
            #htmlString = mpld3.fig_to_html(fig)
            #plt.close()

            #with open('temp.html', 'w') as f:
            #    f.write(htmlString)

            data["figure_markup"] = Markup(fig.to_html(include_plotlyjs=True))
            previousFig = data["figure_markup"]

            return render_template('results.html', data=data)

    # If it is just a get request, we just show the landing
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET', 'POST'])
def autocomplete():
    searchTerm = request.form.get('song-search')

    print(request.form)

    searchResults = spotify.search(searchTerm, type='track', limit=5)

    songNames = [s["name"] for s in searchResults["tracks"]["items"]]

    data = {}
    data["song"] = songNames[0]

    print(songNames)

    if previousFig:
        data["figure_markup"] = previousFig 

    return render_template('results.html', data=data)

def testFunc(arg):
    return list(str(arg))

if __name__ == '__main__':
    app.run()
