# UdG - Seminar 

Prerequisites:
* python 2.7
* pip 9.0.1

To execute the python notebooks in this repository, you should follow the instructions below in a Ubuntu machine. 


1. Install the python requirements `sudo pip install -r requirements.txt` in your physical machine or in a virtual environemnt like `virtualenv`. 
2. Register your twitter app at [apps.twitter.com](https://apps.twitter.com/), get the CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET and update the configuration file `conf/app.json`.
3. Register your google maps app at [developers.google.com/maps](https://developers.google.com/maps/), get the API_KEY and update the configuration file `conf/maps.json`.
4. Activate the widgets extension in jupyter `jupyter nbextension enable --py --sys-prefix widgetsnbextension`
5. Enable the gmaps extension in jupyter `jupyter nbextension enable --py gmaps`
4. Run `jupyter notebook`. This will open the web browser and redirect to localhost:8888
5. Execute first the python notebook called `01_Twitter_crawler.ipynb` to obtain some tweets from Twitter and store them at `data/tweets.json`.