from kiteconnect import KiteConnect
import logging

# Some setting up
API_KEY = '9kxpk63gv7k76agn'
API_SECRET = 'zkgtjepzhmwrmpd9rmhgecoeuntxfk54'

class KiteConnector(object):
    def __init__(self):
        logging.debug("API_KEY: {}".format(API_KEY))
        logging.debug("API_SECRET: {}".format(API_SECRET))
        kite = KiteConnect(api_key=API_KEY)
        # Show the login url
        print("Goto this url and get the request token from the redirect link: {}".format(kite.login_url()), flush=True)
        # Take user input for request token
        request_token = input("Enter the request token: ")
        request_token = str(request_token)
        # Get the access token
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        self.api_key = API_KEY
        self.access_token = access_token
        self.trader = kite
        logging.debug("The connection is established successfully.")
        logging.debug("Access token: {}".format(access_token))
        logging.debug("Output of generate_session: {}".format(data))

        
