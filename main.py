from kiteconnect import KiteConnect
import json

if __name__ == '__main__':
    api_key = '9kxpk63gv7k76agn'
    api_secret = 'zkgtjepzhmwrmpd9rmhgecoeuntxfk54'
    file_name = "data/user_data.json"
    kite = KiteConnect(api_key=api_key)
    # Show the login url
    print(kite.login_url())
    # request access token
    data = kite.generate_session("Db415aaZegcYqglk78R1zk11qTWeQfzn", api_secret=api_secret)
    print(data)
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True, default=str)
    #kite.set_access_token(data["access_token"])
    #out = kite.instruments()
    #with open(file_name, 'w') as outfile:
    #    json.dump(out, outfile, indent=4, sort_keys=True, default=str)



