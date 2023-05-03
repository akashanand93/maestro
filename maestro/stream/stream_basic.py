import datetime
import logging
from kiteconnect import KiteTicker, KiteConnect
from copy import copy
import multiprocessing
import json
import pandas as pd

logging.basicConfig(level=logging.INFO)


def run_initial_setup():
    # Some setting up
    api_key = '9kxpk63gv7k76agn'
    api_secret = 'zkgtjepzhmwrmpd9rmhgecoeuntxfk54'
    kite = KiteConnect(api_key=api_key)
    # Show the login url
    print("Goto this url and get the request token from the redirect link: {}".format(kite.login_url()))
    # Take user input for request token
    request_token = input("Enter the request token: ")
    request_token = str(request_token)
    # Get the access token
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    return api_key, access_token


def on_order_update(ws, data):
    logging.debug("Order update : {}".format(data))


def on_reconnect(ws, attempts_count):
    logging.info("Reconnecting: {}".format(attempts_count))


class FileWriter:
    def __init__(self, log_file_dir=None, queue=None):
        if log_file_dir is None:
            self.log_file = "data/ticker_log_{}.log".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            self.log_file = log_file_dir + "ticker_log_{}.log".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        logging.info("Logging to file: {}".format(self.log_file))
        self.queue = queue

    def start_writing(self):
        """
        continue to listen for messages on the queue and writes to file when receive one
        if it receives a '#done#' message it will exit
        """
        with open(self.log_file, 'w') as f:
            while True:
                m = self.queue.get()
                if m == '#done#':
                    break
                f.write("Ticks: {}".format(m))
                f.write('\n')


def get_instrument_tokens():
    """
    :return: list of instrument tokens
    """
    instruments_file = "/Users/akashanand/repo/maestro/data/instruments.json"
    exchange = "NSE"
    with open(instruments_file, 'r') as infile:
        instruments = json.load(infile)
    instruments_df = pd.DataFrame(instruments)
    # Return instrument_token list for exchange filer
    out = instruments_df.loc[instruments_df["exchange"] == exchange, "instrument_token"].tolist()
    out.sort()
    return out

def divide_instruments_to_workers(instrument_tokens, num_workers):
    """
    :param instrument_tokens: list of instrument tokens
    :param num_workers: number of workers
    :return: list of lists of instrument tokens
    """
    out = []
    for i in range(num_workers):
        out.append([])
    for i, instrument_token in enumerate(instrument_tokens):
        out[i % num_workers].append(instrument_token)
    max_instruments_per_worker = max([len(x) for x in out])
    logging.info("Max instruments per worker: {}".format(max_instruments_per_worker))
    MAX_PERMISSIBLE_INSTRUMENTS_PER_WORKER = 3000
    if max_instruments_per_worker > MAX_PERMISSIBLE_INSTRUMENTS_PER_WORKER:
        raise ValueError("Max instruments per worker is greater than permissible limit: {}".format(
            MAX_PERMISSIBLE_INSTRUMENTS_PER_WORKER))
    return out


class TickerWebHandle:
    def __init__(self, web_socket, list_of_instruments, worker_id=0, queue=None):
        self.list_of_instruments = list_of_instruments
        self.web_socket = web_socket
        self.worker_id = worker_id
        self.queue = queue

    def on_connect(self, ws, response):
        logging.info("Successfully connected. Response: {}".format(response))
        ws.subscribe(self.list_of_instruments)
        ws.set_mode(ws.MODE_FULL, self.list_of_instruments)
        logging.info("Subscribed to tokens in full mode. Number of tokens: {} for worker {}".format(
            len(self.list_of_instruments), self.worker_id))

    def on_ticks(self, ws, ticks):  # noqa
        # Callback to receive ticks.
        # logging.info("Ticks: {}".format(ticks))
        self.queue.put(ticks)

    def on_close(self, ws, code, reason):
        logging.info("Connection closed: Worker id {worker} {code} - {reason}".format(worker=self.worker_id,
                                                                                      code=code, reason=reason))

    def on_error(self, ws, code, reason):
        logging.error("Connection Error: Worker id {worker} {code} - {reason}".format(worker=self.worker_id,
                                                                                      code=code, reason=reason))

    def on_no_reconnect(self, ws):
        logging.info("Reconnecting failed. Stopping. Worker id {}".format(self.worker_id))

    def start(self):
        # Assign the callbacks.
        self.web_socket.on_ticks = self.on_ticks
        self.web_socket.on_connect = self.on_connect
        self.web_socket.on_order_update = on_order_update
        self.web_socket.on_close = self.on_close
        self.web_socket.on_error = self.on_error
        self.web_socket.on_reconnect = on_reconnect
        self.web_socket.on_no_reconnect = self.on_no_reconnect

        # Infinite loop on the main thread. Nothing after this will run.
        # You have to use the pre-defined callbacks to manage subscriptions.
        self.web_socket.connect()


if __name__ == "__main__":
    # initial setup
    num_sockets = 2
    # Get access token and api key
    api_key, access_token = run_initial_setup()
    # Set up logging file
    msg_queue = multiprocessing.Queue()
    file_writer = FileWriter("/Users/akashanand/repo/maestro/data/tick_data/", msg_queue)

    # Get the instruments which we want to listen to
    instrument_tokens = get_instrument_tokens()
    logging.info("Number of instruments: {}".format(len(instrument_tokens)))
    token_lists = divide_instruments_to_workers(instrument_tokens, num_sockets)
    # Initialise websockets
    kws = KiteTicker(api_key, access_token)
    kws_objs = [kws]
    for i in range(num_sockets - 1):
        kws_objs.append(copy(kws))
    token_lists = [[738561], [5633]]
    tick_handles = []
    assert len(kws_objs) == len(token_lists)
    for idx, (kws, token_list) in enumerate(zip(kws_objs, token_lists)):
        tick_handles.append(TickerWebHandle(web_socket=kws, list_of_instruments=token_list, worker_id=idx,
                                            queue=msg_queue))

    processes = []
    for tick_handle in tick_handles:
        proc = multiprocessing.Process(target=tick_handle.start)
        processes.append(proc)
        proc.start()

    # Start writing to file
    file_writer.start_writing()

    # Complete the processes
    for proc in processes:
        proc.join()
