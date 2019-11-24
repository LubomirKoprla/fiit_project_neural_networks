import slack
import pprint
from time import time


def slack_error_message(run, start_time, msg, info):
    with open('../../slack.tkn', 'r') as f:
        client = slack.WebClient(token=f.read())
        client.chat_postMessage(channel='#nn_errors', text='*RUN IN BATCH: #' + str(run) + '* (duration: '
                                                            + str(round((time() - start_time) / 60, 2)) + ' min)\n'
                                                            + '```' + str(msg) + '```' + '\n'
                                                            + pprint.pformat(info))


def slack_info_message(run, start_time, msg, info):
    with open('../../slack.tkn', 'r') as f:
        client = slack.WebClient(token=f.read())
        client.chat_postMessage(channel='#nn_info', text='*RUN IN BATCH: #' + str(run)+ '* (duration: '
                                                            + str(round((time() - start_time) / 60, 2)) + ' min)\n'
                                                            + '`' + str(msg) + '`' + '\n'
                                                            + pprint.pformat(info))
