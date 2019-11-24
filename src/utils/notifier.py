import slack
import pprint


def slack_message(msg, info):
    with open('../../slack.tkn', 'r') as f:
        client = slack.WebClient(token=f.read())
        client.chat_postMessage(channel='#private_nn', text='```' + str(msg) + '```' + '\n\n' + pprint.pformat(info))
