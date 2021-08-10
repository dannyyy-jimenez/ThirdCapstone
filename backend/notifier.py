from twilio.rest import Client
import os
from dotenv import load_dotenv
from exponent_server_sdk import (
    DeviceNotRegisteredError,
    PushClient,
    PushMessage,
    PushServerError,
    PushTicketError,
)
from requests.exceptions import ConnectionError, HTTPError
load_dotenv()
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']

client = Client(account_sid, auth_token)

default_contact_message = "Hello! You've been added as an emergency contact on Phonetix. You will now be notified when your contact is near danger!"


def SMS(phone_number, body):
    """SMS Notifier using twilio

    Parameters
    ----------
    phone_number : string
        phone number of person to notify
    body : string
        message

    Returns
    -------
    object
        twilio sid object

    """
    message = client.messages.create(body=body, from_='+12067361146', to=f'+1{phone_number}')

    return message.sid


# Basic arguments. You should extend this function with the push features you
# want to use, or simply pass in a `PushMessage` object.
def notif(token, message, extra=None):
    """Notify

    Parameters
    ----------
    token : string
        Expo Push Notification Token
    message : string
        message body
    extra : object
        extra info

    Returns
    -------
    None
    """
    if token is None:
        return

    try:
        response = PushClient().publish(
            PushMessage(to=token,
                        body=message,
                        data=extra))
    except PushServerError as exc:
        raise
    except (ConnectionError, HTTPError) as exc:
        # Encountered some Connection or HTTP error - retry a few times in
        # case it is transient.
        raise
    try:
        # We got a response back, but we don't know whether it's an error yet.
        # This call raises errors so we can handle them with normal exception
        # flows.
        response.validate_response()
    except DeviceNotRegisteredError:
        # Mark the push token as inactive
        from notifications.models import PushToken
        PushToken.objects.filter(token=token).update(active=False)
    except PushTicketError as exc:
        raise
