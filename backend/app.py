import json
from flask import Flask, request
from flask_cors import CORS
from db_connection import Connection
import notifier
from wavv import Wavv

app = Flask(__name__)
CORS(app)

db = Connection(__name__ == '__main__')

# Route for flask


@app.route('/')
def home():
    return "ok"

# Load the contacts from the database


@app.route('/api/contacts')
def get_contacts():
    username = request.args.get('username')
    contacts = []
    user_id_query = db.execute(f"""
        SELECT ID FROM Users
        WHERE username = '{username}' """, True)[0][0]

    query = db.execute(f"""
        SELECT username, phone_number FROM Users
        JOIN Contacts
        ON Contacts.A = {user_id_query}
        WHERE Users.ID = Contacts.B """,  True)

    for contact in query:
        username, phone_number = contact

        if username is None:
            contacts.append(phone_number)
        else:
            contacts.append(username)

    # SQL RETURN CONTACTS
    return Respond({"contacts": contacts})


@app.route('/api/login', methods=["POST"])
def login():
    username = request.form.get('username')
    phone_number = request.form.get('phoneNumber')
    notif_token = request.form.get('notif_token')

    query = db.execute(f"""
        SELECT * FROM Users
        WHERE username = '{username}'""", True)

    if len(query) == 0:
        db.execute(f"""
            INSERT INTO Users (username, phone_number, socket)
            VALUES ('{username}', '{phone_number}', '{notif_token}') """)

        print("Inserted!")
    else:
        db.execute(f"""
            UPDATE Users
            SET socket = '{notif_token}', phone_number = '{phone_number}'
            WHERE username = '{username}' """)

        print("Found User So Updated!")

    db.conn.commit()

    # SQL ADD NEW USER TO DB AND THE EXPO NOTIF TOKEN
    return Respond({})


# When danger is reported


@app.route('/api/danger', methods=["POST"])
def danger(username=None, type="Self Reported"):
    if username is None:
        username = request.form.get('username')
    user_id_query = db.execute(f"""
        SELECT ID FROM Users
        WHERE username = '{username}' """, True)[0][0]

    contacts_query = db.execute(f"""
        SELECT username, socket, phone_number FROM Users
        JOIN Contacts
        ON Contacts.A = {user_id_query}
        WHERE Users.ID = Contacts.B """,  True)

    for contact in contacts_query:
        body = f'Heads up! Danger reported near {username}! Type: {type}'
        contact_username,  contact_socket, contact_phone_number = contact

        if contact_username is None or contact_socket is None:
            # notifier.SMS(contact_phone_number, body)
            return Respond({})
        else:
            notifier.notif(contact_socket, body)

    return Respond({})


# Analyze incoming audio data


@app.route('/api/analyze', methods=["POST"])
def analyze():
    username = request.form.get('username')
    audio = request.form.get('audio')
    wavv = Wavv(audio)
    prediction_idx, prediction_name, endangered = wavv.predict()

    if endangered:
        danger(username, prediction_name)

    return Respond({"prediction": prediction_name, "idx": prediction_idx})


@app.route('/api/add_contact', methods=["POST"])
def add_contact():
    username = request.form.get('username')
    contact_phone = request.form.get('phone_number')

    # Add new user to db if it doesnt exist

    query = db.execute(f"""
        SELECT username FROM Users
        WHERE phone_number = '{contact_phone}' """, True)

    if len(query) == 0:
        #  create user with blank username
        db.execute(f"""
            INSERT INTO Users (phone_number)
            VALUES ('{contact_phone}')  """)

    user_id_query = db.execute(f"""
        SELECT ID FROM Users
        WHERE username = '{username}' """, True)[0][0]

    contact_id_query = db.execute(f"""
        SELECT ID FROM Users
        WHERE phone_number = '{contact_phone}' """, True)[0][0]

    multi_contact_check_query = db.execute(f"""
        SELECT id FROM Contacts
        WHERE A = '{user_id_query}' AND B = '{contact_id_query}' """, True)

    # check that a contact association isnt already created

    if len(multi_contact_check_query) == 0:
        db.execute(f"""
            INSERT INTO Contacts (A, B)
            VALUES ({user_id_query}, {contact_id_query}) """)

    db.conn.commit()

    notifier.SMS(contact_phone, notifier.default_contact_message)

    return Respond({})


def Respond(body, error=False):
    return {'_hE': error, '_body': body}


if __name__ == '__main__':
    app.run(port=8080, host='10.0.0.102', debug=True)
