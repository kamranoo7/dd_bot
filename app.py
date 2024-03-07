from flask import Flask, request
from google.api_core.exceptions import BadRequest
import requests
import PyPDF2
from io import StringIO
from pdfminer.layout import LAParams
from pdfminer.high_level import extract_text_to_fp
from pdfminer.high_level import extract_text
import glob
import docx2txt
import iso8601  # This library helps to parse datetime strings to datetime objects
import shutil
import zipfile
import logging
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from flask import Flask, request, render_template, send_file
from datetime import datetime
import time
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
    Response,
)
import weaviate
import openai
from werkzeug.security import generate_password_hash, check_password_hash
import csv
import gspread
from google.oauth2.service_account import Credentials
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import io
import re
import threading
import random
from flask import Flask, request, jsonify
from flask_mail import Mail, Message
import secrets
import uuid
import os
from dotenv import load_dotenv
import ast

app = Flask(__name__)

mail = Mail(app)
app.secret_key = "JobBot"


load_dotenv()


app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587  # Use 465 for SSL
# Ensure correct email
app.config["MAIL_USERNAME"] = "iamkamrankhan00@gmail.com"
app.config["MAIL_PASSWORD"] = "zjwj qhxq askj mxbh"
app.config["MAIL_USE_TLS"] = True  # Set to False if using SSL
app.config["MAIL_USE_SSL"] = False  # Set to True if using SSL
########################################################
global lm_client
lm_client = None

global layer_1
layer_1 = None

global layer_2
layer_2 = None

global openai_flag
openai_flag = True

global layer_1_flag
layer_1_flag = True

global layer_2_flag
layer_2_flag = True

global error_admin_msg
error_admin_msg = "error."
def load_configuration1():
    default_config1 = {
        "openaiKey": "your-default-key",
        "layer1URL": "default layer-1 Url",
        "layer2URL": "default layer-2 url",
        "layer1AuthKey": "default-layer1-auth-key",
        "layer2AuthKey": "default-layer2-auth-key",
        "className1": "default-class-name-1",
        "className2": "default-class-name-2",
    }

    try:
        with open('config1.json', 'r') as file:
            config_data1 = json.load(file)
    except FileNotFoundError:
        config_data1 = default_config1

    return config_data1
def load_configuration():
    default_config = {
        "openaiKey": "your-default-key",
        "layer1URL": "default layer-1 Url",
        "layer2URL": "default layer-2 url",
        "layer1AuthKey": "default-layer1-auth-key",
        "layer2AuthKey": "default-layer2-auth-key",
        "className1": "default-class-name-1",
        "className2": "default-class-name-2",
    }

    try:
        with open('config.json', 'r') as file:
            config_data = json.load(file)
    except FileNotFoundError:
        config_data = default_config

    return config_data



def initiate_clients():
    global openai_flag, layer_1_flag, layer_2_flag, lm_client, layer_1, layer_2, error_admin_msg
    print("hello are u there")
    config_data = load_configuration()
    config_data1=load_configuration1()
    print(config_data,"sss's's's's's's's''s's's's'ss's's'")
    print(config_data1,"qagahghhhhhhhhhhhhhhhhhhhhhhhhhhh")
    try:
        openai.api_key = config_data['openaiKey']
        print(config_data['openaiKey'])
        lm_client = openai.OpenAI(api_key=config_data["openaiKey"])
        msg = [
            {'role': 'system', 'content': "system_message"},
            {'role': 'user', 'content': "user_message"}
        ]

        response = lm_client.chat.completions.create(
            model="gpt-4",
            messages=msg, max_tokens=1000, temperature=0.0,
        )
        openai_flag = False
        print("lm_client working")
    except:
        openai_flag = True
        print("lm_client not working")

    try:
        layer_1 = weaviate.Client(
     url=config_data['layer1URL'],
      auth_client_secret=weaviate.AuthApiKey(api_key=config_data["layer1AuthKey"]),
     additional_headers={"X-OpenAI-Api-Key": config_data["openaiKey"]}
 )
        print("layer 1 working",layer_1)
        layer_1_flag = False
    except:
        layer_1_flag = True
        

    try:
        layer_2 = weaviate.Client(
            url=config_data1['layer2URL'],
            
      auth_client_secret=weaviate.AuthApiKey(api_key=config_data1["layer2AuthKey"]),
            additional_headers={"X-OpenAI-Api-Key": config_data["openaiKey"]}
        )
        layer_2_flag = False
        print("layer 2 working")
    except:
        layer_2_flag = True

    error_admin_msg = ""
    if openai_flag:
        error_admin_msg += "Please check your OpenAI API KEY.\n"
    if layer_1_flag:
        error_admin_msg += "Please check your Layer_1 API KEY.\n"
    if layer_2_flag:
        error_admin_msg += "Please check your Layer_2 API KEY.\n"


initiate_clients()
def update_data(openaiKey,layer1URL, layer2URL, layer1AuthKey, layer2AuthKey, className1, className2):
    global config_data, layer_1, layer_2
    
    # Validate the OpenAI key
    config_data = {
        "openaiKey":openaiKey,
        "layer1URL": layer1URL,
        "layer2URL": layer2URL,
        "layer1AuthKey": layer1AuthKey,
        "layer2AuthKey": layer2AuthKey,
        "className1": className1,
        "className2": className2,
    }

    # Save the updated configuration to 'config.json'
    with open('config.json', 'w') as file:
        json.dump(config_data, file, indent=4)  # Using 4 spaces for JSON indentation for readability
    print(config_data, "config_datatatatatatat")

    # Re-initiate clients with updated configuration
    initiate_clients()

    return render_template('chat.html')

    
#for layer_1
@app.route('/update_config', methods=['POST'])
def update_config():
    global config_data, layer_1, layer_2

    data = request.form

    

    # Validate the OpenAI key
    config_data = {
        "openaiKey": data["openaiKey"],
        "layer1URL": data["layer1URL"],
        
        "layer1AuthKey": data["layer1AuthKey"],
        
        "className1": data["className1"],
        
    }
    openai.api_key = config_data['openaiKey']

    # Save the updated configuration to 'config.json'
    with open('config.json', 'w') as file:
        json.dump(config_data, file, indent=2)
    print(config_data,"config_datatatatatatat")
    # Re-initiate clients with updated configuration
    initiate_clients()
    print(config_data,"done updating thing ssssssssssssssssssssssssssss")
    return render_template('chat.html')

#for layer_2

@app.route('/update_config1', methods=['POST'])
def update_config1():
    global config_data1, layer_1, layer_2

    data = request.form

    

    # Validate the OpenAI key
    config_data1 = {
        
        "layer2URL": data["layer2URL"],
        
        "layer2AuthKey": data["layer2AuthKey"],
        
        "className2": data["className2"],
        
    }
    

    # Save the updated configuration to 'config.json'
    with open('config1.json', 'w') as file:
        json.dump(config_data1, file, indent=2)
    print(config_data1,"config_datatatatatatat")
    # Re-initiate clients with updated configuration
    initiate_clients()
    print(config_data1,"done updating thing ssssssssssssssssssssssssssss")
    return render_template('chat.html')


mail.init_app(app)

global audio_speech
audio_speech = None

global projectName
projectName = "Bot"

global p1
p1 = "provide the answer only in the context of Capria Global  South Fund II"

global p2
p2 = "provide the answer only in the context of Capria Global  South Fund II"

global stop_flag
stop_flag = False

global citations_dictt
citations_dictt = {}
citations_dictt["Capria - Applied GenAI Strategy"] = "https://drive.google.com/file/d/1zfU7DgCqqAISqPvoFrBu2mSMy-8J52tp/view?usp=sharing"
citations_dictt["Capria Global South Fund II - Impact & ESG"] = "https://drive.google.com/file/d/1Rgg6tRebv5n2yabFJDvDuH0N6rwyZrS5/view?usp=sharing"
citations_dictt["Capria Global South Fund II - Investor Long Deck"] = "https://drive.google.com/file/d/1GQO6xuXXjq1pprI1UZe9ekaz39ja6ju8/view?usp=sharing"
citations_dictt["Capria Global South Fund II - People Details"] = "https://drive.google.com/file/d/16OCBAkxDtaJfdEh-E35bXXNj11ww8UP8/view?usp=sharing"
citations_dictt["Capria Ventures Org Chart"] = "https://drive.google.com/file/d/1xl77HtQkwdQW7Se543FUqAzPcTPKQNvd/view?usp=sharing"
citations_dictt["CV Funds Performance & Track Record"] = "https://drive.google.com/file/d/1vX7VuWfaQhLEC5EZKBFWLy2527lRg48K/view?usp=sharing"
citations_dictt['DDQ for Capria Global South Fund II'] = "https://drive.google.com/file/d/15teXiXva51irbEPUFPamTqMRQRVMC-4U/view?usp=sharing"


@app.route("/send_verification", methods=["POST"])
def send_verification():
    data = request.get_json()  # Parse JSON data from the request
    email = data.get("email")  # Get the email from the parsed JSON
    if not email:
        return jsonify({"message": "No email provided"}), 400

    otp = random.randint(100000, 999999)
    session[email] = otp
    print(otp, "otp sent ")

    msg = Message(
        "Email Verification", sender="iamkamrankhan00@gmail.com", recipients=[email]
    )
    msg.body = f"Your verification code is: {otp}"
    mail.send(msg)

    return jsonify({"message": "Verification email sent"})


def replace_in_string(text):
    chars_to_replace = ['\\', '/', '_']
    for char in chars_to_replace:
        text = text.replace(char, ' ')
    text = text.replace('.png', '')
    return text


@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")
    print(otp, "otp ")
    stored_otp = session.get(email)
    print(stored_otp, "stored_otp")

    if stored_otp and str(stored_otp) == str(otp):
        return jsonify({"message": "Email verified successfully"})
    else:
        return jsonify({"message": "Invalid or expired OTP"}), 400




service_account_file = "cred.json"
credentials = service_account.Credentials.from_service_account_file(
    service_account_file
)
dbclient = bigquery.Client(credentials=credentials,
                           project=credentials.project_id)
credentials_path = "credentials.json"
response = "Done."
app.secret_key = "secret_key"

global intro
intro = """
Welcome to the alpha version of Capria's DD Copilot. Trying asking "What is the MOFC for Betterplace" or "Does Capria invest in genai infrastructure?" or "Does Capria consider DEI" or "Who is Capria's tax advisor"
"""


def add_row_to_sheet(data, sheet_id):
    creds = Credentials.from_service_account_file(
        credentials_path, scopes=[
            "https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)

    try:
        sheet = client.open_by_key(sheet_id)
        worksheet = sheet.get_worksheet(0)
        worksheet.append_row(data)
        print("Row added successfully!")
    except Exception as e:
        print("Error: ", e)


# layer_1 = weaviate.Client(
#     url="https://ddbot-a9bg5ni9.weaviate.network",
#     # auth_client_secret=weaviate.AuthApiKey(api    _key="vvZhMKvXbg2iETqRZ1EyLJ8302jWE436t2oG"),
#     additional_headers={"X-OpenAI-Api-Key": key}
# )


def get_user_by_userID(userID):
    try:

        dataset_name = "my-project-41692-400512.jobbot"
        table_name = "users"

        query = """
            SELECT *
            FROM `{0}.{1}`
            WHERE userID = @userID
        """.format(
            dataset_name, table_name
        )

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter(
                "userID", "STRING", userID)]
        )

        query_job = dbclient.query(
            query, job_config=job_config)  # Make an API request

        results = query_job.result()  # Waits for job to complete

        for row in results:
            return {"name": row.name, "password": row.password, "userID": row.userID, 'isAnswer': row.isAnswer, 'role': row.role}

        return None

    except Exception as e:
        print(f"An error occurred while retrieving user: {str(e)}")
        return None


@app.route("/control_panel", methods=["GET", "POST"])
def control_panel():
    global intro, projectName, p1, p2
    if request.method == "POST":
        session["language"] = request.form.get("language", "english")
        session["language"] = get_language(session["language"])
        projectName = request.form.get("title", "Bot")
        session["intro"] = request.form.get("intro", "")
        intro = session["intro"] if session["intro"] != "" else intro

        p1 = request.form.get("prompt_level1", "")
        p2 = request.form.get("prompt_level2", "")

    print("\n\nLanguage being retuned:", session["language"], "\n\n")
    userID = session["userID"]
    user = get_user_by_userID(userID)
    print("users---------", user)
    role = user["role"]
    return render_template(
        "control_panel.html",
        language=session.get("language", "english"),
        intro=intro,
        prompt_level1=p1,
        prompt_level2=p2,
        prompt_level3=p1,
        role=role,
        name=projectName,
    )


@app.route("/trans")
def trans():
    global intro, projectName
    print("/trans-------")
    newList = []
    transwords = [
        projectName,
        "User",
        "Enter Your Query",
        "Feedback",
        "Fast Mode",
        intro,
        "Check level 2?",
        "Submit",
        "Close",
        "Slow Mode",
        "Groq Mode",
    ]

    if session["language"] != "en":
        for item in transwords:
            trans = translate_text(item, session["language"])
            print(trans, "transscripted words ")
            newList.append(trans)

    if not newList:
        newList = transwords

    print(newList, "---------------trans words ")
    return jsonify(newList)


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    try:
        audio_file = request.files["audioFile"]
        if audio_file:
            audio_bytes = audio_file.read()
            audio_file = FileWithNames(audio_bytes)
            session["greet"], session["language"] = transcribe(audio_file)
            session["language"] = (
                request.form["language"]
                if request.form["language"] != "auto"
                else session["language"]
            )
            session["language"] = get_language(session["language"])
            print("\n\n\n", session["language"], "\n\n\n")

            print("\n\n\n", session["language"], "\n\n\n")
    except Exception as e:
        print(e)
        update_logs(e)
        session["language"] = "english"

    return jsonify({"channel": "chat"})


@app.route("/chat")
def chat():
    if "language" not in session or session["language"] is None:
        session["language"] = "en"
        print("Language set to English")
    print("Redirecting...")
    return render_template("chat.html")


@app.route("/")
def index():
    return render_template("signup.html")


custom_functions = [
    {
        "name": "return_response",
        "description": "Function to be used to return the response to the question, and a boolean value indicating if the information given was suffieicnet to generate the entire answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_list": {
                    "type": "array",
                    "description": "List of chunk ids. ONLY the ones used to generate the response to the question being asked. return the id only if the info was used in the response. think carefully.",
                    "items": {"type": "integer"},
                },
                "response": {
                    "type": "string",
                    "description": "This should be the answer that was generated from the context, given the question",
                },
                "sufficient": {
                    "type": "boolean",
                    "description": "This should represent wether the information present in the context was sufficent to answer the question. Return True is it was, else False.",
                },
            },
            "required": ["response", "sufficient", "item_list"],
        },
    }
]

custom_functions_1 = [
    {
        "name": "return_response",
        "description": "Function to be used to return the response to the question, and a boolean value indicating if the information given was suffieicnet to generate the entire answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "boolean",
                    "description": "This should be the answer that was generated from the context, given the question",
                },
                "sufficient": {
                    "type": "boolean",
                    "description": "This should represent wether the information present in the context was sufficent to answer the question. Return True is it was, else False.",
                },
            },
            "required": ["response", "sufficient"],
        },
    }
]


custom_functionsz = [
    {
        "name": "return_response",
        "description": "Function to be used to return the response to the question, and a boolean value indicating if the information given was suffieicnet to generate the entire answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "response_answer": {
                    "type": "string",
                    "description": "This should be the answer that was generated from the context, given the question",
                },
                "item_list": {
                    "type": "array",
                    "description": "List of chunk ids. ONLY the ones used to generate the response to the question being asked. return the id only if the info was used in the response. think carefully.",
                    "items": {"type": "integer"},
                },
                "sufficient": {
                    "type": "boolean",
                    "description": "This should represent wether the information present in the context was sufficent to answer the question. Return True is it was, else False.",
                },
            },
            "required": ["response_answer", "sufficient", "item_list"],
        },
    }
]


def ask_gpt1(question, context, gpt, metadata, language, addition):
    global audio_speech
    user_message = "Question: \n\n" + question + "\n\n\nContext: \n\n" + context
    system_message = "You will be given context from several pdfs, this context is from several chunks, rettrived from a vector DB. each chunk will have a chunk id above it. You will also be given a question. Formulate an answer, ONLY using the context, and nothing else. provide in-text citations within square brackets at the end of each sentence, right after each fullstop. The citation number represents the chunk id that was used to generate that sentence. Do Not bunch multiple citations in one bracket. Uee seperate brackets for each digit. {} Return the response along with a boolean value indicating if the information from the context was enough to answer the question. Return true if it was, False if it wasnt. Return the response, which is th answer to the question asked".format(
        addition
    )

    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    print("-----------------------")
    response = lm_client.chat.completions.create(
        model=gpt,
        messages=msg,
        max_tokens=500,
        temperature=0.0,
        functions=custom_functionsz,
        function_call="auto",
        stream=True
    )

    string = ""
    response_answer = ""
    item_list = ""
    sufficient = ""

    print("====================")
    for chunk in response:
        try:
            delta = chunk.choices[0].delta.function_call.arguments
            string += delta
            if '"item_list":' in string:
                item_list = string.split('"item_list":')[1]
            elif '"response_answer":' in string:
                response_answer = string.split('"response_answer":')[1]
            data = {"response": response_answer,
                    "sufficient": False, "endOfStream": False}
            json_data = json.dumps(data)
            yield f"data: {json_data}\n\n"
        except Exception as e:
            continue

    item_list = item_list.split(',\n')[0]
    item_list = ast.literal_eval(item_list)
    response = response_answer.replace('"item_list', "")

    # response = translate_text(response, language) if language != "en" else response
    # audio_speech = lm_client.audio.speech.create(model="tts-1", voice="alloy", input=response)
    # audio_speech.stream_to_file('output.mp3')

    for item in item_list:
        response += "\n"
        response += "[{}]".format(item)
        response += replace_in_string(metadata[item])

    lst = '\n\n\n list_of_citations = ["https://www.google.com/search?q=1","https://www.google.com/search?q=2","https://www.google.com/search?q=3","https://www.google.com/search?q=4","https://www.google.com/search?q=5","https://www.google.com/search?q=6", "https://www.google.com/search?q=7","https://www.google.com/search?q=8","https://www.google.com/search?q=9","https://www.google.com/search?q=10","https://www.google.com/search?q=11","https://www.google.com/search?q=12"]'
    response += lst

    data = {"response": response, "sufficient": False, "endOfStream": True}
    json_data = json.dumps(data)
    yield f"data: {json_data}\n\n"


def ask_gpt(question, context, gpt, metadata, language, addition, userid):
    global audio_speech, stop_flag, citations_dictt
    user_message = "Question: \n\n" + question + "\n\n\nContext: \n\n" + context
    system_message = "You will be given context from several pdfs, this context is from several chunks, rettrived from a vector DB. each chunk will have a chunk id above it. You will also be given a question. Formulate an answer, ONLY using the context, and nothing else. provide in-text citations within square brackets at the end of each sentence, right after each fullstop. The citation number represents the chunk id that was used to generate that sentence. Do Not bunch multiple citations in one bracket. Uee seperate brackets for each digit. {} Return the response along with a boolean value indicating if the information from the context was enough to answer the question. Return true if it was, False if it wasnt. Return the response, which is th answer to the question asked".format(
        addition
    )

    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    response = lm_client.chat.completions.create(
        model=gpt,
        messages=msg,
        max_tokens=500,
        temperature=0.0,
        functions=custom_functionsz,
        function_call="auto",
        stream=True,
    )

    string = ""
    response_answer = ""
    item_list = ""
    sufficient = ""

    for chunk in response:
        try:
            delta = chunk.choices[0].delta.function_call.arguments
            string += delta
            if '"item_list":' in string:
                item_list = string.split('"item_list":')[1]
            elif '"response_answer":' in string:
                response_answer = string.split('"response_answer":')[1]

            data = {
                "response": response_answer,
                "sufficient": False,
                "endOfStream": stop_flag,
            }
            stop_flag = False
            json_data = json.dumps(data)
            print("streaming..........")
            yield f"data: {json_data}\n\n"
        except Exception as e:
            print(e)
            continue

    item_list = item_list.split(",\n")[0]
    item_list = ast.literal_eval(item_list)
    response = response_answer.replace('"item_list', "")

    response = translate_text(
        response, language) if language != "en" else response
    # audio_speech = lm_client.audio.speech.create(
    #     model="tts-1", voice="alloy", input=response
    # )
    # filename = str(userid) + ".mp3"
    # audio_speech.stream_to_file(filename)

    cits = ["www.google.com"]*(max(item_list)+4)
    for item in item_list:
        response += "\n"
        response += "[{}]".format(item)
        response += replace_in_string(metadata[item-1])
        for key, value in citations_dictt.items():
            if key in metadata[item-1]:
                cits[item] = value

    lst = '\n\n\n list_of_citations = ' + str(cits)
    # print(lst.replace("'", '"'))
    # lst = '\n\n\n list_of_citations = ["https://www.google.com/search?q=1","https://www.google.com/search?q=2","https://www.google.com/search?q=3","https://www.google.com/search?q=4","https://www.google.com/search?q=5","https://www.google.com/search?q=6", "https://www.google.com/search?q=7","https://www.google.com/search?q=8","https://www.google.com/search?q=9","https://www.google.com/search?q=10","https://www.google.com/search?q=11","https://www.google.com/search?q=12"]'
    # print(lst)

    lst = lst.replace("'", '"')
    response += lst
    print(response)
    data = {"response": response, "sufficient": False, "endOfStream": True}
    json_data = json.dumps(data)
    yield f"data: {json_data}\n\n"


def qdb(query, db_client, name, cname, chunk_id, limit):
    context = None
    metadata = []
    try:
        res = (
            db_client.query.get(name, ["text", "metadata"])
            .with_near_text({"concepts": query})
            .with_limit(limit)
            .do()
        )
        context = ""
        metadata = []
        for i in range(limit):
            context += "Chunk ID: " + str(chunk_id) + "\n"
            context += res["data"]["Get"][cname][i]["text"] + "\n\n"
            print(context, "context..............................................")
            metadata.append(res["data"]["Get"][cname][i]["metadata"])
            chunk_id += 1
    except Exception as e:
        print("Exception in DB, dude.")
        print(e)
        time.sleep(3)
        context, metadata = qdb(query, db_client, name, cname, chunk_id, limit)
    return context, metadata


def check_email_exists(email):
    table_id = "my-project-41692-400512.jobbot.users"
    query = """
    SELECT * 
    FROM `{}`
    WHERE email = @email
    """.format(
        table_id
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("email", "STRING", email),
        ]
    )

    query_job = dbclient.query(query, job_config=job_config)

    results = list(query_job.result())
    return len(results) > 0


def insert_user_data(name, password, email, userID):
    table_id = "my-project-41692-400512.jobbot.users"

    rows_to_insert = [
        {"name": name, "password": password, "email": email,
            "userID": userID, "role": 'user', "isAnswer": False}
    ]

    # Make an API request.
    errors = dbclient.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        userID = str(uuid.uuid4())
        name = request.form.get("name")
        password = request.form.get("password")
        email = request.form.get("email")

        # Debugging line
        print(f"Form data received - Name: {name}, Email: {email}")

        if not name or not password:
            print("Name or password not provided")  # Debugging line
            message = "Name and password are required."
            return render_template("signup.html", message=message)

        if check_email_exists(email):
            print("Name already exists")  # Debugging line
            message = "User already exists."
            return render_template("signup.html", message=message)

        hashed_password = generate_password_hash(password)
        insert_user_data(name, password, email, userID)
        print("User registered successfully")  # Debugging line
        return redirect(url_for("login"))

    return render_template("signup.html")


def update_password(email, new_password):
    hashed_password = generate_password_hash(new_password)

    client = dbclient

    table_id = "my-project-41692-400512.jobbot.users"

    query = f"""
    UPDATE `{table_id}`
    SET password = @new_password
    WHERE email = @email
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "new_password", "STRING", new_password),
            bigquery.ScalarQueryParameter("email", "STRING", email),
        ]
    )

    # Make an API request.
    query_job = client.query(query, job_config=job_config)

    query_job.result()

    return "Password updated successfully."


@app.route("/update_password", methods=["GET", "POST"])
def handle_update_password():
    if request.method == "POST":
        email = request.form.get("email")
        new_password = request.form.get("new_password")

        response = update_password(email, new_password)

        return redirect(
            url_for("login")
        )  # Assuming 'login' is the endpoint for your login page

    return render_template("reset_password.html")


def get_user_by_username(name):
    dataset_name = "my-project-41692-400512.jobbot"
    table_name = "users"

    query = """
        SELECT *
        FROM `{0}.{1}`
        WHERE name = @name
    """.format(
        dataset_name, table_name
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("name", "STRING", name)]
    )

    query_job = dbclient.query(
        query, job_config=job_config)  # Make an API request

    results = query_job.result()  # Waits for job to complete

    for row in results:
        return {"name": row.name, "password": row.password, "userID": row.userID, 'isAnswer': row.isAnswer}

    return None


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            name = request.form["name"]
            password = request.form["password"]

            user = get_user_by_username(name)

            if user and user["password"] == password:
                session["userID"] = user["userID"]
                session['language'] = 'en'
                session.modified = True
                print("User Detail", user)
                if user['isAnswer'] == True:
                    return redirect("/chat")
                else:
                    return redirect("/updateProfile")
            else:
                return "Invalid username or password", 401
        except Exception as e:
            print(e, 'error in login')
            return f"Error: {e}", 500
    return render_template("login.html")


@app.route("/forgot_password", methods=["GET"])
def forgot_password():
    return render_template("forgot_password.html")


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    print(data)
    try:
        unique_id = data.get("uniqueId")
        thumbs = data.get("type", "Text")
        l2 = data.get("l2ResponseClicked")
        l3 = data.get("l3ResponseClicked")
        feedback_text = data.get("feedback", "Null")
        level = data.get("level", "test")
        print(thumbs, feedback_text, level)
        if not l2 and not l3:
            add_row_to_sheet(
                [
                    session["transcription"],
                    session["level_1_response"],
                    thumbs,
                    feedback_text,
                ],
                "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y",
            )
        if l2 and not l3:
            add_row_to_sheet(
                [
                    session["transcription"],
                    session["level_2_response"],
                    thumbs,
                    feedback_text,
                ],
                "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y",
            )
        if l2 and l3:
            add_row_to_sheet(
                [
                    session["transcription"],
                    session["level_3_response"],
                    thumbs,
                    feedback_text,
                ],
                "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y",
            )
    except Exception as e:
        update_logs(e)

    return jsonify({"status": "success"})


def transcribe(audio_file):
    try:
        response = lm_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="verbose_json"
        )
        print(response)
        transcription = response.text
        language = response.text + " " + response.language
    except Exception as e:
        print(e)
        update_logs(e)
        transcription = "Error."
        language = "english"

    return transcription, language


class FileWithNames(io.BytesIO):
    name = "audio.wav"


def update_logs(input_string):
    file_exists = os.path.isfile("logs.txt")

    with open("logs.txt", "a" if file_exists else "w") as file:
        if file_exists:
            file.write("\n\n\n\n")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{current_time}\n{input_string}\n")


def process_response(input_string, replacements):
    def replacement(match):
        index = int(match.group(1))
        return (
            f"[{replacements[index]}]" if index < len(
                replacements) else match.group(0)
        )

    try:
        return re.sub(r"\[(\d+)\]", replacement, input_string)
    except:
        return input_string




def translate_text(text, target_language):
    print(target_language)
    api_key = "AIzaSyAtfrkxLhTygIJi9Rb-l0duA8fV9LgKZ7M"  # Replace with your API key

    url = "https://translation.googleapis.com/language/translate/v2"
    data = {"q": text, "target": target_language, "format": "text"}
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    response = requests.post(url, headers=headers, params=params, json=data)
    r = response.json()
    print(r)
    return r["data"]["translations"][0]["translatedText"]


def get_language(lang):
    print("getting lang.")
    lang = lang.lower()
    if "arabic" in lang:
        return "ar"
    if "kannada" in lang:
        return "kn"
    if "telugu" in lang:
        return "te"
    if "spanish" in lang:
        return "es"
    if "hebrew" in lang:
        return "he"
    if "japanese" in lang:
        return "ja"
    if "korean" in lang:
        return "ko"
    if "hindi" in lang:
        return "hi"
    if "bengali" in lang:
        return "bn"
    if "tamil" in lang:
        return "ta"
    if "urdu" in lang:
        return "ur"
    if "chinese" in lang:
        return "zh-CN"
    if "french" in lang:
        return "fr"
    if "german" in lang:
        return "de"

    session["language"] = "english"
    return "en"


@app.route("/level1", methods=["POST"])
def level1():
    print("level 1....\n\n\n")
    session["transcription"] = request.form["query"] if "query" in request.form else ""
    session["prompt_level1"] = (
        "" if "prompt_level1" not in session else session["prompt_level1"]
    )

    if request.form["leng"] != "":
        session["language"] = request.form["leng"]

    session["language"] = (
        "english" if session["language"] == "" else session["language"]
    )

    if request.form["fast"] == "true":
        session["toggle"] = "fast"
    if request.form["slow"] == "true":
        session["toggle"] = "slow"
    if request.form["groq"] == "true":
        session["toggle"] = "groq"

    audio_file = request.files["audio"] if "audio" in request.files else None

    try:
        if audio_file:
            audio_bytes = audio_file.read()
            audio_file = FileWithNames(audio_bytes)
            session["transcription"], session["language"] = transcribe(
                audio_file)
            session["language"] = get_language(session["language"])
            if session["language"].lower() != "en":
                session["transcription"] = translate_text(
                    session["transcription"], "en")
    except Exception as e:
        session["language"] = "en"
        print(e)
        update_logs(e)
        session["transcription"] = "Error."
    return jsonify({"message": "Data received, start streaming"})


@app.route("/level1/stream")
def level1_stream():
    global p1,layer_1
    print(layer_1,"layer 1  dataaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    # Load the configuration to get className1
    config_data = load_configuration()
    class_name1 = config_data["className1"]
    print(class_name1,"class Name is given")
    # Capitalize the first letter of className1
    ClassName1 = class_name1.capitalize()
    print(ClassName1,"what is this mannnnnnnnn")

    try:
        context1 = ""
        metadata1 = []
        # Use class_name1 for ddbot300
        context1, metadata1 = qdb(
            session["transcription"], layer_1, class_name1, ClassName1, chunk_id=1, limit=2)
        print(context1,class_name1,ClassName1,"the value is here should we take")
        # Assuming you also want to apply the same logic to ddbot100
        # Just replace "ddbot100" and "Ddbot100" with appropriate variables if needed
        context2, metadata2 = qdb(
            session["transcription"], layer_1, class_name1,ClassName1, chunk_id=3, limit=2)
        context = context1 + context2
        metadata = metadata1 + metadata2
        sufficient = False
    except Exception as e:
        print("\n\n\nError:    ", e)
        update_logs(e)
        context = "No context"
        metadata = ["1"]

    try:
        resp = Response(
            ask_gpt(
                question=session["transcription"],
                context=context,
                gpt="gpt-4",
                language=session["language"],
                metadata=metadata,
                addition=p1,
                userid=session["userID"],
            ),
            content_type="text/event-stream",
        )
        return resp
    except:
        data = {"response": "Error.", "sufficient": False}
        json_data = json.dumps(data)
        resp = "data: {json_data}\n\n"
        return Response(resp, content_type="text/event-stream")



@app.route("/level2", methods=["POST"])
def level2():
    session["layer_1_response"] = request.form["response"]
    return jsonify({"message": "Data received, start streaming"})


@app.route("/level2/stream")
def level2_stream():
    print("\n\n\nLever 2....\n\n\n")
    config_data1 = load_configuration1()
    class_name2 = config_data1["className2"]
    print(class_name2,"class Name is given")
    # Capitalize the first letter of className1
    ClassName2 = class_name2.capitalize()
    print(ClassName2,"what is this mannnnnnnnn")
    global p2,layer_2
    print(layer_2,"layer_2 is working...................................")
    try:
        context, metadata = qdb(
            session["transcription"], layer_2, class_name2,ClassName2, chunk_id=1, limit=5)
        sufficient = False
    except Exception as e:
        update_logs(e)
        context = "No context"
        metadata = ["1"]

    session["response"] = session["layer_1_response"]

    try:
        resp = Response(
            ask_gpt(
                question=session["transcription"],
                context=context,
                gpt="gpt-4",
                language=session["language"],
                metadata=metadata,
                addition=p2,
                userid=session["userID"],
            ),
            content_type="text/event-stream",
        )

        return resp
    except:
        data = {"response": "Error.", "sufficient": False}
        json_data = json.dumps(data)
        resp = "data: {json_data}\n\n"
        return Response(resp, content_type="text/event-stream")


def insert_data(table_name, data):
    table_id = f"my-project-41692-400512.jobbot.{table_name}"

    if not isinstance(data, list):
        data = [data]

    errors = dbclient.insert_rows_json(table_id, data)

    if errors == []:
        print(f"Data added successfully into {table_name}")
    else:
        print(
            f"Encountered errors while inserting into {table_name} : {errors}")


def delete_data(table_name, identifier_column, identifier_value):

    table_id = f"my-project-41692-400512.jobbot.{table_name}"

    sql_query = f"""
        DELETE FROM `{table_id}`
        WHERE `{identifier_column}` = '{identifier_value}'
    """

    query_job = dbclient.query(sql_query)  # Make an API request.
    query_job.result()  # Waits for the query to finish

    print(
        f"Rows deleted in {table_name} where {identifier_column} is {identifier_value}."
    )


@app.route("/add_question", methods=["POST"])
def add_question():
    data = request.json

    question_id = str(uuid.uuid4())
    question = data.get("question")
    optionA = data.get("optionA")
    optionB = data.get("optionB")
    optionC = data.get("optionC")
    optionD = data.get("optionD")

    dataset_name = "my-project-41692-400512.jobbot"
    table_name = "questions"
    table_id = f"{dataset_name}.{table_name}"

    query = """
        INSERT INTO `{0}` (questionID, question, optionA, optionB, optionC, optionD)
        VALUES (@question_id, @question, @optionA, @optionB, @optionC, @optionD)
    """.format(
        table_id
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "question_id", "STRING", question_id),
            bigquery.ScalarQueryParameter("question", "STRING", question),
            bigquery.ScalarQueryParameter("optionA", "STRING", optionA),
            bigquery.ScalarQueryParameter("optionB", "STRING", optionB),
            bigquery.ScalarQueryParameter("optionC", "STRING", optionC),
            bigquery.ScalarQueryParameter("optionD", "STRING", optionD),
        ]
    )

    try:
        query_job = dbclient.query(
            query, job_config=job_config)  # Make an API request
        return jsonify({"message": "Question added successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {e}"}), 500


def get_question_by_name(question_name):
    # Ensure this matches your actual BigQuery table ID
    table_id = "my-project-41692-400512.jobbot.questions"

    # Construct the query using a safe parameterized approach
    query = """
    SELECT *
    FROM `{}`
    WHERE question = @question_name
    """.format(
        table_id
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "question_name", "STRING", question_name),
        ]
    )

    query_job = dbclient.query(query, job_config=job_config)

    # Fetch the first row from the RowIterator
    try:
        result = next(query_job.result())
    except StopIteration:
        return None

    # Convert the result into a dictionary
    question_details = {
        "questionID": result[0],
        "question": result[1],
        "optionA": result[2],
        "optionB": result[3],
        "optionC": result[4],
        "optionD": result[5],
    }

    return question_details


@app.route("/questions", methods=["GET", "DELETE"])
def get_questions():
    try:
        # Construct the query to retrieve questions from BigQuery table
        dataset_name = (
            "my-project-41692-400512.jobbot"  # Replace with your actual dataset ID
        )
        table_name = "questions"
        table_id = f"{dataset_name}.{table_name}"

        query = """
            SELECT questionID, question, optionA, optionB, optionC, optionD
            FROM `{0}`
        """.format(
            table_id
        )

        # Execute the query
        query_job = dbclient.query(query)

        # Fetch the results
        results = query_job.result()

        # Convert results to a list of dictionaries
        questions = []
        for row in results:
            questions.append(
                {
                    "questionID": row.questionID,
                    "question": row.question,
                    "optionA": row.optionA,
                    "optionB": row.optionB,
                    "optionC": row.optionC,
                    "optionD": row.optionD,
                }
            )

        # Return the list of questions as JSON response
        return jsonify(questions), 200

    except Exception as e:
        return jsonify({"message": f"Error: {e}"}), 500


@app.route("/users")
def users():
    # Ensure this matches your actual BigQuery table ID
    table_id = "my-project-41692-400512.jobbot.users"

    # Construct the query using a safe parameterized approach
    sql_query = """
    SELECT * 
    FROM `{}`
    """.format(
        table_id
    )

    query_job = dbclient.query(sql_query)  # Make an API request.
    results = query_job.result()
    # Convert the results to a list of dictionaries
    user_data = []
    for row in results:
        user_data.append(dict(row))

    return jsonify(user_data)


def update_user_is_answered(userID, retry_count=3):
    print(userID, 'user updated')
    try:
        # Define the update query
        query = f"""
        UPDATE `my-project-41692-400512.jobbot.users`
        SET isAnswer = true
        WHERE userID = '{userID}'
        """

        # Execute the query
        query_job = dbclient.query(query)
        query_job.result()  # Wait for the query to complete
        print('user isAnswered updated')

    except BadRequest as e:
        # Handle specific error related to the streaming buffer
        if 'streaming buffer' in str(e) and retry_count > 0:
            print(
                f"Streaming buffer issue encountered. Retrying... ({retry_count} attempts left)")
            # Retry the operation after a short delay
            time.sleep(20)
            update_user_is_answered(userID, retry_count - 1)
        else:
            # Retry count exceeded or other error occurred
            print(
                f"An error occurred while updating user isAnswered: {str(e)}")


def update_answer(userID, answerID, answer):
    # Define the update query
    query = f"""
    UPDATE `my-project-41692-400512.jobbot.answers`
    SET answer = '{answer}'
    WHERE userID = '{userID}' AND answerID = '{answerID}'
    """

    # Execute the query
    query_job = dbclient.query(query)
    query_job.result()  # Wait for the query to complete
    print('Answer updated successfully')


@app.route("/updateProfile", methods=["GET", "POST", "PUT"])
def create_profile():
    if request.method == "POST":
        # Handle POST request to create new data
        data = request.json
        print(data, "-------------")
        userID = None

        if 'userID' in session:
            userID = session['userID']
        else:
            print("userID not found in session")

        # Update user's profile data
        for item in data:
            answerID = str(uuid.uuid4())
            question = item.get("question")
            answer = item.get("answer")
            fullQuestion = get_question_by_name(question)
            table_data = {
                "answerID": answerID,
                "questionID": fullQuestion['questionID'],
                "answer": answer,
                "userID": userID,
            }
            insert_data('answers', table_data)

        # Update user's isAnswered field to True in BigQuery
        update_user_is_answered(userID)

        return jsonify({"message": "Data received successfully"})

    elif request.method == "PUT":
        # Handle PUT request to update existing data
        data = request.json
        userID = None

        if 'userID' in session:
            userID = session['userID']
        else:
            print("userID not found in session")

        # Update user's profile data
        for item in data:
            # Assuming answerID is provided in the request
            answerID = item.get("answerID")
            answer = item.get("answer")
            update_answer(userID, answerID, answer)

        return jsonify({"message": "Data updated successfully"})

    else:
        return render_template("profile.html")


@app.route('/audioInterval')
def audio_interval():
    def generate_audio():
        global audio_speech
        audio_speech = False
        if audio_speech:
            audio_file_path = './output.mp3'
            audio_speech = None
            with open(audio_file_path, 'rb') as audio_file:
                while True:
                    chunk = audio_file.read(1024)
                    if not chunk:
                        break
                    print('sending audio')
                    yield chunk
        else:
            print('No Audio')
            return {'error': 'null'}
    return Response(generate_audio(), mimetype='audio/mpeg')


@app.route("/answers")
def get_user_answers():
    if 'userID' not in session:
        return jsonify({"error": "User ID not found in session"}), 400

    # Get user ID from session
    userID = session['userID']

    # Define the query to retrieve user's answers
    query = f"""
    SELECT answerID, answer, questionID
    FROM `my-project-41692-400512.jobbot.answers`
    WHERE userID = '{userID}'
    """

    # Execute the query
    query_job = dbclient.query(query)

    # Fetch all results
    results = query_job.result()

    # Prepare the response data
    answers = []
    for row in results:
        answer = {
            "answerID": row["answerID"],
            "answer": row["answer"],
            # Assuming you have a function to get question text
            "question": get_question_text(row["questionID"])
        }
        answers.append(answer)

    # Return the answers as JSON response
    return jsonify({"answers": answers})


def get_question_text(questionID):
    # Define the query to retrieve the question text based on questionID
    query = f"""
    SELECT question
    FROM `my-project-41692-400512.jobbot.questions`
    WHERE questionID = '{questionID}'
    """

    # Execute the query
    query_job = dbclient.query(query)

    # Fetch all results
    results = query_job.result()

    # Extract the question text from the result
    for row in results:
        question_text = row["question"]
        return question_text

    return "Question not found"


# Google drive Integration
from google.oauth2.credentials import Credentials


# Your credentials
CREDENTIALS_FILE = 'drive.json'

# If modifying these scopes, delete the file token.json.

# Global variables
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'drive.json'
DOWNLOAD_DIRECTORY = "download_files"
INFO_FILE_PATH = os.path.join(DOWNLOAD_DIRECTORY, "file_info.json")


@app.route("/new_panel")
def drive():
    return render_template("new_panel.html")
  # Replace with your actual folder ID


def get_google_drive_service():
    """Authenticate and return Google Drive service."""
    creds = None
    # Check if token.json exists and contains valid credentials
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If credentials are not valid, refresh them or log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('drive', 'v3', credentials=creds)
    return service


logging.basicConfig(level=logging.DEBUG)

directory="download_files"


def download_folder(folder_url):
    
    folder_id = folder_url.split('/')[-1]
    service = get_google_drive_service()

    query = f"'{folder_id}' in parents"
    save_directory = "download_files"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_info_list = []
    page_token = None

    try:
        while True:
            response = service.files().list(q=query,
                                            spaces='drive',
                                            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                                            pageToken=page_token, supportsAllDrives=True,
                                            includeItemsFromAllDrives=True).execute()
            items = response.get('files', [])
            for item in items:
                file_id = item['id']
                file_name = item['name']
                modified_time = item['modifiedTime']
                mimeType = item['mimeType']
                file_path = os.path.join(save_directory, file_name)

                if mimeType.startswith('application/vnd.google-apps.'):
                    # Logic to handle Google Docs formats
                    continue  # You'll need to adjust this part based on your needs

                drive_request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, drive_request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                fh.seek(0)

                with open(file_path, 'wb') as f:
                    f.write(fh.read())
                    app.logger.debug(f"{file_name} has been downloaded.")

                file_info_list.append(
                    {'id': file_id, 'name': file_name, 'modifiedTime': modified_time})

            page_token = response.get('nextPageToken')
            if not page_token:
                break

        # After processing all items
        info_file_path = os.path.join(save_directory, "file_info.json")
        with open(info_file_path, 'w') as json_file:
            json.dump(file_info_list, json_file, indent=4)
            app.logger.debug("JSON file written successfully.")

        return {"status": "success", "message": "Files downloaded successfully.", "files": file_info_list}

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

    

directory="download_files_2"


def download_folder_2(folder_url):
    
    folder_id = folder_url.split('/')[-1]
    service = get_google_drive_service()

    query = f"'{folder_id}' in parents"
    save_directory = "download_files_2"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_info_list = []
    page_token = None

    try:
        while True:
            response = service.files().list(q=query,
                                            spaces='drive',
                                            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                                            pageToken=page_token, supportsAllDrives=True,
                                            includeItemsFromAllDrives=True).execute()
            items = response.get('files', [])
            for item in items:
                file_id = item['id']
                file_name = item['name']
                modified_time = item['modifiedTime']
                mimeType = item['mimeType']
                file_path = os.path.join(save_directory, file_name)

                if mimeType.startswith('application/vnd.google-apps.'):
                    # Logic to handle Google Docs formats
                    continue  # You'll need to adjust this part based on your needs

                drive_request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, drive_request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                fh.seek(0)

                with open(file_path, 'wb') as f:
                    f.write(fh.read())
                    app.logger.debug(f"{file_name} has been downloaded.")

                file_info_list.append(
                    {'id': file_id, 'name': file_name, 'modifiedTime': modified_time})

            page_token = response.get('nextPageToken')
            if not page_token:
                break

        # After processing all items
        info_file_path = os.path.join(save_directory, "file_info_2.json")
        with open(info_file_path, 'w') as json_file:
            json.dump(file_info_list, json_file, indent=4)
            app.logger.debug("JSON file written successfully.")

        return {"status": "success", "message": "Files downloaded successfully.", "files": file_info_list}

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

    


info_file_path = r'C:\Users\iamka\OneDrive\Desktop\Jobbot\JobBot\download_files\file_info.json'



def get_file_info():
    """Load file information from JSON, safely handling unexpected formats."""
    if not os.path.exists(INFO_FILE_PATH):
        return {}

    with open(INFO_FILE_PATH, 'r') as json_file:
        try:
            file_info_data = json.load(json_file)
        except json.JSONDecodeError:
            print("Error decoding JSON from file_info.json.")
            return {}

    # Ensure the data is in the expected list of dictionaries format
    if isinstance(file_info_data, list) and all(isinstance(item, dict) for item in file_info_data):
        # Key by 'id' for uniqueness
        file_info_dict = {item['id']: item for item in file_info_data}
    else:
        print(
            "Unexpected data structure in file_info.json. Expected a list of dictionaries.")
        return {}

    return file_info_dict


def check_file_modifications():
    monitored_files = get_file_info()  # Load initial file information
    while True:
        current_files = get_file_info()  # Reload file information on each check
        has_change = False
        for file_name, last_known_time in current_files.items():
            if file_name not in monitored_files or monitored_files[file_name] != last_known_time:
                print(f"{file_name} has been modified. Yes")
                has_change = True
            else:
                print(f"{file_name} has not been modified. No")
        if not has_change:
            print("No changes detected. No")
        else:
            monitored_files = current_files
        time.sleep(10)





# 333333333333333333333333333333333333333333333333333333333333333333333333333333333

def update_file_info_json(file_info):
    """Update the file_info.json with the latest file information."""
    # Prepare data for JSON serialization
    for file_id, info in file_info.items():
        # Convert datetime to string in ISO format
        if isinstance(info['modifiedTime'], datetime):
            info['modifiedTime'] = info['modifiedTime'].isoformat()

    updated_info_list = [value for key, value in file_info.items()]
    try:
        with open(INFO_FILE_PATH, 'w') as json_file:
            json.dump(updated_info_list, json_file, indent=4)
    except Exception as e:
        print(f"Failed to update JSON file: {e}")

directory="download_files"
def download_and_save_file(service, item):
    """Download and save file from Google Drive."""
    file_id = item['id']
    file_name = item['name']
    modified_time = iso8601.parse_date(item['modifiedTime'])
    mimeType = item['mimeType']
    file_path = os.path.join(DOWNLOAD_DIRECTORY, file_name)

    if mimeType.startswith('application/vnd.google-apps.'):
        export_mimeType = 'application/pdf'  # Default for Google Docs
        if mimeType == 'application/vnd.google-apps.spreadsheet':
            export_mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            file_name += '.xlsx'
        elif mimeType == 'application/vnd.google-apps.presentation':
            export_mimeType = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            file_name += '.pptx'
        file_path = os.path.join(DOWNLOAD_DIRECTORY, file_name)
        request = service.files().export_media(fileId=file_id, mimeType=export_mimeType)
    else:
        request = service.files().get_media(fileId=file_id)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    with open(file_path, 'wb') as f:
        fh.seek(0)
        f.write(fh.read())
        print(f"{file_name} has been downloaded.")
    try:
        pdf_vectorization(directory)
    except Exception as e:
        print(e,"new also got vectorized...................................")        

    return {'id': file_id, 'name': file_name, 'modifiedTime': modified_time}


def monitor_and_download_new_files(folder_id):
    service = get_google_drive_service()
    # Load initial file information, assuming it's keyed by file ID for uniqueness
    file_info = get_file_info()
    # Make sure to exclude files that are in the trash
    query = f"'{folder_id}' in parents and trashed = false"

    while True:
        response = service.files().list(q=query, spaces='drive',
                                        fields="files(id, name, mimeType, modifiedTime)").execute()
        items = response.get('files', [])
        # Set of current Google Drive file IDs
        drive_file_ids = set([item['id'] for item in items])
        local_file_ids = set(file_info.keys())  # Set of local file IDs

        # Detect new or updated files
        new_files_info = {}
        for item in items:
            file_id = item['id']
            file_modified_time = iso8601.parse_date(item['modifiedTime'])
            if file_id not in file_info or file_modified_time > iso8601.parse_date(file_info[file_id]['modifiedTime']):
                new_file_info = download_and_save_file(service, item)
                new_files_info[file_id] = new_file_info
                print(
                    f"New or updated file detected and downloaded: {item['name']}")

        # Detect deleted files
        deleted_file_ids = local_file_ids - drive_file_ids
        for file_id in deleted_file_ids:
            file_name = file_info[file_id]['name']
            file_path = os.path.join(DOWNLOAD_DIRECTORY, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(
                    f"File deleted locally because it was removed from Google Drive: {file_name}")

        # Update local file information
        if new_files_info or deleted_file_ids:
            file_info = {file_id: file_info[file_id] for file_id in (
                local_file_ids & drive_file_ids)}  # Keep only non-deleted files
            file_info.update(new_files_info)  # Add or update files
            update_file_info_json(file_info)
            print("File information updated.")

        time.sleep(10)

#pdf related thing
        

def split_pdf_text_by_page(pdf_path):
    pages=[]
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            pages.append(text)
    return pages

def load_documents(directory, glob_patterns):
    documents = []
    for glob_pattern in glob_patterns:
        file_paths = glob.glob(os.path.join(directory, glob_pattern))
        for fp in file_paths:
            try:
                if fp.endswith('.docx'):
                    text = docx2txt.process(fp)
                    pages = [text]  # Treat the whole document as a single "page"
                elif fp.endswith('.pdf'):
                    pages = split_pdf_text_by_page(fp)
                else:
                    print(f"Warning: Unsupported file format for {fp}")
                    continue
                documents.extend([(page, os.path.basename(fp), i+1) for i, page in enumerate(pages)])
            except Exception as e:
                print(f"Warning: The file {fp} could not be processed. Error: {e}")
    return documents


def split_text(text, file_name, chunk_size, chunk_overlap):
    start = 0
    end = chunk_size
    while start < len(text):
        yield (text[start:end], file_name)
        start += (chunk_size - chunk_overlap)
        end = start + chunk_size

def split_documents(documents, chunk_size, chunk_overlap):
    texts = []
    metadata = []
    for doc_text, file_name, page_number in documents:
        for chunk in split_text(doc_text, file_name, chunk_size, chunk_overlap):
            sentence = chunk[0]
            # if len(sentence) <= 1200:
            #     sentence = sentence.replace("what the heck do i do with my life?","")
            #     sentence = sentence.replace("1385_What the Heck_.indd","")
            #     if len(sentence) > 300:
            #         texts.append(sentence)
            #         metadata.append(str(file_name) + " Pg: " + str(page_number))
            #     else: pass
            # else:
            texts.append(sentence)
            metadata.append(str(file_name) + " Pg: " + str(page_number))

    return texts, metadata

#for Layer_1

def pdf_vectorization(directory):
    config_data = load_configuration()  # Load the current configuration

    # Extract configuration values
    class_name = config_data["className1"]
    layer_1_url = config_data["layer1URL"]
    layer_1_auth_key = config_data["layer1AuthKey"]
    openai_key = config_data["openaiKey"]
    print(class_name,"ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
    client = weaviate.Client(
        url=layer_1_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=layer_1_auth_key),
        additional_headers={"X-OpenAI-Api-Key": openai_key}
    )

    print(client,"client is the value of i am getting")


    glob_patterns = ["*.docx", "*.pdf"]
    documents = load_documents(directory, glob_patterns)

    chunk_size = 400
    chunk_overlap = 0
    texts, metadata = split_documents(documents, chunk_size, chunk_overlap)

    print("-----------------------------")
    data_objs = [{"text": tx, "metadata": met}  for tx,met in zip(texts, metadata)]
    print(len(data_objs))

    i = 0

    class_obj = {
    "class": class_name,

    "properties": [
        {
            "name": "text",
            "dataType": ["text"],
        },
        {
            "name": "metadata",
            "dataType": ["text"],
        },
    ],

    "vectorizer": "text2vec-openai",

    "moduleConfig": {
        "text2vec-openai": {
            "vectorizeClassName": False,
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
        },
        
    },
    }
    try:
        client.schema.create_class(class_obj)
    except Exception as e:
        print("Errrrorrrrrrrrrrrrrrrrrrrr",e )
        print("--------*--------**")

    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for data_obj in data_objs:
            i+= 1
            if i>-1:
                print("--",i)
                batch.add_data_object(data_obj, class_name)
            else:
                print(i)



#for Layer_2                
def pdf_vectorization_layer_2 (directory):
    config_data1 = load_configuration1()  # Load the current configuration
    config_data=load_configuration()
    # Extract configuration values
    class_name = config_data1["className2"]
    layer_2_url = config_data1["layer2URL"]
    layer_2_auth_key = config_data1["layer2AuthKey"]
    openai_key = config_data["openaiKey"]
    print(class_name,"ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
    client = weaviate.Client(
        url=layer_2_url,
        auth_client_secret=weaviate.AuthApiKey(api_key=layer_2_auth_key),
        additional_headers={"X-OpenAI-Api-Key": openai_key}
    )

    print(client,"client is the value of i am getting")


    glob_patterns = ["*.docx", "*.pdf"]
    documents = load_documents(directory, glob_patterns)

    chunk_size = 400
    chunk_overlap = 0
    texts, metadata = split_documents(documents, chunk_size, chunk_overlap)

    print("-----------------------------")
    data_objs = [{"text": tx, "metadata": met}  for tx,met in zip(texts, metadata)]
    print(len(data_objs))

    i = 0

    class_obj = {
    "class": class_name,

    "properties": [
        {
            "name": "text",
            "dataType": ["text"],
        },
        {
            "name": "metadata",
            "dataType": ["text"],
        },
    ],

    "vectorizer": "text2vec-openai",

    "moduleConfig": {
        "text2vec-openai": {
            "vectorizeClassName": False,
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
        },
        
    },
    }

    print("--------*--------**")

    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for data_obj in data_objs:
            i+= 1
            if i>-1:
                print("--",i)
                batch.add_data_object(data_obj, class_name)
            else:
                print(i)                

@app.route('/process_drive_folder', methods=['GET'])
def process_drive_folder():
    folder_url = request.args.get('folder_url')
    folder_id = folder_url.split('/')[-1]
    if not folder_url:
        return jsonify({"error": "Missing folder_url parameter"}), 400

    app.logger.debug(f"Processing Folder URL: {folder_url}")
    
    try:
        response_data = download_folder(folder_url)
        if response_data.get('status') != 'success':
            return jsonify({"error": "Failed to download files"}), 500

        download_directory = "download_files"
        # Assume pdf_vectorization is a function you define elsewhere
        pdf_vectorization(directory=download_directory)
        app.logger.debug("Files downloaded and vectorized.")
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    try:
        monitor_and_download_new_files( folder_id)
    except Exception as e:
        print(e,"new file added and got vectorized also................................")   
       
    # finally:
    #      if os.path.exists(download_directory):
    #          shutil.rmtree(download_directory)
    #          app.logger.debug("Downloaded files deleted.")
    
    return jsonify({"message": "Files processed successfully."})



   

@app.route('/process_drive_folder_2', methods=['GET'])
def process_drive_folder_2():
    folder_url = request.args.get('folder_url')
    if not folder_url:
        return jsonify({"error": "Missing folder_url parameter"}), 400

    app.logger.debug(f"Processing Folder URL: {folder_url}")
    
    try:
        response_data = download_folder_2(folder_url)
        if response_data.get('status') != 'success':
            return jsonify({"error": "Failed to download files"}), 500

        download_directory = "download_files_2"
        # Assume pdf_vectorization is a function you define elsewhere
        pdf_vectorization_layer_2(directory=download_directory)
        app.logger.debug("Files downloaded and vectorized.")
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    # finally:
    #      if os.path.exists(download_directory):
    #          shutil.rmtree(download_directory)
    #          app.logger.debug("Downloaded files deleted.")
    
    return jsonify({"message": "Files processed successfully."})



   

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('/login'))


if __name__ == "__main__":
    
    app.run(debug=True)
