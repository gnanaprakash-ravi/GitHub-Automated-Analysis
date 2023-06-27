# import git
import os 
import requests
import json
import openai 
# import tiktoken
from collections import Counter
# import re
import time
import math as m

import streamlit as st

from tokenizers import Encoding
from tokenizers.implementations import BaseTokenizer
from tokenizers.normalizers import Lowercase, Sequence
from tokenizers.pre_tokenizers import Whitespace

from git.repo import Repo
import nbformat
from nbconvert import PythonExporter

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Set up OpenAI API credentials
# openai.api_key = 'hided'
st.title('Github complexity analyser')

# Set your OpenAI API key here or leave it empty
openai_api_key = ''

if openai_api_key:
    st.text_input('OpenAI API Key', value=openai_api_key, type='password')
if not openai_api_key:
    openai_api_key = st.text_input('Enter OpenAI API Key', type='password')


# Set up OpenAI API credentials
openai.api_key = openai_api_key

# username = 'gnanaprakash-ravi'
# link_aadress = 'https://github.com/gnanaprakash-ravi'
# link_aadress = 'https://github.com/code2prab'

link_aaddress = st.text_input('Enter User\'s Github Profile link', '').split('/')


if len(link_aaddress)>=1:
    username = link_aaddress[-1]
    print(username)

# Function to fetch user repositories from GitHub
def fetch_user_repositories(github_url):
    # Extracting username from the GitHub URL
    username = github_url.split("/")[-1]

    # GitHub API endpoint to fetch user repositories
    api_url = f"https://api.github.com/users/{username}/repos"
    try:
        # Sending GET request to the GitHub API
        response = requests.get(api_url)

        if response.status_code == 200:
            # Parsing the JSON response
            repositories = response.json()

            # Extracting repository names and URLs
            repository_names = []
            repository_urls = []

            for repo in repositories:
                repository_names.append(repo["name"])
                repository_urls.append(repo["html_url"])
            return repository_names, repository_urls

        else:
            print("Failed to fetch user repositories. Please check the GitHub URL or try again later.")
            return None, None

    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching user repositories:", str(e))
        return None, None

def clone_repositories(github_url):
    username = github_url.split("/")[-1]
    repository_names, repository_urls = fetch_user_repositories(github_url)

    if repository_names and repository_urls:
        os.makedirs(f"{username}", exist_ok=True)
        os.chdir(f"{username}")

        for repo_name, repo_url in zip(repository_names, repository_urls):
            repo_path = os.path.join(os.getcwd(), repo_name)

            # Check if the repository directory already exists
            if os.path.exists(repo_path):
                print(f"Skipping repository: {repo_name}. Already cloned.")
                continue

            try:
                Repo.clone_from(repo_url, repo_name)
                print(f"Cloned repository: {repo_name}")
            except Exception as e:
                print(f"Failed to clone repository: {repo_name}")
                print("Error:", str(e))
        
        os.chdir("..")
    return repository_names


def openai_bot(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a user seeking advice on programming language."},
            {"role": "user", "content": prompt}
        ],
        temperature = 0.6
    )

    chatbot_response = response.choices[0].message.content
    return chatbot_response

# def give_prompt(code_snippet): 
#     output = ""

#     encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

#     num_tokens = len(encoding.encode(code_snippet))
#     print(num_tokens+50)

#     n = m.ceil((num_tokens+50)/4096)
#     print(n)

#     leng = len(code_snippet)
#     print(leng)
#     for i in range(n):
#         text = code_snippet[int(i*leng/n):int((i+1)*leng/n)]

#         prompt = """ 

#             ```python
#             {}
#             ```

#         Just give 30 words description of the code with its complexity(in percentage) just ignore if code is incomplete but give only proper information
#         """.format(code_snippet)

#         response = openai_bot(prompt)
#         print(response)
#         output = output + response
#     return output

def give_prompt(code_snippet):
    output = ""

    tokenizer = BaseTokenizer(
        tokenizer=Whitespace(),
        normalizer=Sequence([Lowercase()])
    )

    encoding: Encoding = tokenizer.encode(code_snippet)
    print(encoding.num_tokens + 50)

    n = m.ceil((encoding.num_tokens + 50) / 4096)
    print(n)

    leng = len(code_snippet)
    print(leng)
    for i in range(n):
        text = code_snippet[int(i * leng / n):int((i + 1) * leng / n)]

        prompt = f"""
        ```
        {text}
        ```

        Just give 30 words description of the code with its complexity (in percentage). Just ignore if the code is incomplete but provide only proper information.
        """

        response = openai_bot(prompt)
        print(response)
        output = output + response
    return output


# def preprocess_code(code):
#     if code.endswith('.ipynb'):
#         # Load the Jupyter notebook
#         with open(code, 'r', encoding='utf-8') as file:
#             nb = nbformat.read(file, as_version=4)

#         # Create a PythonExporter instance
#         exporter = PythonExporter()

#         # Convert the notebook to Python code
#         (python_code, _) = exporter.from_notebook_node(nb)

#         code = python_code

#     return code

def preprocess_code1(code):
    max_chunk_size = 1000  # Define the maximum number of tokens per chunk

    # Convert Jupyter notebook to Python script
    if code.endswith('.ipynb'):
        # Load the Jupyter notebook
        with open(code, 'r', encoding='utf-8') as file:
            nb = nbformat.read(file, as_version=4)

        # Create a PythonExporter instance
        exporter = PythonExporter()

        # Convert the notebook to Python code
        (python_code, _) = exporter.from_notebook_node(nb)

        code = python_code

    # Split the code into chunks of maximum chunk size
    chunks = [code[i:i + max_chunk_size] for i in range(0, len(code), max_chunk_size)]
    processed_code = ""

    # Process each chunk separately
    for chunk in chunks:
        # Perform your preprocessing steps on each chunk of code
        # For example, you can apply tokenization, cleaning, or other transformations
        # Here, you can perform any desired preprocessing steps on the chunk
        processed_chunk = chunk

        # Append the processed chunk to the final processed code
        processed_code += processed_chunk

    return processed_code


if username != '':

    repos = []

    # # Create a URL to the GitHub API endpoint
    url = "https://api.github.com/users/{}/repos".format(username)

    # Make a GET request to the API endpoint
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the response data as JSON
        data = json.loads(response.content)

        # Iterate over the repositories
        for repo in data:
            # Print the repository name
            # print(repo["name"])
            repos.append(repo["name"])
            
    else:
        st.error("Error: HTTP {}.".format(response.status_code))

    github_url = f"https://github.com/{username}"
    
    repositories = clone_repositories(github_url)

    d = {}

    for repo_name in repos:
        time_stamp = time.time()
        repo = {}
        folders = []
        files = []
        folder_path = os.path.join(username,repo_name)
        print(folder_path)
        final = ""
        for item in os.listdir(folder_path):
            # print(item)
            item_path = os.path.join(folder_path, item)

            if os.path.isfile(item_path):  # Check if it's a file
                try:

                        with open(item_path, 'r') as file:
                            contents = file.read()
                        if item_path.endswith('.ipynb'):
                            # with open(item_path, 'r') as file:
                            contents = preprocess_code1(item_path)
                            # contents = preprocess_code1(item_path)

                            final+=contents
        #                     l,f,c = extract_names_from_code(item_path)

        #                     library_names.extend(l)
        #                     function_names.extend(f)
        #                     class_names.extend(c)
                except:
                    pass
                
            elif os.path.isdir(item_path):  # Check if it's a folder
                folders.append(item)

        # Iterate through the folders and fetch files recursively

        for folder in folders:
            for item in os.listdir(os.path.join(folder_path, folder)):
                item_path = os.path.join(folder_path, folder, item)

                if os.path.isfile(item_path):  # Check if it's a file
                    try:

                        with open(item_path, 'r') as file:
                            contents = file.read()
                        if item_path.endswith('.ipynb'):
                            # with open(item_path, 'r') as file:
                            contents = preprocess_code1(item_path)
                            # contents = preprocess_code1(item_path)

                            final+=contents
        #                         l,f,c = extract_names_from_code(item_path)
        #                         library_names.extend(l)
        #                         function_names.extend(f)
        #                         class_names.extend(c)
                    except:
                        pass

                elif os.path.isdir(item_path):  # Check if it's a folder
                    pass
        # Initialize the parser with the code
        parser = PlaintextParser.from_string(final, Tokenizer("english"))

        # Initialize the LexRank summarizer
        summarizer = LexRankSummarizer()

        # Summarize the code
        summary = summarizer(parser.document, sentences_count=80) 
        # print(summary)
        tur = ""
        for i in summary:
            tur += str(i)
        print(tur)
        prompt = """ 

                    ```python
                    {}
                    ```
                Just give one line of the codes in terms of technical complexity in magnitude do exceed 15 words tell one of these three complexity like low, medium high level of complexity
                """.format(tur)
        
        response = openai_bot(prompt)
        d[repo_name] = response
        st.write(repo_name+": "+response)
        # user_input = prompt
        if time.time() - time_stamp < 30:
            time.sleep(30-time.time() + time_stamp)
            
        # response = chat_with_chatbot(user_input)
        time_stamp = time.time()

    prompt = f'''json that has key the repository name and complexity corresponding  to that in values
    you need to find the highest complexity repo

    {d}

    no need of code and explanation just repo name only'''

    response = openai_bot(prompt)
    st.success(response)
    # st.balloons()