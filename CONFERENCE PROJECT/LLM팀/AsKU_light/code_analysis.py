from httpx import NetworkError
import time
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import PyPDF2
import os
import ast
import feedparser
import fitz  # PyMuPDF
import re
import subprocess
import tiktoken
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer

class NetworkError(Exception):
    """Custom exception for network errors."""
    pass

class CodeAnalysis:
    """
    A class to interact with scientific papers using the Semantic Scholar API and arxiv.

    Attributes:
    -----------
    ss_api_key : str
        The API key for accessing the Semantic Scholar API.
    openai_key : str
        The API key for accessing the ChatOpenAI.
    path_db : str
        The path to the local database where paper is stored. Defaults to './papers_db'.
    code_db : str
        The path to the local database where github repository is sotred. Defaults to './code_db'.
    """

    def __init__(self, ss_api_key, openai_key, path_db='./papers_db', code_db='./code_db'):
        self.ss_api_key = ss_api_key
        self.openai_key = openai_key
        self.path_db= path_db
        self.code_db = code_db

########## PDF 안의 github repository 링크를 찾아서 cloning
    def get_paper_id_from_title(self, title):
        """논문의 제목으로 정보를 가져오는 함수"""
        # Define the API endpoint URL
        url = 'https://api.semanticscholar.org/graph/v1/paper/search?query={}&fields=paperId,title,abstract,authors,citations,fieldsOfStudy,influentialCitationCount,isOpenAccess,openAccessPdf,publicationDate,publicationTypes,references,venue'

        headers = {'x-api-key': self.ss_api_key}
        try:
            response = requests.get(url.format(title), headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Error fetching data from Semantic Scholar API: {e}")

        try:
            data = response.json()
        except ValueError:
            raise NetworkError("Error parsing JSON response from Semantic Scholar API")

        if 'data' in data and data['data']:
            paper = data['data'][0]
            external_ids = paper.get('openAccessPdf', {})
            if external_ids and 'url' in external_ids:
                arxiv_id = external_ids['url']
                if 'http' in arxiv_id:
                    arxiv_id = arxiv_id.split('/')[-1]
                return arxiv_id
            else:
                raise NetworkError("No paper found for the given title")
        else:
            raise NetworkError("No paper found for the given title")

    def get_paper_info_by_title_arxiv(self, query, max_results=1):
        # arxiv api 는 : 이 포함된 query를 검색하지 못함. : 이후의 문자열로 검색.
        query = query.split(':')[-1]

        base_url = 'http://export.arxiv.org/api/query?'
        query_url = f'search_query=all:{query}&start=0&max_results={max_results}'
        response = requests.get(base_url + query_url)

        if response.status_code != 200:
            raise Exception('Error fetching data from arXiv API')

        feed = feedparser.parse(response.content)

        if 'entries' in feed and len(feed.entries) > 0:
            paper = feed.entries[0]
            arxiv_id = paper.id.split('/')[-1]
            return arxiv_id
        else:
            raise NetworkError("No paper found for the given title")
    # def get_paper_id_from_title(self, title, api_key):
        # search_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        # params = {
        #     "query": title,
        #     "fields": "paperId",
        #     "limit": 1
        # }
        # headers = {
        #     "x-api-key": api_key
        # }
        # response = requests.get(search_url, params=params, headers=headers)

        # if response.status_code == 403:
        #     raise Exception("API request failed with status code 403: Forbidden. Please check your API key and permissions.")
        # elif response.status_code == 429:
        #     time.sleep(3)
        #     self.get_paper_id_from_title(title, api_key)
        # elif response.status_code != 200 and response.status_code != 429:
        #     raise Exception(f"API request failed with status code {response.status_code}")
        
        # data = response.json()

        # papers = data.get('data', [])
        # if not papers:
        #     return None

        # first_paper = papers[0]
        # paper_id = first_paper.get('paperId')

        # return paper_id

    def get_arxiv_pdf_url(self, title, api_key):
        ### 
        try:
            arxiv_id = self.get_paper_id_from_title(title)
        except NetworkError as e:
            try:
                arxiv_id = self.get_paper_info_by_title_arxiv(title)
            except NetworkError as e:
                return f"Error: No paper was founded in database"
        ###
    # def get_arxiv_pdf_url(self, paper_id, api_key):
        # url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=externalIds"
        # response = requests.get(url, headers={"x-api-key": api_key})

        # if response.status_code == 403:
        #     raise Exception("API request failed with status code 403: Forbidden. Please check your API key and permissions.")
        # elif response.status_code != 200 :
        #     raise Exception(f"API request failed with status code {response.status_code}")

        # data = response.json()

        # external_ids = data.get('externalIds', {})
        # arxiv_id = external_ids.get('ArXiv')

        if not arxiv_id:
            return None

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return pdf_url

    def download_pdf(self, pdf_url):
        time.sleep(1)
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF with status code {response.status_code}")

        return response.content

    def extract_github_links_from_pdf(self, pdf_content):
        # PDF를 메모리에 로드
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

        # 첫 번째 페이지 추출
        first_page = pdf_document[0]
        text = first_page.get_text()

        # GitHub 링크 추출 (정규 표현식 사용)
        github_links = re.findall(r'(https?://github\.com/[^\s]+)', text)

        return github_links

    def clone_github_repository(self, github_url):
        # GitHub URL에서 리포지토리 이름 추출
        parsed_url = urlparse(github_url)
        repo_name = os.path.basename(parsed_url.path).replace('.git', '')

        # 클론 디렉토리 설정
        clone_dir = os.path.join(self.code_db, repo_name)

        if not os.path.exists(clone_dir):
            os.makedirs(clone_dir)

            # GitHub 리포지토리 클론
            result = subprocess.run(['git', 'clone', github_url, clone_dir], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to clone repository: {result.stderr}")
            self.repo_path = clone_dir
            print(f"Repository cloned into {clone_dir}")
            return self.repo_path
        else : 
            self.repo_path = clone_dir
            print(f"Repository already clonded into {clone_dir}")
            return self.repo_path

    def Git_cloning(self, title:str, github_link:str): 
        try:
            # paper_id = self.get_paper_id_from_title(title, self.ss_api_key)
            if not github_link :
                raise Exception("Github 링크가 필요합니다.")
            self.repo_path = self.clone_github_repository(github_link) # Github Link가 제시 되면 바로 git clone
            # else:
            #     pdf_url = self.get_arxiv_pdf_url(title, self.ss_api_key)
                
            #     if not pdf_url:
            #         print("ArXiv PDF URL not found.")
            #     else:
            #         # PDF 다운로드
            #         pdf_content = self.download_pdf(pdf_url)

            #         # PDF에서 GitHub 링크 추출
            #         github_links = self.extract_github_links_from_pdf(pdf_content)
            #         print(f"GitHub Links: {github_links}")

            #         if len(github_links) > 1 :
            #             self.repo_path = self.clone_github_repository(github_links[0])

            #         elif len(github_links) == 1 :
            #             self.repo_path = self.clone_github_repository(github_links[0])
                            
        except Exception as e:
            print(e)

    def generate_code_from_content(self, content):
        prompt=f"Based on the following content from a research paper, write the corresponding Python code that implements the described concept.\n\nPaper Content: \"{content}\""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    # Function to extract code from .py files in a directory
    def extract_code_from_repo(self, repo_path):
        code_files = {}
        for subdir, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_files[file_path] = f.read()
        return code_files

    def split_code_into_functions(self, code):
        pattern = re.compile(r"def\s+\w+\(.*\):")  # Simplistic pattern to match function definitions
        lines = code.split('\n')
        functions = {}
        current_func_name = None
        current_func_code = []

        for line in lines:
            if pattern.match(line):
                if current_func_name:
                    functions[current_func_name] = '\n'.join(current_func_code)
                current_func_name = line.split('(')[0].replace('def ', '').strip()
                current_func_code = [line]
            elif current_func_name:
                current_func_code.append(line)

        if current_func_name:
            functions[current_func_name] = '\n'.join(current_func_code)

        return functions

    def calculate_cosine_similarity(self, code1, code2):
        vectorizer = TfidfVectorizer().fit_transform([code1, code2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0, 1]
    
    def calculate_similarity_codet5(self, code1, code2):
        checkpoint = "Salesforce/codet5p-110m-embedding"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

        first_code = tokenizer.encode(code1, return_tensors="pt")
        secode_code = tokenizer.encode(code2, return_tensors="pt")

        first_embedding = model(first_code)[0]
        secode_code = model(secode_code)[0]
        cos_sim = util.cos_sim(first_embedding, secode_code)

        # 유사도 점수를 0~1 사이로 정규화
        normalized_score = float(cos_sim.item()) / 2 + 0.5

        return normalized_score
    
    def answer_quality_score(self, code1, code2) :
        """질문과 답변 코드의 유사도를 기반으로 품질 점수를 계산하는 함수"""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # 질문과 답변을 임베딩
        question_embedding = model.encode(code1, convert_to_tensor=True)
        answer_embedding = model.encode(code2, convert_to_tensor=True)

        # 코사인 유사도 계산
        cos_sim = util.cos_sim(question_embedding, answer_embedding)

        # 유사도 점수를 0~1 사이로 정규화
        normalized_score = float(cos_sim.item()) / 2 + 0.5

        return normalized_score
    
    def count_tokens(self, text):
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        return token_count    
    
    def code_analysis(self, title:str, contents:str, github_link:str):
        """
        input : title of paper,
                contents in paper,
                generated code by GPT (response)
        output : explanation about how to implement based on contents
        """
        if not github_link : 
            raise Exception("Github 링크가 필요합니다.")
        # Generate code from paper content
        self.Git_cloning(title, github_link) # Git clone하는 과정

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Extracted code from the repository
        repo_code_files = self.extract_code_from_repo(self.repo_path)
        instruction = f"""
I have two variables that I need help with:\n
code_files: A dictionary where keys are file paths and values are strings of actual code.\n
contents: A string containing a section from a paper that describes a particular implementation detail.\n

I need to achieve the following:\n\n

1.Compare the contents against the values in code_files to find the most similar code snippet.\n
2.Determine which code snippet in code_files best matches the implementation described in contents.\n
3.Provide the file path and the actual code of the selected code snippet.\n
4.Provide an explanation of how the described implementation detail from the paper is realized in the actual code snippet.\n\n

Here are the variables:\n\n
code_files : {repo_code_files}\n\n
contents : {contents}\n\n

Please provide:\n

1.The file path and the most similar code snippet from code_files to the implementation described in contents.\n
2.A detailed explanation of how the selected code snippet implements the concept described in contents.\n
3.The actual code of the selected code snippet in code_files. Please start your response with the following sentence :  "위의 내용들을 기반으로 구현한 예시 코드는 다음과 같습니다." \n\n

Please explain in korean.
"""        
        tokens = self.count_tokens(instruction)
        first_question = f"""Based on the following content from a research paper, write the corresponding Python code that implements the described concept. Provide only the Python code without any additional text or explanation.\n\n
                        Paper Content: \"{contents}\" \n\n
                        """

        generated_code = llm.predict(first_question)
        # 토큰 개수 확인
        if tokens <= 60000:
            return instruction
        
        else:
            # Calculate cosine similarity for each file in the repository
            similarity_scores = {}
            for file_path, code in repo_code_files.items():
                functions = self.split_code_into_functions(code)
                for func_name, func_code in functions.items():
                    similarity = self.calculate_cosine_similarity(generated_code, func_code) # Vectorizer를 이용한 단순한 유사도 측정
                    # similarity = self.answer_quality_score(generated_code, func_code) # Sentence transformer를 이용한 유사도 측정
                    # similarity = self.calculate_similarity_codet5(generated_code, func_code) # CodeT5 모델을 사용한 유사도 측정
                    similarity_scores[(file_path, func_name)] = similarity

            # max_score = max(similarity_scores.values())
            # most_relevant_file, most_relevant_function = [(file_path, func_name) for (file_path, func_name), score in similarity_scores.items() if score == max_score][0]
            # highest_similarity_score = max_score

            most_relevant_file, most_relevant_function = max(similarity_scores, key=similarity_scores.get)
            highest_similarity_score = similarity_scores[(most_relevant_file, most_relevant_function)]

            # 관련 있는 함수가 존재하는 py 파일의 경로 + 함수 + 유사도 제시, 이후 부가적인 설명을 요청하는 prompt
            instruction = f""" First, Please start your response with the following sentence : "Cosine similarity를 기반으로 Github 내에서 논문의 내용과 가장 유사한 함수와 파일 경로는 다음과 같습니다. \n file path : {most_relevant_file}\n , function: {most_relevant_function} \n 더불어 실제로 구현하기 위한 예시 코드는 다음과 같습니다. \n {generated_code}." \n\n\n
            Then, explain in detail how the implementation in the code reflects the theoretical framework or experimental setup described in the paper. Identify any key algorithms or processes in the code that are particularly significant and discuss their importance in the context of the research. 한국말로 설명해. \n
            contents: \"{contents}\" \n
            most relevant code: \"{most_relevant_function}\" \n
            """
            return instruction