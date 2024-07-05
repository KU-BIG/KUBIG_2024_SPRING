from httpx import NetworkError
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import PyPDF2
import re
import os
import io
import feedparser

import arxiv
import tarfile
import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class NetworkError(Exception):
    """Custom exception for network errors."""
    pass

class GetPaper:
    """
    A class to interact with scientific papers using the Semantic Scholar API, ar5iv and arxiv.

    Attributes:
    -----------
    ss_api_key : str
        The API key for accessing the Semantic Scholar API.
    ar5iv_mode : bool, required.
        If False, download the paper in pdf file in path_db and read it. Defaults to True.
    path_db : str, optional
        The path to the local database where paper is stored. Defaults to './papers_db'.
    page_limit : int, optional
        The maximum number of pages to retrieve when querying papers. 
        Recommend for GPT3.5 is 5, and -1 for GPT4o.
        Defaults to 5.
    """
    def __init__(self, ss_api_key, ar5iv_mode = True, path_db='./papers_db', page_limit = 5):
        self.ss_api_key = ss_api_key
        self.ar5iv_mode = ar5iv_mode
        self.path_db= path_db
        self.page_limit = page_limit

    def get_paper_info_by_title_ss(self, title):
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

    # arxiv api 이용, paper search
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


    def get_ar5iv_url(self, arxiv_id):
        "논문의 ar5iv 주소를 받아오는 함수"
        return f"https://ar5iv.org/abs/{arxiv_id}"

    def get_soup_from_url(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # HTTP 에러가 발생하면 예외를 발생시킴
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        except RequestException as e:
            print(f"Error fetching the URL: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
        
    def get_header_from_soup(self, soup):
        # h1부터 h6까지 태그 추출
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        # 태그와 내용을 계층 구조로 저장
        header_list = [(header.name, header.text.strip()) for header in headers]
        title = header_list[0][1]
        header_list = header_list[1:]
        return title, header_list


    def extract_text_under_headers(self, soup, section_list):
        # 결과를 저장할 변수
        results = []

        # 텍스트 리스트를 순회하며 각 텍스트에 해당하는 헤더와 그 아래의 텍스트를 추출
        for text in section_list:
            header_tag = soup.find(lambda tag: tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and text in tag.get_text())
            if header_tag:
                header_text = header_tag.get_text(strip=False)
                header_level = int(header_tag.name[1])
                current_header = {'tag': header_tag.name, 'text': header_text, 'subsections': []}
                results.append(current_header)

                next_element = header_tag.find_next_sibling()
                while next_element:
                    if next_element.name and next_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        next_level = int(next_element.name[1])
                        if next_level <= header_level:
                            break

                        # If it's a tag and within our header range
                        if next_element.name and next_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            next_text = next_element.get_text(strip=False)
                            next_subheader = {'tag': next_element.name, 'text': next_text, 'subsections': []}
                            current_header['subsections'].append(next_subheader)
                            current_header = next_subheader
                            header_level = next_level
                    else:
                        if 'subsections' not in current_header:
                            current_header['subsections'] = []
                        current_header['subsections'].append({'tag': 'p', 'text': next_element.get_text(strip=False)})

                    next_element = next_element.find_next_sibling()

        content = ''
        for x in results:
            content += x['text']
            for y in x['subsections']:
                content += y['text']
        content = re.sub(r'\n{3,}', '\n\n', content) # 3번 이상 \n 이 연속되면 2번으로 줄이기
        return content

    def list_section(self, header_list):
        section_list = ''
        for tag, text in header_list:
            level = int(tag[1])  # 태그에서 레벨을 추출 (h1 -> 1, h2 -> 2, ..)
            section_list += '  ' * (level - 1) + text +'\n'
        return section_list


    def download_pdf(self, arxiv_id):
        """
        Download the PDF of a paper given its arXiv ID, if it does not already exist.
        """
        if not os.path.exists(self.path_db):
            os.makedirs(self.path_db)
        
        file_path = os.path.join(self.path_db, f'{arxiv_id}.pdf')
        
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping download.")
            return file_path

        pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        response = requests.get(pdf_url)

        if response.status_code != 200:
            raise Exception('Error downloading PDF from arXiv')

        with open(file_path, 'wb') as file:
            file.write(response.content)
        return file_path

    def read_pdf(self, arxiv_id, end_page=None):
        pdf_content = ""
        file_path = f'{self.path_db}/{arxiv_id}.pdf'
    
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                if end_page is None or end_page > total_pages:
                    end_page = total_pages

                for page_num in range(1, end_page):
                    page = reader.pages[page_num-1]
                    pdf_content += page.extract_text()
                    if page_num == self.page_limit:
                        print('Page limit reached at', self.page_limit + 1)
                        break

        except FileNotFoundError:
            return f"Error: The file {file_path} does not exist."
        except Exception as e:
            return f"An error occurred while reading the file: {e}"

        pdf_content = re.sub(r'\s+', ' ', pdf_content).strip()
        return pdf_content

    ### figure visualization part
    def download_arxiv_source(self, arxiv_id):
        """
        <figure download>
        figure가 포함되어 있는 source를 반환한다.
        """
        # 검색한 arXiv 논문 정보 가져오기
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        # 소스 파일 다운로드 링크
        source_url = paper.pdf_url.replace('pdf', 'e-print')
        
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(self.path_db):
            os.makedirs(self.path_db)

        # 파일 다운로드
        response = requests.get(source_url)
        output_path = os.path.join(self.path_db, f"{arxiv_id}.tar.gz")
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        file_path = f'{output_path}'
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=self.path_db)

        output_source = os.path.join(self.path_db, arxiv_id)

        print(f"Source files downloaded to: {output_source}")
        return output_source

    def find_pdf_files(self, root_folder):
        '''
        <figure path find>
        source 폴더 내의 모든 figure를 찾아서 {name:path} 형태의 dictionary 반환
        '''
        print('ROOT',root_folder)
        pdf_files = {}
        for dirpath, dirnames, filenames in os.walk(root_folder):
            print('FILES',filenames)
            print('DIRPATH',dirpath)
            print('DIRNAMES',dirnames)
            for filename in filenames:
                if filename.lower().endswith('.pdf'):
                    name = filename.replace('.pdf', '').split('/')[-1]
                    pdf_files[name] = os.path.join(dirpath, filename)
        print('PDF files :', pdf_files)
        return pdf_files
    
    def query_name_matching(self, query, pdf_files):
        '''
        <figure name matching>
        query와 pdf_files를 받아 query와 가장 유사한 이름을 가진 pdf 파일의 이름을 반환
        '''
        figure_name = list(pdf_files.keys())
        pdf_names = list(pdf_files.keys())
        pdf_names = [name for name in pdf_names]
        pdf_names.insert(0,query)

        encoded_input = self.tokenizer(pdf_names, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        rec = cosine_similarity(sentence_embeddings)
        rec = rec[0][1:]
        max_index = rec.argmax()

        return pdf_names[max_index + 1]

    def display_figure(slef, pdf_files, name):
        """
        <figure display>
        name과 pdf_files를 받아 figure를 display
        """
        pdf_path = pdf_files[name]
        print(pdf_path)
        # PDF 파일 열기
        pdf_document = fitz.open(pdf_path)
        
        # 첫 페이지 가져오기
        page = pdf_document.load_page(0)
        pix = page.get_pixmap()
        
        # 이미지 변환
        image_bytes = pix.tobytes()
        image = Image.open(io.BytesIO(image_bytes))

        plt.imshow(image)
        plt.axis('off')  # 축을 표시하지 않도록 설정합니다
        plt.show()

    ## main code
    def load_paper(self, title:str, sections:list=None, arxiv_id:str=None):
        '''
        INPUT : title of paper,
                list of sections in paper
        OUTPUT : text of the paper
        '''

        if arxiv_id == None or arxiv_id == '':
            try:
                arxiv_id = self.get_paper_info_by_title_ss(title)
            except NetworkError as e:
                try:
                    arxiv_id = self.get_paper_info_by_title_arxiv(title)
                except NetworkError as e:
                    return f"Error: No paper was founded in database"

        url = self.get_ar5iv_url(arxiv_id)
        soup = self.get_soup_from_url(url) if self.ar5iv_mode else None

        if (soup):
            title, header_list = self.get_header_from_soup(soup)
            if sections == None or sections == []:
                sections_list = self.list_section(header_list)
                instruction_for_agent = f'Here is the title and section of the paper in HTML\ntitle\n{title}\nsections\n{sections_list}\n\n Use the \'loadpaper\' tool again, specifying the section list you want to view in detail.'
                return instruction_for_agent
            else:
                content = self.extract_text_under_headers(soup, sections)
                if content == '': # Section list was incorrect
                    sections_list = self.list_section(header_list)
                    content = f'Section list was incorrect.\nHere is the title and section of the paper\ntitle\n{title}\nsections\n{sections_list}\n\n Use the \'loadpaper\' tool again, specifying the section list you want to view in detail.'
                
                return content
            
        else: # case for ar5iv is not exist or request error. assume that arxiv_id is correct
            try:
                download_path = self.download_pdf(arxiv_id)
                pdf_content = self.read_pdf(arxiv_id)

                return pdf_content
            except:
                raise Exception(f'Error downloading PDF from arXiv using arxiv id {arxiv_id}')

