import warnings
warnings.filterwarnings('ignore')
import torch
import os
import numpy as np
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import time



from httpx import NetworkError
import requests
from requests.exceptions import RequestException

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class NetworkError(Exception):
    """Custom exception for network errors."""
    pass

class RecommendPaper:
    """
    A class to recommend papers based on the given target paper.
    When type is 'citation' it returns the papers that cite the target paper.(target paper를 인용한 후속 논문 추천)
    When type is 'reference' it returns the papers that are referenced by the target paper.(target paper가 인용한 이전 논문 추천)
    """
    def __init__(self, ss_api_key, threshold = 0.6):
        """
        Parameters
        ----------
        ss_api_key : str
            Semantic Scholar API key.
        tokenizer : AutoTokenizer
        model : AutoModel
        """
        self.ss_api_key = ss_api_key
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # references paper list
    def query2references(self, query, num=20):
        """
        <parameter>
        - query : target paper name
        - num : 반환할 reference paper 최대 개수
        <return>
        - target paper의 정보
        - reference paper(target paper가 인용한 논문)들의 정보 include context, intend
        """    
        # Define the API endpoint URL
        url = 'https://api.semanticscholar.org/graph/v1/paper/search?fields=paperId,title,abstract'

        # paper name 기입
        query_params = {'query': query}
        headers = {'x-api-key': self.ss_api_key}
        try:
            try:
                response = requests.get(url, params=query_params, headers=headers)
                response.raise_for_status()
            except:
                time.sleep(1)
                response = requests.get(url, params=query_params, headers=headers)
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Error fetching data from Semantic Scholar API: {e}")
        try:
            response = response.json()
        except ValueError:
            raise NetworkError("Error parsing JSON response from Semantic Scholar API")
        paper_id = response['data'][0]['paperId']


        fields = '?fields=title,publicationDate,influentialCitationCount,contexts,intents,abstract'
        """
        context ; snippets of text where the reference is mentioned
        intents ; intents derived from the contexts in which this citation is mentioned.
        """

        url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references'+ fields

        # Send the API request
        response2 = requests.get(url=url, headers=headers).json()
        # response

        limit = 10
        references = []
        for elements in response2['data']:
            try:
                if elements['citedPaper']['influentialCitationCount'] > limit:
                    references.append(elements)
                else: pass
            except: pass

        return response['data'], sorted(references, key=lambda x: x['citedPaper']['influentialCitationCount'], reverse=True)[:num]
    

    def reference_recommend(self, query:str, rec_num, num=20, sorting=True):
        """
        <References Recommendation>
        target paper에서 reference paper가 인용된 부분(context)와 의도(intend) 또한 함께 고려하여 논문 추천
        query(target paper name)가 input으로 들어오면, target paper의 reference 중 유사한 논문들을 추천
        1. target paper와 reference paper의 abstract간의 cosine similarity를 계산
        2. 유사도가 threshold 이상인 논문들을 추천 + recommend 개수만큼 top K filtering
        3. sorting=True 이면, 날짜순으로 정렬 
        """
        recommend = rec_num
        recommend += 1

        ### 상위 num=20개의 reference이 많은 논문들을 select
        target_response, reference = self.query2references(query=query, num=num)
        reference_response = [ref['citedPaper'] for ref in reference]
        reference_context = [ref['contexts'] for ref in reference]
        reference_intent = [ref['intents'] for ref in reference]


        ## target 논문과 num개의 reference 사이의 유사도 계산
        abs_dict = {}
        abs_dict[target_response[0]['title']] = target_response[0]['abstract']

        for keyword in reference_response:
            paper_id, title = keyword['paperId'], keyword['title']
            abstract = str(keyword['abstract'])
            abs_dict[title] = abstract

        sentences = list(abs_dict.values())
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        rec = cosine_similarity(sentence_embeddings)
        
        ## 유사도가 threshold 이상인 논문들을 추출
        indices = np.where(rec[0][1:] > self.threshold)[0]
        rec_lst = [reference_response[i] for i in indices]
        rec_context = [reference_context[i] for i in indices]
        rec_intent = [reference_intent[i] for i in indices]

        for item in rec_lst:
            if item['publicationDate'] is None:
                item['publicationDate'] = datetime.max

        # intent와 context를 list에 추가
        for i in range(len(rec_lst)):
            try:
                rec_lst[i]['intent'] = ' '.join(rec_intent[i])
                rec_lst[i]['context'] = rec_context[i][0]
            except: 
                rec_lst[i]['intent'] = 'None'
                rec_lst[i]['context'] = 'None'
        if recommend:
            if len(rec_lst) > recommend:
                rec_lst = rec_lst[1:recommend]

        # 날짜순으로 정렬
        if sorting==True:
            rec_lst = sorted(rec_lst, key=lambda x: x['influentialCitationCount'], reverse=True)[:num]
            rec_lst = sorted(rec_lst, key=lambda x: datetime.strptime(x['publicationDate'], '%Y-%m-%d') if isinstance(x['publicationDate'], str) else x['publicationDate'])
        
        updated_result = []
        for paper in rec_lst:
            paper.pop('paperId', None)  # 'paperId'가 없는 경우에도 에러가 발생하지 않도록 함
            updated_result.append(paper)

        return updated_result
    

    def query2citations(self, query, num=20):
        '''
        <Citations>
        query(target paper name)가 input으로 들어오면, target paper의 citation(target paper를 인용한 논문)을 가져온다.
            최신 논문들은 citation이 많이 없어 따로 influentialCitationCount filtering X
            num : 반환할 citation paper 최대 개수
        Citation paper를 Citation 순으로 정렬
        '''
        ## Citation paper list
        url = 'https://api.semanticscholar.org/graph/v1/paper/search'

        # paper name 기입
        query_params = {'query': query,'fields': 'citations,citations.influentialCitationCount,citations.title,citations.publicationDate,citations.abstract'}

        headers = {'x-api-key': self.ss_api_key}
        try:
            response = requests.get(url, params=query_params, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Error fetching data from Semantic Scholar API: {e}")
        try:
            citations_response = response.json()
            paper_id = citations_response['data'][0]['paperId']
        except ValueError:
            raise NetworkError("Error parsing JSON response from Semantic Scholar API")

        ## target paper information
        url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=abstract,title'
        try:
            response = requests.get(url, params=query_params, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"현재 Semantic Scholar API가 불안정해서 에러가 발생했습니다. 잠시후에 다시 진행해주세요(API 키가 무료버전이여서 그래요 ㅜㅜ)")
            # raise NetworkError(Error fetching data from Semantic Scholar API: {e}")

        try:
            target_response = response.json()
        except ValueError:
            raise NetworkError("Error parsing JSON response from Semantic Scholar API")

        def get_citation_count(item):
            influential_citation_count = item.get('influentialCitationCount')
            if influential_citation_count is not None:
                return influential_citation_count
            else:
                return 0
            
        return target_response, sorted(citations_response['data'][0]['citations'], key=get_citation_count, reverse=True)[:num]
    
    def citation_recommend(self, query, rec_num, num=20):
        '''
        <Citations Recommendation>
        query(target paper name)가 input으로 들어오면, target paper의 citation 중 유사한 논문들을 추천
        1. target paper와 citation paper의 abstract간의 cosine similarity를 계산
        2. 유사도가 threshold 이상인 논문들을 추천 + recommend 개수만큼 top K filtering
        3. publicationDate를 기준으로 날짜순으로 정렬
        '''
        recommend = rec_num
        recommend += 1

        ### 상위 20개의 citation이 많은 논문들을 select
        target_response, citation_response = self.query2citations(query=query, num=num)

        ## target 논문과 20개의 citation 사이의 유사도 계산
        abs_dict = {}
        abs_dict[target_response['title']] = target_response['abstract']

        for keyword in citation_response:
            paper_id, title = keyword['paperId'], keyword['title']
            abstract = str(keyword['abstract'])
            abs_dict[title] = abstract

        sentences = list(abs_dict.values())
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        rec = cosine_similarity(sentence_embeddings)
        
        ## 유사도가 threshold 이상인 논문들을 추출
        indices = np.where(rec[0][1:] > self.threshold)[0]
        rec_lst = [citation_response[i] for i in indices]

        for item in rec_lst:
            if item['publicationDate'] is None:
                item['publicationDate'] = datetime.max

        if len(rec_lst) > recommend:
            rec_lst = rec_lst[1:recommend]

        # 날짜순으로 정렬
        rec_lst = sorted(rec_lst, key=lambda x: datetime.strptime(x['publicationDate'], '%Y-%m-%d') if isinstance(x['publicationDate'], str) else x['publicationDate'])

        updated_result = []
        for paper in rec_lst:
            paper.pop('paperId', None)  # 'paperId'가 없는 경우에도 에러가 발생하지 않도록 함
            updated_result.append(paper)

        return updated_result
    

    def query2recommend_paper(self, query, rec_type, rec_num=5):
        '''
        type에 따라 target paper에 대한 citation 혹은 reference를 추천
        '''
        if rec_type == 'citation':
            return self.citation_recommend(query=query, rec_num=rec_num, num=30)
        elif rec_type == 'reference':
            return self.reference_recommend(query=query, rec_num=rec_num, num=30)
        else:
            raise Exception('citation 논문들을 추천받을지 reference 논문들을 추천받을지 입력해줘야 합니다!')