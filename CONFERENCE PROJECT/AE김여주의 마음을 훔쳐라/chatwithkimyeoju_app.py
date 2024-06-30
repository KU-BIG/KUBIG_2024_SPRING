import streamlit as st
from streamlit_chat import message
import base64
import json
import http.client
import ssl
import requests
import re

class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request_first(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/tasks/a3r5loz8/completions', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def _send_request_second(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/tasks/fv4465n4/search', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def _send_request_third(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }
        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/tasks/hayct9t3/search', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result
  

    def execute_first(self, completion_request):
        res = self._send_request_first(completion_request)  # _send_request_first 메서드 호출
        if res['status']['code'] == '20000':
            return res['result']['text']
        else:
            return 'Error'

    def execute_second(self, completion_request):
        res = self._send_request_second(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['outputText']
        else:
            return 'Error'

    def execute_third(self, completion_request):
        res = self._send_request_third(completion_request) 
        if res['status']['code'] == '20000':
            return res['result']['outputText']
        else:
            return 'Error'




st.title("김여주의 마음을 훔쳐라!✉️")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

input_2 = ""; input_3 = ""

st.sidebar.title('AI 김여주와 대화해 보세요!')
with st.sidebar.form('form', clear_on_submit=True):
    user_input = st.text_area('You: ', '', key='input', height=50).strip()
    submitted = st.form_submit_button('Send')

    if submitted and user_input:
        with st.spinner("김여주 님이 입력중..."):

            # First API Call
            completion_executor_1 = CompletionExecutor(
                host='clovastudio.apigw.ntruss.com',
                api_key='NTA0MjU2MWZlZTcxNDJiYzpCdGaklIyXuEPpOZHGfsvYxzac629fvGv456Y10sgW',
                api_key_primary_val = 'H9SKYi4EHagBD0Nep7SWJ0T9LD4Z5vurEIuSQnpY',
                request_id='b23967f0-2a8d-433d-96a1-3ae90e2b4a3b'
            )

            preset_text = user_input

            request_data_1 = {
                'text': preset_text,
                'start': '',
                'restart': '',
                'includeTokens': False,
                'topP': 0.8,
                'topK': 4,
                'maxTokens': 300,
                'temperature': 0.85,
                'repeatPenalty': 8.0,
                'stopBefore': ['<|endoftext|>'],
                'includeAiFilters': True,
                'includeProbs': False
            }

            response_text_1 = completion_executor_1.execute_first(request_data_1)

            st.session_state.past.append(user_input)  # Update past to current user input
            st.session_state.generated.append(response_text_1)  # Clear previous generated and start with new

            input_3 += user_input; input_3 += response_text_1
            input_2 += user_input; input_2 += response_text_1       

            # Second API Call (Emotion Score API)
            completion_executor_2 = CompletionExecutor(
                host='clovastudio.apigw.ntruss.com',
                api_key='NTA0MjU2MWZlZTcxNDJiY1QtvFM+zuln9DqRrtQMT5Xq1wrs69nE9dzZ3+hGRAH8',
                api_key_primary_val='jgsmH29fjM0b3GEEPQV008G7bAyzEG48L6DMXxvb',
                request_id='7619af78-5dfb-47e9-90fc-07fa15e2b101'
            )

            preset_text = input_2

            request_data_2 = {
                'text': preset_text,
                'includeAiFilters': True
            }

            response_text_2 = completion_executor_2.execute_second(request_data_2)

            st.session_state.generated.append(response_text_2)  # Store second API result

            # Third API Call (Cumulative Emotion API)
            completion_executor_3 = CompletionExecutor(
                host='clovastudio.apigw.ntruss.com',
                api_key='NTA0MjU2MWZlZTcxNDJiY1QtvFM+zuln9DqRrtQMT5Xq1wrs69nE9dzZ3+hGRAH8',
                api_key_primary_val='jgsmH29fjM0b3GEEPQV008G7bAyzEG48L6DMXxvb',
                request_id='7e89c200-0e68-4f73-a46f-18155afb2c1c'
            )

            preset_text = input_3

            request_data_3 = {
                'text': preset_text,
                'includeAiFilters': True
            }

            response_text_3 = completion_executor_3.execute_third(request_data_3)

            st.session_state.generated.append(response_text_3)  # Store second API result

with st.expander("소중한 김여주와의 대화 기록"):
    if len(st.session_state['past']) > 0 and len(st.session_state['generated']) > 0:
        past_length = len(st.session_state['past'])
        generated_length = len(st.session_state['generated'])

        i = 0
        j = 0

        while i < past_length or j < generated_length:
            if i < past_length:
                message(st.session_state['past'][i], is_user=True, key=f"past_{i}")
                i += 1
            if j < generated_length:
                message(st.session_state['generated'][j], key=f"generated_{j}")
                j += 1
                if j < generated_length:
                    message(st.session_state['generated'][j], key=f"generated_{j}")
                    j += 1
                    if j < generated_length:
                        message(st.session_state['generated'][j], key=f"generated_{j}")
                        j += 1

