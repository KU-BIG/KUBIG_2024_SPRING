import huggingface_hub
huggingface_hub.login(token="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")

import warnings
warnings.filterwarnings('ignore')
import os
# from langchain_openai import ChatOpenAI
# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from typing import Literal, Optional
from langchain.tools import StructuredTool


from getpaper import GetPaper
from getpaper_v2 import GetPaper_v2 # figure include
from recommendpaper import RecommendPaper
from code_analysis import CodeAnalysis


ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

getpapermodule_v2 = GetPaper_v2(ss_api_key, ar5iv_mode = True, path_db = './papers_db', page_limit = 5)

class load_paper_input(BaseModel):
    title: str = Field(description="target paper title")
    sections: list = Field(description='list of sections', default = None)
    arxiv_id: Optional[str] = Field(default=None, description="arxiv id of the paper. use it when the prompt contain arxiv id. arxiv IDs are unique identifiers for preprints on the arxiv repository, formatted as `YYMM.NNNNN`. For example, `1706.03762` refers to a paper submitted in June 2017, and `2309.10691` refers to a paper submitted in September 2023.")
    show_figure: Optional[bool] = Field(default=False, description="show figure in the paper")


# loadpaper를 사용하여 우선 target paper의 section list를 불러온 후, 각 section의 content를 불러오게끔 설정
loadpaper = StructuredTool.from_function(
    func=getpapermodule_v2.load_paper,
    name="loadpaper",
    description="""
        The `loadPaper` tool is designed to facilitate the process of retrieving and reading academic papers based on a given search title. \
        The `title` parameter is a string representing the title of the paper. The 'sections' parameter is a list representing the list of the sections in the paper. \
        The 'arxiv_id' parameter is a string representing the arxiv id. \
        The 'show_figure' parameter is a boolean value that determines whether to display the figures in the paper. \
        If the sections parameter is none, you can get the section list of the paper. If the sections parameter get the section list, you can load the paper's content. \
        Use this tool several times to get the section first and then get the detail content of each section. \
        Do NOT show the figures when 'sections' parameter is None. \
    """,
    args_schema=load_paper_input
)



getpapermodule = GetPaper(ss_api_key, ar5iv_mode = True, page_limit = None)

class load_paper_input_wo_figure(BaseModel):
    title: str = Field(description="target paper title")
    sections: list = Field(description='list of sections', default = None)
    arxiv_id: Optional[str] = Field(default=None, description="arxiv id of the paper. use it when the prompt contain arxiv id. arxiv IDs are unique identifiers for preprints on the arxiv repository, formatted as `YYMM.NNNNN`. For example, `1706.03762` refers to a paper submitted in June 2017, and `2309.10691` refers to a paper submitted in September 2023.")


# loadpaper를 사용하여 우선 target paper의 section list를 불러온 후, 각 section의 content를 불러오게끔 설정
loadpaper_wo_figure = StructuredTool.from_function(
    func=getpapermodule.load_paper,
    name="loadpaper",
    description="""
        The `loadPaper` tool is designed to facilitate the process of retrieving and reading academic papers based on a given search title. \
        The `title` parameter is a string representing the title of the paper. The 'sections' parameter is a list representing the list of the sections in the paper. \
        The 'arxiv_id' parameter is a string representing the arxiv id. \
        If the sections parameter is none, you can get the section list of the paper. If the sections parameter get the section list, you can load the paper's content. \
        Use this tool several times to get the section first and then get the detail content of each section. \
    """,
    args_schema=load_paper_input_wo_figure
)


# load paper without section, without figure.
getpapermodule_wo_figure_wo_section = GetPaper(ss_api_key, ar5iv_mode = False, page_limit = 9)

class load_paper_input_wo_figure_wo_section(BaseModel):
    title: str = Field(description="target paper title")
    arxiv_id: Optional[str] = Field(default=None, description="arxiv id of the paper. use it when the prompt contain arxiv id. arxiv IDs are unique identifiers for preprints on the arxiv repository, formatted as `YYMM.NNNNN`. For example, `1706.03762` refers to a paper submitted in June 2017, and `2309.10691` refers to a paper submitted in September 2023.")


# loadpaper를 사용하여 우선 target paper의 section list를 불러온 후, 각 section의 content를 불러오게끔 설정
loadpaper_wo_figure_wo_section = StructuredTool.from_function(
    func=getpapermodule_wo_figure_wo_section.load_paper,
    name="loadpaper",
    description="""
        The `loadPaper` tool is designed to facilitate the process of retrieving and reading academic papers based on a given search title. \
        The 'arxiv_id' parameter is a string representing the arxiv id. \
    """,
    args_schema=load_paper_input_wo_figure_wo_section
)



recommendpapermodule = RecommendPaper(ss_api_key, threshold = 0.6)

## (reference paper/citation paper) recommendation
class recommend_paper_input(BaseModel):
    query: str = Field(description="target paper title")
    rec_type: Literal['reference', 'citation'] = Field(description="reference or citation paper recommendation")
    rec_num: Optional[int] = Field(default=5, description="number of recommended papers default is 5")  # 기본값 5로 설정

recommendpaper = StructuredTool.from_function(
    func=recommendpapermodule.query2recommend_paper,
    name="recommendpaper",
    description="""
        This 'recommendpaper' tool is designed to recommend relevant academic papers based on a given query, \
        focusing on either the papers cited by the target paper (references) or the papers citing the target paper (citations).\
        The `query` parameter is a string representing the title of the paper.\
        The `rec_type` parameter specifies whether the recommendation should be based on references or citations.\
        The `rec_num` parameter specifies the number of recommended papers. if the number is NOT MENTIONED set rec_num=5\
        The recommendation is based on the cosine similarity between the abstracts of the target paper and its references or citations.\
        Users can specify whether they want recommendations from references or citations. \
        The tool returns the top relevant papers sorted by publication date or influential citation count.
    """,
    args_schema=recommend_paper_input
)

codeanalysismodule = CodeAnalysis(ss_api_key, openai_key, path_db = './code_db')

class code_analysis_inputs(BaseModel):
    title: str = Field(description="target paper title")
    contents : str = Field(description = "Contents in paper")
    github_link : Optional[str] = Field(description = "Generated code by GPT", default = None)

code_matching = StructuredTool.from_function(
    func=codeanalysismodule.code_analysis,
    name="code_matching",
    description="""
    'code_matching' tool provides references for the most closely matching parts between the content of a research paper and the actual implemented code. \
    'title' is a parameter that takes the title of the research paper. \
    'contents' is a parameter where the user inputs the parts of the paper they are curious about how to implement in code. \
    'response' refers to the code generated by GPT based on 'contents'. \
    The 'code_matching' tool takes the title and content of a research paper as input and compares the GPT-generated code with actual implemented codes to identify the parts that are most similar, 
    thus providing references for the implementation process.""",
    args_schema = code_analysis_inputs
)