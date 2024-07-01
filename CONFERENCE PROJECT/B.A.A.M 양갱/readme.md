팀명: B.A.A.M 양갱 = Beta Alpha Amino-acid Model

팀원: 18기 신인수, 19기 이동주, 19기 이지운

# 딥러닝을 이용한 단백질 2차구조 예측


## Motivation
딥러닝을 이용한 단백질 구조 예측이 2021년도 Nature Method of the year로 성정되었다. 우리 팀은 또한 딥러닝을 이용한 단백질 구조예측을 시도해보고자 본 프로젝트를 계획하게 되었다.

## 프로젝트 목표
현재 사용되는 AlphaFold, RoseTTAfold 등은 pytorch 기반이 아니기 때문에, pytorch 기반 단백질 구조 예측 모델을 만들기로 했다. 또한, '단백질 구조 예측'의 범위는 넓기 때문에 좁힐 필요가 있다. Protein-ligand interaction (PLI), protein-protein interaction (PPI), contact prediction, fold classification 등 제한을 두지 않으면 과제가 무궁무진하다. 우리는 비교적 간단한 분류예측 문제인 2차구조 예측을 주제로 성정하였다.

또다른 목표로 프로젝트 시작 당시 4월 부터 CASP16을 observer 자격으로 참여하는 것이 목표였으나, 프로젝트 도중 진행하기 어려울 것을 판단하여 하지 않았다.

## 선행연구
AlphaFold, RoseTTAfold, ColabFold 모두 FASTA file로 단백질 서열을 넣으면 HMM (hidden Markov model)을 이용하여 유사한 서열의 단백질을 데이터베이스(PDB등)로부터 불러온다. 
이를 바탕으로 MSA (multiple sequence alignmen) 파일을 바탕으로 training을 하게 된다.

## Method

4글자 PDB 쿼리를 이용하여 DSSP (dictionary of secondary structure protein) [링크](https://swift.cmbi.umcn.nl/gv/dssp/) 파일을 불러온다. DSSP는 단백질의 2차구조를 다음과 같이 8가지로 분류한다. 

- H = α-helix
- B = residue in isolated β-bridge
- E = extended strand, participates in β ladder
- G = 3-helix (310 helix)
- I = 5 helix (π-helix)
- T = hydrogen bonded turn
- S = bend

DSSP의 핵심이라고 할 수 있는 부분은 Structure 열인데, 그에 대한 설명은 [이 레퍼런스 참조](https://pubs.acs.org/doi/full/10.1021/ci5000856) (거의 유일한 레퍼런스).

구조의 특징에 대한 9글자 문자열인데, 각 글자의 의미는 다음과 같다.

- {' ', '<', '>', 'P'}: structure_detail1-> helix 종류
- {' ', '3', 'X', '<', '>'}: structure_detail2 -> helix 종류
- {' ', '4', 'X', '<', '>'}: structure_detail3 -> helix 종류
- {' ', 'X', '<', '>', '5'}: structure_detail4 -> helix 종류
- {' ', 'S'}: structure_detail5 -> Bend 위치
- {' ', '-', '+'}: structure_detail6 -> chirality
- {'U', 's', 'Q', 'O', 'e', 'g', 'R', 'd', 'D', 'Z', 'H', 'w', 'v', 'V', 'G', 'a', 'p', 'W', 'l', 'M', 'C', 'o', 'S', 'j', 'F', 'i', 'u', 'L', 'K', 'q', 'N', 'm', 'k', 'X', 'z', 'c', 'A', 'Y', 'y', 'b', 't', 'P', ' ', 'f', 'I', 'h', 'n', 'r', 'J', 'E', 'B', 'x', 'T'}: : structure_detail7 -> beta-bridge label
- {'U', 's', 'e', 'O', 'Q', 'g', 'R', 'd', 'D', 'Z', 'H', 'w', 'v', 'V', 'G', 'p', 'a', 'W', 'l', 'M', 'C', 'o', 'S', 'j', 'F', 'i', 'u', 'L', 'K', 'q', 'm', 'N', 'k', 'X', 'z', 'c', 'Y', 'A', 'y', 'b', 't', 'P', ' ', 'f', 'I', 'h', 'n', 'r', 'J', 'E', 'B', 'x', 'T'} structure_detail8-> beta bridge label
 

## 시행착오 
