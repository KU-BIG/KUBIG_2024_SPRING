팀명: B.A.A.M 양갱 = Beta Alpha Amino-acid Model

팀원: 18기 신인수, 19기 이동주, 19기 이지운

# 딥러닝을 이용한 단백질 2차구조 예측


## Motivation
딥러닝을 이용한 단백질 구조 예측이 2021년도 Nature Method of the year로 선정되었다. 우리 팀은 또한 딥러닝을 이용한 단백질 구조예측을 시도해보고자 본 프로젝트를 계획하게 되었다.

## 프로젝트 목표
현재 사용되는 AlphaFold, RoseTTAfold 등은 pytorch 기반이 아니기 때문에, pytorch 기반 단백질 구조 예측 모델을 만들기로 했다. 또한, '단백질 구조 예측'의 범위는 넓기 때문에 좁힐 필요가 있다. Protein-ligand interaction (PLI), protein-protein interaction (PPI), contact prediction, fold classification 등 제한을 두지 않으면 과제가 무궁무진하다. 우리는 비교적 간단한 분류예측 문제인 2차구조 예측을 주제로 성정하였다.

또다른 목표로 프로젝트 시작 당시 4월 부터 CASP16을 observer 자격으로 참여하는 것이 목표였으나, 프로젝트 도중 진행하기 어려울 것을 판단하여 하지 않았다.

## 선행연구
AlphaFold, RoseTTAfold, ColabFold 모두 FASTA file로 단백질 서열을 넣으면 HMM (hidden Markov model)을 이용하여 유사한 서열의 단백질을 데이터베이스(PDB등)로부터 불러온다. 
이를 바탕으로 MSA (multiple sequence alignment) 파일을 바탕으로 training을 하게 된다.

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

딥러닝 모델은 pytorch의 LSTM을 사용하였다. 

전체적인 분석과정은

1. DSSP 파일이 있는 단백질 4 글자 쿼리를 바탕으로 DSSP 파일 다운로드
2. DSSP 파일을 전처리하여 LSTM 모델에 traning

데이터는 DSSP가 있는 파일 216606개 중 20000개를 불러 와서 140000개를 train set, 60000개를 test set으로 선정하였다. 
데이터가 너무 크기 때문에 2000개의 단백질을 묶어 pkl 파일로 저장하였다. 전부 다운받으면 86GB 정도의 크기이다.

모델 구조는 다음과 같다.
- input size = 60
- hidden size = 128
- layers = 2
- output size = 8

## 결과
10 epoch training 결과 average accuracy = 96.63%

50 epoch training 결과 average accucary = 96.80% 
 

## 시행착오 

1. HMM 기법으로 가장 대중적은 방법은 MMSeq2인데, 이를 Google Colab에 실행하는 과정이 잘 안 되어, HMM을 포기
2. DSSP를 불러오면 시간이 오래 걸리는데, Biopython을 이용해서 Bio.PDB.DSSP를 사용하면 DSSP를 불러올 수 있다. 그러나 이것은 약소화된 형식으로 본 프로젝트에 사용하기는 부적절
3. transformer 계열 모델을 추가적으로 training해서 비교하려고 했다. 특히 BERT를 시도를 했는데, 다음과 같은 이유로 부적절하다고 판단
   - BERT는 input이 문자열이기 때문에 추가적이 데이터가 있는 현재 setting에서는 진행하기 어려운 것으로 판단
   - BERT 계열로 단백질 구조 예측을 한 연구는 주로 FASTA -> 구조 순서로 예측하기 때문에, 우리의 setting인 query -> DSSP -> 2차구조 예측 순서와 다름

## 프로젝트 한계
모델의 한계도 명확한 편이다. 

1. 우선 DSSP를 바탕으로 했기 때문에, DSSP 데이터가 있는 단백질의 구조 예측만 가능하다. -> 개선하려면 mmcif 파일로 training
2. testing set이 따로 없어서 전체 데이터를 다운로드 받고, train-test를 나눴다. -> CASP 등 대회를 통해 testing 하기
3. CASP 대회의 경우 형식이 FASTA input, mmcif output이기 때문에 DSSP를 사용하면 해당 task는 수행 불가 -> DSSP외의 최신 데이터베이스 사용
4. DSSP 데이터베이스 자체의 한계
   - mmcif의 경우 모든 원자의 X, Y, Z 좌표가 명시되어, 더 많은 정보 보유
   - DSSP의 경우 Cα의 X, Y, Z 좌표만 있다 -> mmcif output으로 예측 어려움

## 프로젝트 Contribution

**본 프로젝트는 KUBIG 최초의 생물정보학 프로젝트로서 의의를 가진다.** 

후속적으로 관련 프로젝트를 진행할 경우, 본 프로젝트의 시행착오가 도움이 될 것으로 생각된다.

2년 후에 CASP17이 열릴텐데, 이를 도전할 경우 FASTA input을 받고, mmcif output이 나오도록 모델을 훈련시켜야 할 것이다.
DSSP 데이터베이스를 사용하면 CASP 대회에 도전하기 어려울 가능성이 클 것으로 보인다.
