*Chapter1*

AI -> ML -> DL 순으로 영역 좁아짐. ML의 핵심은 데이터 기반으로 학습한다는 점임. (<-> 규칙 기반)

**AI 번역**

- 번역 문제의 하나의 데이터는 원문과 그에 대응하는 번역문으로 구성된 쌍.
- 학습에 사용되지 않은 새로운 문장을 입력해 번역된 문장의 정확성을 확인해 모델 성능 평가함(Perplexity: 예측 능력 수치화, BLEU Score: 번역 품질 수치화).
- 토크나이징: 텍스트가 숫자가 아니기 때문에 처리하는 전처리 과정. 텍스트를 적절한 단위로 나누는 작업을 의미함.
ex) 나는 학생입니다 -> 나는/학생/입니다 의 세 토큰으로 나눔.
- 정리하면 자연어 처리의 번역 작업은 '텍스트 -> 숫자 시퀀스 변환 후 입력 -> 숫자 시퀀스 출력 -> 텍스트로 변환'임.

**지도 학습(Supervised Learning)**

- 머신러닝의 학습 방식은 지도 학습, 비지도 학습, 자기 지도 학습, 강화 학습으로 나눌 수 있음.
- 지도 학습이란 정답을 알고 있는 상태에서 학습하는 방식.
- 각 입력 데이터에 대한 정답을 부여하는 데이터 라벨링이 필수적
- 회귀(Regression, 연속적 레이블)와 분류(Classification, 이산적 레이블)로 나뉘어짐, 기계 번역은 토큰 분류라는 관점에서 분류이다.

컴퓨터 비전 분야 예시
- 분류(Classification): 이미지가 무엇을 나타내는지 하나의 클래스로 나타내는 것. 세 개 이상의 클래스를 포함하는 다중 분류도 있음.
- 위치 추정(Localization): 분류와 함께 수행되며 이미지 내 객체의 클래스를 판단하고 동시에 위치를 출력하는 것(박스의 중심 좌표, 높이, 너비를 나타냄). 박스 정보 예측은 회귀임.
- 객체 탐지(Object Detection): 한 이미지 내 여러 객체의 분류 및 위치 추정을 동시 수행.
ex) YOLO는 이미지를 그리드로 나누어 각 그리드 셀에서 분류 및 각각의 바운딩 박스에 대한 신뢰도(그리드 셀 내 객체의 중심이 존재할 가능성과 예측된 박스가 실제 객체와 겹치는 정도를 곱한 값)와 박스 정보(중심 좌표, 높이, 너비)를 예측함.
- 분할(Segmentation): 이미지의 모든 픽셀을 대상으로 각 픽셀이 어떤 클래스에 해당하는지 판단(이제 네모가 아닌 객체의 모양으로 출력 가능)
- 인스턴스 분할(Instance Segmentation): 같은 클래스의 서로 다른 객체 구분.
- 자세 추정(Pose Estimation): 사람의 주요 신체 부위 좌표 예측. -------------|
- 얼굴 랜드마크 탐지(Facial Landmark Detection): 얼굴의 주요 특징점 예측. --|-> 이 두 개는 각 좌표에 대한 정확한 레이블링이 필요해 비용이 상당하다..

**자기 지도 학습(Self-Supervised Learning)**
- 지도 학습의 가장 큰 단점인 대량의 레이블링된 데이터를 준비하기 위해 비용과 시간이 많이 든다는 것을 해결하기 위해 등장.
- 레이블이 없는 데이터로 학습.
- 사전 학습(Pre-Training)과 미세 조정(Fine-Tuning)의 두 단계로 진행됨.

- 사전 학습 단계: 실제 풀고자 하는 진짜 문제(Downstream Task) 대신 가짜 문제(Pretext Task)를 새롭게 정의하여 해결(이 과정에서 레이블이 없는 데이터 활용).
- 미세 조정 단계: 레이블이 있는 데이터를 이용하여 일반적인 지도 학습 방식으로 모델 조정.

컴퓨터 비전 분야 예시(Context Prediction, Contrastive Learning)
Context Prediction: 사전 학습 단계에서 이미지의 구조 및 객체 간 위치 관계 학습 -> 이미지의 전반적인 구조 파악.
- 학습 과정: 이미지에서 무작위로 위치를 선정해 특정 크기의 파란색 패치를 둠 -> 파란색 패치 주변에 동일한 크기의 패치들을 배치 -> 모델이 파란색 패치와 주변 패치들의 상대적 위치 관계를 예측하도록 학습.
- 장점: 레이블 없는 데이터에 적용 가능, 패란색 패치의 위치를 임의로 선정할 수 있기에 학습 데이터를 무한히 생성 가능.

Contrastive Learning: 출처가 같다면 당기고, 출처가 다르면 밀어내는 학습 방식 -> 이미지 간 유사도 파악.
- 학습 방식: 하나의 이미지에 서로 다른 두 가지 변형(Augmentation)을 가한 후 변형된 이미지 쌍의 출처가 같은지 다른지 인식하도록 학습.
- 학습 규칙: 같은 이미지에서 변형된 쌍은 출력값을 서로 가깝게 만들고 다른 이미지에서 변형된 쌍은 출력값을 서로 멀어지게 만듦.

자연어 처리 분야 예시(GPT, BERT) <- 텍스트 그 자체가 입력이자 정답이 되므로 인위적 레이블링 필요없음.
GPT: 다음 단어를 예측하는 방식으로 학습
BERT: 문장에 빈칸을 만들고 빈칸에 알맞은 토큰을 예측(Masked Token Prediction)하는 방식과 두 문장이 연속된 문장인지 예측(Next Sentence Prediction)하는 방식을 동시에 사용.

**비지도 학습(Unsupervised Learning)**
- 정답이 주어지지 않은 상태에서 데이터의 특징을 스스로 학습하는 방식.

군집화(Clustering)
- 비슷한 특성을 가진 데이터들을 그룹으로 묶는 방법.
- K-means, DBSCAN 등이 있음.

차원 축소
- 데이터의 중요 특성을 유지하면서 데이터의 복잡성을 줄임.
- PCA, SVD 등이 있음.

**강화 학습(Reinforcement Learning)**
- 특정한 행동을 강화시키는 학습 방식.
- 행동(Action)에 대한 보상(Reward)을 줌으로써 그 행동을 강화하는 것이 핵심.
- 알파고도 강화학습의 예시임(이기는 수를 뒀을 때 해당 행동에 Reward를 줌).
- 규칙을 알려주지 않아도 적절한 보상을 주는 것만으로 학습시키는 것이 핵심.
- 용어1: Agent(행동을 취하는 주체), Action(Agent가 취할 수 있는 모든 행동), Reward(Agent가 Action에 따라 받게 되는 보상. 강화 학습의 핵심 전제는 Agent가 Reward를 최대화하려 한다는 것), Environment(강화 학습이 일어나는 공간)
- 용어2: State(환경의 현재 상태를 나타냄), 행동 가치 함수 Q(특정 State에서 특정 Action을 했을 때 현재와 미래에 얻을 수 있는 Reward 합의 기댓값), Episode(완료까지의 하나의 완전한 시행)
- 용어3: Q-Learning(Episode 내에서 Action을 여러 번 수행하여 Q 값을 반복적으로 업데이트해 최적의 행동 가치를 학습하는 방법. 현재 State와 Action, Reward, 다음 State의 정보를 사용하여 Q값 갱신), 심층 강화 학습(Deep Reinforcement Learning: Q-Learning에서 Q값을 딥러닝을 이용해 학습)
- 용어4: Exploration(기존에 학습하지 않은 새로운 방법을 찾는 것, &epsilon;-Greedy 기법의 &epsilon;값을 0과 1 사이로 조정해 일탈 빈도 조정), Exploitation(기존 지식을 활용하는 것, Exploration과 균형을 이루어야 함), Discount Factor &gamma;(Q값을 현재 시점으로 가져올 때 곱하는 0과 1 사이의 값. 1에 가까울수록 미래의 보상을 중요하게 여김)

*Chapter2*

**인공 신경망(Artificial Neural Network)**
- 곱하고 더하고 액티베이션하는 과정의 연속
- 활성화 함수(Activation Function): 들어오는 값에 따라 나가는 값 결정, Unit Step Function(계단 함수: 들어온 값 총합 양수면 1, 아니면 0), Linear Activation(선형 액티베이션: 들어온 값 그대로 출력) 등 존재.
- weight, bias를 넣어 민감도 조절.
- 노드끼리 모두 연결된 층은 FC(Fully-Connected) 레이어라고 함
- MLP(Multi-Layer Perceptron): 인풋, 아웃풋, 히든 레이어를 하나 이상 가지면서 모든 레이어가 FC 레이어인 신경망
- 딥러닝의 핵심은 weight와 bias를 어떻게 조정하는지이고 이것을 학습이라 함(weight만 조정하는 것이 아님!).

**Loss**
- Loss를 단순히 각각 데이터 세트의 loss를 더한 값으로 하면 양수와 음수 모두 포함되기 때문에 정확한 loss값 추정이 어려울 수 있음 -> 제곱해서 더하거나 절댓값을 더해야 함. 단, 제곱 시에는 Outlier(이상치)에 더 민감하게 반응 -> 절댓값 에러보다 이상치에 더 가깝게 예측선을 수정함.
MSE(Mean Squared Error) Loss: 에러를 제곱한 후 더해 평균냄.
MAE(Mean Absolute Error) Loss: 에러에 절댓값을 씌운 후 더해 평균냄.

**경사 하강법**
- Gradient: 함숫값이 가장 가파르게 증가하는 방향.
- Learning Rate: Gradient 반대 방향으로 이동할 때의 보폭 -> 너무 작으면 최소점에 도달하는데 오래 걸리고 너무 크면 최소점을 지나 발산하거나 불안정한 모습 보임. 일반적으로 0.1, 0.01, 0.001 등을 사용.
- 두 가지 문제점: 계산 속도가 느리다(모든 데이터를 고려하기 때문), 좋지 않은 Local Minimum에 빠질 수 있다. -> SGD를 이용하여 해결.

확률적 경사 하강법(Stochastic Gradient Descent)
- 모든 데이터의 Loss를 고려하는 GD와 달리 단 하나의 데이터만을 무작위로 선택해 Loss 계산
- 기존 GD의 속도 문제와 Local Minimum에 빠지는 문제 완화 가능
- Mini-Batch Gradient Descent: 단 하나의 데이터가 아닌 Mini-Batch 데이터에 대한 Loss 계산하는 방식. 대규모 데이터셋에서는 단순 SGD보다 성능 좋음.
- GD와 SGD는 Loss함수의 등고선이 타원형일 때 효율적인 학습이 어려움.
- GD와 SGD는 현재의 Gradient만 고려함 -> 과거의 Gradient를 고려하는 새로운 방법들을 아래에 소개함.

Momentum
- 이전 Gradient들을 누적하여 현재의 이동 방향 결정(관성과 비슷).

RMSProp(Root Mean Squared Propagation)
- Momentum과 달리 각 파라미터에 대한 편미분값을 제곱하여 누적.
- 가파른 축으로는 조심스럽게, 완만한 축으로는 과감하게 이동하는 효과를 줌.
- 급격한 이동 방지 및 평평한 영역을 빠르게 탈출 가능. -> 학습의 안정성을 높임.

Adam(Adaptive Moment Estimation)
- Momentum과 RMSProp의 장점을 결합한 최적화 알고리즘
- Momentum의 관성 효과, PMSProp의 적응적 이동 방향 조정, 학습 초반 편향 문제 모두 해결

**웨이트 초기화(Weight Initialization)**
- 세 가지 방식이 널리 알려짐. 공통적으로 weight를 평균이 0인 랜덤한 값으로 초기화하며 분산이 다른 차이점이 있음.
- 모든 층의 웨이트를 하나의 정규분포로 초기화하는것이 아닌 각 층마다 정규분포로 초기화하는것임.

Yann LeCun 초기화
- ![image](https://github.com/user-attachments/assets/4e4d6929-f9e5-479c-a658-b254dab6af64) 또는 ![image](https://github.com/user-attachments/assets/73fea43e-783c-43a3-9fc9-c03a8038a925)
- 두 분포 모두 평균 0, 분산 1/N_in, 단 가우시안 분포는 0 주변에 더 집중된 값을 선택함.

Kaiming He 초기화
- ![image](https://github.com/user-attachments/assets/a03ff620-be0e-4446-b4bd-ad276018d53e)
 또는 ![image](https://github.com/user-attachments/assets/f19f932e-c5c3-47ff-8480-8a2fa4a0950c)
- 두 분포 모두 평균 0, 분산 2/N_in, LeCun 방식보다 2배 더 큰 분산임.
- ReLU 활성화 함수 사용할 때 특히 효과적인 것으로 알려져 있음.

Xavier 초기화
- ![image](https://github.com/user-attachments/assets/365b09e2-b21e-48b4-af4d-7655bf3bc8a5)
 또는 ![image](https://github.com/user-attachments/assets/a98d0a92-3b8e-4e79-b272-a72f60b564b6)
- 두 분포 모두 평균 0, 분산 2/(N_in + N_out), 다른 초기화와 달리 N_out도 고려.
- 다른 방식들보다 작은 분산으로, 0에 더 가깝게 초기화.
- Sigmoid나 tanh과 같은 활성화 함수 사용할 때 특히 효과적인 것으로 알려져 있음.

- N_in을 고려하는 이유는 activation에 들어가는 개수가 많아질수록 분산이 커지므로 이를 줄여주기 위해서임 -> 분산이 너무 크면 그래디언트 소실(Vanishing Gradient: 그래디언트가 0에 가까워짐) 문제 발생.
- N_out을 고려하는 이유는 역전파 과정을 위함(이후 자세히 다룸) -> 그래디언트 폭발(Exploding Gradient)문제 발생.

**Batch Size와 Learning Rate의 조절**
- Batch Size는 하나의 Batch에 몇 개의 데이터가 들어갈건지를 의미함(등분x).
- 일반적으로 Batch Size가 커질수록 Validation error가 증가한다. 왜냐하면 배치 크기가 클수록 모델이 훈련 데이터에 과적합되는 경향이 있기 때문임. (배치 크기가 크면 특정한 최소점 도달이 더 쉽고 배치마다의 특성이 비슷해지기 때문)
- 이를 해결하기 위해서는 Linear Scaling Rule(Batch Size를 늘리면 Learning Rate도 비례해서 키움), Learning Rate Warmup(학습 초기에 Learning Rate를 0에서 시작하여 점진적으로 증가시킴)을 사용함(Learning Rate를 조절하는 것을 Learning Rate Scheduling이라 부름).

**K-fold cross validation**
- 데이터가 부족할 때 검증 데이터를 더 효과적으로 활용하기 위해 사용 -> 훈련 데이터 개수가 너무 적으면 편향 문제가 발생할 수 잇기 때문임.
- K개의 훈련/검증 데이터 조합을 만들고 각 조합에 대한 평균 Loss를 구함 -> K배의 학습 시간이 필요한 단점이 있음.
- 보통 하이퍼파라미터를 케이스를 나누고 해당 케이스마다 K-fold를 적용해 평균 loss가 가장 낮은 케이스의 하이퍼파라미터를 채용함. 그 이후에는 해당 케이스로 검증 데이터 없이 전부 훈련에 사용해 모델을 만들거나 K개의 모델을 앙상블하는 방식 중 선택하게 됨. 

*Chapter3*

**선형 Activation과 비선형 Activation**
- activation이 비선형인 이유는 선형이라면 층을 복잡하게 해도 1개 층을 가지는것과 같기 때문임.. 이러면 비선형 관계를 나타낼 수 없다.
- 선형 activation과 비선형 activation을 섞어써도 선형 층들은 하나로 축약되어버림.
- 그래도 선형 activation은 필요하다.. 특히 회귀 문제의 경우 마지막 층에 사용되어야 함. 출력값의 범위가 제한되면 안되기 때문.
- 특히 모델 중간에서 사용되어도 유용할 수 있음. 노드 수가 줄어드는 레이어에서는 비선형성을 포기하되 정보 손실을 막고 노드 수가 늘어나는 레이어는 정보 손실을 최소화하며 충분한 비선형성을 얻기 좋기 때문임. (노드 수가 줄어드는 층에서는 차원 축소로 인한 정보 손실이 발생함.)

ReLU(Rectified Linear Unit)
- 양수 입력은 그대로 출력, 음수 입력은 0.
- 음수 입력을 0으로 만들기 때문에 정보 손실이 일어날 수 있음.
- 노드 수가 줄어드는 층에서는 음수 입력과 차원 축소로 인한 두 가지 손실이 결합되어 중요한 정보가 많이 사라질 수 있음. -> 이 때는 선형 액티베이션을 쓰자.

- 결론: 비선형 activation은 네트워크의 복잡도를 높이지만 대신 정보 손실을 야기할 수 있음. 선형 activation은 정보 손실이 없지만 복잡도를 증가시키지 못함. 이 둘을 조합해서 사용해야 함.

- 추가적으로 노드 수가 증가할 때는 정보 손실이 없지만 추가적인 정보가 생성되는 것은 아님.

**역전파(Backpropagation)**
- 다시 돌아와서 보자.

*Chapter4*

- 이진 분류(Binary Classification)에는 Sigmoid 사용, 다중 분류(Multiclass Classification)에는 Softmax 사용.
- 이진 분류에는 BCE(Binary Cross-Entropy) Loss 사용, 다중 분류에는 Cross-Entropy Loss 사용.
- 퍼셉트론: hidden layer 없이 unit step function을 활성화 함수로 사용허눈 단층 신경망 모델.
- 선형 분류: 분류 경계가 선형인 경우. (입력과 출력의 관계는 비선형적일수 있음.)

Unit Step Function을 사용한 모델의 두 가지 문제점 -> Sigmoid 도입해서 해결.
- 미분 불가능: 역전파 과정에서 모든 파라미터에 대한 편미분이 0이 되어 학습 불가능.
- 극단적 분류: 출력값이 0 또는 1이기에 분류 경계선 근처의 미묘한 차이 반영X.

Sigmoid function
- ![image](https://github.com/user-attachments/assets/1d923ba8-0562-48e9-a46a-abbf043d0dc9)
- 사실 이 수식은 다양한 Sigmoid 중 Logistic function임. (그래프가 이와 같이 S자형을 갖는 모든 함수들을 Sigmoid라고 총칭함.)
- 전 구간 미분 가능 -> 그래디언트 최적화 가능.
- 출력값의 범위가 0 ~ 1이므로 확률로 해석 가능. -> 더 합리적인 분류 경계선을 찾기 용이하다.

BCE Loss(MLE: Maximum Likelihood Estimation 관점에서 해석)
- ![image](https://github.com/user-attachments/assets/c6594c0b-1e28-4493-b1cc-c167b1e1b783)
- N은 샘플의 총 개수, y는 실제 레이블, y_hat은 모델이 예측한 확률.
- Underflow 문제: ![image](https://github.com/user-attachments/assets/bb542001-5064-4bc5-9ffb-6130b5f8b348) 기본적으로는 이와 같이 식이 나오는데, 곱하는 과정에서 값이 계속 작아지기 때문에 로그를 취함(컴퓨터의 부동소수점 문제 해결).
- 로그 내부의 값이 0과 1 사이이므로 로그 앞에 -1을 곱해 양수로 만들어줌. -> 굳이 양수로 만드는 이유는 '최소화'문제로 바꾸기 위함임.

로지스틱 회귀(Logistic Regression)
- 입력과 출력 사이의 관계를 확률 함수로 표현하고 이 함수를 은닉층이 없는 인공 신경망으로 놓고 추정하는 방법.
- 분류 문제를 다루지만 회귀라는 이름을 가짐(근본적으로 두 방식이 같은 접근 방식이기 때문).
- 로지스틱 회귀는 입력과 출력 사이의 관계를 Logistic 함수로 놓고 이 함수의 파라미터를 추정하는 것을 목표로 함.
- 또 다른 관점에서는 Logit(= Log-Odds, Odds에 로그를 취한 값)을 선형 회귀를 통해 구하는 것으로 해석함.
- Odds: 승리 확률(q)을 패배 확률로 나눈 값. ![image](https://github.com/user-attachments/assets/872ae4f7-ca24-44eb-ae63-bb5d646cb774)
- Logit: ![image](https://github.com/user-attachments/assets/31010a4f-ceca-453b-b337-4db7548dbf2e)
- Logit의 q에 대해 정리하면 ![image](https://github.com/user-attachments/assets/e37458aa-8db9-4eb4-b85d-07b11142c4fa)가 됨.
- 즉, 로지스틱 회귀는 입력을 받아 Logit을 출력하는 신경망(선형 회귀)과 Logit을 확률로 변환하는 Sigmoid 함수의 두 단계로 나뉘어져있음.
- 즉, 인공 신경망의 역할은 입력값과 Logit 사이의 선형 관계를 찾는 것이며 Sigmoid는 Logit을 확률로 변환하고 BCE Loss를 계산하기 위해 사용되는 함수임.
- 따라서 로지스틱 회귀는 선형 회귀를 통해 Logit을 예측하고 이를 확률로 변환하여 이진 분류 문제를 해결하는 방법임.

이진 분류에 MSE Loss를 도입하면 어떻게 될까?
- 실험을 위해 레이블이 1인 데이터 하나에 대한 MSE Loss와 BCE Loss를 비교해 보겠음.
- 이 때 MSE Loss는 ![image](https://github.com/user-attachments/assets/fd7c118d-1646-4b9f-b353-acc615042b24), BCE Loss는 ![image](https://github.com/user-attachments/assets/8d68feb7-c833-4788-b98e-077264b3d142) 가 됨.
- 그래프로 표현하면 0 < q < 1의 범위에서 MSE Loss는 이차함수, BCE Loss는 로그함수의 형태를 띄며 BCE Loss는 0으로 다가갈수록 무한대로 발산함.
- 여기서 BCE가 상대적으로 예측 오류에 더 민감하게 반응한다는 것을 알 수 있음 -> BCE는 잘못된 예측에 더 강한 페널티를 부과함.
- 또한 웨이트 w에 대해 q로 정리 시 MSE Loss는 Non-Convex이고 BCE는 Convex이다. (Convex: 아래로 볼록한 함수로 단 하나의 Minimum인 Global Minimum을 가짐.)
- 물론 출력층 이전 층들의 w에 대해서는 둘 모두 Non-Convex이겠지만 같은 조건에서는 BCE가 Non-Convex한 정도가 덜해 최적화 과정의 안정성 면에서 BCE가 유리함.
- 결론: 이진 분류 문제에서는 BCE Loss가 잘못된 예측에 더 강력한 페널티를 부과하고 최적화 과정에서 더 안정적인 특성을 보여주기 때문에 사용됨.
- 물론 상황에 따라 다를 수는 있다..

딥러닝과 MLE
- Loss를 최소화하는 파라미터를 찾는 학습 과정은 MLE를 최대로 하는 파라미터를 찾는 과정과 같음.
- 사실 BCE와 MLE도 이 맥락에서는 Likelihood 최대화라는 공통점을 갖는다.
- 자세한 수학적 개념은 MML 공부하면서 다룰 예정. (생략함)

Loss 함수와 NLL(Negative Log-Likelihood)
- ![image](https://github.com/user-attachments/assets/cf6c4708-7489-4ade-b5ff-5e102521a939) 이 식은 베르누이 분포임..
- 그리고 이 식은 개별 시행이 독립시행임을 가정했음.
- 여기서 -1/n log를 취한게 바로 BCE Loss.
- NLL의 정의는 이런식으로 Likelihood에 -log를 취한것임.
- MSE Loss도 가우시안 분포를 따른다고 가정하고 NLL을 구하면 얻을 수 있음.
- 즉, 베르누이 분포에 NLL 취하면 BCE Loss, 가우시안 분포에 NLL 취하면 MSE Loss가 됨.
- 따라서 BCE와 MSE는 가정한 분포가 다를 뿐 NLL을 취한 공통점이 있음.
- MAE Loss 또한 라플라스 분포에서 NLL을 취했을 뿐.

- 정리하면 MSE는 가우시안, BCE는 베르누이, MAE는 라플라스 분포에 NLL을 취한 것.
- 이진 분류 문제는 베르누이 분포(이산적 값)에 적합하기 때문에 BCE를 쓰는 것이라 해석 가능.
- 회귀에서는 가우시안 분포(연속적 값)에 적합하기 때문에 MSE를 쓰는 것이라 해석 가능.
- 라플라스 분포는 가우시안 분포에 비해 꼬리 부분의 확률밀도가 더 크기 때문에 Outlier에 영향 덜 받음 -> 이상치 많으면 MAE Loss를 쓰는것이 좋음. (이전에 논의한 것과 같은 결과)
- 추가적으로 다중 분류는 Cross-Entropy Loss가 유리. 카테고리 분포(Categorical Distribution)에서 따옴.

- 결론적으로 분포를 f(x)라고 하면 Loss함수의 일반적인 표현식을 ![image](https://github.com/user-attachments/assets/75ad92a2-c043-4beb-b24e-394e3bef208a) 라 표현할 수 있음.
- 따라서 문제의 특성에 맞는 확률 분포를 알맞게 가정하고 이에 기반한 Loss 함수를 직접 설계할 수 있어야 함.

다중 분류
- One-Hot Encoding을 이용해 각 클래스를 표현함. (클래스 내의 우선순위를 부여하지 않게 됨. 클래스별로 0, 1, 2...식으로 표현하면 거리가 표현된다는 문제가 생김.)
- 이 때 출력층이 0과 1 사이에 나오지 않으면 문제가 생김. 이를 위해 Softmax함수를 이용함.
- Softmax를 사용하면 출력의 합이 1이 되므로 나머지 값들은 자연스럽게 0이 됨 -> 정답 출력값을 담당하는 노드 이외의 출력값은 고려하지 않아도 됨. -> softmax를 사용하면 해당 분류에서는 각 노드의 웨이트는 해당하는 클래스의 학습만 관여함. (더욱 독립적.)
- Sigmoid를 각 노드에 적용한다면 출력값의 합이 1이 되지 않아도 값들의 상대적 크기를 비교해 분류가 가능함. 단, One-Hot Encoding의 특성을 제대로 살릴 수 없어 효율적이지 않음.
  
- 단, 예외적으로 하나의 이미지에 여러 클래스가 있는 다중 레이블 분류(Multi-label Classification)에는 Sigmoid가 더 적합함! (레이블 벡터의 성분 합이 1이 아니기 때문. ex.[0, 1, 1, 0, 0] 가능.)

Cross-Entropy Loss
- 다중 분류에서 레이블이 카테고리 분포(Categorical Distribution)을 따른다고 가정하고 NLL을 구한 Loss.
- 카테고리 분포는 베르누이 분포를 확장한 개념으로, 멀티누이 분포(Multinoulli Distribution)라고도 부름.
- 베르누이 분포는 0또는 1의 단일 값에 대한 확률을 다루지만 카테고리 분포는 [1, 0, 0]같은 랜덤 벡터에 대한 확률을 다룸.
- 수식적으로는 ![image](https://github.com/user-attachments/assets/e8439d5e-0f44-418a-8aac-601272360809)임.
- 예를 들어 세 개의 클래스의 경우에는 ![image](https://github.com/user-attachments/assets/12036992-4b43-4efc-bf39-0da4b19ae7fa)와 같다.
- 여기에 NLL 씌우면 그게 Cross-Entropy Loss임.
- CE의 재미있는 성질 중 하나는 항상 실제 분포의 Entropy보다 크거나 같다는 것임. (궁금하면 증명해보자.) 이 성질은 CE를 줄일수록 q가 y에 가까워진다는 것을 의미함.

Softmax 회귀(Softmax Regression, Logistic Regression을 여러 클래스로 확장했다는 의미에서 Multinomial Logistic Regression이라고도 함)
- 입력과 출력 사이의 관계를 여러 클래스에 대한 확률 분포 함수로 표현하고 이 함수를 은닉층이 없는 인공 신경망으로 놓고 추정하는 방법.
- Logit들을 선형 회귀를 통해 구하는 것(로지스틱 회귀와 같음).
- 두 단계로 나누면 데이터를 입력받아 Logit들을 출력하는 신경망(선형 회귀) 단계와 이후 Logit들을 확률 분포로 변환하는 Softmax 함수가 있는 단계로 해석 가능.
- 결론적으로 Softmax 회귀는 선형 회귀를 통해 Logit들을 예측하고, 이를 확률 분포로 변환하여 다중 분류 문제를 해결하는 방법임.

*Chapter4*

Universal Approximation Theorem
- 딥러닝은 결국 입력과 출력 간의 관계를 나타내는 함수를 찾는 문제임.
- MLP는 히든 레이어가 단 한 층만 있어도 제한된 범위 안의 어떤 연속 함수든 나타낼 수 있음이 증명되어있음.
- 단, 히든 레이어가 충분한 수의 노드를 가져야 하고 활성화 함수가 다항 함수가 아니어야 한다는 전제가 있음.
