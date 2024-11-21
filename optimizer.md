# Loss optimizer

- 딥러닝 모델의 weight 최적화를 위해 사용되는 기법

# Loss function

- 해당 task에 있어 모델의 오류 정도를 수학적으로 나타내는 함수

- 고등학생때까지는 function의 최솟값을 찾기 위해서는 미분 후 0이 되는 해를 찾아 최솟값을 찾았지만 실제로는 변수가 많아지고 다항함수로 나타낼 수 없는 function이 많음 -> 다양한 근사적 방법을 이용해야함
l(x) = y 에서 x는 weight, y는 loss값

# Gradient descent

- 현재 weight에서 정의된 loss를 줄이는 방향으로 weight를 조정하는 방식
- loss function을 편미분해 gradient의 반대 방향(가장 가파른 방향)으로 조정함

![image](https://github.com/user-attachments/assets/75d816ef-78fc-4f1f-9725-e440fc5658ad)

learning rate α: 한 번의 업데이트에서 weight의 이동 정도를 정함(step size라고도 함)
- 너무 크면: 진동(수렴X) 또는 minimum을 탈출하는 문제 발생
- 너무 작으면: 학습률이 떨어지고 학습이 끝날 때까지 underfitting되는 문제 발생 가능

- 기본적인 Gradient descent는 모든 데이터를 고려한 loss function에 대해 수행되기 때문에 느리다는 문제가 있음
- 또한 출발지점과 가장 가까운 local minimum에 빠질 위험이 있다(특히 해당 local minimum이 매우 평평한 정도라면 학습이 거의 되지 않음) -> SGD를 이용하여 해결한다.

  # Stochastic gradient descent

- Gradient descent의 문제점인 느린 속도와 출발지점과 가장 가까운 local minimum에 빠지는 문제 해결을 위해 사용될 수 있는 방법

- 데이터를 N개로 나눈 batch마다의 loss에 대해서 Gradient descent를 수행

![image](https://github.com/user-attachments/assets/7d29105a-23d3-479f-babf-16497abad952)

- 랜덤하게 데이터를 뽑아 만들어진 loss function에서 조정되므로 전체 데이터에 대한 loss landscape와 지형이 다름 -> local minimum에 빠졌다가 탈출 가능(평평할수록 잘 빠져나온다)
- 더 빠른 속도로 학습

  +) SGD는 GD의 문제점을 해결하기 위해 일부분의 데이터만 뽑아서 GD를 적용시킴 -> 학습 자체를 덜하게 되는게 아닌지 의문이 들 수 있음(예를 들어 문제집을 5회 반복하는것과 일부분씩 뽑아서 5회 반복하는것 중 학습은 전체 5회 반복이 더 학습을 많이할텐데?)

A. SGD는 GD와 기울기의 '대략적인 방향은 같음' -> 근사적으로 비슷한 방향으로 적용됨. 또한 같은 시간 당 더 많은 횟수의 반복이 가능하므로 충분한 학습량을 확보할 수 있음.

다만 GD와 SGD는 모두 fixed step size를 가진다는 점에서 한계가 존재 할 수 있음 -> Momentum, RMS-Prop, Adam 등의 기법 이용 가능.

+) 정확한 SGD의 정의는 모든 데이터 중 단 한 개의 샘플만 사용한 것이며 GD는 전부 사용한 것 -> 실제 학습 시에는 미니 배치 단위로 끊어 optimization을 적용함 (둘의 중간 지점)
+) batch: 전체 데이터셋, mini-batch: 전체 데이터셋의 부분 집합



































※ 해당 정리본은 유튜버 혁펜하임님의 강의를 참고한 것입니다.
