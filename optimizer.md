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
- 또한 출발지점과 가장 가까운 local minimum에 빠질 위험이 있다 -> SGD를 이용하여 해결한다.

  # Stochastic gradient descent













































※ 해당 정리본은 유튜버 혁펜하임님의 강의를 참고한 것입니다.
