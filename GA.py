import numpy as np
import copy
from numpy.random import *
# weight
weight = [9, 7, 8, 2, 10, 7, 7, 8, 5, 4,
	  7, 5, 7, 5, 9, 9, 9, 8, 8, 2,
	  7, 7, 9, 8, 4, 7, 3, 9, 7, 7,
	  9, 5, 10, 7, 10, 10, 7, 10, 10, 10,
	  3, 8, 3, 4, 2, 2, 5, 3, 9, 2]

# price
price = [20, 28, 2, 28, 15, 28, 21, 7, 28, 12,
	 21, 4, 31, 28, 24, 36, 33, 2, 25, 21,
	 35, 14, 36, 25, 12, 14, 40, 36, 2, 28,
	 33, 40, 22, 2, 18, 22, 14, 22, 15, 22,
	 40, 7, 4, 21, 21, 28, 40, 4, 24, 21]

# WEIGHT_LIMIT: 制限
# GENE: 遺伝子数
# SET: セット数
# LEARNING_EPOCH: 学習回数
WEIGHT_LIMIT = 60
GENE = 50
SET = 50
LEARNING_EPOCHS = 30000
reverse_i = [GENE - i - 1 for i in range(GENE)]
weight_sum = 0
price_sum = 0


def random_0to1():
    return(randint(100) / 100)

def random_0to50():
    return(randint(50))

def sort_up_array(up_array):
	# 値段で昇順にソート
	for i in range(SET):
		x = np.argmax(up_array[i:SET, -1]) + i
		up_array = np.insert(up_array, i, list(up_array[x]), axis=0)
		up_array = np.delete(up_array, x+1, axis=0)
	return(up_array)

def sum_weight_assess(weight_array):
	for i in range(SET):
		sum_weight = 0
		for j in range(GENE):
			if weight_array[i][j] == 1:
				sum_weight += weight[j]
			if sum_weight > WEIGHT_LIMIT:
				weight_array[i,-1] = 0
	return(weight_array)

def calculate_price(price_array):
	for i in range(SET):
		sum_price = 0
		for j in range(GENE):
			if price_array[i][j] == 1:
				sum_price += price[j]
		price_array[i,-1] = sum_price
	return(price_array)

# 初期値の設定(binary:0~1)
array = [[] for _ in range(SET)]
for i in range(SET):
	sum_weight = 0
	for j in range(GENE):
		rand = random_0to1()
		if rand < 0.5:
			sum_weight += weight[j]
			if sum_weight > WEIGHT_LIMIT:
				sum_weight -= weight[j]
				array[i].append(0)
			else:
				array[i].append(1)
		else:
			array[i].append(0)
# 値段の計算
sum_price = [0 for _ in range(SET)]
for i in range(SET):
	for j in range(GENE):
		if array[i][j] == 1:
			sum_price[i] += price[j]
	array[i].append(sum_price[i])
n2array = np.array(array)

# 学習
for epoch in range(LEARNING_EPOCHS):
	if epoch%500 == 0:
		print(str(epoch)+"epoch")
		print(n2array[:, -1])

	copy_n2array = copy.deepcopy(n2array)
	sum = np.sum(n2array)
	# 累積和
	cumsum_n2array = np.cumsum(n2array[:, -1])
	rate = cumsum_n2array / sum
	# ルーレット選択
	for i in range(SET):
	    rand = random_0to1()
	    for j in reverse_i:
	        if rand > rate[j]:
	            n2array[i] = copy_n2array[j]
	            break
	n2array = sort_up_array(n2array)
	# 交差
	for i in reverse_i:
		rand = random_0to1()
		if rand < 0.8:
			# 2点交叉:div1,div2
			div1 = randint(0, GENE)  # 1〜GENEの乱数生成
			div2 = randint(div1, GENE) + 1
			n2array[i][div1:div2] = n2array[i-1][div1:div2]

	# 突然変異
	for i in range(SET):
		rand_0to1 = random_0to1()
		rand_0to50 = random_0to50()
		if rand < 0.3:
			if n2array[i][rand_0to50] == 1:
				n2array[i][rand_0to50] = 0
			else:
				n2array[i][rand_0to50] = 1

	#エリート保存戦略
	if epoch != 0:
		min_index = np.argmin(n2array[:,-1])
		#print(min_index)
		n2array[min_index] = copy_n2array[0]

	n2array = calculate_price(n2array)
	#重さの制限
	n2array = sum_weight_assess(n2array)
	n2array = sort_up_array(n2array)

# 学習結果
print("学習結果")
print(n2array[:,-1])
# In[]:
print("エリート個体の中身")
print(n2array[0])
sum_weight=0
for j in range(GENE):
	if n2array[0][j] == 1:
		sum_weight += weight[j]
print(sum_weight)
