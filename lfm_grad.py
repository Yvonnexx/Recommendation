import sys;
import math;
import random;

FACTOR_NUM = 10;

### ALS method for sloving factors.
Alpha1 = 0.05;
Alpha2 = 0.05;
learn_rate = 0.1;

item_users = {};
user_items = {};

user_len = {};
item_len = {};

train_data = [];

train_index = 0;
f2 = file(sys.argv[1]);
for line in f2:
	items = line.strip().split('\t');
	uid = int(items[0]);
	tid = int(items[1]);
	rating = float(items[2]);
	
	if(not item_users.has_key(tid)):
		item_users[tid] = {};
		item_len[tid] = [0, 0, 0];
	item_users[tid][uid] = rating;
	item_len[tid][0] += 1;
	item_len[tid][1] += rating;
	item_len[tid][2] += rating * rating;

	if(not user_items.has_key(uid)):
		user_items[uid] = {};
		user_len[uid] = [0, 0, 0];
	user_items[uid][tid] = rating;
	user_len[uid][0] += 1;
	user_len[uid][1] += rating;
	user_len[uid][2] += rating * rating;
	train_data.append([uid, tid, rating]);
	train_index += 1;
f2.close();
print "User Num", len(user_len);
print "Item Num", len(item_len);
print "rating Num", train_index;

user_factor = {};
item_factor = {};
for uid in user_len:
	user_factor[uid] = [0] * FACTOR_NUM;
	for i in range(0, FACTOR_NUM):
		user_factor[uid][i] = random.random();

for tid in item_len:
	item_factor[tid] = [0] * FACTOR_NUM;
	for i in range(0, FACTOR_NUM):
		item_factor[tid][i] = random.random();


def Matrix_Multiplication(mat, vec, row, col, res):
	for i in range(0, col):
		result = 0;
		for j in range(0, row):
			result = result + mat[i * row + j] * vec[j];
		res[i] = result;

def Inner_Product(vec1, vec2, col):
	sum = 0;
	for i in range(0, col):
		sum += vec1[i] * vec2[i];
	return sum;

def Matrix_Add_Vector(mat, vec, size, a1, a2):
	for i in range(0, size):
		for j in range(0,size):
			v = vec[i] * vec[j];
			mat[i * size + j] = a1 * mat[i * size + j] + a2 * v;

def Vector_Add(vec1, vec2, a1, a2, size):
	for i in range(0, size):
		vec1[i] = a1 * vec1[i] + a2 * vec2[i];

def L2(vec1, vec2, size):
	error = 0;
	for i in range(0,size):
		error += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	return error;

old_rmse = 0;
old_mae = 0;
for iter in range(0,30):
	#### iteration for users.
	train_error = 0;

	for train_sample in train_data:
		uid = train_sample[0];
		tid = train_sample[1];
		rat = train_sample[2];
		erat = 0;
		for i in range(0, FACTOR_NUM):
			erat += user_factor[uid][i] * item_factor[tid][i];
		error = rat - erat;
		train_error += error * error;
		for i in range(0, FACTOR_NUM):
			ug = Alpha1 * user_factor[uid][i] - error * item_factor[tid][i];
			ig = Alpha2 * item_factor[tid][i] - error * user_factor[uid][i];
			user_factor[uid][i] += - learn_rate * ug;		
			item_factor[tid][i] += - learn_rate * ig;
	print "training error", math.sqrt(train_error * 1.0 / len(train_data));
	
	MAE = 0;
	RMSE = 0;
	num = 0;
	f1 = file(sys.argv[2]);
	for line in f1:
		items = line.strip().split('\t');
		uid = int(items[0]);
		tid = int(items[1]);
		grat = float(items[2]);
		erat = 0;

		if(user_len.has_key(uid) and item_len.has_key(tid)):
			for f in range(0, FACTOR_NUM):
				erat += user_factor[uid][f] * item_factor[tid][f];
		num += 1;
		MAE += math.fabs(grat-erat);
		RMSE += (grat - erat) * (grat - erat);
	f1.close();
	print "iter ",iter ,"MAE ", MAE / num;
	print "iter ",iter ,"RMSE ", math.sqrt(RMSE/ num);

	if(iter >= 1):
		if(old_rmse < RMSE):
			learn_rate = learn_rate / 2.0;

	old_rmse = RMSE;
	old_mae = MAE;
MAE = 0;
RMSE = 0;
num = 0;
f1 = file(sys.argv[2]);
f3 = file(sys.argv[3],'w');
for line in f1:
	items = line.strip().split('\t');
	uid = int(items[0]);
	tid = int(items[1]);
	grat = float(items[2]);
	erat = 0;

	if(user_len.has_key(uid) and item_len.has_key(tid)):
		for f in range(0, FACTOR_NUM):
			erat += user_factor[uid][f] * item_factor[tid][f];
	f3.write(str(uid)+"\t"+str(tid)+"\t"+str(grat-erat)+"\t"+str(grat)+"\n");
	num += 1;
	MAE += math.fabs(grat-erat);
	RMSE += (grat - erat) * (grat - erat);

f1.close();
f3.close();
print "MAE ", MAE / num;
print "RMSE ", math.sqrt(RMSE/ num);
