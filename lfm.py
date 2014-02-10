import sys;
import math;
import random;

FACTOR_NUM = 20;

### ALS method for sloving factors.
Alpha1 = 10.0;
Alpha2 = 10.0;

item_users = {};
user_items = {};

user_len = {};
item_len = {};

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

def Updata_User_Factor(uid):
	global FACTOR_NUM;
	global Alpha1;
	global Alpha2;
	global user_items;
	global item_factor;
	global user_factor;

	inver_matrix = [0] * FACTOR_NUM * FACTOR_NUM;
	for i in range(0, FACTOR_NUM):
		inver_matrix[i * FACTOR_NUM + i] = 1.0 / Alpha1;

	sum_vec = [0] * FACTOR_NUM;

	hz = [0] * FACTOR_NUM;
	for tid in user_items[uid]:
		Matrix_Multiplication(inver_matrix, item_factor[tid], FACTOR_NUM, FACTOR_NUM, hz);
		sum = 1.0 + Inner_Product(hz, item_factor[tid], FACTOR_NUM);
		Matrix_Add_Vector(inver_matrix, hz, FACTOR_NUM, 1.0, - 1.0 / sum);
		rating = user_items[uid][tid];
		Vector_Add(sum_vec, item_factor[tid], 1.0, rating, FACTOR_NUM);
	u_record = [0] * FACTOR_NUM;
	Vector_Add(u_record, user_factor[uid], 0, 1.0, FACTOR_NUM);
	Matrix_Multiplication(inver_matrix, sum_vec, FACTOR_NUM, FACTOR_NUM, user_factor[uid]);
	error = L2(u_record, user_factor[uid], FACTOR_NUM);
	return error;

def Updata_Item_Factor(tid):
	global FACTOR_NUM;
	global Alpha1;
	global Alpha2;
	global item_users;
	global item_factor;
	global user_factor;

	inver_matrix = [0] * FACTOR_NUM * FACTOR_NUM;
	for i in range(0, FACTOR_NUM):
		inver_matrix[i * FACTOR_NUM + i] = 1.0 / Alpha2;

	sum_vec = [0] * FACTOR_NUM;

	hz = [0] * FACTOR_NUM;
	for uid in item_users[tid]:
		Matrix_Multiplication(inver_matrix, user_factor[uid], FACTOR_NUM, FACTOR_NUM, hz);
		sum = 1.0 + Inner_Product(hz, user_factor[uid], FACTOR_NUM);
		Matrix_Add_Vector(inver_matrix, hz, FACTOR_NUM, 1.0, - 1.0 / sum);
		rating = item_users[tid][uid];
		Vector_Add(sum_vec, user_factor[uid], 1.0, rating, FACTOR_NUM);
	i_record = [0] * FACTOR_NUM;
	Vector_Add(i_record, item_factor[tid], 0, 1.0, FACTOR_NUM);
	Matrix_Multiplication(inver_matrix, sum_vec, FACTOR_NUM, FACTOR_NUM, item_factor[tid]);
	error = L2(i_record, item_factor[tid], FACTOR_NUM);
	return error;

for iter in range(0,30):
	#### iteration for users.
	uerror = 0;
	for uid in user_len:
		uerror += Updata_User_Factor(uid);
	print "user update", math.sqrt( uerror * 1.0 / len(user_len));

	#### iteration for items.
	ierror = 0;
	for tid in item_len:
		ierror += Updata_Item_Factor(tid);
	print "item update", math.sqrt( ierror * 1.0 / len(item_len));

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
