import sys;
#import io;
import math;
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


user_cograph = {};
def push_user_graph(uid1, uid2, r1, r2):
	global user_cograph;
	
	if(not user_cograph.has_key(uid1)):
		user_cograph[uid1] = {};
	
	if(not user_cograph[uid1].has_key(uid2)):
		user_cograph[uid1][uid2] = [0,0,0,0];
	
	user_cograph[uid1][uid2][0] += 1;
	user_cograph[uid1][uid2][1] += r1*r2;
	user_cograph[uid1][uid2][2] += r1*r1;
	user_cograph[uid1][uid2][3] += r2*r2;


uid_index = 0;
for uid in user_items:
	for tid in user_items[uid]:
		for couid in item_users[tid]:
			if(uid <= couid):
				continue;
			push_user_graph(uid, couid, user_items[uid][tid], user_items[couid][tid]);
	uid_index += 1;
	if(uid_index % 300 == 0):
		print "construct graph", uid_index;

def pearson_similarity(mXY, mN, mX, mY, mXX, mYY):
	a = (mXY - mX * mY  *1.0 / mN);
	b = (mXX - mX * mX * 1.0 / mN) * (mYY - mY * mY * 1.0 / mN);
	return  a / math.sqrt(b + 0.000001);

def local_pearson_similarity(mXY, mXlen, mYlen, mX, mY, mCX, mCY, mCom, mCXX, mCYY):
	avgx = mX * 1.0 / mXlen;
	avgy = mY * 1.0 / mYlen;
	a = mXY - avgx * mCY - avgy * mCX + mCom * avgx * avgy;
	b = (mCXX - 2*mCX * avgx + mCom * avgx * avgx) * (mCYY - 2 * mCY * avgy + mCom * avgy * avgy);
	return a / math.sqrt(b + 0.00000001);


def cosine_similarity(mXY, mXX, mYY):
	return mXY * 1.0 / math.sqrt(mXX * mYY + 0.000000001);

def jarcard_similarity(mcom, mXlen, mYlen):
	return mcom * 1.0 / (mXlen + mYlen - mcom);

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
	rat = 0;
	rat_wei = 0;
	if(user_items.has_key(uid) and item_users.has_key(tid)):
		for his_uid in item_users[tid]:
			fid = his_uid;
			eid = uid;
			if(his_uid < uid):
				fid = uid;
				eid = his_uid;
			if(user_cograph.has_key(fid) and user_cograph[fid].has_key(eid)):
				mCom = user_cograph[fid][eid][0];
				mXlen = user_len[fid][0];
				mYlen = user_len[eid][0];
				
				
				jsim = jarcard_similarity(mCom, mXlen, mYlen);
				
				a = user_cograph[fid][eid][1];
				
				b = user_len[fid][2];
				c = user_len[eid][2]; 

				psim = jsim * a * 1.0 / math.sqrt(b * c + 0.000000001);
				rat = rat + psim * item_users[tid][his_uid];
				rat_wei = rat_wei + abs(psim);
	e = rat * 1.0 / (rat_wei + 0.0000001);
	f3.write(str(uid)+"\t"+str(tid)+"\t"+str(grat-e)+"\t"+str(grat)+"\n");
	num += 1;
	MAE += math.fabs(grat-e);
	RMSE += (grat - e) * (grat - e);
f1.close();
f3.close();
print "MAE ", MAE / num;
print "RMSE ", math.sqrt(RMSE/ num);
