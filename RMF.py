import sys;
import math;
import random;

## Rank based Matrix Factoriation Method.

FACTOR_NUM = int(sys.argv[3]);
user_factor = {};
user_bias = {};
learn_rate = 0.1;
Alpha = 0.02;

rw_alpha = 0.2;
NRANGE = 2;
NTRIAL = 5; # sample negative samples.

user_graph = {};
train_data = [];
link_num = 0;

def get_neighbor_canditors(uid, neis):
	global NRANGE;
	global user_graph;
	global rw_alpha;
	if(user_graph.has_key(uid)):
		seed_users = {};
		seed_users[uid] = 1;

		neighbors = {};
		xvector = {};
		xvector[uid] = 1;

		for iter in range(0, NRANGE):
			xstack = {};
			for suid in xvector:
				prob = (1 - rw_alpha) * xvector[suid];
				if(not user_graph.has_key(suid)):
					continue;	

				trans = 1.0 / len(user_graph[suid]);
				for uuid in user_graph[suid]:
					if(not xstack.has_key(uuid)):
						xstack[uuid] = 0;
					xstack[uuid] += prob * trans;
			for suid in seed_users:
				prob = rw_alpha * seed_users[suid];
				if(not xstack.has_key(suid)):
					xstack[suid] = 0;
				xstack[suid] += prob;	
			xvector = xstack;
		for uuid in xvector:
			if(not user_graph.has_key(uuid)):
				continue;
			if(user_graph[uid].has_key(uuid)):
				continue;
			if(uuid == uid):
				continue;
			neis[uuid] = xvector[uuid];
		#print "neighs",neis;
edge_num = 0;

user_list = [];
f1 = file(sys.argv[1]);
for line in f1:
	items = line.strip().split('\t');
	uid1 = int(items[0]);
	uid2 = int(items[1]);
	if(not user_graph.has_key(uid1)):
		user_graph[uid1] = {};
		user_list.append(uid1);

	if(not user_graph[uid1].has_key(uid2)):
		user_graph[uid1][uid2] = 1;

	if(not user_graph.has_key(uid2)):
		user_graph[uid2] = {};
		user_list.append(uid2);

	if(not user_graph[uid2].has_key(uid1)):
		user_graph[uid2][uid1] = 1;
	train_data.append([uid1,uid2]);
	#train_data.append([uid2,uid1]);
	edge_num += 1;
f1.close();

for uid in user_graph:
	user_factor[uid] = [0] * FACTOR_NUM;
	for i in range(0, FACTOR_NUM):
		user_factor[uid][i] = random.random() * 0.1;
	user_bias[uid] = 0;

def Pred_Link(uid1, uid2):
	global user_factor;
	global user_bias;
	global FACTOR_NUM;

	u1b = 0;
	u2b = 0;
	uin = 0;
	if(user_bias.has_key(uid1)):
		u1b = user_bias[uid1];
	if(user_bias.has_key(uid2)):
		u2b = user_bias[uid2];
	if(user_factor.has_key(uid1) and user_factor.has_key(uid2)):
		for i in range(0, FACTOR_NUM):
			uin += user_factor[uid1][i] * user_factor[uid2][i];
	pred = u1b + u2b + uin; #1.0 / (1.0 + math.exp(-));
	return pred;


print "Loading Canditors Neighbors ....";

test_users = {};
test_users_cand = {};
test_users_targ = {};
f2 = file(sys.argv[2]);
for line in f2:
	items = line.strip().split('\t');
	uid1 = int(items[0]);
	uid2 = int(items[1]);
	if(not test_users.has_key(uid1)):
		test_users[uid1] = 1;
		test_users_targ[uid1] = {};
		test_users_cand[uid1] = {};
		get_neighbor_canditors(uid1, test_users_cand[uid1]);
	test_users_targ[uid1][uid2] = 1;
f2.close();

print "Loading Canditors Neighbors Done";

def Precision(sorted_dict, target_dict):
	total_num = 0;
	target_num = 0;
	precision = 0;
	for mitem in sorted_dict:
		jid = mitem[0];
		total_num = total_num + 1;
		if(target_dict.has_key(jid)):
			target_num = target_num + 1;
			precision = precision + target_num * 1.0 / total_num;
		if(total_num >= 20):
			break;
	s = len(target_dict);
	if(s >= 20):
		s = 20;
	return precision * 1.0 / s;

def Positive_Update(uid1, uid2):
	global Alpha;
	global user_bias;
	global learn_rate;
	global user_factor;

	erat = Pred_Link(uid1, uid2);
	u1bg = (1.0 - erat) - Alpha * user_bias[uid1];
	u2bg = (1.0 - erat) - Alpha * user_bias[uid2];
	user_bias[uid1] += learn_rate * u1bg;
	user_bias[uid2] += learn_rate * u2bg;

	
	for i in range(0, FACTOR_NUM):
		u1g = Alpha * user_factor[uid1][i] - (1.0 - erat) * user_factor[uid2][i];
		u2g = Alpha * user_factor[uid2][i] - (1.0 - erat) * user_factor[uid1][i];
		user_factor[uid1][i] += - learn_rate * u1g;		
		user_factor[uid2][i] += - learn_rate * u2g;
	return math.log(erat);

def Negative_Update(uid1, uid2):
	global Alpha;
	global user_bias;
	global learn_rate;
	global user_factor;

	erat = Pred_Link(uid1, uid2);
	u1bg = ( - erat) - Alpha * user_bias[uid1];
	u2bg = ( - erat) - Alpha * user_bias[uid2];
	user_bias[uid1] += learn_rate * u1bg;
	user_bias[uid2] += learn_rate * u2bg;

	for i in range(0, FACTOR_NUM):
		u1g = Alpha * user_factor[uid1][i] - ( - erat) * user_factor[uid2][i];
		u2g = Alpha * user_factor[uid2][i] - ( - erat) * user_factor[uid1][i];
		user_factor[uid1][i] += - learn_rate * u1g;		
		user_factor[uid2][i] += - learn_rate * u2g;
	return math.log( 1 - erat);

def log_add(v1, v2):
	if(v1 > v2):
		return v1 + math.log( 1 + math.exp(v2 - v1));
	if(v2 >= v1):
		return v2 + math.log( 1 + math.exp(v1 - v2));

def Model_Update(uid1, uid2, neg_ids, neg_num):
	global FACTOR_NUM;
	global user_factor;
	global user_bias;
	global Alpha;
	global learn_rate;

	base_sim = 0;
	sim1 = [];
	sim2 = [];
	base_sim = Pred_Link(uid1, uid2);

	sum1 = base_sim;
	sum2 = base_sim;
	for nid in neg_ids:
		s1 = Pred_Link(uid1, nid);
		s2 = Pred_Link(uid2, nid);
		sim1.append(s1);
		sim2.append(s2);
		sum1 = log_add(sum1, s1);
		sum2 = log_add(sum2, s2);
	perror = 2* base_sim - sum1 - sum2;
	  
	gradi = [0] * FACTOR_NUM;
	grad_biasi = 2;
	for nid in range(0, len(neg_ids)):
		grad_biasi -= math.exp(sim1[nid] - sum1);
	grad_biasi -= math.exp(base_sim - sum1);
	grad_biasi -= math.exp(base_sim - sum2);

	for i in range(0, FACTOR_NUM):
		gradi[i] = 2 * user_factor[uid2][i];
		for nid in range(0, len(neg_ids)):
			gradi[i] -= math.exp(sim1[nid] - sum1) * user_factor[neg_ids[nid]][i];
		gradi[i] -= math.exp(base_sim - sum1) * user_factor[uid2][i];		
		gradi[i] -= math.exp(base_sim - sum2) * user_factor[uid2][i];

	gradj = [0] * FACTOR_NUM;
	grad_biasj = 2;
	for nid in range(0, len(neg_ids)):
		grad_biasj -= math.exp(sim2[nid] - sum2);
	grad_biasj -= math.exp(base_sim - sum2);
	grad_biasj -= math.exp(base_sim - sum1);

	for i in range(0, FACTOR_NUM):
		gradj[i] = 2 * user_factor[uid1][i];
		for nid in range(0, len(neg_ids)):
			gradj[i] -= math.exp(sim2[nid] - sum2) * user_factor[neg_ids[nid]][i];
		gradj[i] -= math.exp(base_sim - sum2) * user_factor[uid1][i];		
		gradj[i] -= math.exp(base_sim - sum1) * user_factor[uid1][i];
	
	for nid in range(0, len(neg_ids)):
		gb = - math.exp(sim1[nid] - sum1) - math.exp(sim2[nid] - sum2);
		user_bias[neg_ids[nid]] += learn_rate * ( - Alpha * user_bias[neg_ids[nid]] + gb );
		for i in range(0, FACTOR_NUM):
			gf = - math.exp(sim1[nid] - sum1) * user_factor[uid1][i] - math.exp(sim2[nid] - sum2) * user_factor[uid2][i];
			user_factor[neg_ids[nid]][i] += learn_rate * ( -Alpha * user_factor[neg_ids[nid]][i]  + gf);
	
	user_bias[uid1] += learn_rate * ( - Alpha * user_bias[uid1] + grad_biasi);
	user_bias[uid2] += learn_rate * ( - Alpha * user_bias[uid2] + grad_biasj);
	for i in range(0, FACTOR_NUM):
		user_factor[uid1][i] += learn_rate * ( - Alpha * user_factor[uid1][i] + gradi[i]);
		user_factor[uid2][i] += learn_rate * ( - Alpha * user_factor[uid2][i] + gradj[i]);
	return perror;


old_prec = 0;
######### Bias SGD + Sparse Matrix Factorization.
for iter in range(0,30):
	train_error = 0;
	for train_sample in train_data:
		uid1 = train_sample[0];
		uid2 = train_sample[1];
		neg_ids = [];
		for neg in range(0, NTRIAL):
			neg_user = random.randint(0, len(user_list) - 1);
			negid = user_list[neg_user];
			neg_ids.append(negid);

		train_error += Model_Update(uid1, uid2, neg_ids, NTRIAL);

		#train_error += Positive_Update(uid1, uid2);
		#train_error += Negative_Update(uid1, negid);
		#train_error += Negative_Update(uid2, negid);

	print "train error", train_error / len(train_data) / ( 1.0 + NTRIAL);

	mix_global_prec = 0;

	global_prec = 0;
	for uid in test_users_targ:
		neighs = {};
		mix_neighs = {};
		for cand_nei in test_users_cand[uid]:
			prior_v = test_users_cand[uid][cand_nei];
			erat = Pred_Link(uid, cand_nei);
			#print "say something", prior_v, erat;
			neighs[cand_nei] = erat;
			mix_neighs[cand_nei] = erat * 0.001 + prior_v;

		sort_neighs = sorted(neighs.items(), lambda x, y:cmp(x[1],y[1]), reverse=True);
		mp = Precision(sort_neighs, test_users_targ[uid]);
		global_prec = global_prec + mp;

		mix_sort_neighs = sorted(mix_neighs.items(), lambda x, y:cmp(x[1],y[1]), reverse=True);
		mix_mp = Precision(mix_sort_neighs, test_users_targ[uid]);
		mix_global_prec = mix_global_prec + mix_mp;

	global_prec = global_prec * 1.0 / len(test_users);
	mix_global_prec = mix_global_prec * 1.0 / len(test_users);
	print "precision ", global_prec;
	print "mix precision",mix_global_prec; 
	print "learn rate", learn_rate;
	print "iteration", iter;
	if(iter > 0):
		if(old_prec > global_prec):
			learn_rate = learn_rate / 2.0;
	old_prec = global_prec;
