from utility import read_from_h5_file  , get_hot_encode , load_model_from_name , get_rank , get_pow_rank, get_rank_list_from_prob_dist
from utility import XorLayer , InvSboxLayer
from utility import METRICS_FOLDER 
from train_models_ResNet import model_multi_task_single_target,model_single_task    , model_multi_task_single_target_one_shared_mask, model_multi_task_single_target_not_shared,model_multi_task_single_target_one_shared_mask_shared_branch, model_multi_task_affine, model_multi_task_single_target_one_shared_mask_shared_branch_general_masking
from gmpy2 import mpz,mul
import argparse 
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import pickle 
   



class Attack:
    def __init__(self,training_type, n_experiments = 1000,n_traces = 10000,model_type = False, model_name = None, resnet = False):
        
        self.models = {}
        self.n_traces = n_traces
        
        traces , labels_dict, metadata  = read_from_h5_file(n_traces = self.n_traces,dataset = 'attack',load_plaintexts = True)
        self.traces = traces
        self.n_traces = traces.shape[0]
        input_length = traces.shape[1]

        if training_type == 'single_task_subin' or training_type == 'single_task_subout':
            for byte in range(2,16):
                model = model_single_task(input_length = input_length,summary = False, resnet = resnet)   
                model = load_model_from_name(model,'model_{}_{}.weights.h5'.format(training_type,byte))
                self.models[byte] = model
            
            target = 's' if 'subout' in training_type else 't'
            self.name = training_type


        elif training_type == 'multi_task_single_target':
    
            model = model_multi_task_single_target(input_length = input_length, resnet = resnet)                 
            target = 's'
  
        elif training_type == 'multi_task_single_target_not_shared':
    
            model = model_multi_task_single_target_not_shared(input_length = input_length, resnet = resnet)  
            target = 's'

        elif training_type == 'multi_task_single_target_one_shared_mask':
            
            model = model_multi_task_single_target_one_shared_mask(input_length = input_length, resnet = resnet)   
            target = 't'               

        elif training_type == 'multi_task_single_target_one_shared_mask_shared_branch':
            
            model = model_multi_task_single_target_one_shared_mask_shared_branch(input_length = input_length, resnet = resnet)    
            target = 't'                   
    
        elif training_type == 'multi_task_single_target_one_shared_mask_shared_branch_general_masking':
            
            model = model_multi_task_single_target_one_shared_mask_shared_branch_general_masking(input_length = input_length, resnet = resnet)    
            target = 't'
    
        elif 'multi_task_affine' in training_type:
            shared_branch = 'shared' in training_type
            model = model_multi_task_affine(input_length = input_length, resnet = resnet, shared_branch=shared_branch)
            target = 's'
    
        else:
            print('Some error here')       


        self.n_experiments = n_experiments
        self.powervalues = {}
        
        
        self.correct_guesses = {}
        self.history_score = {}
        self.traces_per_exp = 100
        
        plaintexts = np.array(metadata['plaintexts'],dtype = np.uint8)[:self.n_traces]
        self.plaintexts = get_hot_encode(plaintexts)
        batch_size = self.n_traces//10
  
        
        
     
        master_key = [0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF]
        self.subkeys = master_key

        self.powervalues = np.expand_dims(traces,2)
        predictions = {}
        if 'single_task' in training_type:
            for byte in range(2,16):
                predictions['output_{}'.format(byte)] = self.models[byte].predict({'traces':self.powervalues})['output']
        else:
            base_name = model_name if model_name else 'model_{}'.format(training_type)
            # self.name = '{}_{}'.format('model' if not model_type else 'multi_model',training_type)
            self.name = base_name
            model = load_model_from_name(model, self.name + '_all.weights.h5')

            predictions = model.predict({'traces':self.powervalues})
        self.predictions = np.empty((self.n_traces, 14,256),dtype =np.float32) 
        batch_size = 1000
        
        xor_op = XorLayer(name = 'xor')
        inv_op = InvSboxLayer(name = 'iinv')
        for batch in range(0,self.n_traces//batch_size):
            print('Batch of prediction {} / {}'.format(batch + 1,self.n_traces//batch_size))
            for byte in tqdm(range(2,16)):                  
                    if target == 't': 
                        self.predictions[batch*batch_size:(batch+1)*batch_size,byte-2] = xor_op([predictions['output_{}'.format(byte)][batch*batch_size:(batch+1)*batch_size],self.plaintexts[batch*batch_size:(batch+1)*batch_size,byte]])
                    else:
                        inv_pred = inv_op(predictions['output_{}'.format(byte)][batch*batch_size:(batch+1)*batch_size])
                        self.predictions[batch*batch_size:(batch+1)*batch_size,byte-2] = xor_op([inv_pred,self.plaintexts[batch*batch_size:(batch+1)*batch_size,byte]])
                    
        print(f"\n=== Per-byte s1 accuracy ===")
        for byte in range(14):
            _ , acc , _, _  = get_rank_list_from_prob_dist(self.predictions[:,byte],np.repeat(self.subkeys[byte+2],self.predictions.shape[0]))
            print(f"  Byte {byte + 2:2d}: {acc:.2f}%")
        
    def run(self,print_logs = False):

       for experiment in range(self.n_experiments):
           self.history_score[experiment] = {}
           self.history_score[experiment]['total_rank'] = []
           self.subkeys_guess = {}
           for i in range(2,16):
               self.subkeys_guess[i] = np.zeros(256,)
               self.history_score[experiment][i] = []
           traces_order = np.random.permutation(self.n_traces)[:self.traces_per_exp]
           count_trace = 1

           for trace in traces_order:
               all_recovered = True
               ranks = {}
               total_rank = mpz(1)

               for byte in range(2,16):
                   self.subkeys_guess[byte] += np.log(self.predictions[trace][byte-2] + 1e-36)
                   ranks[byte-2] = get_rank(self.subkeys_guess[byte],self.subkeys[byte])
                   self.history_score[experiment][byte].append(ranks[byte-2])
                   total_rank = mul(total_rank,mpz(ranks[byte-2]))
                   if np.argmax(self.subkeys_guess[byte]) != self.subkeys[byte]:
                       all_recovered = False

               self.history_score[experiment]['total_rank'].append(get_pow_rank(total_rank))

               if all_recovered:
                   for elem in range(count_trace,self.traces_per_exp):
                       for i in range(2,16):
                           self.history_score[experiment][i].append(ranks[i-2])
                       self.history_score[experiment]['total_rank'].append(0)
                   break
               else:
                   count_trace += 1

           if experiment % max(1, self.n_experiments // 10) == 0:
               print(f"  Exp {experiment}: final GE = 2^{self.history_score[experiment]['total_rank'][-1]}")

       array_total_rank = np.empty((self.n_experiments,self.traces_per_exp))
       for i in range(self.n_experiments):
           for j in range(self.traces_per_exp):
               array_total_rank[i][j] = self.history_score[i]['total_rank'][j]
       mean_ge = np.mean(array_total_rank,axis=0)
       whe = np.where(mean_ge < 2)[0]
       threshold = int(np.min(whe)) if whe.shape[0] >= 1 else self.traces_per_exp

       print(f"\n=== Fixed-key results ({self.n_experiments} experiments) ===")
       print(f"GE < 2 at trace: {threshold}")
       print(f"Final mean GE: 2^{mean_ge[-1]:.1f}")

       result_path = METRICS_FOLDER + 'history_attack_experiments_{}_{}'.format(self.name,self.n_experiments)
       file = open(result_path,'wb')
       pickle.dump(self.history_score,file)
       file.close()
       print(f"\nResults saved to {result_path}")


   
def run_attack(training_type,model_type, model_name = None, resnet = False):                
    attack = Attack(training_type,model_type = model_type, model_name = model_name, resnet = resnet)    
    attack.run()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--MULTI_MODEL', action="store_true", dest="MULTI_MODEL",
                        help='Classical training of the intermediates', default=False)
    parser.add_argument('--SINGLE',   action="store_true", dest="SINGLE", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    parser.add_argument('--model_type', type=str, default=None, 
                        help='Specific model type to attack (e.g., multi_task_single_target_one_shared_mask)')
    parser.add_argument('--model_name', type=str, default=None, 
                        help='Specific model name to attack (e.g., model_multi_task_single_target_one_shared_mask)')
    parser.add_argument('--resnet', action="store_true", dest="resnet", help='Using ResNet', default=False)
    args            = parser.parse_args()
  

    MULTI_MODEL        = args.MULTI_MODEL
    SINGLE        = args.SINGLE
    MULTI = args.MULTI
    MODEL_TYPE = args.model_type
    MODEL_NAME = args.model_name
    RESNET = args.resnet
 

    TARGETS = {}
    if MODEL_TYPE:
        training_types = [MODEL_TYPE]
    elif SINGLE:
        training_types = ['multi_task_single_target']

    elif MULTI:
        training_types = ['single_task_subin','single_task_subout']
    else:
        print('No training mode selected. Defaulting to multi_task_single_target_one_shared_mask')
        training_types = ['multi_task_single_target_one_shared_mask']


    for training_type in training_types:
        process_eval = Process(target=run_attack, args=(training_type,MULTI_MODEL, MODEL_NAME, RESNET))
        process_eval.start()
        process_eval.join()   
            
            
    
    
            
            
        
        