import os
import random
import wandb
import torch
import math
import pickle
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from math import ceil
# from torch.optim import Adam, AdamW
# from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.utils import *
from utils.functions import *
from model import VRPModel
from envs.env import MTVRPEnv, get_dataloader
from envs.transformer import StateAugmentation

# Utils function
def normalize_coord(coord:torch.Tensor) -> torch.Tensor: 
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    scale = max((x_max - x_min) , (y_max - y_min))
    # x_scaled = (x - x_min) / (x_max - x_min) 
    # y_scaled = (y - y_min) / (y_max - y_min)
    x_scaled = (x - x_min) / scale
    y_scaled = (y - y_min) / scale
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    # return coord_scaled
    return coord_scaled, scale

def metric2str(metric_label, metric_list):
    metric_info = '|'.join([f'{metric_label[i]} {metric_list[i]:.4f}' for i in range(len(metric_label))])
    return metric_info

def transform_dict_to_mean(dict_):
    for k, v in dict_.items():
        dict_[k] = torch.tensor(v).mean().item()

def cal_model_size(model, args):
    param_count = sum(param.nelement() for param in model.parameters())
    buffer_count = sum(buffer.nelement() for buffer in model.buffers())
    args.log('Total number of parameters: {}'.format(param_count))
    args.log('Total number of buffer elements: {}'.format(buffer_count))

class VRPTrainer:
    def __init__(self,args):
        self.args=args
        # cuda
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # Main Components
        self.model = VRPModel(args) # args.log(f'current device:{self.model.device}')
        cal_model_size(self.model, args) #
        # exit(0)
        self.env = MTVRPEnv(**args.env)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = args.optimizer_params['optimizer']['lr'],
            weight_decay=args.optimizer_params['optimizer']['weight_decay'] if 'weight_decay' in args.optimizer_params['optimizer'] else 0,
        )
        if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones = args.optimizer_params['scheduler']['milestones'],
                gamma = args.optimizer_params['scheduler']['gamma']
            )
        else:
            raise NotImplementedError
        # Restore
        self.start_epoch = 1
        model_load = args.trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location='cuda')
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict, strict=True)
            self.start_epoch = 1 + model_load['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            args.log(f'Saved Model Loaded from {checkpoint_fullname}.')
            # set random state
            if not args.ddp: # ddp
                self.env.__setstate__(checkpoint['env_state_dict'])
                torch.set_rng_state(checkpoint['rng_state_dict']['torch.rng_state'].cpu())
                torch.cuda.set_rng_state(checkpoint['rng_state_dict']['torch.cuda.rng_state'].cpu())
                random.setstate(checkpoint['rng_state_dict']['random.state'])
            else:
                self.env.__setstate__(checkpoint['env_state_dict'], set_seed=False)
            self.env.data_dir = args.env['data_dir']
        # ddp model
        if args.ddp:
            torch.distributed.barrier()
            self.model = DistributedDataParallel(self.model)  # , find_unused_parameters=True)
            # assure different ddp workers have the same param
            for param in self.model.parameters(): # if all ddp.worker load the same ckpt, this may be redundant
                dist.broadcast(param.data, src=0)
            args.log(f'use ddp, current device:{torch.cuda.current_device()}')
        # test dataloader
        if args.test:
            self.test_dataloader = get_dataloader(
                self.env.dataset(phase='test', data_size=args.env['test_episodes']),
                batch_size=args.env['test_batch_size'],
                ddp=args.ddp,
                num_workers=args.num_workers
            )
            self.augmentation = StateAugmentation()
        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        # test before train
        if args.test_lib: 
            self.test_lib(self.start_epoch - 1)
            return 
        if args.test: self.test(self.start_epoch - 1)
        if args.test_only: exit(0)
        # begin train
        for epoch in range(self.start_epoch, args.trainer_params['epochs']+1):
            args.log('=================================================================')
            if args.wandb != '' and not args.mute: wandb.log({f'lr': self.optimizer.param_groups[0]['lr']}, step=epoch) #
            ########################## one epoch ###########################
            self.model.train()
            train_pbar = tqdm(range(args.trainer_params['train_step']) , bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}', leave=False)
            train_label = f"Train|Epoch{str(epoch).zfill(3)}/{str(args.trainer_params['epochs']).zfill(3)}"
            all_metric = [] # metric: [score loss grad] # metric: [score loss grad]
            for step in train_pbar: # task_param (p,n)
                if args.skip and step > 2: break
                self.optimizer.zero_grad()
                n_loc = args.n_size
                batch_size = args.batch_size
                self.env.generator.reset_n_loc(n_loc)
                td = self.env.reset(batch_size=batch_size).to('cuda') # sample
                if args.ddp: torch.distributed.barrier()
                out = self.model(td, self.env)
                reward = out["reward"].view(-1, batch_size) # repeat_num, batch_size
                log_likelihood = out["log_likelihood"].view(-1, batch_size)  # repeat_num, batch_size
                # loss
                advantage = reward - reward.mean(dim=0, keepdims=True) # repeat_num, batch_size
                loss = -(advantage * log_likelihood).mean()
                # Score
                max_pomo_reward, _ = reward.max(dim=0)  # bs   get best results from pomo
                score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
                if args.ddp: torch.distributed.barrier()
                loss.backward()
                grad_norms, grad_norms_clipped = clip_grad_norms(self.optimizer.param_groups, 1.)
                self.optimizer.step()
                # Log
                metric_list = [loss.item(), 
                               score_mean.item(),
                               grad_norms[0].item()]
                # metric_list = [loss.item(), score_mean.item(), 1.]
                all_metric.append(metric_list)
                metric_info = '|'.join([f'{args.metric_label[i]} {metric_list[i]:.4f}' for i in range(len(args.metric_label))])
                train_pbar.set_description(f"🙏> {train_label}|{metric_info}")
            if args.ddp: torch.distributed.barrier()
            # Log Once, for each epoch
            ## on each device
            metric_tensor = torch.tensor(all_metric).mean(dim=0)
            metric_list = metric_tensor.tolist()
            metric_info = '|'.join([f'{args.metric_label[i]} {metric_list[i]:.4f}' for i in range(len(args.metric_label))])
            elapsed = f"{tqdm.format_interval(train_pbar.format_dict['elapsed'])}"
            args.log(f"{train_label}|{elapsed}|{metric_info}|LR {self.optimizer.param_groups[0]['lr']:.2e}")
            ## on all devices
            if args.ddp:
                metric_tensor_ = metric_tensor.to('cuda')
                dist.reduce(metric_tensor_, dst=0)  # op=ReduceOp.SUM
                if args.rank == 0:
                    metric_avg = metric_tensor_ / dist.get_world_size()  # average over all workers
                    metric_info = '|'.join([f'{args.metric_label[i]} {metric_avg[i]:.4f}' for i in range(len(args.metric_label))])
                    args.log(f'***ddp_reduce*** {train_label}|{elapsed}|{metric_info}')
                    if args.wandb != '' and args.rank == 0:
                        wandb.log({f'{args.metric_label[i]}_train': metric_list[i] for i in range(len(args.metric_label))}, step= epoch) # epoch*2
                torch.distributed.barrier()
            elif args.wandb != '':
                wandb.log({f'{args.metric_label[i]}_train': metric_list[i] for i in range(len(args.metric_label))}, step= epoch) #  epoch*2
            # test during train
            if args.test and (epoch % args.env['test_interval'] == 0 or epoch in args.env['test_epoch']): self.test(epoch)
            ########################## one epoch end ###########################
            # MultiStepLR LR Decay
            if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
                self.scheduler.step()
            # Remain times
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, args.trainer_params['epochs'])
            args.log("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, args.trainer_params['epochs'], elapsed_time_str, remain_time_str))
            # Save Model
            if ((epoch == args.trainer_params['epochs']) or (epoch % args.trainer_params['model_save_interval']) == 0):
                if not args.mute:
                    args.log("Saving trained_model")
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict() if not self.args.ddp else self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'env_state_dict': self.env.__getstate__(),
                        'rng_state_dict': {'torch.rng_state': torch.get_rng_state(),
                                           'torch.cuda.rng_state': torch.cuda.get_rng_state(),
                                           'random.state': random.getstate()}
                    }
                    torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(args.result_dir, epoch))
            # end of epoch
            if args.ddp: torch.distributed.barrier()
        args.log(" *** Training Done *** ")

    @torch.no_grad()
    def test(self, epoch):
        args = self.args
        self.model.eval()

        results = []

        dataset_num = len(self.test_dataloader)

        for data_idx, (dataset_name, test_dataloader) in enumerate(self.test_dataloader.items()):

            eval_label = (
                f"Eval {dataset_name:20s} "
                f"{str(data_idx).zfill(3)}/{str(dataset_num).zfill(3)} "
                f"|Epoch{str(epoch).zfill(3)}"
            )

            eval_pbar = tqdm(
                test_dataloader,
                bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}',
                leave=False
            )

            instance_offset = 0

            for step, inp in enumerate(eval_pbar):

                if args.skip and step > 1:
                    break

                batch_size = inp.batch_size[0]
                td = self.env.reset(td=inp.to('cuda'))
                td = self.augmentation(td)

                if args.ddp:
                    torch.distributed.barrier()

                out = self.model(td, self.env)

                all_reward = out["reward"].view(
                    -1,
                    self.augmentation.num_augment,
                    batch_size
                )

                all_reward, _ = all_reward.max(dim=0)
                score = -all_reward[0, :].float()
                aug_reward, _ = all_reward.max(dim=0)
                aug_score = -aug_reward.float()

                for i in range(batch_size):
                    results.append({
                        "dataset": dataset_name,
                        "instance_id": instance_offset + i,
                        "cost": score[i].item(),
                        "aug_cost": aug_score[i].item(),
                    })

                instance_offset += batch_size

                eval_pbar.set_description(
                    f"{eval_label} | "
                    f"cost {score.mean().item():.4f} | "
                    f"aug_cost {aug_score.mean().item():.4f}"
                )

            args.log(
                f"{eval_label} | "
                f"instances {instance_offset}"
            )

        if not args.mute:

            result_df = pd.DataFrame(results)

            save_path = os.path.join(
                args.result_dir,
                f"cada_results_{args.n_size}.csv"
            )

            result_df.to_csv(
                save_path,
                index=False
            )

            args.log(
                f"Saved CADA results to {save_path}"
            )

        if args.ddp:
            torch.distributed.barrier()


    @torch.no_grad()
    def test_lib(self, epoch):
        # with torch.amp.autocast("cuda"):
        import vrplib
        from einops import rearrange

        args = self.args
        self.model.eval()
        all_test_dataset = [
            # 'A',
            # 'B',
            # # 'E',
            # 'F',
            # 'M',
            # 'P',
            'X',
            # 'XML'
        ]
        size_limit = 251 #  if problem_size > problem_size_limit: continue
        all_dataset_dict = {'A': [], 'B': [], 'F': [], 'M': [], 'P': [],
                            'X-100-300': [], 'X-300-500': [], 'X-500-700': [], 'X-700-1000': []
                            }  # opt score gap
        # 
        for dataset in all_test_dataset:
            if dataset != 'XML':    
                dataset_dir = f'../data/lib_data/{dataset}' 
                sol_dir = f'../data/lib_data/{dataset}'
            else:
                dataset_dir = '../data/lib_data/cvrplib-Set-XML100/generated_instances'
                sol_dir = '../data/lib_data/cvrplib-Set-XML100/generated_solutions'
            path_list = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
            aug_factor = 8
            batch_size = 1
            all_gap, all_aug_gap = [], []
            gap_dict, aug_gap_dict = {},{}
            aug_score_dict = {}
            time_dict = {}
            opt_score_dict = {}
            if dataset == 'X':
                X_opt_score_list = [[], [], [], []]  # 100-300, 300-500, 500-700, 700-1000
                X_aug_score_dic = [[], [], [], []]  # 100-300, 300-500, 500-700, 700-1000
                X_aug_gap_dict = [[], [], [], []]  # 100-300, 300-500, 500-700, 700-1000
            for path in path_list:
                # self.logger.info(path)
                if path.endswith(".sol"):
                    continue
                # begin _solve_cvrplib                
                # get problem
                problem = vrplib.read_instance(path)
                coords = torch.tensor(problem['node_coord']).float()
                coords_norm, scale = normalize_coord(coords)
                original_capacity = problem['capacity']
                demand = torch.tensor(problem['demand'][1:]).float() / original_capacity
                original_capacity = torch.tensor(original_capacity)[None]                  
                td_instance = TensorDict({
                    "locs": coords_norm.unsqueeze(0), 
                    "demand_linehaul": demand.unsqueeze(0), 
                    "capacity_original": original_capacity.unsqueeze(0),
                }, batch_size=[1])
                td_reset = self.env.reset(td_instance,lib_data=True).to('cuda')
                # # get p_s_tag
                # keep_mask = torch.zeros((td_reset.shape[0],5), dtype=torch.bool) # 'c', 'o', 'tw', 'l', 'b'
                # keep_mask[:, 0] = True
                # td_reset['p_s_tag'] = torch.cat((
                #     keep_mask.float(),
                #     torch.full_like(td_reset['open_route'], td_reset['locs'].shape[1]/2000, dtype=torch.float32,device=td_instance.device)
                # ),dim=-1)
                num_features = 5
                num_combinations = 2 ** num_features  
                codes = torch.tensor([[(i >> j) & 1 for j in range(num_features)] for i in range(num_combinations)], dtype=torch.bool) # 32,5
                td_reset = batchify(td_reset, codes.shape[0])
                td_reset['p_s_tag'] = torch.cat((
                    codes.float(),
                    torch.full_like(td_reset['open_route'], td_reset['locs'].shape[1]/2000, dtype=torch.float32,device=td_instance.device)
                ),dim=-1)
                batch_size = codes.shape[0]
                #
                if size_limit is not None and td_reset['locs'].shape[1] > size_limit:
                    continue
                # get opt cost 
                instance_name = os.path.basename(path).split('.')[0]
                if dataset == 'XML':
                    sol_path = os.path.join(sol_dir, f"{instance_name}.vrp.sol")
                else:
                    sol_path = os.path.join(sol_dir, f"{instance_name}.sol")
                solution = vrplib.read_solution(sol_path)
                opt = solution['cost'] # note that this cost is somehow slightly lower than the one calculated from the distance matrix
                # solve
                start_time = time.time()
                td = self.augmentation(td_reset) #=td.expand(8, batch_size).contiguous().view(8*batch_size,)
                if args.ddp: torch.distributed.barrier()
                out = self.model(td, self.env) # repeat_num, 8, batch_size
                use_time = time.time()-start_time
                all_reward = out["reward"].view(-1, self.augmentation.num_augment, batch_size)  # repeat_num, 8, batch_size
                all_reward = rearrange(all_reward, 'r a b -> (r b) a').unsqueeze(-1)
                
                all_reward, _ = all_reward.max(dim=0)  # aug, batch_size=1
                score = -all_reward[0, :].float().item() # batch_size=1
                aug_reward, _ = all_reward.max(dim=0)  # batch=1
                aug_score = -aug_reward.float().item() # batch=1
                score, aug_score = ceil(score * scale), ceil(aug_score * scale)
                gap = (score - opt) / opt * 100
                aug_gap = (aug_score - opt) / opt * 100
                args.log(f"{instance_name}, aug score {aug_score:.1f}, aug gap {aug_gap:.1f}%")
                # update
                all_gap.append(gap), all_aug_gap.append(aug_gap)
                gap_dict[instance_name]=gap
                aug_gap_dict[instance_name] = aug_gap
                aug_score_dict[instance_name] = aug_score
                time_dict[instance_name] = use_time
                opt_score_dict[instance_name] = opt

                num = int(instance_name.split('_')[0][3:]) if dataset == 'XML' else int(instance_name.split('-')[1][1:])
                if dataset == 'X':
                    idx_ = 0 if num <= 300 else (1 if num <= 500 else (2 if num <= 700 else 3))
                    X_opt_score_list[idx_].append(opt)
                    X_aug_score_dic[idx_].append(aug_score)
                    X_aug_gap_dict[idx_].append(aug_gap)

            args.log(f"Avg aug Gap {sum(all_aug_gap) / len(all_aug_gap) :.2f}%")
            # save excel
            data = {'Instance Name': list(gap_dict.keys()),
                    'Opt Score': [opt_score_dict[name] for name in gap_dict.keys()],
                    'Aug Score': [aug_score_dict[name] for name in gap_dict.keys()],
                    'Aug Gap': [aug_gap_dict[name] for name in gap_dict.keys()],
                    'Use Time': [time_dict[name] for name in gap_dict.keys()],
                    }
            df = pd.DataFrame(data)
            df = df.sort_values('Instance Name')
            df.to_excel(f'{args.result_dir}/{dataset}.xlsx', index=False, engine='openpyxl')
            # update all xlsx
            if dataset != 'X' and dataset != 'XML':
                all_dataset_dict[dataset] = [np.mean(list(opt_score_dict.values())),
                                             np.mean(list(aug_score_dict.values())),
                                             np.mean(list(aug_gap_dict.values())),
                                             ]
            elif dataset == 'X':
                for idx_, num_ in enumerate(['100-300', '300-500', '500-700', '700-1000']):
                    all_dataset_dict[f'X-{num_}'] = [np.mean(X_opt_score_list[idx_]),
                                                     np.mean(X_aug_score_dic[idx_]),
                                                     np.mean(X_aug_gap_dict[idx_])
                                                     ]
        # save all dataset xlsx
        all_dataset_dict_ =  {}     
        for key, value in  all_dataset_dict.items():
            if len(value) !=0:
                all_dataset_dict_[key] = value
        df = pd.DataFrame(all_dataset_dict_)
        df.to_excel(f'{args.result_dir}/cvrplib.xlsx', index=False, engine='openpyxl')

