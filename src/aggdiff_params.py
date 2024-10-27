import argparse
"""

"""
best_params_dict_aggdiff= {
# Done
'texas': {'reaction_term':'aggdiff-gat', 'alpha_dim':'sc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'euler', 'time': 1.386920368271074,'step_size': 0.6,
          'epoch':200, 'lr': 0.01 ,'decay': 0.0005,
          'block':'constant', 'hidden_dim': 128 , 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.5, 'dropout': 0,
          'use_mlp': False, 'm2_mlp' : True, 'XN_activation': False, 
          'diffusion_rate1': 0.3, 'diffusion_rate2': 0.2, 'time_encoding': 'mlp', 'heads':1,
           }, 
# Done
'wisconsin': {'reaction_term':'aggdiff-sin', 'alpha_dim':'vc', 'beta_dim':'sc', 'beta_diag':True, 
          'method':'euler', 'time': 1.6787937304960685,'step_size': 0.45,
          'epoch':200, 'lr': 0.01449628926673464 ,'decay': 0.00901220554965404,
          'block':'attention', 'hidden_dim': 32 , 'data_norm':'gcn', 'self_loop_weight':1,
          'input_dropout': 0.5394876953124689, 'dropout': 0.4849875143400876,
          'use_mlp': False, 'm2_mlp' : False, 'XN_activation': True, 
          'diffusion_rate1': 0.2, 'diffusion_rate2': 0.5, 'time_encoding': 'mlp'
           },
# Done
'cornell': {'reaction_term':'aggdiff-gat', 'alpha_dim':'vc', 'beta_dim':'sc', 'beta_diag':True, 
          'method':'euler', 'time': 0.3245737289644446,'step_size': 0.4,
          'epoch':200, 'lr': 0.008225031932075681 ,'decay': 0.028046982280138487,
          'block':'constant', 'hidden_dim': 64 , 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.48912249722614337, 'dropout': 0.3159670329306962,
          'use_mlp': True, 'm2_mlp' : False, 'XN_activation': True, 
          "diffusion_rate1":0.75, 'diffusion_rate2':0.95, 'time_encoding': 'None'
           },
# Done
'film': {'reaction_term':'aggdiff-sin', 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'euler', 'time': 0.17706405373935158,'step_size': 0.2,
          'epoch':200, 'lr': 0.007850471605952366 ,'decay': 0.0014341594128526826,
          'block':'constant', 'hidden_dim': 128 , 'data_norm':'rw', 'self_loop_weight':0,
          'input_dropout': 0.416670279482016, 'dropout': 0.6450760407185196,
          'use_mlp': False, 'm2_mlp' : True,'XN_activation': True, 
          "diffusion_rate1":0.8200000000000001, 'diffusion_rate2':0.27, 'time_encoding': 'None'
           },
# Done
'squirrel': {'reaction_term':'aggdiff-gauss', 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'euler', 'time': 3.7681007348220046,'step_size':0.8,
          'epoch':200, 'lr': 0.01713941076746923 ,'decay': 2.8396246589758592e-06,
          'block':'constant', 'hidden_dim': 256, 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.5156955697758989, 'dropout': 0.09328362336851624,
          'use_mlp': False, 'm2_mlp': True, 'XN_activation': True, 
          "diffusion_rate1":0.8, 'diffusion_rate2':0.75, 'time_encoding': 'None', 'layer_norm': True
           },
# Done---
'chameleon': {'reaction_term':'aggdiff-gat', 'alpha_dim':'sc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'euler', 'time': 2.0073496872209295,'step_size':2,
          'epoch':200, 'lr': 0.0067371581757143285 ,'decay': 7.736946152049231e-05,
          'block':'attention', 'hidden_dim': 256, 'data_norm':'gcn', 'self_loop_weight':1,
          'input_dropout': 0.6759632513264229, 'dropout': 0.09328362336851624,
          'use_mlp': False, 'm2_mlp': True, 'XN_activation': True, 
            "diffusion_rate1":1, 'diffusion_rate2':0.59, 'time_encoding': 'None', 
           },
# Done
'Cora': {'reaction_term':'aggdiff-gat', 'alpha_dim':'sc', 'beta_dim':'sc', 'beta_diag':True, 
          'method':'euler', 'time': 4.004545316695705,'step_size':0.8,
          'epoch':200, 'lr': 0.011402915506754104 ,'decay': 0.008014968630105014,
          'block':'constant', 'hidden_dim': 256, 'data_norm':'gcn', 'self_loop_weight':1,
          'input_dropout': 0.5043839651430236, 'dropout': 0.4145754297432822,
          'use_mlp': False, 'm2_mlp': True, 'XN_activation': True,
          "diffusion_rate1":0.25, 'diffusion_rate2':0.95, 'time_encoding': 'None'
           },
# Done
'Citeseer': {'reaction_term':'aggdiff-gauss', 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True, 
          'method':'euler', 'time': 2.3711122780523177,'step_size':0.25,
          'epoch':200, 'lr': 0.0029496654117168557 ,'decay': 0.013789766632941278,
          'block':'constant', 'hidden_dim': 32, 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.5224892802449188, 'dropout': 0.46161962752030056,
          'use_mlp': False, 'm2_mlp': True, 'XN_activation': False, 
          "diffusion_rate1":0.1, 'diffusion_rate2':0.95, 'time_encoding': 'None'
           },
# Done
'Pubmed': {'reaction_term':'aggdiff-log', 'alpha_dim':'vc', 'beta_dim':'sc', 'beta_diag':True, 
          'method':'euler', 'time': 1.1013192919418102,'step_size':0.6000000000000001,
          'epoch':200, 'lr': 0.010838870718586332 ,'decay': 0.0005182464582183332,
          'block':'constant', 'hidden_dim': 256, 'data_norm':'gcn', 'self_loop_weight':0,
          'input_dropout': 0.3648432339951884, 'dropout': 0.25687002898139405,
          'use_mlp': True, 'm2_mlp': True, 'XN_activation': True, 
          "diffusion_rate1":0.8, 'diffusion_rate2':0.9, 'time_encoding': 'None', "log_eps": 0.3952429146873153
           },
}

def shared_grand_params(opt):
    opt['block'] = 'constant'
    opt['function'] = 'laplacian'
    opt['optimizer'] = 'adam'
    opt['epoch'] = 200
    opt['lr'] = 0.001
    opt['method'] = 'euler'
    opt['geom_gcn_splits'] = True
    return opt

def shared_gread_params(opt):
    opt['function'] = 'gread'
    opt['optimizer'] = 'adam'
    opt['geom_gcn_splits'] = True
    return opt

def hetero_params(opt):
    #added self loops and make undirected for chameleon & squirrel
    if opt['dataset'] in ['chameleon', 'squirrel']:
        opt['hetero_SL'] = True
        opt['hetero_undir'] = True
    return opt