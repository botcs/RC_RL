
# training
python -W ignore::UserWarning runDDQN.py -timeout=1200 -max_steps=10000 -max_level_steps=10000 -level_switch=repeated -game_name=vgfmri3_sokoban -use_ddp=1 -doubleq=1 

# eval

#python -W ignore::UserWarning runDDQN.py -timeout=1200 -max_steps=100000 -max_level_steps=10000 -level_switch=fmri -game_name=vgfmri3_sokoban -use_ddp=0 -doubleq=1 -pretrain=1 -model_weight_path=model_weights/vgfmri3_sokoban_trial1_repeated.pt
