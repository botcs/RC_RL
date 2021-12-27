
# training

rm reward_histories/vgfmri3_sokoban_reward_history_repeated_trial1.csv
rm model_weights/vgfmri3_sokoban_trial1_repeated.pt

python -W ignore::UserWarning runDDQN.py -timeout=250000 -max_steps=1000000 -max_level_steps=250000 -level_switch=repeated -game_name=vgfmri3_sokoban  -doubleq=1  -batch_size=32  -target_update=1000 -gamma=0.999

# eval

#python -W ignore::UserWarning runDDQN.py -timeout=1200 -max_steps=100000 -max_level_steps=10000 -level_switch=fmri -game_name=vgfmri3_sokoban -use_ddp=0 -doubleq=1 -pretrain=1 -model_weight_path=model_weights/vgfmri3_sokoban_trial1_repeated.pt
