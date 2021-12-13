#python -W ignore::UserWarning runDDQN.py -game_name=vgfmri4_chase

games=( 'vgfmri3_chase' 'vgfmri3_helper' 'vgfmri3_bait' 'vgfmri3_lemmings' 'vgfmri3_plaqueAttack' 'vgfmri3_zelda')
#games=( 'vgfmri4_chase' 'vgfmri4_helper' 'vgfmri4_bait' 'vgfmri4_lemmings' 'vgfmri4_avoidgeorge' 'vgfmri4_zelda')
#games=(  'vgfmri3_helper' 'vgfmri3_bait'  'vgfmri3_zelda')

#games=(  'vgfmri4_lemmings' 'vgfmri4_zelda') # 90000, 2 days
#games=( 'vgfmri4_avoidgeorge' ) # 50000, 2 days

#games=( 'vgfmri4_helper' ) n 140000 , 4 days
#games=( 'vgfmri4_bait' ) # 140000, 4 days
#games=( 'vgfmri4_chase' ) # 50000, 4 days

echo ---------------- >> jobs.txt
echo --- $(date): Running runDDQN  >> jobs.txt
echo ---------------- >> jobs.txt
git log | head -n 1 >> jobs.txt

for game in ${games[*]}; do
    outfileprefix="${MY_SCRATCH}/VGDL/output/runDDQN_${game}"
    echo ---------------------------------------------------------------------------------
    echo  game ${game},  file prefix = $outfileprefix

    # send the job to NCF
    #
    #sbatch_output=`sbatch -p fasse_gpu --gres=gpu --mem 20001 -t 2-0:20 -o ${outfileprefix}_%j.out -e ${outfileprefix}_%j.err --wrap="source activate pedro; python -W ignore::UserWarning runDDQN.py -timeout=1200 -max_steps=1000000 -max_level_steps=100000 -level_switch=repeated -game_name=${game}"`
    sbatch_output=`sbatch -p fasse_gpu --gres=gpu --mem 20001 -t 2-0:20 -o ${outfileprefix}_%j.out -e ${outfileprefix}_%j.err --wrap="source activate pedro; python -W ignore::UserWarning runDDQN.py -timeout=1200 -max_steps=1000000 -max_level_steps=1200 -level_switch=fmri -game_name=${game} -pretrain=1 -model_weight_path=${game}_trial1_repeated.pt -num_trials=11"`
    # for local testing
    #sbatch_output=`echo Submitted batch job 88725418`
    echo $sbatch_output

    # Append job id to jobs.txt
    #
    sbatch_output_split=($sbatch_output)
    job_id=${sbatch_output_split[3]}
    echo runDDQN.py , game ${game}: ${outfileprefix}_${job_id}.out -- $sbatch_output >> jobs.txt

    echo watch job status with: sacct -j ${job_id}
    echo watch output with: tail -f ${outfileprefix}_${job_id}.out
    echo watch error with: tail -f ${outfileprefix}_${job_id}.err

    sleep 1
done
