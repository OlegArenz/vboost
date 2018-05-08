#!/bin/bash
#SBATCH -A project00490
#SBATCH -J vboost_gmm
#SBATCH --mail-type=ALL
# Bitte achten Sie auf vollständige Pfad-Angaben:
#SBATCH -e /home/j_arenz/jobs/vboost_gmm.err.%j
#SBATCH -o /home/j_arenz/jobs/vboost_gmm.out.%j
#
#SBATCH -n 35      # Anzahl der MPI-Prozesse
#SBATCH -c 1     # Anzahl der Rechenkerne (OpenMP-Threads) pro MPI-Prozess
#SBATCH --mem-per-cpu=400   # Hauptspeicher pro Rechenkern in MByte
#SBATCH -t 1-00:00:00     # in Stunden, Minuten und Sekunden, oder '#SBATCH -t 10' - nur Minuten

srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 0 -vboost -progressDir _output/rank0/seed5/ -seed 5 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 0 -vboost -progressDir _output/rank0/seed6/ -seed 6 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 0 -vboost -progressDir _output/rank0/seed7/ -seed 7 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 0 -vboost -progressDir _output/rank0/seed8/ -seed 8 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 0 -vboost -progressDir _output/rank0/seed9/ -seed 9 &

srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 1 -vboost -progressDir _output/rank1/seed5/ -seed 5 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 1 -vboost -progressDir _output/rank1/seed6/ -seed 6 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 1 -vboost -progressDir _output/rank1/seed7/ -seed 7 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 1 -vboost -progressDir _output/rank1/seed8/ -seed 8 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 1 -vboost -progressDir _output/rank1/seed9/ -seed 9 &

srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 2 -vboost -progressDir _output/rank2/seed5/ -seed 5 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 2 -vboost -progressDir _output/rank2/seed6/ -seed 6 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 2 -vboost -progressDir _output/rank2/seed7/ -seed 7 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 2 -vboost -progressDir _output/rank2/seed8/ -seed 8 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 2 -vboost -progressDir _output/rank2/seed9/ -seed 9 &

srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 3 -vboost -progressDir _output/rank3/seed5/ -seed 5 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 3 -vboost -progressDir _output/rank3/seed6/ -seed 6 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 3 -vboost -progressDir _output/rank3/seed7/ -seed 7 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 3 -vboost -progressDir _output/rank3/seed8/ -seed 8 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 3 -vboost -progressDir _output/rank3/seed9/ -seed 9 &

srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 4 -vboost -progressDir _output/rank4/seed5/ -seed 5 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 4 -vboost -progressDir _output/rank4/seed6/ -seed 6 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 4 -vboost -progressDir _output/rank4/seed7/ -seed 7 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 4 -vboost -progressDir _output/rank4/seed8/ -seed 8 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 4 -vboost -progressDir _output/rank4/seed9/ -seed 9 &

srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 5 -vboost -progressDir _output/rank5/seed5/ -seed 5 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 5 -vboost -progressDir _output/rank5/seed6/ -seed 6 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 5 -vboost -progressDir _output/rank5/seed7/ -seed 7 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 5 -vboost -progressDir _output/rank5/seed8/ -seed 8 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 5 -vboost -progressDir _output/rank5/seed9/ -seed 9 &

srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 10 -vboost -progressDir _output/rank10/seed5/ -seed 5 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 10 -vboost -progressDir _output/rank10/seed6/ -seed 6 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 10 -vboost -progressDir _output/rank10/seed7/ -seed 7 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 10 -vboost -progressDir _output/rank10/seed8/ -seed 8 &
srun -N1 -n1 --time=1-00:00:00 --mem-per-cpu=400 python2.7 mlm_main.py -model GMM_20 -rank 10 -vboost -progressDir _output/rank10/seed9/ -seed 9 &

wait
