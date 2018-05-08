#!/bin/bash
#SBATCH -A project00490
#SBATCH -J vboost_frisk
#SBATCH --mail-type=ALL
# Bitte achten Sie auf vollständige Pfad-Angaben:
#SBATCH -e /home/j_arenz/jobs/vboost_frisk.err.%j
#SBATCH -o /home/j_arenz/jobs/vboost_frisk.out.%j
#
#SBATCH -n 40      # Anzahl der MPI-Prozesse
#SBATCH -c 1     # Anzahl der Rechenkerne (OpenMP-Threads) pro MPI-Prozess
#SBATCH --mem-per-cpu=1024   # Hauptspeicher pro Rechenkern in MByte
#SBATCH -t 2-00:00:00     # in Stunden, Minuten und Sekunden, oder '#SBATCH -t 10' - nur Minuten

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 0 -vboost -progressDir _output/rank0/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 0 -vboost -progressDir _output/rank0/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 0 -vboost -progressDir _output/rank0/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 0 -vboost -progressDir _output/rank0/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 0 -vboost -progressDir _output/rank0/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 1 -vboost -progressDir _output/rank1/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 1 -vboost -progressDir _output/rank1/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 1 -vboost -progressDir _output/rank1/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 1 -vboost -progressDir _output/rank1/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 1 -vboost -progressDir _output/rank1/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 2 -vboost -progressDir _output/rank2/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 2 -vboost -progressDir _output/rank2/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 2 -vboost -progressDir _output/rank2/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 2 -vboost -progressDir _output/rank2/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 2 -vboost -progressDir _output/rank2/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 3 -vboost -progressDir _output/rank3/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 3 -vboost -progressDir _output/rank3/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 3 -vboost -progressDir _output/rank3/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 3 -vboost -progressDir _output/rank3/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 3 -vboost -progressDir _output/rank3/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 4 -vboost -progressDir _output/rank4/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 4 -vboost -progressDir _output/rank4/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 4 -vboost -progressDir _output/rank4/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 4 -vboost -progressDir _output/rank4/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 4 -vboost -progressDir _output/rank4/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 5 -vboost -progressDir _output/rank5/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 5 -vboost -progressDir _output/rank5/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 5 -vboost -progressDir _output/rank5/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 5 -vboost -progressDir _output/rank5/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 5 -vboost -progressDir _output/rank5/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 10 -vboost -progressDir _output/rank10/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 10 -vboost -progressDir _output/rank10/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 10 -vboost -progressDir _output/rank10/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 10 -vboost -progressDir _output/rank10/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 10 -vboost -progressDir _output/rank10/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 20 -vboost -progressDir _output/rank20/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 20 -vboost -progressDir _output/rank20/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 20 -vboost -progressDir _output/rank20/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 20 -vboost -progressDir _output/rank20/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model frisk -rank 20 -vboost -progressDir _output/rank20/seed4/ -seed 4 &




wait
