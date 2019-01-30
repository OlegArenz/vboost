#!/bin/bash
#SBATCH -A project00790
#SBATCH -J vboost_gg
#SBATCH --mail-type=ALL
# Bitte achten Sie auf vollständige Pfad-Angaben:
#SBATCH -e /home/j_arenz/jobs/vboost_gg.err.%j
#SBATCH -o /home/j_arenz/jobs/vboost_gg.out.%j
#
#SBATCH -n 5      # Anzahl der MPI-Prozesse
#SBATCH -c 1     # Anzahl der Rechenkerne (OpenMP-Threads) pro MPI-Prozess
#SBATCH --mem-per-cpu=1024   # Hauptspeicher pro Rechenkern in MByte
#SBATCH -t 2-00:00:00     # in Stunden, Minuten und Sekunden, oder '#SBATCH -t 10' - nur Minuten


srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -rank 10 -vboost -progressDir _output/rank10/seed5/ -seed 5 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -rank 10 -vboost -progressDir _output/rank10/seed6/ -seed 6 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -rank 10 -vboost -progressDir _output/rank10/seed7/ -seed 7 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -rank 10 -vboost -progressDir _output/rank10/seed8/ -seed 8 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -rank 10 -vboost -progressDir _output/rank10/seed9/ -seed 9 &


wait
