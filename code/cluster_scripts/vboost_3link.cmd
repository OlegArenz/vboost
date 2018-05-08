#!/bin/bash
#SBATCH -A project00490
#SBATCH -J vboost_3Link
#SBATCH --mail-type=ALL
# Bitte achten Sie auf vollständige Pfad-Angaben:
#SBATCH -e /home/j_arenz/jobs/vboost_3Link.err.%j
#SBATCH -o /home/j_arenz/jobs/vboost_3Link.out.%j
#
#SBATCH -n 20      # Anzahl der MPI-Prozesse
#SBATCH -c 1     # Anzahl der Rechenkerne (OpenMP-Threads) pro MPI-Prozess
#SBATCH --mem-per-cpu=1024   # Hauptspeicher pro Rechenkern in MByte
#SBATCH -t 2-00:00:00     # in Stunden, Minuten und Sekunden, oder '#SBATCH -t 10' - nur Minuten

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 0 -vboost -progressDir _output/npvi/rank0/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 0 -vboost -progressDir _output/npvi/rank0/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 0 -vboost -progressDir _output/npvi/rank0/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 0 -vboost -progressDir _output/npvi/rank0/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 0 -vboost -progressDir _output/npvi/rank0/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 1 -vboost -progressDir _output/npvi/rank1/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 1 -vboost -progressDir _output/npvi/rank1/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 1 -vboost -progressDir _output/npvi/rank1/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 1 -vboost -progressDir _output/npvi/rank1/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 1 -vboost -progressDir _output/npvi/rank1/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 2 -vboost -progressDir _output/npvi/rank2/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 2 -vboost -progressDir _output/npvi/rank2/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 2 -vboost -progressDir _output/npvi/rank2/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 2 -vboost -progressDir _output/npvi/rank2/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 2 -vboost -progressDir _output/npvi/rank2/seed4/ -seed 4 &

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 3 -vboost -progressDir _output/npvi/rank3/seed0/ -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 3 -vboost -progressDir _output/npvi/rank3/seed1/ -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 3 -vboost -progressDir _output/npvi/rank3/seed2/ -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 3 -vboost -progressDir _output/npvi/rank3/seed3/ -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model planarRobot_3 -rank 3 -vboost -progressDir _output/npvi/rank3/seed4/ -seed 4 &


wait
