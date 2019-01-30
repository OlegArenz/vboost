#!/bin/bash
#SBATCH -A project00790
#SBATCH -J npvi_gg
#SBATCH --mail-type=ALL
# Bitte achten Sie auf vollständige Pfad-Angaben:
#SBATCH -e /home/j_arenz/jobs/npvi_gg.err.%j
#SBATCH -o /home/j_arenz/jobs/npvi_gg.out.%j
#
#SBATCH -n 10      # Anzahl der MPI-Prozesse
#SBATCH -c 1     # Anzahl der Rechenkerne (OpenMP-Threads) pro MPI-Prozess
#SBATCH --mem-per-cpu=1024   # Hauptspeicher pro Rechenkern in MByte
#SBATCH -t 2-00:00:00     # in Stunden, Minuten und Sekunden, oder '#SBATCH -t 10' - nur Minuten

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed5/ -seed 5 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed6/ -seed 6 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed7/ -seed 7 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed8/ -seed 8 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed9/ -seed 9 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed11/ -seed 11 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed12/ -seed 12 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed13/ -seed 13 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed14/ -seed 14 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=1024 python2.7 mlm_main.py -model german_credit -npvi -progressDir _output/npvi/seed15/ -seed 15 &


wait
