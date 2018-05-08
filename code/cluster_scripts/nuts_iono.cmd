#!/bin/bash
#SBATCH -A project00490
#SBATCH -J nuts_iono
#SBATCH --mail-type=ALL
# Bitte achten Sie auf vollständige Pfad-Angaben:
#SBATCH -e /home/j_arenz/jobs/nuts_iono.err.%j
#SBATCH -o /home/j_arenz/jobs/nuts_iono.out.%j
#
#SBATCH -n 5      # Anzahl der MPI-Prozesse
#SBATCH -c 1     # Anzahl der Rechenkerne (OpenMP-Threads) pro MPI-Prozess
#SBATCH --mem-per-cpu=2048   # Hauptspeicher pro Rechenkern in MByte
#SBATCH -t 2-00:00:00     # in Stunden, Minuten und Sekunden, oder '#SBATCH -t 10' - nur Minuten

srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=2048 python2.7 mlm_main.py -model "iono" -mcmc -mcmc_nsamps 10000 -progressDir _output/nuts_samples10k/seed0 -seed 0 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=2048 python2.7 mlm_main.py -model "iono" -mcmc -mcmc_nsamps 10000 -progressDir _output/nuts_samples10k/seed1 -seed 1 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=2048 python2.7 mlm_main.py -model "iono" -mcmc -mcmc_nsamps 10000 -progressDir _output/nuts_samples10k/seed2 -seed 2 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=2048 python2.7 mlm_main.py -model "iono" -mcmc -mcmc_nsamps 10000 -progressDir _output/nuts_samples10k/seed3 -seed 3 &
srun -N1 -n1 --time=2-00:00:00 --mem-per-cpu=2048 python2.7 mlm_main.py -model "iono" -mcmc -mcmc_nsamps 10000 -progressDir _output/nuts_samples10k/seed4 -seed 4 &


wait
