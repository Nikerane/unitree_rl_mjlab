#!/usr/bin/env bash
#SBATCH --job-name=z1-fic-pilot
#SBATCH --account=d2026d06-166-users
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --exclude=gn03,gn34
#SBATCH --no-requeue
#SBATCH --output=/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot-%A_%a.out
#SBATCH --error=/ceph/hpc/home/eunikhilr/z1-fic-joint-pilot-%A_%a.err

printf '%s\n' \
  'FIC_PILOT_RETIRED: scripts/slurm/vega_fic_joint_pilot.sbatch is RETIRED.' \
  'Use scripts/slurm/vega_fic_direct_reference.sbatch instead.' \
  'The banked results remain reproducible from frozen revision 0b270e26fac03cea5b2fedac5a34b60276dd9a4d.' \
  >&2
exit 2
