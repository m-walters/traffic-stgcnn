#!/bin/bash
runpath="/home/michael/msc/summer17/traffic/traffig-stgcnn/veldata/"
runtype="secondring"
#source ~/jupyter_py2/bin/activate

for t in {"0.5","1.0"} # tcutoff
do
	for v in {"1.0",} # velmin
	do
		for l in {10,20} # tg length
		do
			runname=$runtype"_t"$t"v"$v"l"$l
			#sbatch --output=$runpath$runname".log" makestates $t $v $n $s $r $l $runname $runpath
			python -u gen_vels.py -t $t -v $v -l $l --runname=${runname} --runname=${runpath}
		done
	done
done

