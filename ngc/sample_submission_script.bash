#!/bin/bash

# These parameters actually have effects!
# TODO

USE_APEX=1

# Construct experiment identifier
# TODO
EXPERIMENT=""
EXPERIMENT+=""
if [ "$USE_APEX" = "1" ]; then EXPERIMENT+="-apex"; fi

# Setup environment in docker instance
CMD=""
CMD+="git clone https://wookiengc:ETHZurich123@github.com/mcbuehler/Seg2Eye;"
CMD+="cd Seg2Eye;"
CMD+="git checkout openeds;"
CMD+="echo;"
CMD+="git status;"  # Print cloned repo status
CMD+="echo Now on commit:;"
CMD+="git rev-parse --short HEAD;"
CMD+="echo;"
#CMD+="git checkout openeds;"  # Switch git branch
#CMD+="git checkout $COMMIT_HASH;"  # Checkout specified commit
#CMD+="git pull;"
CMD+="pip install --user -r requirements.txt;"

## Copy over HDF files
#CMD+="cp -v /data/GazeCapture.h5 /;"
#CMD+="cp -v /data/MPIIFaceGaze.h5 /;"

OUTPUT_DIR_STEM="/work/marcel/$(date +%y%m%d)_outputs"
OUTPUT_DIR="$OUTPUT_DIR_STEM/$EXPERIMENT"

# Copy source files to output dir
CMD+="mkdir -p ${OUTPUT_DIR};"
CMD+="rsync -a --include '*/' --include '*.py' --exclude '*' ./ ${OUTPUT_DIR}/src;"

# Construct train script call
CMD+="python3 train.py \
	--name $EXPERIMENT \
	\
	--dataroot /data/all.h5 \
	--dataset_mode openeds \
	\
	--no_vgg_loss \
"
if [ "$USE_APEX" == "1" ]; then CMD+=" --use_apex"; fi
CMD+=";"

# Fix permissions
CMD+="chown -R 120000:120000 $OUTPUT_DIR;"

# Strip unnecessary whitespaces
CMD=$(echo "$CMD" | tr -s ' ' | tr  -d '\n' | tr -d '\t')
echo $CMD

# Submit job to NGC
NGC_CMD="ngc batch run \
	--name \"$EXPERIMENT\" \
	--preempt RUNONCE \
	--ace nv-us-west-2 \
	--instance dgx1v.32g.8.norm \
	--image nvidia/pytorch:19.08-py3 \
	\
	--result /results \
	--workspace lpr-seonwook:/work:RW \
	--datasetid XXXXXX:/data \
	\
	--org nvidian \
	--team lpr \
"
if [ "$1" ]; then
	NGC_CMD+=" --apikey $1"
fi
NGC_CMD+=" --commandline \"$CMD\""
echo ""
echo $NGC_CMD
eval $NGC_CMD
