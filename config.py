#!/mnt/lustre_fs/users/kavotaw/apps/anaconda/anaconda3/envs/parsl_py36/bin/python3
##/mnt/lustre_fs/users/kavotaw/apps/anaconda/anaconda3/bin/python3
from parsl.channels import SSHChannel
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher

from parsl.config import Config
from parsl.executors import HighThroughputExecutor,IPyParallelExecutor
from parsl.executors.ipp_controller import Controller
from parsl.addresses import address_by_hostname #add this to your imports

mcullaghcluster = Config(
    executors=[
         HighThroughputExecutor(
            label='mccullagh_cluster',
            cores_per_worker=20,
	    address='192.168.175.16',
            provider=SlurmProvider(
                channel=SSHChannel(
                    hostname='login1.rc.colostate.edu',
                    username='kavotaw',     # Please replace USERNAME with your username
                    script_dir='/home/kavotaw/lustrefs/z15-adfr/adfr_parsl_efficiency_test/',
		    password='' # Password goes here (or use keys)
                ),
                launcher=SrunLauncher(),
		scheduler_options="""#SBATCH --ntasks-per-node=20
#SBATCH --output=/mnt/lustre_fs/users/kavotaw/z15-adfr/adfr_parsl_efficiency_test/test.out
#SBATCH --error=/mnt/lustre_fs/users/kavotaw/z15-adfr/adfr_parsl_efficiency_test/test.err""",
		worker_init="""source activate parsl_py36
export PYTHONPATH="/mnt/lustre_fs/users/kavotaw/z15-adfr/adfr_parsl_efficiency_test/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/lustre_fs/users/kavotaw/adfr-vs-vina/install_mgl/mgltools2_x86_64Linux2_1.1/lib/"
export MGL_ROOT="/mnt/lustre_fs/users/kavotaw/adfr-vs-vina/install_mgl/mgltools2_x86_64Linux2_1.1/"
PATH="$MGL_ROOT/bin:$PATH"
pwd
cd /mnt/lustre_fs/users/kavotaw/z15-adfr/adfr_parsl_efficiency_test/""",
                walltime="24:00:00",
                init_blocks=4,
                max_blocks=4,
                nodes_per_block=1,
		partition='mccullagh',
            ),
        )
    ],
    strategy=None

)
