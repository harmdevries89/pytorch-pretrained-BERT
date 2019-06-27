import argparse
import pathlib
import subprocess
import sys
import socket
import os
import json
import shutil

ALL_NODES = "dgx01,dgx02,dgx03,dgx04,dgx05,dgx06,dgx07"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="name of the current run, used for logging directory",
)
parser.add_argument(
    "--nodes",
    type=str,
    default=ALL_NODES,
    help="names of nodes to launch training on e.g.: dgx01,dgx02",
)
parser.add_argument(
    "--config", type=str, help="which training config to use"
)
parser.add_argument(
    "--workdir",
    type=str,
    default="/home/nathan/code/pytorch-pretrained-BERT",
    help="which training config to use",
)
parser.add_argument(
    "--launch-tensorboard",
    action="store_true",
    help="Also launch tensorboard instance for monitoring run",
)
parser.add_argument("--load-ckpt", type=str, help="checkpoint to resume from")
parser.add_argument("--resume", action="store_true", help="checkpoint to resume from")
parser.add_argument("--stop", action="store_true", help="whether to resume training")
parser.add_argument(
    "--clean", action="store_true", help="whether to clean experiment directory"
)

# distributed args
parser.add_argument("--n-gpus", type=int, default=8, help="number of gpus per node")

parser.add_argument(
    "--master-port", type=int, default=1234, help="number of gpus per node"
)
parser.add_argument("--disable-infiniband", action="store_true", help="whether to tell NCCL to use Infiniband")

args = parser.parse_args()


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def ssh(host, command):
    ssh = subprocess.Popen(
        ["ssh", host, command],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return ssh


def watch(processes):
    if not isinstance(processes, list):
        processes = [processes]

    while processes:
        not_dead = []
        for process in processes:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                continue
            if output:
                not_dead.append(process)
                print(output.strip())

        processes = not_dead


def get_git_revision(base_path):
    git_dir = pathlib.Path(base_path) / ".git"
    with (git_dir / "HEAD").open("r") as head:
        ref = head.readline().split(" ")[-1].strip()

    with (git_dir / ref).open("r") as git_hash:
        return git_hash.readline().strip()


def stop(name, nodes):
    if not query_yes_no("Are you sure you want to stop this experiment?"):
        return

    print("Killing {} on {}".format(name, nodes))
    cmd = 'docker kill $(docker ps -q --filter "name={}")'.format(name)
    processes = []
    for host in nodes:
        processes.append(ssh(host, cmd))

    watch(processes)


def setup_experiment(exp_dir, clean=False):
    if clean:
        shutil.rmtree(exp_dir)

    os.mkdir(exp_dir)
    os.mkdir(os.path.join(exp_dir, "logs"))
    os.mkdir(os.path.join(exp_dir, "checkpoints"))
    os.mkdir(os.path.join(exp_dir, "tensorboard"))


def launch_tensorboard(exp_dir):
    cmd = "docker run -d -p 0.0.0.0:6006:6006 --name tensorboard -v {exp_dir}:/data -it tensorflow/tensorflow tensorboard --logdir /data".format(
        exp_dir=exp_dir
    )
    os.system(cmd)

def get_latest_checkpoint(path):
    filename = max(os.listdir(path), key=lambda f: int(f.split('.')[1]))
    return os.path.join(path, filename)

def main():
    user = os.environ["USER"]
    args.nodes = args.nodes.split(",")
    exp_dir = os.path.join(os.sep, "home", user, "experiments", args.name)

    if args.stop:
        stop(args.name, args.nodes)
        return

    # Launch tensorboard
    if args.launch_tensorboard:
        launch_tensorboard(os.path.join(os.sep, "home", user, "experiments"))
        return


    if args.resume or args.load_ckpt:
        config_path = os.path.join(exp_dir, 'config.json')
    elif args.config:
        config_path = args.config
    else:
        print("Must provide a config.json")
        return

    model_config = {}
    with open(config_path) as config_file:
        model_config = json.load(config_file)

    model_config["exp-dir"] = exp_dir

    # Create directories
    if args.load_ckpt or args.resume:
        if args.resume:
            ckpt_path = get_latest_checkpoint(os.path.join(exp_dir, 'checkpoints'))
        else:
            ckpt_path = os.path.join(exp_dir, 'checkpoints', args.load_ckpt)

        print("Resuming from checkpoint: {}".format(ckpt_path)) 

        if not os.path.exists(ckpt_path):
            print("Checkpoint does not exist. Aborting.")
            return

        model_config['load-ckpt'] = ckpt_path
    else:
        print("Creating experiment directory at {}".format(exp_dir))

        clean = args.clean and query_yes_no(
            "Are you sure you want to clean this experiment? This will delete all logs and checkpoints"
        )
        setup_experiment(exp_dir, clean=clean)

        # Save copy of config json
        shutil.copy(args.config, os.path.join(exp_dir, 'config.json'))

    # Connect to machines and launch
    docker_args = """--rm --name={name} --privileged -v /home/hdvries/:/home/hdvries -v /home/nathan/:/home/nathan -v /dev/infiniband:/dev/infiniband \
-w {workdir} --ipc=host --network=host -e PYTHONPATH={workdir} -e NCCL_DEBUG=INFO -e NCCL_SOCKET_IFNAME="bond0.186" -e NCCL_IB_DISABLE={ib_disable} --dns 192.168.170.100 """.format(
        workdir=args.workdir, name=args.name, ib_disable=int(args.disable_infiniband)
    )
    image = "images.borgy.elementai.net/multi/bert"
    master_ip = socket.gethostbyname(args.nodes[0])

    if model_config:
        model_args = " ".join("--{} {}".format(k, v) for k, v in model_config.items())
    else:
        model_args = ""

    processes = []
    for i, host in enumerate(args.nodes):
        docker_cmd = "nvidia-docker run {} {}".format(docker_args, image)

        distributed_args = "--nproc_per_node {n_gpus} --nnodes {nnodes} --node_rank {node_rank} --master_addr {master_ip} --master_port {master_port} --use_env".format(
            n_gpus=args.n_gpus,
            nnodes=len(args.nodes),
            node_rank=i,
            master_ip=master_ip,
            master_port=args.master_port,
        )

        runner = "python -m torch.distributed.launch {} examples/run_pretraining_bert.py {} >> {exp_dir}/node.{host}.log 2>&1".format(
            distributed_args, model_args, exp_dir=exp_dir, host=host
        )

        cmd = "{} {}".format(docker_cmd, runner)

        print(cmd)

        ssh_handle = ssh(host, cmd)
        processes.append(ssh_handle)

    watch(processes)


if __name__ == "__main__":
    main()
