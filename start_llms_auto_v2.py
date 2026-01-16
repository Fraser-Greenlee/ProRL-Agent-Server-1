import subprocess
import time
import sys
import os

# --- 配置部分 ---
GPFS = "/lustre/"
HAOHOME = "/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh"
ROOT = f"{HAOHOME}/root"
RESULTS = f"{HAOHOME}/results"
PROJECTS = f"{HAOHOME}/projects"
DATA = f"{HAOHOME}/data"
HOME = HAOHOME
IMAGE = f"{HAOHOME}/dockers/nvidian+nemo+verl_v2_enroot_dev0.8.5.sqsh"


def log(message):
    print(f"[start_llms_auto_v2.py] {message}")


def run_cmd(cmd, shell=True, capture_output=True):
    """
    Run command and return the results
    """
    result = subprocess.run(cmd, shell=shell, text=True, capture_output=capture_output)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def submit_srun(node_name):
    """
    Boot-up the container in the node via sbatch
    """
    cmd = ["sbatch", f"--nodelist={node_name}", "start_container.sh"]

    log(f"Requesting node {node_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result


def get_job_id(node_name):
    """
    Get the job id launched in `node_name`
    """
    time.sleep(5)
    for _ in range(10):
        out, _, _ = run_cmd(f"squeue -h -w {node_name} -o %A -u $USER")
        if out and out.strip().isdigit():
            return out.strip()
        time.sleep(2)
    return None


def cancel_job_and_wait(job_id):
    """
    Cancel the Job and block (wait) until it is confirmed that the Job has disappeared from the queue.
    Ensure the environment is clean when retrying.
    """
    if not job_id:
        return

    log(f"Cancelling job {job_id}...")
    run_cmd(f"scancel {job_id}")

    # Loop and check whether the job is still in the queue
    max_wait = 60
    start_time = time.time()

    while time.time() - start_time < max_wait:
        # Check if job exists with the given job_id
        out, _, _ = run_cmd(f"squeue -h -j {job_id}")

        # If nothing shown, job is cancelled
        if not out.strip():
            log(f"Job {job_id} successfully confirmed cancelled.")
            return True

        time.sleep(2)
        log("Waiting for job to terminate...")

    log(f"Warning: Job {job_id} still appears in queue after {max_wait}s. Proceeding anyway.")
    return False


def ssh_exec(node, command):
    """Execute commands on node via SSH"""
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no {node} '{command}'"
    return run_cmd(ssh_cmd)


def get_container_pid(node_name):
    """Log into the node to get PID"""
    # Check if enroot list is working well and `sleep` is being shown as the command for enroot
    cmd1 = 'enroot list -f | head -n 2 | awk "{print \\$9}"'
    while True:
        out, err, code = ssh_exec(node_name, cmd1)
        if len(out.strip().split("\n")) > 1 and "sleep" in out.strip().split("\n")[1].strip():
            break
        time.sleep(1)

    # Fetch Container PID from enroot list
    cmd2 = 'enroot list -f | head -n 2 | awk "{print \\$2}"'
    out, err, code = ssh_exec(node_name, cmd2)

    return out.strip().split("\n")[1].strip()


def check_kvm_permission(node_name, pid):
    """
    Enter the container and check write permission on /dev/kvm
    """
    check_cmd = f"enroot exec {pid} /bin/bash -c 'test -w /dev/kvm && echo success'"
    ssh_cmd = f'ssh -o StrictHostKeyChecking=no {node_name} "{check_cmd}"'
    out, _, _ = run_cmd(ssh_cmd)

    log(f"check_kvm_permission: {out}")
    return "success" in out


def run_main_workload(node_name, pid, job_index):
    """
    Execute the final workload
    """
    vllm_cmd = (
        f"nohup vllm serve /lustre/fsw/portfolios/nvr/users/yidong/data/models/openai/gpt-oss-120b/ "
        f"--async-scheduling --data-parallel-size 8 "
        f"> {RESULTS}/vllm.log 2>&1 &"
    )

    wait_cmd = (
        "echo 'Waiting for vllm server to be ready...' && sleep 120 && "
        "while ! curl -s http://localhost:8000/health > /dev/null 2>&1; "
        "do sleep 10; echo 'Still waiting...'; done && echo 'vllm server is ready!'"
    )

    python_cmd = (
        f"cd {PROJECTS}/Yi_new/ProRL-Agent-Server/ && "
        f"python synthetic_data_generator.py --job-index {job_index} >> log_{job_index} 2>&1"
    )

    full_payload = (
        f"cd /workspace && source .venv/bin/activate && "
        f"export VLLM_ENABLE_RESPONSES_API_STORE=1 && "
        f"{vllm_cmd} {wait_cmd} && {python_cmd}"
    )

    log(f"Executing workload on {node_name} inside PID {pid}...")

    escaped_payload = full_payload.replace("'", "'\\''")
    final_cmd = f"enroot exec {pid} /bin/bash -c '{escaped_payload}'"

    subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=no", node_name, final_cmd])
    log("Workload command sent.")


def wait_for_job_running(job_id, timeout=300):
    """
    Check if job entered running (R) state.
    Handle pending (PD) or Configuring (CF) states.
    """
    log(f"Waiting for Job {job_id} to become RUNNING...")

    while True:
        # Get job status (-o %t: output only the state abbreviation, such as R, PD, CF, CG)
        state, _, _ = run_cmd(f"squeue -h -j {job_id} -o %t")
        state = state.strip()

        if state == "R":
            log(f"Job {job_id} is now RUNNING.")
            return True
        elif state == "PD":
            # Pending
            pass
        elif state == "CF":
            # Configuring: The node is being allocated/started, need to wait.
            log(f"Job {job_id} is CONFIGURING (CF)...")
        elif not state:
            log(f"Job {job_id} not found in queue. It may have failed.")
            return False

        time.sleep(5)


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <NODE_NAME> <JOB_INDEX>")
        sys.exit(1)

    target_node = sys.argv[1]
    job_index = sys.argv[2]

    while True:
        max_retries = 5
        retry_count = 0

        # cancel if job is running on target node under the current user
        job_id = get_job_id(target_node)
        if job_id:
            cancel_job_and_wait(job_id)

        while retry_count < max_retries:
            retry_count += 1
            log(f"=== Attempt #{retry_count} for node {target_node} ===")

            # 1. Request node and boot-up the container in the node
            _ = submit_srun(target_node)

            # 2. Get Job ID
            job_id = get_job_id(target_node)

            # If we can't get Job ID， srun failed - kill the process and retry
            if not job_id:
                log("Could not get Job ID. srun failed or timed out.")
                continue

            log(f"Job ID: {job_id} | Node: {target_node}")

            if not wait_for_job_running(job_id):
                log("Job failed to enter RUNNING state.")
                cancel_job_and_wait(job_id)
                continue

            # Job started, wait for the container to start
            time.sleep(5)

            # 3. Fetch Enroot PID
            pid = get_container_pid(target_node)
            if not pid:
                log("Failed to get enroot PID.")
                cancel_job_and_wait(job_id)
                continue

            log(f"Found container PID: {pid}")

            # 4. Check /dev/kvm permission
            if check_kvm_permission(target_node, pid):
                log("KVM check passed!")

                # 5. Execute main workload
                run_main_workload(target_node, pid, job_index)
                log("Tasks launched. Keeping srun alive locally.")

                # check if job id is still running, if is, wait for it to finish
                while wait_for_job_running(job_id):
                    time.sleep(100)

                break
            else:
                log(f"KVM check failed on {target_node}.")

                cancel_job_and_wait(job_id)

                time.sleep(5)

        if retry_count >= max_retries:
            log("Max retries reached. Exiting.")

if __name__ == "__main__":
    main()


# NOTE: check Reserved nodes
# scontrol show res sla_res_osworld_agent_vlm_cpu_only

